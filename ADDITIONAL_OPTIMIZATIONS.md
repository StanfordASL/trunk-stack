# Additional Optimizations to Reduce 42ms → ~10ms

## Current Bottleneck Analysis

You've improved from **88ms → 42ms** (2x speedup) by adding JIT decorators.

The remaining 42ms is caused by:

1. **Line 92 in dyn_system.py**: Using `jacfwd` instead of `jacrev`
2. **Line 271 in adiabatic_ssm_trunk.py**: Using vmap on individual RBF calls instead of batched computation
3. **Branching logic** in `continuous_dynamics` (lines 251-261 vs 263-283)
4. **Multiple vmap calls** that could be fused
5. **RK4 integration** (if enabled) multiplies cost by 4x

---

## Optimization 1: Switch jacfwd to jacrev ⚡ HIGH IMPACT

**File:** `stack/main/src/executor/executor/dyn_system.py`
**Line:** 92

**CURRENT:**
```python
# Compute Jacobians using JAX automatic differentiation
A, B = jax.vmap(jax.jacfwd(f, argnums=(0, 1)))(x, u)
```

**OPTIMIZED:**
```python
# Compute Jacobians using JAX automatic differentiation
# jacrev is faster when output_dim < input_dim (typical for dynamics)
A, B = jax.vmap(jax.jacrev(f, argnums=(0, 1)))(x, u)
```

**Expected speedup:** 2-3x (42ms → 14-21ms)

**Why this works:**
- Your system: `n_x=2` (state dim), `n_u=6` (control dim), output dim = 2
- `jacfwd`: Computes one forward pass per input dimension = 8 forward passes
- `jacrev`: Computes one backward pass per output dimension = 2 backward passes
- Ratio: 8/2 = 4x theoretical speedup (but overhead means ~2-3x in practice)

---

## Optimization 2: Batch RBF Evaluation ⚡ MODERATE IMPACT

**File:** `stack/main/src/executor/executor/adiabatic_ssm_trunk.py`
**Line:** 271

**CURRENT (slow):**
```python
xbar_batch_full = jax.vmap(lambda ui: rbf_eval_single_jax(ui, self.U_select, self.W, self.epsilon, self.case_rbf).squeeze())(u)
```

**OPTIMIZED:**
Add this new function after `rbf_eval_single_jax`:

```python
@partial(jax.jit, static_argnums=(4,))
def rbf_eval_batch_jax(u_batch, U_select, W, epsilon, case_rbf):
    """
    Vectorized RBF evaluation for entire batch at once.
    Much faster than vmap over individual evaluations.

    Args:
        u_batch: (N, n_u) batch of control inputs
        U_select: (n_u, M) centers
        W: (M, d_prime) weights
        epsilon: shape parameter
        case_rbf: boolean for RBF type

    Returns:
        (N, d_prime) RBF evaluations
    """
    # u_batch: (N, n_u), U_select: (n_u, M)
    # We need distances from each u[i] to each center

    # Expand for broadcasting
    u_expanded = u_batch[:, :, None]  # (N, n_u, 1)
    centers_expanded = U_select[None, :, :]  # (1, n_u, M)

    # Compute squared differences
    diff = u_expanded - centers_expanded  # (N, n_u, M)
    dist_sq = jnp.sum(diff ** 2, axis=1)  # (N, M)
    dist = jnp.sqrt(dist_sq)  # (N, M)

    # Apply RBF
    if case_rbf:
        A = jnp.exp(-(epsilon * dist)**2)  # (N, M)
    else:
        A = dist  # (N, M)

    # Matrix multiply: (N, M) @ (M, d_prime) = (N, d_prime)
    result = A @ W

    return result
```

**Then replace line 271 with:**
```python
xbar_batch_full = rbf_eval_batch_jax(u, self.U_select, self.W, self.epsilon, self.case_rbf)
```

**Expected speedup:** 1.5-2x on top of jacrev change

---

## Optimization 3: Simplify continuous_dynamics branching ⚡ MODERATE

**File:** `stack/main/src/executor/executor/adiabatic_ssm_trunk.py`
**Lines:** 241-283

**Issue:** The `if x.ndim == 1:` branching can cause JAX to compile two separate code paths.

**SOLUTION A - Remove branching (recommended):**
Always ensure inputs are batched, even for single samples:

```python
def continuous_dynamics(self, x, u):
    """
    Continuous-time dynamics f(x,u), broadcasted over the leading dimension.
    Always operates on batched inputs for consistency.
    """
    # Ensure inputs are batched
    if x.ndim == 1:
        x = x[None, :]  # (n_x,) -> (1, n_x)
        u = u[None, :]  # (n_u,) -> (1, n_u)
        squeeze_output = True
    else:
        squeeze_output = False

    # Batch processing (same for all cases)
    xbar_batch_full = rbf_eval_batch_jax(u, self.U_select, self.W, self.epsilon, self.case_rbf)
    xbar_batch = xbar_batch_full[:, :3]

    # Prepare batched inputs
    x_inputs = jnp.concatenate([x, xbar_batch], axis=1)  # shape (batch_size, 8)

    # Apply to all batch elements using vmap
    xr_batch = jax.vmap(lambda xi: self.Rd @ eval_aSSM_exps(xi, self.exps_Rd))(x_inputs)

    # Return shape (batch_size, output_dim) or squeeze if needed
    if squeeze_output:
        return xr_batch.squeeze(axis=0)
    else:
        return xr_batch.squeeze()
```

**SOLUTION B - Use lax.cond (more complex):**
Only if Solution A doesn't work well.

---

## Optimization 4: JIT-compile dynamics_jac ⚡ LOW-MODERATE

**File:** `stack/main/src/executor/executor/dyn_system.py`
**Line:** 80

**Add JIT decorator:**
```python
@partial(jax.jit, static_argnums=(0, 2))  # self and continuous are static
def dynamics_jac(self, x: Union[np.ndarray, jnp.ndarray], u: Union[np.ndarray, jnp.ndarray], continuous=False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Get linearized discrete dynamics matrices A_list, B_list, d_list around several state, action pairs.
    Can be overridden for analytical derivatives.
    """
    if continuous:
        f = partial(self.continuous_dynamics)
    else:
        f = partial(self.discrete_dynamics)

    if len(x.shape) == 2:
        # Compute Jacobians using JAX automatic differentiation
        A, B = jax.vmap(jax.jacrev(f, argnums=(0, 1)))(x, u)  # <-- Also change to jacrev

        # Compute affine terms
        d = jax.vmap(f)(x, u) - jnp.einsum('ijk,ik->ij', A, x) - jnp.einsum('ijk,ik->ij', B, u)
    else:
        A, B = jax.jacrev(f, argnums=(0, 1))(x, u)  # <-- Also change to jacrev
        d = f(x, u) - A @ x - B @ u

    return A, B, d
```

**Note:** This might not work if `dynamics_jac` is a method. You may need to move it to a standalone function or carefully handle `self`.

---

## Optimization 5: Reduce Horizon N (if acceptable) ⚡ LINEAR IMPACT

**File:** `stack/main/src/executor/executor/mpc_initializer_node.py`
**Around line 329**

**CURRENT:**
```python
gusto_config = GuSTOConfig(
    # ...
    N=10,  # Current horizon
)
```

**OPTIMIZED:**
```python
gusto_config = GuSTOConfig(
    # ...
    N=5,  # Reduced horizon for faster computation
    max_gusto_iters=3,  # Also limit iterations
)
```

**Expected speedup:** 2x (since Jacobian is computed for N timesteps)

**Trade-off:** Shorter prediction horizon may reduce control quality

---

## Optimization 6: Verify RK4 is Disabled ⚡ CRITICAL CHECK

**File:** `stack/main/src/executor/executor/adiabatic_ssm_trunk.py`
**Line:** 439

Make sure this is `rk4=False`:
```python
system = aSSM_strategy_rad_bas(
    exps_Rd=exps_Rd, Rd=Rd, U_select=U_select, W=W_sm,
    epsilon=epsilon_sm, case_rbf=case_rbf, dt=dt,
    n_x=2, n_u=6,
    rk4=False  # ← VERIFY THIS IS FALSE!
)
```

If it's `True`, change to `False` for **4x speedup**.

---

## Implementation Priority

### Immediate (5 min, biggest impact):

1. ✅ **Change jacfwd to jacrev** in `dyn_system.py:92`
   - Expected: 42ms → 14-21ms

2. ✅ **Verify rk4=False** in `adiabatic_ssm_trunk.py:439`
   - If True, change to False for 4x speedup

### High Priority (15 min):

3. ✅ **Add rbf_eval_batch_jax** and use it in `continuous_dynamics`
   - Expected: Additional 1.5-2x speedup

4. ✅ **Simplify continuous_dynamics** to remove branching
   - Expected: Small speedup + cleaner code

### Medium Priority (if still needed):

5. **Reduce N from 10 to 5-7** in `mpc_initializer_node.py`
   - Expected: 2x speedup (linear with N)

6. **Add max_gusto_iters=3** to limit SQP iterations
   - Expected: Faster convergence for real-time control

---

## Expected Results

### After jacfwd → jacrev only:
- Dynamics linearization: **14-21 ms** (down from 42ms)
- Total GuSTO iteration: **~30-40 ms**

### After all optimizations:
- Dynamics linearization: **5-10 ms**
- Total GuSTO iteration: **10-20 ms**
- **MPC frequency: 50-100 Hz** ✅

---

## Quick Test Script

```python
import time
import jax.numpy as jnp
from executor.adiabatic_ssm_trunk import setup_aSSM

# Setup
model, *_ = setup_aSSM(dt=0.01, rk4=False)

# Test data
x_test = jnp.zeros((10, 2))
u_test = jnp.zeros((10, 6))

# Warm-up
_ = model.dynamics_jac(x_test, u_test)

# Benchmark
times = []
for i in range(20):
    x = jnp.ones((10, 2)) * i * 0.01
    u = jnp.ones((10, 6)) * i * 0.01

    t0 = time.time()
    A, B, d = model.dynamics_jac(x, u)
    elapsed = time.time() - t0
    times.append(elapsed)

    if i < 5 or i > 15:  # Print first and last few
        print(f"Run {i+1}: {elapsed*1000:.2f} ms")

avg = jnp.mean(jnp.array(times))
std = jnp.std(jnp.array(times))
print(f"\n{'='*40}")
print(f"Average: {avg*1000:.2f} ms")
print(f"Std: {std*1000:.2f} ms")
print(f"Expected MPC freq: {1.0/(avg*3):.1f} Hz")
print(f"{'='*40}")

if avg < 0.015:
    print("✅ EXCELLENT! Under 15ms")
elif avg < 0.030:
    print("✅ GOOD! Under 30ms")
elif avg < 0.050:
    print("⚠️  OK but could be better (30-50ms)")
else:
    print("❌ STILL TOO SLOW")
```

---

## Summary

The main remaining issue is **using `jacfwd` instead of `jacrev`**. This single change should give you **2-3x speedup**.

Combined with the batched RBF evaluation, you should reach **~10-15ms per Jacobian** computation, enabling **real-time MPC at 50-100 Hz**.
