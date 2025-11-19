# MPC Performance Optimization Guide

## Problem Summary
Your GuSTO iterations take ~200ms, but LOCP QP solve only takes ~1ms.
**The bottleneck is Jacobian computation (88-91ms for dynamics, negligible for performance).**

## Root Cause Analysis

From your timing output:
- Dynamics linearization: **87.96 ms** (first) / **91.25 ms** (iteration 1)
- Performance linearization: **0.03 ms** (first) / **0.05 ms** (iteration 1)
- LOCP solve: **1 ms**
- **Total GuSTO iteration: 204 ms**

The dynamics Jacobian via JAX autodiff is recomputing the entire forward pass + derivatives for N=10 timesteps.

## Optimization Strategies (Ranked by Impact)

### 🔥 CRITICAL - Strategy 1: Cache & Reuse Jacobians (Expected speedup: 5-10x)

**Problem:** You're recomputing Jacobians at every GuSTO iteration even when the trajectory barely changes.

**Solution:** Only recompute Jacobians when the trajectory changes significantly.

```python
# Add to GuSTO.__init__
self.cached_A = None
self.cached_B = None
self.cached_d = None
self.cached_x = None
self.cached_u = None
self.jac_recompute_thresh = 1e-3  # Threshold for recomputation

def _should_recompute_jacobians(self, x_new, u_new):
    """Check if Jacobians need recomputation."""
    if self.cached_x is None:
        return True

    x_diff = jnp.max(jnp.abs(x_new - self.cached_x))
    u_diff = jnp.max(jnp.abs(u_new - self.cached_u))

    return (x_diff > self.jac_recompute_thresh) or (u_diff > self.jac_recompute_thresh)

# Modify solve() method
if new_solution:
    self.x_k = x_next.copy()
    self.u_k = u_next.copy()

    # Only recompute if trajectory changed enough
    if self._should_recompute_jacobians(self.x_k, self.u_k):
        A_d, B_d, d_d = self._get_dynamics_linearizations(self.x_k, self.u_k)
        self.cached_A, self.cached_B, self.cached_d = A_d, B_d, d_d
        self.cached_x, self.cached_u = self.x_k.copy(), self.u_k.copy()
    else:
        # Reuse cached Jacobians
        A_d, B_d, d_d = self.cached_A, self.cached_B, self.cached_d
```

### 🔥 CRITICAL - Strategy 2: Use jacrev instead of jacfwd (Expected speedup: 2-3x)

**Problem:** `jacfwd` computes one forward pass per input dimension. For dynamics with n_x=2, n_u=6, this means 8 forward passes.

**Solution:** Use `jacrev` (reverse-mode AD) which is faster when output_dim < input_dim.

```python
# In _perform_dynamics_linearization
A, B = jax.jacrev(f, argnums=(0, 1))(x, u)  # Instead of jacfwd
```

### 🔥 IMPORTANT - Strategy 3: Reduce Horizon N (Expected speedup: Linear with N)

**Problem:** You have N=10 timesteps, each requiring Jacobian computation.

**Solution:** Test with smaller horizon first.

```python
# In your config
gusto_config = GuSTOConfig(
    # ... other params ...
    N=5,  # Try reducing from 10 to 5
)
```

**Trade-off:** Smaller horizon = less optimal trajectory, but 2x faster.

### Strategy 4: Pre-JIT Compile Critical Functions

**Problem:** First call to JIT functions includes compilation time.

**Solution:** Add warm-up calls in `__init__`:

```python
# In GuSTO.__init__, after creating LOCP
if start_with_solve:
    # Warm-up: JIT compile critical functions
    dummy_x = jnp.zeros((self.N, self.n_x))
    dummy_u = jnp.zeros((self.N, self.n_u))

    # Force compilation
    _ = self._perform_dynamics_linearization(dummy_x[0], dummy_u[0])
    _ = self._perform_perf_mapping_linearization(dummy_x[0], dummy_u[0])

    print('JIT compilation complete.')
    print('First solve may take a while due to factorization and caching.')
```

### Strategy 5: Optimize RBF Evaluation

**Problem:** Your RBF evaluation in `performance_mapping` is called repeatedly.

**Current:**
```python
xs_batch_full = jax.vmap(
    lambda ui: jnp.ravel(self.rbf_eval_single_jax(ui))
)(u)
```

**Optimized - Batch all at once:**
```python
@partial(jax.jit, static_argnums=(0,))
def rbf_eval_batch_jax(self, u_batch):
    """Vectorized RBF evaluation for entire batch."""
    # u_batch: (N, n_u)
    # U_select: (d, M) - centers
    # W_sm: (M, d_prime) - weights

    # Compute all distances at once
    # u_batch: (N, d) needs to be compared to U_select.T: (M, d)
    u_expanded = u_batch[:, :, None]  # (N, d, 1)
    centers = self.U_select.T[None, :, :]  # (1, M, d)

    diff = u_expanded - centers  # (N, 1, d) - (1, M, d) = (N, M, d)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=2))  # (N, M)

    # A: (N, M), W_sm: (M, d_prime) -> result: (N, d_prime)
    result = dist @ self.W_sm

    return result

# Then in performance_mapping:
xs_batch_full = self.rbf_eval_batch_jax(u)  # Single call instead of vmap
```

### Strategy 6: Simplify Dynamics Model

**Problem:** Your aSSM model has expensive monomial evaluations.

**Check:** Is RK4 integration necessary?

```python
# In adiabatic_ssm_trunk.py setup
system = aSSM_strategy_rad_bas(..., rk4=False)  # Try Euler instead
```

RK4 requires 4 function evaluations per timestep. Euler only needs 1.

### Strategy 7: Reduce GuSTO Convergence Iterations

**Problem:** You might be over-converging.

**Solution:**
```python
gusto_config = GuSTOConfig(
    # ... other params ...
    max_gusto_iters=3,  # Reduce from 500
    convg_thresh=0.05,  # Increase from 0.01 (less strict)
)
```

For real-time MPC, 1-3 SQP iterations is often sufficient.

## Implementation Priority

### Immediate Actions (Implement Now):

1. **Switch to jacrev** (5 min, 2-3x speedup)
2. **Reduce N from 10 to 5-7** (1 min, 2x speedup)
3. **Limit max_gusto_iters to 3** (1 min, reduces worst-case time)
4. **Switch from RK4 to Euler** (1 min, 4x speedup if acceptable)

### Medium Priority (This week):

5. **Implement Jacobian caching** (30 min, 5-10x speedup for steady-state)
6. **Batch RBF evaluation** (20 min, 2x speedup)

### Advanced (If still too slow):

7. **Analytical Jacobians** - Derive analytical derivatives for your aSSM model
8. **Model Reduction** - Simplify the monomial terms
9. **Parallel Processing** - Use JAX `pmap` for multi-core

## Quick Test Script

Add this to test performance:

```python
import time
import jax.numpy as jnp

# Test Jacobian computation
x_test = jnp.zeros((10, 2))
u_test = jnp.zeros((10, 6))

# Warm-up
_ = gusto._get_dynamics_linearizations(x_test, u_test)

# Benchmark
n_runs = 10
times = []
for _ in range(n_runs):
    t0 = time.time()
    _ = gusto._get_dynamics_linearizations(x_test, u_test)
    times.append(time.time() - t0)

print(f"Average Jacobian time: {jnp.mean(jnp.array(times))*1000:.2f} ms")
print(f"Std: {jnp.std(jnp.array(times))*1000:.2f} ms")
```

## Expected Results

With all optimizations:
- **Before:** ~200 ms per GuSTO iteration
- **After (conservative):** ~20-40 ms per GuSTO iteration
- **After (optimistic):** ~5-10 ms per GuSTO iteration

This would bring you to **50-100 Hz MPC update rate**, suitable for real-time control.

## Debugging Tips

1. Add timing to individual components:
```python
t0 = time.time()
A, B = jax.jacfwd(f, argnums=(0, 1))(x, u)
print(f"jacfwd time: {(time.time()-t0)*1000:.2f} ms")

t0 = time.time()
d = f(x, u)
print(f"forward pass time: {(time.time()-t0)*1000:.2f} ms")
```

2. Check if recompilation is happening:
```python
# JAX will print compilation messages if you enable logging
import jax
jax.config.update('jax_log_compiles', True)
```

3. Profile with JAX profiler:
```python
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_trace=True):
    gusto.solve(...)
# Then open chrome://tracing and load the trace
```
