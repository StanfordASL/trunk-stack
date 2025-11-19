# Aggressive Optimizations: 0.095s → 0.008s (12x speedup needed!)

## Target Breakdown

To achieve **0.008s (8ms)** total solve time with N=10 horizon:

| Component | Target Time | Strategy |
|-----------|-------------|----------|
| Initial Jacobian | 2-3 ms | Cache + jacrev + batch RBF |
| Per iteration Jacobian | 2-3 ms | Same + skip recomputation |
| LOCP solve | 1-2 ms | Already fast |
| Other overhead | 1-2 ms | Reduce iterations |
| **TOTAL** | **8 ms** | **1-3 GuSTO iterations max** |

## Critical Insight

0.095s = 95ms is likely caused by:
1. **Multiple GuSTO iterations** (if 3 iters × 30ms each = 90ms)
2. **Jacobian recomputation every iteration**
3. **Still using jacfwd** instead of jacrev

## CRITICAL FIX 1: Limit GuSTO to 1-2 Iterations ⚡⚡⚡

**Why:** For real-time MPC, you DON'T need full convergence!

**File:** `stack/main/src/executor/executor/mpc_initializer_node.py`

```python
gusto_config = GuSTOConfig(
    Qz=Qz,
    Qzf=Qzf,
    R=R,
    x_char=x_char,
    f_char=f_char,
    N=10,
    H=H,
    max_gusto_iters=2,      # ← CRITICAL: Limit to 1-2 iterations!
    convg_thresh=0.1,       # ← Relax convergence (was 0.01)
    epsilon=0.1,            # ← Relax constraint tolerance (was 0.01)
)
```

**Expected impact:** If currently 3+ iterations, this alone gives 2-3x speedup!

---

## CRITICAL FIX 2: Jacobian Caching (Reuse Between MPC Calls)

**File:** `stack/main/src/controller/controller/mpc/gusto_upgrade_trunk.py`

Add to `__init__` (after LOCP creation):

```python
# Jacobian caching for warm-starting
self.cached_A = None
self.cached_B = None
self.cached_d = None
self.cached_H = None
self.cached_G = None
self.cached_c = None
self.last_x_k = None
self.last_u_k = None
```

Add this method:

```python
def _should_recompute_jacobians(self, x_new, u_new, threshold=0.05):
    """
    Check if trajectory changed enough to warrant Jacobian recomputation.

    For real-time MPC with warm-starting, trajectories don't change much
    between consecutive MPC solves, so we can reuse Jacobians.
    """
    if self.last_x_k is None:
        return True

    x_diff = jnp.max(jnp.abs(x_new - self.last_x_k))
    u_diff = jnp.max(jnp.abs(u_new - self.last_u_k))

    return (x_diff > threshold) or (u_diff > threshold)
```

Modify `solve()` method around line 544:

```python
# If valid solution, update and recompute dynamics
if new_solution:
    self.x_k = x_next.copy()
    self.u_k = u_next.copy()

    if self.max_gusto_iters >= 1:
        # Smart Jacobian recomputation
        if self._should_recompute_jacobians(self.x_k, self.u_k):
            t_iter_jac_start = time.time()
            A_d, B_d, d_d = self._get_dynamics_linearizations(self.x_k, self.u_k)
            self.cached_A, self.cached_B, self.cached_d = A_d, B_d, d_d
            self.last_x_k, self.last_u_k = self.x_k.copy(), self.u_k.copy()
            t_iter_dyn = time.time() - t_iter_jac_start

            t_iter_perf_start = time.time()
            if self.nonlinear_perf_mapping:
                H_d, G_d, c_d = self._get_perf_mapping_linearizations(self.x_k, self.u_k)
                self.cached_H, self.cached_G, self.cached_c = H_d, G_d, c_d
            else:
                H_d, G_d, c_d = None, None, None
            t_iter_perf = time.time() - t_iter_perf_start

            print(f'Iteration {itr} Jacobian RECOMPUTED:')
            print(f'  Dynamics: {t_iter_dyn*1000:.2f} ms, Performance: {t_iter_perf*1000:.2f} ms')
        else:
            # Reuse cached Jacobians (FAST!)
            A_d = self.cached_A
            B_d = self.cached_B
            d_d = self.cached_d
            H_d = self.cached_H
            G_d = self.cached_G
            c_d = self.cached_c
            print(f'Iteration {itr}: REUSING cached Jacobians (0 ms)')
```

**Expected impact:** After first MPC solve, subsequent solves can skip Jacobian computation entirely if trajectory is similar!

---

## CRITICAL FIX 3: Reduce Horizon N

**File:** `mpc_initializer_node.py`

```python
gusto_config = GuSTOConfig(
    # ... other params ...
    N=5,  # ← Reduce from 10 to 5
)
```

**Impact:**
- 2x faster Jacobian computation (linear with N)
- 2x faster LOCP solve (scales with N)
- **Total: 2x speedup**

**Trade-off:** Shorter prediction horizon (50ms instead of 100ms at dt=0.01)

---

## CRITICAL FIX 4: Use jacrev (If Not Already Applied)

**File:** `stack/main/src/executor/executor/dyn_system.py` lines 92, 97

```python
# Line 92:
A, B = jax.vmap(jax.jacrev(f, argnums=(0, 1)))(x, u)  # ← Must be jacrev!

# Line 97:
A, B = jax.jacrev(f, argnums=(0, 1))(x, u)  # ← Must be jacrev!
```

**Expected impact:** 2-3x faster Jacobian if currently using jacfwd

---

## CRITICAL FIX 5: Skip Initial Solve in GuSTO __init__

**File:** `stack/main/src/controller/controller/mpc/gusto_upgrade_trunk.py` line ~121

```python
def __init__(self, model, config, x0, u_init, x_init,
            z=None, u=None, zf=None, U=None, X=None, Xf=None, dU=None,
            start_with_solve=False,  # ← Change to False!
            exps_M=None, M=None, U_select=None, W_sm=None, epsilon_sm=None, case_rbf=None, **kwargs):
```

**Or in mpc_solver_node.py when creating GuSTO:**

```python
self.gusto = GuSTO(
    self.model,
    self.config,
    x0,
    self.u_init,
    self.x_init,
    z=z_ref_win,
    zf=z_ref_win[-1],
    U=U,
    dU=dU,
    start_with_solve=False,  # ← Set to False to skip initial solve
    exps_M=self.exps_M,
    M=self.M,
    # ... rest
)
```

**Why:** The initial solve in `__init__` is just for warming up. Skip it for faster startup.

---

## MODERATE FIX 6: Batch RBF Evaluation

Add this function in `adiabatic_ssm_trunk.py` after `rbf_eval_single_jax`:

```python
@partial(jax.jit, static_argnums=(4,))
def rbf_eval_batch_jax(u_batch, U_select, W, epsilon, case_rbf):
    """Vectorized RBF evaluation - much faster than vmap."""
    # u_batch: (N, n_u), U_select: (n_u, M), W: (M, d_prime)

    # Compute distances from each u[i] to each center
    # u_batch: (N, n_u, 1), U_select: (1, n_u, M)
    u_exp = u_batch[:, :, None]  # (N, n_u, 1)
    centers = U_select[None, :, :]  # (1, n_u, M)

    # Squared distance: (N, n_u, M) -> (N, M)
    diff_sq = (u_exp - centers) ** 2
    dist = jnp.sqrt(jnp.sum(diff_sq, axis=1))  # (N, M)

    # Apply RBF
    if case_rbf:
        A = jnp.exp(-(epsilon * dist) ** 2)
    else:
        A = dist

    # (N, M) @ (M, d_prime) = (N, d_prime)
    return A @ W
```

Then replace line 271 in `continuous_dynamics`:

```python
# OLD:
xbar_batch_full = jax.vmap(lambda ui: rbf_eval_single_jax(ui, self.U_select, self.W, self.epsilon, self.case_rbf).squeeze())(u)

# NEW:
xbar_batch_full = rbf_eval_batch_jax(u, self.U_select, self.W, self.epsilon, self.case_rbf)
```

---

## MODERATE FIX 7: Pre-allocate Arrays in LOCP

**File:** `locp_upgrade.py`

The LOCP update might be copying arrays. Ensure warm_start is True:

In `mpc_solver_node.py`:

```python
self.gusto = GuSTO(
    # ... params ...
    warm_start=True,  # ← Ensure this is True
    solver='CLARABEL',  # or 'OSQP' - try both
)
```

---

## ADVANCED FIX 8: Skip Trust Region Checks (Risky!)

**Only if desperate and willing to sacrifice robustness:**

In `gusto_upgrade_trunk.py` around line 490:

```python
# Comment out or simplify trust region check
# tr_satisfied = True  # Always assume satisfied (risky!)
```

**NOT RECOMMENDED** unless you're okay with potentially unstable solutions.

---

## Implementation Checklist (Priority Order)

### Immediate (Will get you to ~20-30ms):

- [ ] **Set max_gusto_iters=2** (biggest impact if currently >2 iterations)
- [ ] **Set convg_thresh=0.1** (faster convergence)
- [ ] **Verify jacrev is being used** (not jacfwd)
- [ ] **Set start_with_solve=False** in GuSTO creation

### High Priority (Will get you to ~10-15ms):

- [ ] **Implement Jacobian caching** (reuse between iterations)
- [ ] **Reduce N from 10 to 5-7** (2x speedup)

### If Still Needed (~5-10ms):

- [ ] **Add batched RBF evaluation**
- [ ] **Verify warm_start=True for LOCP**
- [ ] **Try different solver** (OSQP vs CLARABEL vs GUROBI)

---

## Expected Timeline

| After Fix | Time | Speedup |
|-----------|------|---------|
| Current | 95 ms | 1x |
| max_iters=2 + convg_thresh=0.1 | 40-50 ms | 2x |
| + jacrev | 20-30 ms | 3-5x |
| + Jacobian caching | 10-15 ms | 6-10x |
| + N=5 | **5-10 ms** | **10-20x** |

---

## Diagnostic Command

Before optimizing, run this to see where time is spent:

```python
# In your test script, enable verbose output:
self.gusto = GuSTO(
    # ... params ...
    verbose=2,  # ← Enable detailed logging
)

# And check:
print(f"Number of GuSTO iterations: {itr}")
print(f"Average time per iteration: {t_gusto/itr:.3f}s")
```

---

## Nuclear Option: Reduce to N=3-5 + max_iters=1

If you absolutely MUST hit 8ms and nothing else works:

```python
gusto_config = GuSTOConfig(
    # ... other params ...
    N=3,  # ← Extremely short horizon
    max_gusto_iters=1,  # ← Single SQP iteration
    convg_thresh=1.0,  # ← Don't check convergence
)
```

This will give you **~5-8ms** but with:
- Very short prediction (30ms lookahead)
- No SQP convergence guarantee
- Potentially suboptimal controls

**Only use if real-time performance is more critical than optimality!**

---

## Summary

The key insight: **You don't need full GuSTO convergence for real-time MPC!**

1. Limit to 1-2 SQP iterations
2. Cache Jacobians between solves
3. Use jacrev
4. Consider reducing N

This should get you from **95ms → 8-15ms** which is close to your 8ms target!
