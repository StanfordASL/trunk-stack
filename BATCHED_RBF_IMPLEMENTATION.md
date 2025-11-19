# Batched RBF Implementation for Performance

## Problem Identified

Both `adiabatic_ssm_trunk.py` and `gusto_upgrade_trunk.py` use:
```python
jax.vmap(lambda ui: rbf_eval_single_jax(ui, ...))(u)
```

This creates **N separate RBF evaluations** instead of one batched operation.

For N=10: 10 small matrix multiplies instead of 1 large one.

---

## Solution: Add Batched RBF Function

### Step 1: Add to `adiabatic_ssm_trunk.py`

Add this function after `rbf_eval_single_jax` (around line 150):

```python
@jax.jit
def rbf_eval_batch_jax(u_batch, U_select, W, epsilon, case_rbf):
    """
    Batched JAX-compatible RBF evaluation.
    Evaluates RBF for multiple points at once - much faster than vmap.

    Args:
        u_batch: (N, d) - batch of evaluation points
        U_select: (d, M) - each column is a center
        W: (M, d_prime) - weights
        epsilon: scalar - shape parameter (unused in current implementation)
        case_rbf: bool - RBF type flag (unused in current implementation)

    Returns:
        final_value: (N, d_prime) - batch of RBF evaluations
    """
    # Ensure u_batch is 2D
    if u_batch.ndim == 1:
        u_batch = u_batch[None, :]  # (d,) -> (1, d)

    # u_batch: (N, d), U_select: (d, M)
    # We need distances from each u[i] to each center

    # Expand dimensions for broadcasting
    u_expanded = u_batch[:, :, None]  # (N, d, 1)
    centers_expanded = U_select[None, :, :]  # (1, d, M)

    # Compute squared differences
    # (N, d, 1) - (1, d, M) -> (N, d, M)
    diff = u_expanded - centers_expanded

    # Sum over dimension axis and take sqrt
    # (N, d, M) -> (N, M)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=1))

    # RBF kernel (currently just distance)
    A = dist  # (N, M)

    # Single batched matrix multiply
    # (N, M) @ (M, d_prime) = (N, d_prime)
    result = A @ W

    return result
```

### Step 2: Update `continuous_dynamics` in `adiabatic_ssm_trunk.py`

Replace line 271:
```python
# OLD (line 271):
xbar_batch_full = jax.vmap(lambda ui: rbf_eval_single_jax(ui, self.U_select, self.W, self.epsilon, self.case_rbf).squeeze())(u)

# NEW:
xbar_batch_full = rbf_eval_batch_jax(u, self.U_select, self.W, self.epsilon, self.case_rbf)
```

Also update the single-sample case (line 253) to use batched version:
```python
# OLD (line 253):
xbar_full = rbf_eval_single_jax(u, self.U_select, self.W, self.epsilon, self.case_rbf)

# NEW (convert to batch of 1, then extract):
xbar_full = rbf_eval_batch_jax(u[None, :], self.U_select, self.W, self.epsilon, self.case_rbf)[0]
```

---

### Step 3: Add to `gusto_upgrade_trunk.py`

Add this method to the `GuSTO` class (after `rbf_eval_single_jax`, around line 204):

```python
@partial(jax.jit, static_argnums=(0,))
def rbf_eval_batch_jax(self, u_batch):
    """
    Batched RBF evaluation using self.U_select and self.W_sm.
    Much faster than vmap over rbf_eval_single_jax.

    Args:
        u_batch: (N, n_u) - batch of control inputs

    Returns:
        (N, d_prime) - batch of RBF evaluations
    """
    # Ensure 2D
    if u_batch.ndim == 1:
        u_batch = u_batch[None, :]

    # u_batch: (N, n_u), U_select: (n_u, M)
    # Expand for broadcasting
    u_expanded = u_batch[:, :, None]  # (N, n_u, 1)
    centers_expanded = self.U_select[None, :, :]  # (1, n_u, M)

    # Compute distances
    diff = u_expanded - centers_expanded  # (N, n_u, M)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=1))  # (N, M)

    # RBF kernel
    A = dist  # (N, M)

    # Batched matrix multiply
    result = A @ self.W_sm  # (N, M) @ (M, d_prime) = (N, d_prime)

    return result
```

### Step 4: Update `performance_mapping` in `gusto_upgrade_trunk.py`

Replace lines 296-299:
```python
# OLD (lines 296-299):
xs_batch_full = jax.vmap(
    lambda ui: jnp.ravel(
        self.rbf_eval_single_jax(ui)
    ))(u)

# NEW:
xs_batch_full = self.rbf_eval_batch_jax(u)
```

---

## Expected Performance Impact

### Before (vmap approach):
- N separate RBF calls
- Each: small (1, M) @ (M, d_prime) matmul
- Total: N × matmul_cost
- **For N=10, M=50, d_prime=6: ~5-8ms**

### After (batched approach):
- Single RBF call
- One large (N, M) @ (M, d_prime) matmul
- Better cache utilization
- XLA can optimize single large operation
- **For same dimensions: ~2-3ms** (2-3× faster)

### Impact on Total Solve Time:
- Dynamics linearization: 40ms → 35-37ms (saves 3-5ms)
- Performance linearization: already fast, minimal impact
- **Total GuSTO: 95ms → 88-92ms**

---

## Why This Works

1. **Fewer function calls:** 1 instead of N
2. **Better vectorization:** Single large matmul vs N small ones
3. **Cache efficiency:** Sequential memory access
4. **XLA optimization:** Easier to optimize one op than N ops
5. **Less JAX overhead:** No vmap dispatching overhead

---

## Additional Optimization: Fuse RBF + Monomial

If you want to go further, you could fuse the RBF evaluation with the subsequent monomial evaluation:

```python
@partial(jax.jit, static_argnums=(0,))
def fused_rbf_and_monomial(self, x, u):
    """
    Fuse RBF evaluation and monomial evaluation into single operation.
    Reduces intermediate array allocations.
    """
    # Batched RBF
    xbar_batch = self.rbf_eval_batch_jax(u)[:, :3]

    # Concatenate and evaluate monomials in one go
    x_inputs = jnp.concatenate([x, xbar_batch], axis=1)

    # Batched monomial evaluation
    xr_batch = jax.vmap(lambda xi: self.Rd @ eval_aSSM_exps(xi, self.exps_Rd))(x_inputs)

    return xr_batch.squeeze()
```

This could save another 1-2ms by reducing intermediate arrays.

---

## Summary

**Changes needed:**

1. Add `rbf_eval_batch_jax` function to `adiabatic_ssm_trunk.py`
2. Replace vmap call in `continuous_dynamics` (line 271)
3. Add `rbf_eval_batch_jax` method to `GuSTO` class
4. Replace vmap call in `performance_mapping` (lines 296-299)

**Expected speedup:**
- 3-5ms per Jacobian computation
- ~6-10ms total per MPC solve

**Next step after this:**
- Implement Jacobian caching/reuse for another 30-40ms savings
- Together: 95ms → 50-60ms (or better with caching)
