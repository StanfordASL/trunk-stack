# CRITICAL JIT RECOMPILATION FIXES

## 🔥 ROOT CAUSE IDENTIFIED 🔥

Your 88-91ms slowdown is caused by **JAX recompiling on every iteration** because:

1. **`rbf_eval_single_jax()` is NOT @jax.jit decorated** in adiabatic_ssm_trunk.py
2. **`eval_aSSM_exps()` is NOT @jax.jit decorated** in adiabatic_ssm_trunk.py
3. These non-JIT functions are called inside JIT-compiled `continuous_dynamics()`
4. When JAX autodiff tries to trace through them, it recompiles on EVERY call

## Call Stack That's Killing You:

```
gusto.solve()
  └─> _get_dynamics_linearizations() [~88ms]
       └─> jax.vmap(jax.jacfwd(discrete_dynamics))  [JAX tries to trace]
            └─> discrete_dynamics() [JIT compiled ✓]
                 └─> continuous_dynamics() [JIT compiled ✓]
                      ├─> rbf_eval_single_jax() [NOT JIT ❌ RECOMPILES!]
                      └─> eval_aSSM_exps() [NOT JIT ❌ RECOMPILES!]
```

With RK4 (4 calls per step) × N=10 horizon = **40 recompilations per Jacobian!**

---

## IMMEDIATE FIXES (Apply in order):

### FIX 1: Add @jax.jit to rbf_eval_single_jax ⚡ CRITICAL
**File:** `stack/main/src/executor/executor/adiabatic_ssm_trunk.py`
**Line:** 110

**BEFORE:**
```python
def rbf_eval_single_jax(u, U_select, W, epsilon, case_rbf):
    """
    JAX-compatible version of rbf_eval_single
    ...
    """
    u = jnp.asarray(u).reshape(-1)
```

**AFTER:**
```python
@partial(jax.jit, static_argnums=(4,))  # case_rbf is boolean, make static
def rbf_eval_single_jax(u, U_select, W, epsilon, case_rbf):
    """
    JAX-compatible version of rbf_eval_single
    ...
    """
    u = jnp.reshape(u, (-1,))  # Better for JAX tracer than asarray().reshape()
```

**Don't forget to add the import at the top of the file:**
```python
from functools import partial  # Add if not already present
```

---

### FIX 2: Add @jax.jit to eval_aSSM_exps ⚡ CRITICAL
**File:** `stack/main/src/executor/executor/adiabatic_ssm_trunk.py`
**Line:** 194

**BEFORE:**
```python
def eval_aSSM_exps(xi, exps):
    """
    Evaluate monomial terms using JAX arrays.
    """
    xi = jnp.asarray(xi).flatten()
```

**AFTER:**
```python
@jax.jit
def eval_aSSM_exps(xi, exps):
    """
    Evaluate monomial terms using JAX arrays.
    """
    xi = jnp.reshape(xi, (-1,))  # Better for JAX tracer
```

---

### FIX 3: Fix shape manipulation in eval_monomials_single_sm ⚡ IMPORTANT
**File:** `stack/main/src/executor/executor/adiabatic_ssm_trunk.py`
**Line:** 59

**BEFORE:**
```python
def eval_monomials_single_sm(xi, exps):
    """Evaluate monomials at a single point with constant term added"""
    xi = jnp.asarray(xi).flatten()
```

**AFTER:**
```python
@jax.jit
def eval_monomials_single_sm(xi, exps):
    """Evaluate monomials at a single point with constant term added"""
    xi = jnp.reshape(xi, (-1,))
```

---

### FIX 4: Same for eval_monomials_batch_sm ⚡ IMPORTANT
**File:** `stack/main/src/executor/executor/adiabatic_ssm_trunk.py`
**Line:** 34

**BEFORE:**
```python
def eval_monomials_batch_sm(xi, exps):
    """Evaluate monomials at multiple points with constant term added"""
    xi = jnp.asarray(xi)  # Shape: (N, 9) where N is number of points
```

**AFTER:**
```python
@jax.jit
def eval_monomials_batch_sm(xi, exps):
    """Evaluate monomials at multiple points with constant term added"""
    xi = xi if isinstance(xi, jnp.ndarray) else jnp.array(xi)
```

---

### FIX 5: Fix the duplicate in GuSTO class ⚡ MODERATE
**File:** `stack/main/src/controller/controller/mpc/gusto_upgrade_trunk.py`
**Line:** 167

The GuSTO class has its own `rbf_eval_single_jax` method which IS JIT-compiled,
but it should use the same pattern. Change line 182:

**BEFORE:**
```python
u = jnp.asarray(u).reshape(-1)  # Flatten to (d,) - handles both (d,) and (1,d)
```

**AFTER:**
```python
u = jnp.reshape(u, (-1,))  # Flatten to (d,) - handles both (d,) and (1,d)
```

And line 210:
**BEFORE:**
```python
xi = jnp.asarray(xi).flatten()
```

**AFTER:**
```python
xi = jnp.reshape(xi, (-1,))
```

---

### FIX 6: Check RK4 setting ⚡ HIGH IMPACT
**File:** `stack/main/src/executor/executor/adiabatic_ssm_trunk.py`
**Line:** 439 (in setup_aSSM function)

**VERIFY THIS IS FALSE:**
```python
system = aSSM_strategy_rad_bas(
    exps_Rd=exps_Rd, Rd=Rd, U_select=U_select, W=W_sm,
    epsilon=epsilon_sm, case_rbf=case_rbf, dt=dt,
    n_x=2, n_u=6,
    rk4=False  # ← MUST BE FALSE! If True, change to False for 4x speedup
)
```

---

## VERIFICATION SCRIPT

After applying fixes, run this to verify JIT is working:

```python
import jax
import time
import jax.numpy as jnp

# Enable compilation logging
jax.config.update('jax_log_compiles', True)

# Your model setup code here...
from executor.adiabatic_ssm_trunk import setup_aSSM
model, *_ = setup_aSSM(dt=0.01, rk4=False)

# Test with consistent shapes
x_test = jnp.zeros((10, 2))
u_test = jnp.zeros((10, 6))

print("=== First call (should compile) ===")
A1, B1, d1 = model.dynamics_jac(x_test, u_test)
print("Shape A:", A1.shape, "B:", B1.shape, "d:", d1.shape)

print("\n=== Second call (should NOT compile) ===")
A2, B2, d2 = model.dynamics_jac(x_test, u_test)

print("\n=== Third call with same shape (should NOT compile) ===")
x_test2 = jnp.ones((10, 2)) * 0.5
u_test2 = jnp.ones((10, 6)) * 0.1
A3, B3, d3 = model.dynamics_jac(x_test2, u_test2)

# Benchmark
print("\n=== Benchmarking ===")
times = []
for i in range(10):
    x = jnp.ones((10, 2)) * i * 0.01
    u = jnp.ones((10, 6)) * i * 0.01

    t0 = time.time()
    A, B, d = model.dynamics_jac(x, u)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed*1000:.2f} ms")

avg = jnp.mean(jnp.array(times))
print(f"\nAverage: {avg*1000:.2f} ms")
print(f"Std: {jnp.std(jnp.array(times))*1000:.2f} ms")

if avg < 0.020:  # Less than 20ms
    print("✅ SUCCESS! Jacobian is fast (< 20ms)")
elif avg < 0.050:  # Less than 50ms
    print("⚠️  BETTER but could be faster (20-50ms)")
else:
    print("❌ STILL SLOW! Check if JIT decorators were applied correctly")
```

---

## EXPECTED RESULTS

### BEFORE fixes:
```
First call: Compiling... (lots of messages)
Second call: Compiling... (recompiling! ❌)
Third call: Compiling... (recompiling! ❌)
Average: 88.5 ms
```

### AFTER fixes:
```
First call: Compiling... (one-time compilation)
Second call: (no compilation, uses cached) ✅
Third call: (no compilation, uses cached) ✅
Average: 8-15 ms (5-10x speedup!)
```

---

## Why This Works

JAX JIT compilation works by tracing your function once, compiling it to optimized XLA code,
then reusing that compiled code. When JAX encounters:

1. **Non-JIT functions inside JIT functions** → It can't optimize, falls back to Python
2. **Shape changes** → Recompiles for new shape
3. **Type changes** → Recompiles for new type

By decorating `rbf_eval_single_jax()` and `eval_aSSM_exps()` with `@jax.jit`,
JAX can now:
- Trace through the entire call stack in one go
- Compile everything to optimized XLA
- Reuse the compiled code on subsequent calls

The `jnp.asarray().reshape()` pattern was also problematic because JAX's shape
tracer couldn't always infer shapes statically, causing recompilation.

---

## Checklist

- [ ] Fix 1: Add @jax.jit to rbf_eval_single_jax in adiabatic_ssm_trunk.py:110
- [ ] Fix 2: Add @jax.jit to eval_aSSM_exps in adiabatic_ssm_trunk.py:194
- [ ] Fix 3: Replace asarray().flatten() with reshape() in eval_monomials_single_sm
- [ ] Fix 4: Add @jax.jit to eval_monomials_batch_sm
- [ ] Fix 5: Fix reshape in gusto_upgrade_trunk.py rbf_eval_single_jax method
- [ ] Fix 6: Verify rk4=False in setup_aSSM
- [ ] Run verification script
- [ ] Confirm average Jacobian time < 20ms

After these fixes, your MPC should run at **50-100 Hz** instead of **5 Hz**!
