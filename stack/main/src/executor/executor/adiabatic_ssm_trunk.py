import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.mplot3d import Axes3D  # Add this import at the top
from functools import partial
import os
import pandas as pd

from scipy.spatial.distance import cdist

from .dyn_system import System

@jax.jit
def eval_monomials_batch_sm(xi, exps):
    """Evaluate monomials at multiple points with constant term added"""
    xi = jnp.asarray(xi)  # Shape: (N, 9) where N is number of points
    
    # Ensure xi is 2D
    if xi.ndim == 1:
        xi = xi.reshape(1, -1)
    
    N, n_dims = xi.shape
    x = xi.reshape(N, n_dims, 1)  # Shape: (N, 9, 1)
    
    # Add dimension to exps for broadcasting
    exps_expanded = exps[:, :, None]  # Shape: (219, 9, 1)
    
    # Broadcasting: (N, 9, 1) with (219, 9, 1) -> (N, 219, 9, 1)
    # We need to reshape for proper broadcasting
    x_expanded = x[:, None, :, :]  # Shape: (N, 1, 9, 1)
    exps_broadcast = exps_expanded[None, :, :, :]  # Shape: (1, 219, 9, 1)
    
    powered = x_expanded ** exps_broadcast  # Shape: (N, 219, 9, 1)
    u = jnp.prod(powered, axis=2)  # Product over dimension axis, Shape: (N, 219, 1)
    u = u.squeeze(axis=-1)  # Shape: (N, 219)
    
    # Add constant term (1) for each point
    ones_term = jnp.ones((N, 1))  # Shape: (N, 1)
    u = jnp.concatenate([u, ones_term], axis=1)  # Shape: (N, 220)
    
    return u

@jax.jit
def eval_monomials_single_sm(xi, exps):
    """Evaluate monomials at a single point with constant term added"""
    xi = jnp.asarray(xi).flatten()
    x = xi.reshape(1, -1, 1)  # Shape: (1, 9, 1)
    
    # Add dimension to exps for broadcasting
    exps_expanded = exps[:, :, None]  # Shape: (219, 9, 1)
    
    powered = x ** exps_expanded  # Now broadcasting works: (1, 9, 1) ** (219, 9, 1)
    u = jnp.prod(powered, axis=1, keepdims=True)  # Shape: (219, 1)
    # Reshape u to 2D before concatenating
    u = u.squeeze(axis=-1)  # Shape: (219, 1)
    
    # Add constant term (1)
    u = jnp.vstack([u, jnp.ones((1, 1))])
    return u

def rbf_eval_batch(u, U_select, W, epsilon, case_rbf):
    """
    RBF evaluation function for multiple points (batch)
    
    Inputs:
        u: (N, d) array, where each row is a d-dimensional evaluation point
        U_select: (M, d) array, where each row is a d-dimensional center
        W: (M, d_prime) array, where each row is the weight vector for a center
        epsilon: shape parameter for the Gaussian RBF
        case_rbf: boolean, if True use Gaussian RBF, else use distance
    
    Output:
        final_value: (N, d_prime) array, where each row is the RBF interpolation
                     evaluated at the corresponding row of u
    """
    u = np.asarray(u)
    
    # Ensure u is 2D
    if u.ndim == 1:
        u = u.reshape(1, -1)
    
    N, d = u.shape           # N: number of evaluation points, d: dimension
    M, d_check = U_select.shape  # M: number of centers
    M_check, d_prime = W.shape   # d_prime: dimension of output values
    
    # Validate dimensions
    if d_check != d:
        raise ValueError(f'Dimension of U_select ({d_check}) must match dimension of u ({d})')
    if M_check != M:
        raise ValueError(f'Number of rows in W ({M_check}) must match number of centers in U_select ({M})')
    
    # Compute pairwise Euclidean distances between u and U_select
    # dist: (N, M) matrix, where dist[i,j] is the distance between u[i] and U_select[j]
    dist = cdist(u, U_select, metric='euclidean')  # Shape: (N, M)
    
    # Compute Gaussian RBF: phi(r) = exp(-(epsilon * r)^2)
    # A: (N, M) matrix, where A[i,j] = phi(||u[i] - U_select[j]||)
    # if case_rbf:
    #     A = np.exp(-(epsilon * dist)**2)
    # else:
    #     A = dist
    A = dist
    
    # Compute RBF interpolation: final_value = A @ W
    # A: (N, M), W: (M, d_prime) -> final_value: (N, d_prime)
    final_value = A @ W  # Shape: (N, d_prime)
    
    return final_value

@jax.jit
def rbf_eval_single_jax(u, U_select, W, epsilon, case_rbf):
    """
    JAX-compatible version of rbf_eval_single
    Expects MATLAB-style format: U_select is (d, M), W is (M, d_prime)
    
    Args:
        u: (d,) or (1, d) - evaluation point
        U_select: (d, M) - each column is a center
        W: (M, d_prime) - weights
        epsilon: scalar - shape parameter
        case_rbf: bool - if True use Gaussian RBF, else use distance
    
    Returns:
        final_value: (d_prime, 1) - RBF evaluation result
    """
    u = jnp.asarray(u).reshape(-1)  # Flatten to (d,) - handles both (d,) and (1,d)
    
    # U_select: (d, M) - each column is a center
    # u: (d,) - evaluation point
    # Compute distances from u to each center
    u_expanded = u[:, None]  # Shape: (d, 1)
    diff = U_select - u_expanded  # Broadcasting: (d, M) - (d, 1) = (d, M)
   # Define scaling factors as JAX array
    scaling = jnp.array([30.0,90.0, 50.0, 90.0, 50.0, 30.0])[:, None]  # Shape: (d, 1)
    
    # Apply scaling (element-wise division)
    diff = diff / scaling  # Broadcasting: (d, M) / (d, 1) = (d, M)
    
    dist = jnp.sqrt(jnp.sum(diff**2, axis=0, keepdims=True))  # Shape: (1, M)
    
    # Compute RBF
    # if case_rbf:
    #     A = jnp.exp(-(epsilon * dist)**2)  # Shape: (1, M)
    # else:
    #     A = dist  # Shape: (1, M)
    A = dist

    # Compute interpolation
    # A: (1, M), W: (M, d_prime) -> result: (1, d_prime)
    result = A @ W  # Shape: (1, d_prime)
    final_value = result.T  # Shape: (d_prime, 1)
    
    return final_value

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

def rbf_eval_single(u, U_select, W, epsilon, case_rbf):
    """
    RBF evaluation function for a single point
    
    Inputs:
        u: (d,) array, a single d-dimensional evaluation point
        U_select: (M, d) array, where each row is a d-dimensional center
        W: (M, d_prime) array, where each row is the weight vector for a center
        epsilon: shape parameter for the Gaussian RBF
        case_rbf: boolean, if True use Gaussian RBF, else use distance
    
    Output:
        final_value: (d_prime, 1) array, the RBF interpolation evaluated at u
    """
    u = np.asarray(u).flatten()  # Ensure u is 1D
    u_reshaped = u.reshape(1, -1)  # Shape: (1, d)
    
    d = u.shape[0]
    M, d_check = U_select.shape  # M: number of centers
    M_check, d_prime = W.shape   # d_prime: dimension of output values
    
    # Validate dimensions
    if d_check != d:
        raise ValueError(f'Dimension of U_select ({d_check}) must match dimension of u ({d})')
    if M_check != M:
        raise ValueError(f'Number of rows in W ({M_check}) must match number of centers in U_select ({M})')
    
    # Compute pairwise Euclidean distances between u and U_select
    # dist: (1, M) matrix
    dist = cdist(u_reshaped, U_select, metric='euclidean')  # Shape: (1, M)
    
    # Compute Gaussian RBF: phi(r) = exp(-(epsilon * r)^2)
    # A: (1, M) matrix
    # if case_rbf:
    #     A = np.exp(-(epsilon * dist)**2)
    # else:
    #     A = dist
    A = dist
    
    # Compute RBF interpolation: final_value = A @ W
    # A: (1, M), W: (M, d_prime) -> result: (1, d_prime)
    result = A @ W  # Shape: (1, d_prime)
    
    # Reshape to (d_prime, 1) to match the monomial function style
    final_value = result.T  # Shape: (d_prime, 1)
    
    return final_value

@jax.jit
def eval_aSSM_exps(xi, exps):
    """
    Evaluate monomial terms using JAX arrays.
    """
    xi = jnp.asarray(xi).flatten()
    x = xi.reshape(1, -1, 1)
    exps_expanded = exps[:, :, None]  # Add the missing dimension
    powered = x ** exps_expanded      # Now broadcasting works
    u = jnp.prod(powered, axis=1, keepdims=True)
    # Reshape u to 2D before concatenating
    u = u.squeeze(axis=-1)  # Shape: (*, 1)
    return u


def to_jax_array(data):
    """Convert MATLAB data (potentially sparse) to JAX array"""
    if hasattr(data, 'toarray'):  # It's a sparse matrix
        return jnp.asarray(data.toarray())
    else:  # It's already dense
        return jnp.asarray(data)
    
@jax.tree_util.register_static
class aSSM_strategy_rad_bas(System):
    """
    A d-dimensional aSSM-reduced model that can be auto‐differentiated by JAX.
    Controls: u , n_u dimensional
    Reduced state: x , d dimensional
    """

    def __init__(self, exps_Rd, Rd, U_select, W, epsilon, case_rbf, dt, n_x=2, n_u=6, rk4=False):
        super().__init__(dt=dt, n_x=n_x, n_u=n_u, rk4=rk4)
        self.exps_Rd = exps_Rd
        self.Rd = Rd
        self.U_select = U_select
        self.W = W
        self.epsilon = epsilon
        self.case_rbf = case_rbf
        # ============ NEW: Pre-compile Jacobian functions ============
        self._jac_single = None
        self._jac_batch = None
        self._compile_jacobians()
        self.d_dynamics = jax.jit(self.discrete_dynamics)
        self.vmap_dynamics = jax.jit(jax.vmap(self.discrete_dynamics))
        
    
    def _compile_jacobians(self):
        """Pre-compile Jacobian functions for both single and batch cases."""
        
        # Single sample Jacobian (x: (n_x,), u: (n_u,))
        def f_single(x, u):
            return self.discrete_dynamics(x, u)
        
        # Use jacrev instead of jacfwd (faster for small n_x)
        self._jac_single = jax.jit(jax.jacrev(f_single, argnums=(0, 1)))
        
        # Batch Jacobian (x: (N, n_x), u: (N, n_u))
        self._jac_batch = jax.jit(jax.vmap(jax.jacrev(f_single, argnums=(0, 1))))

        # Pre-compile affine term computation
        def compute_affine(A, B, x, u, fx):
            Ax = jnp.sum(A * x[:, None, :], axis=2)
            Bu = jnp.sum(B * u[:, None, :], axis=2)
            return fx - Ax - Bu
        
        self._compute_affine_batch = jax.jit(compute_affine)

    # def _compile_jacobians(self):
    #     """Pre-compile Jacobian functions that exploit RBF caching."""
        
    #     # For single sample: compute RBF once, then take Jacobian w.r.t. x only
    #     def f_single_cached(x, xbar_cache):
    #         """Dynamics with pre-computed RBF - only x varies."""
    #         return self._discrete_dynamics_with_rbf_cache(x, None, xbar_cache)
        
    #     # Compile Jacobian w.r.t. x only (u doesn't vary during Jacobian computation)
    #     jac_x_single = jax.jit(jax.jacrev(f_single_cached, argnums=0))
        
    #     # For control Jacobian, we need to differentiate the RBF computation
    #     def f_for_B_single(u):
    #         """Full dynamics for computing B matrix."""
    #         # Need a dummy x for signature
    #         return lambda x: self.discrete_dynamics(x, u)
        
    #     # Store compiled versions
    #     self._jac_x_single = jac_x_single
    #     self._jac_u_single = jax.jit(lambda x, u: jax.jacrev(f_for_B_single(u))(x))
        
    #     # Batch versions
    #     self._jac_x_batch = jax.jit(jax.vmap(jac_x_single))
        
    #     # For batch control Jacobian
    #     def compute_B_batch(x_batch, u_batch):
    #         """Compute B matrices for batch."""
    #         def single_B(x, u):
    #             return jax.jacrev(lambda ui: self.discrete_dynamics(x, ui))(u)
    #         return jax.vmap(single_B)(x_batch, u_batch)
        
    #     self._jac_u_batch = jax.jit(compute_B_batch)
        
    # def _discrete_dynamics_with_rbf_cache(self, x, u, xbar_cache):
    #     """
    #     Discrete dynamics that uses pre-computed RBF values.
        
    #     Args:
    #         x: state (n_x,) or (N, n_x)
    #         u: control (n_u,) or (N, n_u) - NOT USED, only for signature compatibility
    #         xbar_cache: pre-computed RBF values (3,) or (N, 3)
    #     """
    #     if not self.rk4:
    #         if x.ndim == 1:
    #             x_input = jnp.concatenate([x, xbar_cache])
    #             return x + self.dt * (self.Rd @ eval_aSSM_exps(x_input, self.exps_Rd)).squeeze()
    #         else:
    #             x_inputs = jnp.concatenate([x, xbar_cache], axis=1)
    #             xr_batch = jax.vmap(lambda xi: self.Rd @ eval_aSSM_exps(xi, self.exps_Rd))(x_inputs)
    #             return x + self.dt * xr_batch.squeeze()
        
    #     # RK4 with cached RBF
    #     if x.ndim == 1:
    #         def f_rk4(x_val):
    #             x_input = jnp.concatenate([x_val, xbar_cache])
    #             xr = self.Rd @ eval_aSSM_exps(x_input, self.exps_Rd)
    #             return xr.squeeze()
            
    #         k1 = f_rk4(x)
    #         k2 = f_rk4(x + 0.5*self.dt*k1)
    #         k3 = f_rk4(x + 0.5*self.dt*k2)
    #         k4 = f_rk4(x + self.dt*k3)
            
    #         return x + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    #     else:
    #         def f_rk4_batch(x_val):
    #             x_inputs = jnp.concatenate([x_val, xbar_cache], axis=1)
    #             xr_batch = jax.vmap(lambda xi: self.Rd @ eval_aSSM_exps(xi, self.exps_Rd))(x_inputs)
    #             return xr_batch.squeeze()
            
    #         k1 = f_rk4_batch(x)
    #         k2 = f_rk4_batch(x + 0.5*self.dt*k1)
    #         k3 = f_rk4_batch(x + 0.5*self.dt*k2)
    #         k4 = f_rk4_batch(x + self.dt*k3)
            
    #         return x + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
     
    # def dynamics_jac(self, x, u, continuous=False):
    #     """
    #     Optimized Jacobian computation with RBF caching.
        
    #     Key insight: During Jacobian computation, u is constant but x varies.
    #     We compute RBF(u) once and reuse it for all x perturbations.
    #     """
    #     if continuous:
    #         # Continuous case (fallback to standard approach)
    #         f = lambda xi, ui: self.continuous_dynamics(xi, ui)
    #         if x.ndim == 1:
    #             A, B = jax.jacrev(f, argnums=(0, 1))(x, u)
    #             d = f(x, u) - A @ x - B @ u
    #         else:
    #             A, B = jax.vmap(jax.jacrev(f, argnums=(0, 1)))(x, u)
    #             d = jax.vmap(f)(x, u) - jnp.sum(A * x[:, None, :], axis=2) - jnp.sum(B * u[:, None, :], axis=2)
    #         return A, B, d
        
    #     # Discrete case with RBF caching optimization
    #     if x.ndim == 1:
    #         # === SINGLE SAMPLE ===
    #         # Compute RBF once
    #         xbar_full = rbf_eval_single_jax(u, self.U_select, self.W, self.epsilon, self.case_rbf)
    #         xbar = xbar_full[:3].squeeze()
            
    #         # Compute A using cached RBF (much faster!)
    #         A = self._jac_x_single(x, xbar)
            
    #         # Compute B (needs to differentiate through RBF)
    #         B = jax.jacrev(lambda ui: self.discrete_dynamics(x, ui))(u)
            
    #         # Forward pass and affine term
    #         fx = self.discrete_dynamics(x, u)
    #         d = fx - A @ x - B @ u
            
    #     else:
    #         # === BATCH CASE ===
    #         # Compute RBF once for entire batch
    #         xbar_batch_full = jax.vmap(
    #             lambda ui: rbf_eval_single_jax(ui, self.U_select, self.W, self.epsilon, self.case_rbf).squeeze()
    #         )(u)
    #         xbar_batch = xbar_batch_full[:, :3]
            
    #         # Compute A matrices using cached RBF
    #         A = self._jac_x_batch(x, xbar_batch)
            
    #         # Compute B matrices (needs to differentiate through RBF)
    #         B = self._jac_u_batch(x, u)
            
    #         # Forward pass and affine term
    #         fx = jax.vmap(self.discrete_dynamics)(x, u)
    #         Ax = jnp.sum(A * x[:, None, :], axis=2)
    #         Bu = jnp.sum(B * u[:, None, :], axis=2)
    #         d = fx - Ax - Bu
        
    #     return A, B, d
    # ============ OVERRIDE dynamics_jac in parent class ===========


    def dynamics_jac(self, x, u, continuous=False):
        """
        Optimized Jacobian computation with pre-compiled functions.
        """
        #import time
        #print(f"dynamics_jac called with x.shape={x.shape}, u.shape={u.shape}")
        if continuous:
            f = lambda xi, ui: self.continuous_dynamics(xi, ui)
            if x.ndim == 1:
                A, B = jax.jacrev(f, argnums=(0, 1))(x, u)
                d = f(x, u) - A @ x - B @ u
            else:
                A, B = jax.vmap(jax.jacrev(f, argnums=(0, 1)))(x, u)
                d = jax.vmap(f)(x, u) - jnp.sum(A * x[:, None, :], axis=2) - jnp.sum(B * u[:, None, :], axis=2)
        else:
            if x.ndim == 1:
                #t0 = time.perf_counter()
                A, B = self._jac_single(x, u)
                #t1 = time.perf_counter()
                fx = self.d_dynamics(x, u)
                #t2 = time.perf_counter()
                d = fx - A @ x - B @ u
                #t3 = time.perf_counter()
                
                #print(f"  Jac A,B: {(t1-t0)*1000:.2f}ms, Forward: {(t2-t1)*1000:.2f}ms, Affine: {(t3-t2)*1000:.2f}ms")
            else:
                #t0 = time.perf_counter()
                A, B = self._jac_batch(x, u)
                #t1 = time.perf_counter()
                # fx = jax.vmap(self.discrete_dynamics)(x, u)
                fx = self.vmap_dynamics(x, u)
                #t2 = time.perf_counter()
                # Ax = np.sum(A * x[:, None, :], axis=2)
                # Bu = np.sum(B * u[:, None, :], axis=2)
                # d = fx - Ax - Bu
                d = self._compute_affine_batch(A, B, x, u, fx)
                #t3 = time.perf_counter()
                
                #print(f"  Jac A,B: {(t1-t0)*1000:.2f}ms, Forward: {(t2-t1)*1000:.2f}ms, Affine: {(t3-t2)*1000:.2f}ms")
        
        return A, B, d

    # def dynamics_jac(self, x, u, continuous=False):
    #     """
    #     Optimized Jacobian computation with pre-compiled functions.
    #     """
        

    #     if continuous:
    #         # Continuous case (less common, not optimized)
    #         f = lambda xi, ui: self.continuous_dynamics(xi, ui)
    #         if x.ndim == 1:
    #             A, B = jax.jacrev(f, argnums=(0, 1))(x, u)
    #             d = f(x, u) - A @ x - B @ u
    #         else:
    #             A, B = jax.vmap(jax.jacrev(f, argnums=(0, 1)))(x, u)
    #             d = jax.vmap(f)(x, u) - jnp.sum(A * x[:, None, :], axis=2) - jnp.sum(B * u[:, None, :], axis=2)
    #     else:
    #         # Discrete case (main use) - use pre-compiled versions
    #         if x.ndim == 1:
    #             # Single sample
    #             A, B = self._jac_single(x, u)
    #             fx = self.discrete_dynamics(x, u)
    #             d = fx - A @ x - B @ u
    #         else:
    #             # Batch
    #             A, B = self._jac_batch(x, u)
    #             fx = jax.vmap(self.discrete_dynamics)(x, u)
    #             # Optimized affine term (faster than einsum)
    #             Ax = jnp.sum(A * x[:, None, :], axis=2)
    #             Bu = jnp.sum(B * u[:, None, :], axis=2)
    #             d = fx - Ax - Bu
        
    #     return A, B, d    
    
    def continuous_dynamics(self, x, u):
        """
        Continuous-time dynamics f(x,u), broadcasted over the leading dimension.
    
        Supports both single-sample input  (x.shape = (6,),  u.shape = (6,))
        and batched input                 (x.shape = (N,6), u.shape = (N,6)).
        """
       
        # If we have a single sample (rank-1), do direct computation
        #print(x.ndim)
        if x.ndim == 1:
            # Single sample case: x shape (6,), u shape (6,)
            xbar_full =  rbf_eval_single_jax(u, self.U_select, self.W, self.epsilon, self.case_rbf)
            #xbar_full = rbf_eval_batch_jax(u[None, :], self.U_select, self.W, self.epsilon, self.case_rbf)[0]
            xbar = xbar_full[:3].squeeze()
            x_input = jnp.concatenate([x, xbar])  # shape (8,)
           
            # Evaluate aSSM
            xr = self.Rd @ eval_aSSM_exps(x_input, self.exps_Rd)
            #xu = self.B @ eval_aSSM_exps(u_input, self.exps_B)
        
            return (xr.squeeze())  # shape (output_dim,)
    
        # Otherwise, handle batched data: x.shape = (N, 6), u.shape = (N, 6)
        batch_size = x.shape[0]
    
        # Broadcast xbar and ubar to match batch size
        #xbar_batch = jnp.tile(xbar, (batch_size, 1))  # shape (batch_size, 2)
        #xbar_0_batch = jnp.tile(xbar_0, (batch_size, 1))  # shape (batch_size, 2)
        #ubar_batch = jnp.tile(ubar, (batch_size, 1))  # shape (batch_size, 6)
        
        xbar_batch_full = jax.vmap(lambda ui: rbf_eval_single_jax(ui, self.U_select, self.W, self.epsilon, self.case_rbf).squeeze())(u)
        #xbar_batch_full = rbf_eval_batch_jax(u, self.U_select, self.W, self.epsilon, self.case_rbf)

        xbar_batch = xbar_batch_full[:, :3]
        
        # Prepare batched inputs
        x_inputs = jnp.concatenate([x, xbar_batch], axis=1)  # shape (batch_size, 8)
        #u_inputs = jnp.concatenate([u - ubar_batch, xbar_batch], axis=1)  # shape (batch_size, 8)
    
        # Apply to all batch elements using vmap
        xr_batch = jax.vmap(lambda xi: self.Rd @ eval_aSSM_exps(xi, self.exps_Rd))(x_inputs)
        #xu_batch = jax.vmap(lambda ui: self.B @ eval_aSSM_exps(ui, self.exps_B))(u_inputs)
    
        # Return shape (batch_size, output_dim) - same pattern as SimpleCarModel
        return (xr_batch).squeeze()
    
    @partial(jax.jit, static_argnums=(0,))
    def discrete_dynamics(self, x, u):
        """
        RK4 integration with RBF hoisting optimization.
        
        Since RBF depends only on u (not x), and u doesn't change across
        RK4 stages, we compute RBF(u) once and reuse it for k1, k2, k3, k4.
        
        This gives ~4x speedup for RBF computation.
        """
        if self.rk4:
            # === SINGLE SAMPLE CASE ===
            if x.ndim == 1:
                # Compute RBF once (not 4 times!)
                xbar_full = rbf_eval_single_jax(u, self.U_select, self.W, self.epsilon, self.case_rbf)
                xbar = xbar_full[:3].squeeze()
                
                # Define RK4 function with pre-computed xbar
                def f_rk4(x_val):
                    """Continuous dynamics with cached RBF."""
                    x_input = jnp.concatenate([x_val, xbar])
                    xr = self.Rd @ eval_aSSM_exps(x_input, self.exps_Rd)
                    return xr.squeeze()
                
                # RK4 stages - all use same xbar
                k1 = f_rk4(x)
                k2 = f_rk4(x + 0.5*self.dt*k1)
                k3 = f_rk4(x + 0.5*self.dt*k2)
                k4 = f_rk4(x + self.dt*k3)
                
                return x + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            # === BATCH CASE ===
            else:
                # Compute RBF once for entire batch
                xbar_batch_full = jax.vmap(
                    lambda ui: rbf_eval_single_jax(ui, self.U_select, self.W, self.epsilon, self.case_rbf).squeeze()
                )(u)
                xbar_batch = xbar_batch_full[:, :3]
                
                # Define RK4 function with pre-computed xbar_batch
                def f_rk4_batch(x_val):
                    """Batched continuous dynamics with cached RBF."""
                    x_inputs = jnp.concatenate([x_val, xbar_batch], axis=1)
                    xr_batch = jax.vmap(lambda xi: self.Rd @ eval_aSSM_exps(xi, self.exps_Rd))(x_inputs)
                    return xr_batch.squeeze()
                
                # RK4 stages - all use same xbar_batch
                k1 = f_rk4_batch(x)
                k2 = f_rk4_batch(x + 0.5*self.dt*k1)
                k3 = f_rk4_batch(x + 0.5*self.dt*k2)
                k4 = f_rk4_batch(x + self.dt*k3)
                
                return x + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        else:
            # Euler integration - use parent class implementation
            return x + self.continuous_dynamics(x, u) * self.dt


def setup_aSSM(dt, rk4=False):
    """
    Setup function to create an aSSM_strategy_rad_bas model from MATLAB data.
    """                          

    mat_data = scipy.io.loadmat('/home/trunk/Documents/trunk-stack/stack/main/data/models/ssm/aSSM_model_trunk_5D.mat')  # Replace with your .mat file path
    #mat_data = scipy.io.loadmat('FO_aSSM_model_elastica.mat')  # Replace with your .mat file path
    # 
    # Extract exps_R and R
    exps_Rd = to_jax_array(mat_data['exps_Rd'])
    Rd = to_jax_array(mat_data['Rd'])
    
    exps_M = to_jax_array(mat_data['exps_M'])
    M = to_jax_array(mat_data['M'])

    V = to_jax_array(mat_data['V'])
    U_select = to_jax_array(mat_data['U_select'])
    W_sm = to_jax_array(mat_data['W'])
    epsilon_sm = float(mat_data['epsilon'])
    case_rbf = int(mat_data['case_rbf'])
    print(U_select.shape)
    print(W_sm.shape)
    # U_select = np.load('/home/trunk/Documents/trunk-stack/stack/main/data/models/ssm/rbf_centers.npy')
    # # W_sm = np.load('/home/trunk/Documents/trunk-stack/stack/main/data/models/ssm/rbf_weights.npy')
    # U_select = jnp.array(U_select)
    # W_sm = jnp.array(W_sm)
    # print(U_select.shape)
    # print(W_sm.shape)

    system = aSSM_strategy_rad_bas(exps_Rd = exps_Rd, Rd = Rd, U_select = U_select, W = W_sm, epsilon = epsilon_sm, case_rbf = case_rbf ,dt=dt, n_x=2, n_u=6, rk4=False)
    return system, exps_M, M, U_select, W_sm, epsilon_sm, case_rbf, V
