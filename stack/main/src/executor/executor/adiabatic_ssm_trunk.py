import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.mplot3d import Axes3D  # Add this import at the top

import os
import pandas as pd

from scipy.spatial.distance import cdist

from .dyn_system import System


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
    if case_rbf:
        A = np.exp(-(epsilon * dist)**2)
    else:
        A = dist
    
    # Compute RBF interpolation: final_value = A @ W
    # A: (N, M), W: (M, d_prime) -> final_value: (N, d_prime)
    final_value = A @ W  # Shape: (N, d_prime)
    
    return final_value

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
    dist = jnp.sqrt(jnp.sum(diff**2, axis=0, keepdims=True))  # Shape: (1, M)
    
    # Compute RBF
    if case_rbf:
        A = jnp.exp(-(epsilon * dist)**2)  # Shape: (1, M)
    else:
        A = dist  # Shape: (1, M)
    
    # Compute interpolation
    # A: (1, M), W: (M, d_prime) -> result: (1, d_prime)
    result = A @ W  # Shape: (1, d_prime)
    final_value = result.T  # Shape: (d_prime, 1)
    
    return final_value

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
    if case_rbf:
        A = np.exp(-(epsilon * dist)**2)
    else:
        A = dist
    
    # Compute RBF interpolation: final_value = A @ W
    # A: (1, M), W: (M, d_prime) -> result: (1, d_prime)
    result = A @ W  # Shape: (1, d_prime)
    
    # Reshape to (d_prime, 1) to match the monomial function style
    final_value = result.T  # Shape: (d_prime, 1)
    
    return final_value

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
        
        xbar_batch = xbar_batch_full[:, :3]
        
        # Prepare batched inputs
        x_inputs = jnp.concatenate([x, xbar_batch], axis=1)  # shape (batch_size, 8)
        #u_inputs = jnp.concatenate([u - ubar_batch, xbar_batch], axis=1)  # shape (batch_size, 8)
    
        # Apply to all batch elements using vmap
        xr_batch = jax.vmap(lambda xi: self.Rd @ eval_aSSM_exps(xi, self.exps_Rd))(x_inputs)
        #xu_batch = jax.vmap(lambda ui: self.B @ eval_aSSM_exps(ui, self.exps_B))(u_inputs)
    
        # Return shape (batch_size, output_dim) - same pattern as SimpleCarModel
        return (xr_batch).squeeze()


def setup_aSSM(dt, rk4=False):
    """
    Setup function to create an aSSM_strategy_rad_bas model from MATLAB data.
    """                          

    mat_data = scipy.io.loadmat('/home/trunk/Documents/trunk-stack/stack/main/data/models/ssm/aSSM_model_trunk_upgrade.mat')  # Replace with your .mat file path
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
    

    system = aSSM_strategy_rad_bas(exps_Rd = exps_Rd, Rd = Rd, U_select = U_select, W = W_sm, epsilon = epsilon_sm, case_rbf = case_rbf ,dt=dt, n_x=2, n_u=6, rk4=True)
    return system, exps_M, M, U_select, W_sm, epsilon_sm, case_rbf, V
