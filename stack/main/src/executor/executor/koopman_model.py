import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.io
# from mpl_toolkits.mplot3d import Axes3D  # Add this import at the top
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


def filter_for_I_cm(x, I_set):
    cm = x[:, I_set]           # Extract columns: [[1, 3], [5, 7]]
    cm_reversed = cm[:, ::-1]  # Reverse column order: [[3, 1], [7, 5]]
    cm_vec = cm_reversed.flatten(order='F')  # Flatten column-wise: [3, 7, 1, 5]
    return cm_vec


def to_jax_array(data):
    """Convert MATLAB data (potentially sparse) to JAX array"""
    if hasattr(data, 'toarray'):  # It's a sparse matrix
        return jnp.asarray(data.toarray())
    else:  # It's already dense
        return jnp.asarray(data)
    
"""""
def delayed_state(states, num_delays=5):
    
    Create a delay-embedded state representation from the state trajectory.

    Args:
        states: JAX array of shape (T, state_dim), where each row is the state at time t.
        num_delays: Number of delays (default 5, including current state).

    Returns:
        delayed_states: JAX array of shape (T, state_dim * num_delays).
                       For time t, the row is [state(t), state(t-1), ..., state(t-(num_delays-1))].
                       If t-d < 1, state(1) is used for padding.
    
    T, state_dim = states.shape
    delayed_states = jnp.zeros((T, state_dim * num_delays))

    for d in range(num_delays):
        # Compute indices for delay d: t - d for t = 0, ..., T-1
        indices = jnp.maximum(0, jnp.arange(T) - d)
        # Select states at these indices (state[0] for negative indices)
        valid_states = states[indices, :]
        # Assign to the corresponding slice
        delayed_states = delayed_states.at[:, d * state_dim:(d + 1) * state_dim].set(valid_states)

    return (delayed_states.T).squeeze()
"""

def delayed_state(states, num_delays=5):
    """
    Create a delay-embedded state representation for the latest timestep only.

    Args:
        states: JAX array of shape (T, state_dim), where each row is the state at time t.
        num_delays: Number of delays (default 5, including current state).

    Returns:
        latest_delayed_state: JAX array of shape (state_dim * num_delays,).
                             Contains [state(T-1), state(T-2), ..., state(T-num_delays)].
                             If t-d < 0, state(0) is used for padding.
    """
    T, state_dim = states.shape
    latest_delayed_state = jnp.zeros(state_dim * num_delays)

    for d in range(num_delays):
        # For the latest timestep (T-1), get index (T-1-d)
        idx = jnp.maximum(0, T - 1 - d)
        # Get the state at that index
        valid_state = states[idx, :]
        # Assign to the corresponding slice
        latest_delayed_state = latest_delayed_state.at[d * state_dim:(d + 1) * state_dim].set(valid_state)

    return latest_delayed_state.squeeze()

class Koopman_pregain(System):
    """
    A d-dimensional aSSM-reduced model that can be auto‐differentiated by JAX.
    Controls: u , n_u dimensional
    Reduced state: x , d dimensional
    """

    def __init__(self, A_koop, B_koop, us_xs, dt, n_x=6, n_u=6, rk4=False):
        super().__init__(dt=dt, n_x=n_x, n_u=n_u, rk4=rk4)
        self.A_koop = A_koop
        self.B_koop = B_koop
        self.us_xs = jnp.asarray(us_xs) if not callable(us_xs) else us_xs
        
    def update_param(self, us_xs):
        self.us_xs = jnp.asarray(us_xs) if not callable(us_xs) else us_xs

    """
    def continuous_dynamics(self, x, u):
        xbar = self.us_xs[-2:]   # Gets the last 2 elements (indices 6,7)
        ubar = self.us_xs[:6]    # Gets the first 6 elements (indices 0,1,2,3,4,5)
    
        #print(f"xbar: {xbar}, shape: {xbar.shape}")  # Should be shape (2,)
        #print(f"ubar: {ubar}, shape: {ubar.shape}")  # Should be shape (6,)
    
        # Fix x and u shapes for concatenation
        x_flat = x.flatten()  # Convert (1, 6) to (6,)
        u_flat = u.flatten()  # Convert (1, 6) to (6,)
        # debug 
        # Debug the concatenated inputs
        #x_input = jnp.concatenate([x_flat, xbar.flatten()])
        #u_input = jnp.concatenate([u_flat - ubar, xbar.flatten()])
    
        #print(f"x_flat shape: {x_flat.shape}")
        #print(f"xbar shape: {xbar.shape}")
        #print(f"x_input shape: {x_input.shape}")  # This is (38,) but should be (8,)
        #print(f"exps_Rd shape: {self.exps_Rd.shape}")  # Second dim should match x_input length
    
        # Same for u_input
        #print(f"u_input shape: {u_input.shape}")
        #print(f"exps_B shape: {self.exps_B.shape}")
        #print(f"Rd shape: {self.Rd.shape}")
        #print(f"B shape: {self.B.shape}")
        # Evaluate the aSSM reduced dynamics
        xr = self.Rd @ eval_aSSM_exps(jnp.concatenate([x_flat, xbar.flatten()]), self.exps_Rd)
        xu = self.B @ eval_aSSM_exps(jnp.concatenate([u_flat - ubar, xbar.flatten()]), self.exps_B)
        #print(f"xr shape: {xr.shape}, xu shape: {xu.shape}")
        #xd = xr.flatten() + xu.flatten()  # Combine the reduced dynamics contributions
        #print(f"xd shape: {xd.shape}")
        return xr.flatten() + xu.flatten()
    """
    """
    def continuous_dynamics(self, x, u):
        
        Handle batch processing for continuous dynamics.
        x: shape (15, 6) - batch of states
        u: shape (15, 6) - batch of controls
        
        xbar = self.us_xs[-2:]   # Gets the last 2 elements (indices 6,7) - shape (2,)
        ubar = self.us_xs[:6]    # Gets the first 6 elements (indices 0,1,2,3,4,5) - shape (6,)
    
        # Broadcast xbar and ubar to match batch size
        batch_size = x.shape[0]  # 15
        xbar_batch = jnp.tile(xbar, (batch_size, 1))  # shape (15, 2)
        ubar_batch = jnp.tile(ubar, (batch_size, 1))  # shape (15, 6)
        print("xbar_batch shape:", xbar_batch.shape)  # Should be (15, 2
        print("ubar_batch shape:", ubar_batch.shape)  # Should be (15, 6
        print(f"x shape: {x.shape}, u shape: {u.shape}")  # Debugging line
        # Prepare batched inputs
        x_inputs = jnp.concatenate([x, xbar_batch], axis=1)  # shape (15, 8)
        u_inputs = jnp.concatenate([u - ubar_batch, xbar_batch], axis=1)  # shape (15, 8)
        
        print(f"x_inputs shape: {x_inputs.shape}, u_inputs shape: {u_inputs.shape}")
        print("u_inputs:", u_inputs)  # Debugging line
        # Apply to all batch elements using vmap
        xr_batch = jax.vmap(lambda xi: self.Rd @ eval_aSSM_exps(xi, self.exps_Rd))(x_inputs)
        xu_batch = jax.vmap(lambda ui: self.B @ eval_aSSM_exps(ui, self.exps_B))(u_inputs)
        
        # Combine results
        return (xr_batch + xu_batch).squeeze()  # Remove extra dimensions if needed
    """ 
    def continuous_dynamics(self, x, u):
        """
        Continuous-time dynamics f(x,u), broadcasted over the leading dimension.

        Uses linear Koopman dynamics: xd = (A_koop - I)*x + B_koop*u with dt = 1/60

        Supports both single-sample input  (x.shape = (6,),  u.shape = (6,))
        and batched input                 (x.shape = (N,6), u.shape = (N,6)).
        """

        dt_c = dt  # Time step
        xbar = jnp.tile(self.us_xs[-3:], 2)  # shape (6,) - repeat last 3 elements twice
        ubar = self.us_xs[:6]    # shape (6,) - get first 6 elements for input equilibrium
        
        # Create identity matrix for state dimension (6x6)
        I = jnp.eye(6)

        # Compute the dynamics matrix (A_koop - I)
        A_diff = self.A_koop - I

        # If we have a single sample (rank-1), do direct computation
        if x.ndim == 1:
            # Single sample case: x shape (6,), u shape (6,)
            # xd = (A_koop - I)*(x-xbar) + B_koop*(u-ubar)
            xd = A_diff @ (x - xbar) + self.B_koop @ (u - ubar)
            return xd / dt_c

        # Otherwise, handle batched data: x.shape = (N, 6), u.shape = (N, 6)
        # Broadcast equilibrium points to match batch size
        batch_size = x.shape[0]
        xbar_batch = jnp.tile(xbar, (batch_size, 1))  # shape (batch_size, 6)
        ubar_batch = jnp.tile(ubar, (batch_size, 1))  # shape (batch_size, 6)
        
        # For batched computation, use matrix multiplication
        # (x-xbar): (N, 6), A_diff: (6, 6) -> (x-xbar) @ A_diff.T gives (N, 6)
        # (u-ubar): (N, 6), B_koop: (6, 6) -> (u-ubar) @ B_koop.T gives (N, 6)
        xd_batch = (x - xbar_batch) @ A_diff.T + (u - ubar_batch) @ self.B_koop.T

        return xd_batch / dt_c
    

def setup_koopman(dt, rk4=False):
    """
    Setup function to create an aSSM_strategy_rad_bas model from MATLAB data.
    """                          
    mat_data = scipy.io.loadmat('/home/trunk/Documents/trunk-stack/stack/main/data/models/koopman/Koopman_hardware_trunk_Jan30.mat')  # Replace with your .mat file path
    #mat_data = scipy.io.loadmat('FO_aSSM_model_elastica.mat')  # Replace with your .mat file path
   
    # Extract exps_R and R
    A_koop = to_jax_array(mat_data['A_Koop'])
    B_koop = to_jax_array(mat_data['B_Koop'])
    exps_I = to_jax_array(mat_data['exps_I_lin'])
    I = to_jax_array(mat_data['I_linear'])

    return A_koop, B_koop, exps_I, I 




