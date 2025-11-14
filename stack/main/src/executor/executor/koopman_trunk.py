# ----------------------------------------------------------
# testing_mpc.py
# ----------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.mplot3d import Axes3D  # Add this import at the top
from misc import HyperRectangle
import os
from control import dlqr

# Import the GuSTO config and MPC policy
# from gusto import GuSTOConfig
# from mpc_policy import MPCPolicy
# from arm_struct_main import simulate_arm_motion
# from elastica_trunk import ArmSimulator

# 1) Import the generic System base class from dyn_system.py
#    Adjust import path to match your own project structure
from .dyn_system import System

class SimpleCarModel(System):
    """
    A 4D car model with states (px, py, heading, velocity)
    and controls (steering, acceleration). Inherits from the
    System base class so that it automatically handles:
      - discrete_dynamics (Euler or RK4)
      - multistep_dynamics
      - dynamics_jac (via JAX)
    """

    def __init__(self, dt=0.1, rk4=False):
        """
        Args:
            dt: Sampling time (s).
            rk4: If True, uses RK4 integration in discrete_dynamics; 
                 otherwise uses simple Euler integration.
        """
        # n_x=4, n_u=2
        super().__init__(dt, n_x=4, n_u=2, rk4=rk4)

    def continuous_dynamics(self, x, u):
        """
        Continuous-time dynamics f(x,u), broadcasted over the leading dimension.

        States:  x = [ px, py, heading, v     ] 
        Controls:     [ steer, a              ]
        
        Returns: dx/dt of the same shape as x.

        Supports both single-sample input  (x.shape = (4,),  u.shape = (2,))
        and batched input                 (x.shape = (N,4), u.shape = (N,2)).
        """
        # If we have a single sample (rank-1), just do the direct computation
        if x.ndim == 1:
            px, py, heading, v = x
            steer, a = u
            L = 1.0  # wheelbase

            dx = v * jnp.cos(heading)
            dy = v * jnp.sin(heading)
            dtheta = v * jnp.tan(steer) / L
            dv = a

            return jnp.array([dx, dy, dtheta, dv])

        # Otherwise, handle batched data:
        # x.shape = (N, 4), u.shape = (N, 2)
        px, py, heading, v = x.T     # each shape: (N,)
        steer, a = u.T              # each shape: (N,)
        L = 1.0

        dx = v * jnp.cos(heading)
        dy = v * jnp.sin(heading)
        dtheta = v * jnp.tan(steer) / L
        dv = a

        # Return shape (N,4)
        return jnp.array([dx, dy, dtheta, dv]).T

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

        dt_c = 1/60  # Time step
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


def run_mpc_demo():
    """
    Demonstration of using the refactored SimpleCarModel with GuSTO-based MPC.
    """

    

    # 5) Set an initial state and define a simple reference path
    n_elem = 20
    dt_ref = 1.0 / 60
    initial_position = jnp.zeros((3, n_elem + 1))
    initial_position = initial_position.at[1].set(jnp.linspace(0, 1, n_elem + 1))  # Straight rod along y-axis
    initial_velocity = jnp.zeros((3, n_elem + 1))

    x0 = jnp.concatenate([initial_position, initial_velocity], axis=1)   # stacked like [positions, velocities]
    print("Initial state shape:", x0.shape)

    # Build a (T + N + 1)-long reference for px, py
    """"
    final_time = 10.0
    time_ref = np.linspace(0, final_time, int(final_time / dt_ref) + 1)
    T = len(time_ref)
    num_points = T
    radius = 1.0
    center_y = 0  # Sphere center at y=0.5
    num_arms = 3    # Number of star arms

    # Star-shaped trajectory
    theta = np.linspace(0, 2 * np.pi, num_points)
    r_star = radius * (0.45 + 0.45 * np.cos(num_arms * theta + np.pi))   # Star radius modulation
    xt_ref = r_star * np.cos(theta)
    zt_ref = r_star * np.sin(theta)
    yt_ref = center_y + np.sqrt(np.maximum(0, radius**2 - xt_ref**2 - zt_ref**2))

    z_ref = np.stack([xt_ref, yt_ref, zt_ref], axis=1)  # shape (T+N+1, 3)
    print("z_ref shape:", z_ref.shape)
    """

    # Build a (T + N + 1)-long reference for px, py
    np.random.seed(50)
    final_time = 10.0
    time_ref = np.linspace(0, final_time, int(final_time / dt_ref) + 1)
    T = len(time_ref)
    num_points = T
    radius = 1.0
    center_y = 0  # Sphere center at y=0

    # Random track generation parameters
    num_segments = 15  # Number of waypoints
    theta_waypoints = np.random.uniform(0, 2 * np.pi, num_segments)
    phi_waypoints = np.random.uniform(0, 0.5, num_segments) * np.pi  # Only upper hemisphere (0 to pi/2)

    # Convert spherical to Cartesian for waypoints (ensuring y > 0)
    x_waypoints = radius * np.sin(phi_waypoints) * np.cos(theta_waypoints)
    y_waypoints = center_y + radius * np.cos(phi_waypoints)  # This will be positive
    z_waypoints = radius * np.sin(phi_waypoints) * np.sin(theta_waypoints)

    # Decide which segments are straight vs curved
    segment_types = np.random.choice(['smooth', 'sharp', 'straight'], 
                                    size=num_segments-1, 
                                    p=[0.4, 0, 0.6])

    # Interpolate between waypoints
    xt_ref = []
    yt_ref = []
    zt_ref = []

    points_per_segment = num_points // (num_segments - 1)

    for i in range(num_segments - 1):
        start = np.array([x_waypoints[i], y_waypoints[i], z_waypoints[i]])
        end = np.array([x_waypoints[i+1], y_waypoints[i+1], z_waypoints[i+1]])
        
        if segment_types[i] == 'straight':
            # Linear interpolation (geodesic on sphere)
            t = np.linspace(0, 1, points_per_segment)
            for j in range(len(t)):
                point = start + t[j] * (end - start)
                # Project back onto sphere
                norm = np.sqrt(point[0]**2 + (point[1] - center_y)**2 + point[2]**2)
                point = point / norm * radius
                point[1] = center_y + abs(point[1] - center_y)  # Ensure y > 0
                xt_ref.append(point[0])
                yt_ref.append(point[1])
                zt_ref.append(point[2])
        
        elif segment_types[i] == 'sharp':
            # Sharp turn - just jump to next point
            t = np.linspace(0, 1, points_per_segment)
            for j in range(len(t)):
                if t[j] < 0.8:
                    point = start
                else:
                    point = end
                xt_ref.append(point[0])
                yt_ref.append(point[1])
                zt_ref.append(point[2])
        
        else:  # 'smooth'
            # Smooth interpolation via great circle
            t = np.linspace(0, 1, points_per_segment)
            start_centered = start - np.array([0, center_y, 0])
            end_centered = end - np.array([0, center_y, 0])
            angle = np.arccos(np.clip(np.dot(start_centered, end_centered) / 
                                    (np.linalg.norm(start_centered) * 
                                        np.linalg.norm(end_centered)), -1, 1))
            
            for j in range(len(t)):
                if angle > 0.01:
                    interp = (np.sin((1-t[j])*angle)/np.sin(angle)) * start + \
                            (np.sin(t[j]*angle)/np.sin(angle)) * end
                else:
                    interp = start + t[j] * (end - start)
                
                # Ensure on sphere and y > 0
                norm = np.sqrt(interp[0]**2 + (interp[1] - center_y)**2 + interp[2]**2)
                interp = interp / norm * radius
                interp[1] = center_y + abs(interp[1] - center_y)  # Force y > 0
                
                xt_ref.append(interp[0])
                yt_ref.append(interp[1])
                zt_ref.append(interp[2])

    # Convert to numpy arrays and trim/pad to exact length
    xt_ref = np.array(xt_ref[:num_points])
    yt_ref = np.array(yt_ref[:num_points])
    zt_ref = np.array(zt_ref[:num_points])

    # Pad if necessary
    if len(xt_ref) < num_points:
        pad_length = num_points - len(xt_ref)
        xt_ref = np.concatenate([xt_ref, np.full(pad_length, xt_ref[-1])])
        yt_ref = np.concatenate([yt_ref, np.full(pad_length, yt_ref[-1])])
        zt_ref = np.concatenate([zt_ref, np.full(pad_length, zt_ref[-1])])

    z_ref = np.stack([xt_ref, yt_ref, zt_ref], axis=1)  # shape (T+N+1, 3)
    print("z_ref shape:", z_ref.shape)

    # new refrence traj 
    """"
    mat_data = scipy.io.loadmat('ref_traj.mat')  # Replace with your .mat file 
    truth = np.array(mat_data['Truth'])  # Assuming 'truth' is the key for the trajectory
    time_ref = np.array(mat_data['testy_time'])  # Assuming 'time' is the key for the time vector
    truth = truth.squeeze()  # Remove any singleton dimensions
    time_ref = time_ref.squeeze()  # Remove any singleton dimensions
    print("After squeeze - Truth shape:", truth.shape)
    truth = truth.T
    # Extract coordinates
    xt_ref = truth[:, 0]  # Extract x-coordinates
    yt_ref = truth[:, 1]  # Extract y-coordinates  
    zt_ref = truth[:, 2]  # Extract z-coordinates

    # Stack into final array
    z_ref = np.stack([xt_ref, yt_ref, zt_ref], axis=1)  # shape (T, 3)

    # Set T based on the actual data length
    T = len(time_ref)

    # Verify shapes
    print("z_ref shape:", z_ref.shape)
    print("time_ref shape:", time_ref.shape)
    print("T =", T)
    mat_data = scipy.io.loadmat('ref_inputs.mat')
    u_ref = np.array(mat_data['uData_pred'])  # Assuming 'u_ref' is the key for the control inputs
    """
    # load aSSM-reduced model
    # Load the .mat file
    #mat_data = scipy.io.loadmat('aSSM_model_elastica.mat')  # Replace with your .mat file path
    mat_data = scipy.io.loadmat('Koopman_pregain_elastica.mat')  # Replace with your .mat file path
    # 
    # Extract exps_R and R
    A_koop = to_jax_array(mat_data['A_Koop'])
    B_koop = to_jax_array(mat_data['B_Koop'])
    exps_I = to_jax_array(mat_data['exps_I_lin'])
    I = to_jax_array(mat_data['I_linear'])

    #mat_data = scipy.io.loadmat('aSSM_model_elastica.mat')  # Replace with your .mat file path
    #I = to_jax_array(mat_data['I'])
    #exps_I = to_jax_array(mat_data['exps_I'])

    # Verify shapes
    print("A_koop shape:", A_koop.shape)
    print("B_koop shape:", B_koop.shape)
    print("exps_I shape:", exps_I.shape)
    print("I shape:", I.shape)
    print("I:", I)
    print("exps_I:", exps_I)

    indexSet = [n_elem]
    #indexSet = [2,(n_elem + 1) // 2, n_elem]
    cm = filter_for_I_cm(x0,indexSet) 
    print("x0:", x0)
    print("cm shape:", cm.shape)
    print("cm:", cm)
    us_xs_0_ubar = I @ eval_monomials_single_sm(cm, exps_I)  
    us_xs_0_xbar = cm[0:3]
    us_xs_0 = jnp.concatenate([us_xs_0_ubar.flatten(), us_xs_0_xbar.flatten()], axis=0)  # shape (n_u + n_x,) 
    print("us_xs_0:", us_xs_0)  
    delay_embed_x0 = jnp.tile(cm[0:3], 2)
    reduced_x0 = delay_embed_x0
    print("reduced_x0 shape:", reduced_x0.shape)
    system = Koopman_pregain(A_koop = A_koop, B_koop = B_koop, us_xs=us_xs_0,dt=dt_ref, n_x=6, n_u=6, rk4=False)

    # 3) Prepare a GuSTO config
    #    We'll track position only: z = [px, py].
    #    So H is 2x4, extracting just px, py from [px,py,heading,v].
    H = jnp.eye(3, 6)
    Qz  = 10.0 * jnp.eye(6)    # position tracking cost
    Qzf = 10.0 * jnp.eye(6)    # terminal position cost
    #R   = 0.5 * jnp.eye(6)     # control effort cost for zeroth order and aSSM
    #R   = 1.0 * jnp.eye(6)     # control effort cost for first order aSSM
    R   = 1.5 * jnp.eye(6)     # control effort cost for first order aSSM
    # Characteristic scales for x, f (used in the GuSTO trust region steps)
    x_char = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*10
    f_char = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*10

    Q_full = np.zeros((6,6))
    Q_full = Qz  # Adjust slice as needed for your subset
    #Q_full += 1e-6 * np.eye(6)
    # Now compute LQR
    K, S, E = dlqr(A_koop, B_koop, Q_full, R)
    print("LQR gain K:", K)
    print("LQR gain shape:", K.shape)
    K_red = K  # Only use position feedback
    

    config = GuSTOConfig(
        Qz=Qz,
        Qzf=Qzf,
        R=R,
        x_char=x_char,
        f_char=f_char,
        N=10,        # MPC horizon
        H=H
    )

    U = HyperRectangle([1.0]*6, [-1.0]*6)
    #U = None 
    # 4) Create the MPC policy

    mpc = MPCPolicy(model=system, config=config, xs=cm[0:3], U=U, dU=None, init_guess_type='shift')

    #mpc.reset(reduced_x0, obs=reduced_x0, z_ref=z_ref, start_with_solve=True)


    # 7) Simulate the closed-loop system
    states = []
    controls = []
    current_state = x0
    end_effector_pos = []

    # sim
    cl_time = dt_ref  # Short simulation for testing
    sim_dt = 2.0e-4
    total_steps = int(cl_time / sim_dt)
    number_of_control_points = 3
    n_elem = 20

    arm_simulator = ArmSimulator(
    n_elem=n_elem,
    number_of_control_points=number_of_control_points,
    alpha=15.0,
    beta=15.0,
    sim_dt=sim_dt,
    E=1e7,
    nu=7,
    rendering_fps=60,
    max_rate_of_change_of_activation=np.infty,
    initial_position=initial_position,
    initial_velocity=initial_velocity
    )


    #u_guess = I @ eval_monomials_batch_sm(z_combined, exps_I).T 
    #print("u_guess shape:", u_guess.shape)  # Should be (3*(T+N+1), 219)
    #print("u_guess:", u_guess)
    #u_guess = u_ref
    #print("u_guess shape:", u_guess.shape)
    


    for t in range(T):
        cm_r = filter_for_I_cm(current_state,indexSet) # uncommet for aSSM
        print("cm_r:", cm_r)
        print("cm_r shape:", cm_r.shape)
        #cm = filter_for_I_cm(x0,indexSet) # ucommet this line for zeroth-order aSSM     
        #cm = np.concatenate([z_ref[t, :], cm_r[3:]])  # Use the reference trajectory for the current step  
        cm = z_ref[t, :]
        print("z_ref[t, :]:", z_ref[t, :])
        print("cm:", cm)
        print("cm shape:", cm.shape)
        us_xs_0_ubar = I @ eval_monomials_single_sm(cm, exps_I)  
        us_xs_0_xbar = cm_r[0:3]
        u = -K_red @ np.tile((us_xs_0_xbar - z_ref[t, :]).reshape(-1, 1), (2, 1)) + us_xs_0_ubar.reshape(-1, 1)# LQR control law around current position
        #u = us_xs_0_ubar.reshape(-1, 1)
        u_clipped = np.clip(u, -1.0, 1.0)
        u = u_clipped.flatten()
        print("Step:", t)
        print("us_xs_0_xbar:", us_xs_0_xbar)
        print("us_xs_0_ubar:", us_xs_0_ubar)
        print("z_ref[t, :]:", z_ref[t, :])
        print("u:", u)
        #print("s:",-K_red @ (us_xs_0_xbar - z_ref[t, :]))
        #us_xs_0 = jnp.concatenate([us_xs_0_ubar.flatten(), us_xs_0_xbar.flatten()], axis=0)
        #print("us_xs_0:", us_xs_0) 
        #print("cm:", cm)
        #mpc.update_model_param(us_xs_0,cm[0:3]) # uncommet for aSSM and first-order aSSM

        # Initial reduced state
        end_effector_pos.append(current_state[:, n_elem])  # Store the end effector position
        #print("end_effector_pos:", end_effector_pos[-1])
        #print("end_effector_pos:", end_effector_pos)
        #check =delayed_state(jnp.array(end_effector_pos))
        #print("check shape:", check.shape)
        #print("check:", check)
        #check2 = check - jnp.tile(cm[0:3], 5)
        #print("check2 shape:", check2.shape)
        #print("check2:", check2)
        #reduced_x = delayed_state(jnp.array(end_effector_pos), num_delays=2)  # Reduced state for the aSSM model
        #print("reduced_x shape:", reduced_x.shape)
        #print("reduced_x:", reduced_x)
        
        #u, info = mpc.compute_control(reduced_x)
        #print("Control u shape:", u.shape)
        #print("Control u:", u)
        #print("Control info:", info)
        # only thing to sort is the parametrization map and the performance mapping to nonlinear
        # plus sort out the control bounds.

        # simulate arm in dt-ref window with sim_dt spacing
        #normal_controls =np.tile(np.array(u[:3]), (total_steps, 1))
        #binormal_controls = np.tile(np.array(u[-3:]), (total_steps, 1))
        #u = u_guess[t, :].reshape(-1, 6)  # Reshape to (total_steps, 6) for controls
        #u = u_guess[:, t]
        #print("u shape:", u.shape)  # Should be (3*(T+N+1), 6)

        normal_controls = np.array([u[0], u[2], u[4]])    # positions 0, 2, 4
        binormal_controls = np.array([u[1], u[3], u[5]])  # positions 1, 3, 5
    
        # Simulate one control interval using the ArmSimulator
        interval_results = arm_simulator.simulate_interval(
            normal_control_values=normal_controls,
            binormal_control_values=binormal_controls,
            interval_duration=cl_time
        )

        # Extract the final state from the simulation results
        next_state_pos = jnp.array(interval_results['final_position'])
        next_state_vel = jnp.array(interval_results['final_velocity'])
        next_state = jnp.concatenate([next_state_pos, next_state_vel], axis=1)
    
        # Debug prints
        #print("current_state shape:", current_state.shape)
        #print("initial_position shape:", initial_position.shape) 
        #print("initial_velocity shape:", initial_velocity.shape)
        #print("next_state_pos shape:", next_state_pos.shape)
        #print("next_state_vel shape:", next_state_vel.shape)
        #print("next_state shape:", next_state.shape)
    
        # Store results
        states.append(np.array(current_state))
        controls.append(np.array(u))
    
        # Update current state for next iteration
        current_state = next_state
    
        # Update initial conditions for next simulation step
        # (The simulator maintains its state, but we need to update our tracking)
        initial_position = np.array(next_state_pos)
        initial_velocity = np.array(next_state_vel)

    
    # Store final end effector position
    end_effector_pos.append(current_state[:, n_elem])
    end_effector_pos = np.array(end_effector_pos)
    states = np.array(states)
    controls = np.array(controls)

    # Visualization: compare actual position vs. reference
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(end_effector_pos[:,0], end_effector_pos[:,1], end_effector_pos[:,2], 'b--', label='Koopman MPC solution')
    ax.plot(xt_ref, yt_ref, zt_ref, 'r--', label='Reference')
    ax.set_title('Closed-loop end effector trajectory')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot the end effector position over time and reference
    plt.figure(figsize=(8, 4))
    plt.plot(end_effector_pos[:, 0], label='x position')
    plt.plot(end_effector_pos[:, 1], label='y position')
    plt.plot(end_effector_pos[:, 2], label='z position')
    plt.plot(xt_ref, 'r--', label='Reference x position')
    plt.plot(yt_ref, 'g--', label='Reference y position')
    plt.plot(zt_ref, 'b--', label='Reference z position')
    plt.title('End Effector Position Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Define directory once
    save_dir = 'Koopman_LQR_larger_harder_track'

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Your arrays (assuming they're already defined)
    end_effector_pos = np.array(end_effector_pos)
    states = np.array(states)
    controls = np.array(controls)
    z_ref = np.stack([xt_ref, yt_ref, zt_ref], axis=1)  # shape (T, 3)

    min_length = min(len(end_effector_pos), len(z_ref))
    end_effector_trimmed = end_effector_pos[:min_length]
    z_ref_trimmed = z_ref[:min_length]

    #    Calculate squared differences
    squared_errors = np.sum((end_effector_trimmed - z_ref_trimmed) ** 2, axis=1)

    # Calculate RMSE
    rmse = np.sqrt(np.mean(squared_errors))

    print(f"RMSE between end effector and reference trajectory: {rmse:.6f}")
   
    print("states shape:", states.shape)

    # Method 1: Save as individual NumPy files
    np.save(f'{save_dir}/end_effector_pos.npy', end_effector_pos)
    np.save(f'{save_dir}/states.npy', states)
    np.save(f'{save_dir}/controls.npy', controls)
    np.save(f'{save_dir}/z_ref.npy', z_ref)

    # Method 2: Save as one compressed archive
    np.savez_compressed(f'{save_dir}/data.npz', 
                   end_effector_pos=end_effector_pos,
                   states=states,
                   controls=controls,
                   z_ref=z_ref)

    # Method 3: Save as CSV files
    # End effector positions
    if end_effector_pos.shape[1] == 3:
        ee_header = 'x_pos,y_pos,z_pos'
    else:
        ee_header = ','.join([f'pos_{i}' for i in range(end_effector_pos.shape[1])])

    np.savetxt(f'{save_dir}/end_effector_pos.csv', end_effector_pos, 
           delimiter=',', header=ee_header)

    # States - positions only in interleaved format
    if states.ndim == 3:
        print(f"Original states shape: {states.shape}")
        
        # Extract positions (first 21 elements from each dimension)
        positions = states[:, :, :21]  # Shape: (601, 3, 21)
        
        # Reorder to x_0,y_0,z_0,x_1,y_1,z_1,... format
        positions_interleaved = positions.transpose(0, 2, 1).reshape(positions.shape[0], -1)
        
        np.savetxt(f'{save_dir}/states_positions.csv', positions_interleaved, delimiter=',')
        print(f"Positions saved in x,y,z interleaved format: {positions.shape} -> {positions_interleaved.shape}")
    else:
        np.savetxt(f'{save_dir}/states.csv', states, delimiter=',')

    # Controls and reference
    np.savetxt(f'{save_dir}/controls.csv', controls, delimiter=',')
    np.savetxt(f'{save_dir}/z_ref.csv', z_ref, 
            delimiter=',', header='x_ref,y_ref,z_ref')

    print(f"Files saved successfully in '{save_dir}/' directory!")
    print("\nSaved files:")
    print("- Individual .npy files: end_effector_pos.npy, states.npy, controls.npy, z_ref.npy")
    print("- Compressed archive: data.npz")
    print("- CSV files: end_effector_pos.csv, states_positions.csv, controls.csv, z_ref.csv")

    # Easy extraction examples:
    print("\n" + "="*50)
    print("EASY EXTRACTION EXAMPLES:")
    print("="*50)

    print(f"\n1. Load individual NumPy files:")
    print(f"end_effector_pos = np.load('{save_dir}/end_effector_pos.npy')")
    print(f"states = np.load('{save_dir}/states.npy')")
    print(f"controls = np.load('{save_dir}/controls.npy')")
    print(f"z_ref = np.load('{save_dir}/z_ref.npy')")

    print(f"\n2. Load from compressed archive:")
    print(f"data = np.load('{save_dir}/data.npz')")
    print("end_effector_pos = data['end_effector_pos']")
    print("states = data['states']")
    print("controls = data['controls']")
    print("z_ref = data['z_ref']")


if __name__ == "__main__":
    run_mpc_demo()
