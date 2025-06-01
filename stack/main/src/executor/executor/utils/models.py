"""
Reduced order models of controlled systems.
"""
import os
import pickle
import jax
import jax.numpy as jnp
from functools import partial
from .misc import trajectories_delay_embedding, trajectories_derivatives, RK4_step, compute_rmse


class ReducedOrderModel:
    """
    Base class for reduced order models.
    """
    def __init__(self, n_x, n_u, n_y, n_z):
        self.n_x = n_x
        self.n_u = n_u
        self.n_y = n_y
        self.n_z = n_z

    def continuous_dynamics(self, x, u):
        """
        Continuous dynamics of the system.
        """
        raise NotImplementedError

    def discrete_dynamics(self, x, u, dt=0.01):
        """
        Discrete-time dynamics of the system.
        """
        raise NotImplementedError

    def rollout(self, x0, u, dt=0.01):
        """
        Rollout of the model with a given control sequence at an initial condition.
        """
        raise NotImplementedError

    def performance_mapping(self, x):
        """
        Performance mapping maps the state, x, to the performance output, z.
        """
        raise NotImplementedError
    
    @property
    def H(self):
        """
        Linear transformation from the state, x, to the performance variable, z.
        """
        raise NotImplementedError


class Residual_dynamics:
    def __init__(self, ssm_basis):
        self.basis = ssm_basis

    def __call__(self, delayed_ref_vec):
        return self.basis.T @ delayed_ref_vec


class control_SSMR(ReducedOrderModel):
    """
    SSMR model combining a delay SSM with a residual dynamics model.
    """

    def __init__(self, delay_config, ssm_path):

        model_file_path = ssm_path

        # Check if the file exists
        if not os.path.exists(model_file_path):
            print(f"Error: The file {model_file_path} does not exist.")
            exit(1)

        # Load the saved model from the pickle file.
        with open(model_file_path, "rb") as f:
            ssm = pickle.load(f)

        # Autonomous dynamics model
        self.ssm = ssm

        # Residual dynamics model
        self.residual_dynamics = Residual_dynamics(ssm.SSM_basis)

        # Observation-performance matrix maps the observations, y, to the performance variable, z
        perf_var_dim = delay_config["perf_var_dim"] * (1 + ssm.specified_params["include_velocity"])
        meas_var_dim = len([x - 1 for x in ssm.specified_params["measured_rows"] if x <= 18])
        obs_perf_matrix = jnp.zeros((perf_var_dim, (meas_var_dim * (1 + int(ssm.specified_params["include_velocity"]))
                                                    + (ssm.specified_params["num_u"] * int(delay_config["also_embedd_u"])))
                                    * (1 + int(ssm.specified_params["embedding_up_to"]))))

        self.obs_perf_matrix = obs_perf_matrix.at[:perf_var_dim, :perf_var_dim].set(jnp.eye(perf_var_dim))

        n_x = ssm.SSM_basis.shape[1]  # n_x: reduced state dimension
        n_u = ssm.specified_params["num_u"]  # n_u: number control variables
        n_z, n_y = obs_perf_matrix.shape  # n_z: number performance varliables; n_y: full state dimension

        super().__init__(n_x, n_u, n_y, n_z)

    @partial(jax.jit, static_argnums=(0,))
    def continuous_dynamics(self, x, u):
        """
        Continuous dynamics of reduced system.
        """
        return self.ssm.reduced_dynamics(x) + self.residual_dynamics(u)

    @partial(jax.jit, static_argnums=(0,))
    def discrete_dynamics_helper(self, x, u, dt=0.01):
        """
        Discrete-time dynamics of reduced system using RK4 integration.
        """
        return RK4_step(self.continuous_dynamics, x, u, dt)

    @partial(jax.jit, static_argnums=(0,))
    def discrete_dynamics(self, x_tilde, u, dt=0.01):
        """
        Augmenting the discrete function with the past u reference to correctly augment it
        x_tilde is assumed to contain [x_delay_aug, u_past]
        num_delay * shift_steps past ref values will be needed
        """
        if self.ssm.specified_params["embedding_up_to"] == 0:
            u_ref_ext = jnp.vstack([jnp.zeros((self.n_y - u.shape[0], 1)), -self.ssm.lam @ u.reshape(-1, 1)]).flatten()
            return self.discrete_dynamics_helper(x_tilde, u_ref_ext, dt)
        else:
            x, u_past = x_tilde[:self.n_x], x_tilde[self.n_x:]
            u_past_shifted = u_past.reshape(-1, self.n_u)  # [::self.aug_ssm.specified_params["shift_steps"]]

            # Proceed with vstack
            u_ref_ext = jnp.vstack(
                [jnp.vstack([jnp.zeros((len(self.ssm.specified_params["measured_rows"]), 1)), -self.ssm.lam @ u.reshape(-1, 1)])] +
                [jnp.vstack([jnp.zeros((len(self.ssm.specified_params["measured_rows"]), 1)), -self.ssm.lam @ u_past_shifted[i].reshape(-1, 1)])
                 for i in range(0, u_past_shifted.shape[0], (1 + self.ssm.specified_params["shift_steps"]))]
            ).flatten()
            u_flat = u.flatten()  # Shape (2,)
            u_past_flat = u_past[:-self.n_u].flatten()  # Shape (8,)
            u_stacked = jnp.concatenate([u_flat, u_past_flat])
            result = jnp.concatenate([self.discrete_dynamics_helper(x, u_ref_ext, dt).flatten(), u_stacked.flatten()])
            return result

    def dynamics_step(self, x, u_dt):
        """
        Perform a single step of the reduced dynamics.
        X is required to be augmented with the past reference values
        """
        u, dt = u_dt[:-1], u_dt[-1]
        return self.discrete_dynamics(x, u, dt), x

    def rollout(self, x0, u, dt=0.01):
        """
        Rollout of the discrete-time dynamics model, with u being an array of length N.
        Note that if u has length N, then the output will have length N+1.
        x0 is in the reduced coordinates
        """
        u_dt = jnp.column_stack([u, jnp.full(u.shape[0], dt)])  # shape of u is (N, n_u)
        final_state, xs = jax.lax.scan(self.dynamics_step, x0, u_dt)
        return jnp.vstack([xs, final_state])

    def performance_mapping(self, x):
        """
        Performance mapping maps the state, x, to the performance output, z, through
        z = C @ y = C @ w(x).
        """
        return self.obs_perf_matrix @ self.decode(x)

    @property
    def H(self):
        """
        Linear mapping from the state, x, to the performance variable, z.
        """
        raise AttributeError("SSMR uses a nonlinear performance mapping, hence H is not defined.")

    def encode(self, y):
        """
        Encode the observations, y, into the reduced state, x.
        """
        return self.ssm.encode(y)

    def decode(self, x):
        """
        Decode the reduced state, x, into the observations, y.
        """
        red_coordinates = x[:self.n_x]
        return self.ssm.decode(red_coordinates)

    def save_model(self, path):
        """
        Save the SSMR model to a file.
        """
        raise NotImplementedError


def get_residual_labels(delay_ssm, trajs, ts, u_func=None, rnd_key=jax.random.PRNGKey(0), us=None):
    """
    Get labels for B_r learning.
    """
    # Either provide the control inputs or the control function
    if us is None and u_func is None:
        raise ValueError("Either control inputs or control function must be provided.")

    N_trajs = len(trajs)
    ys = trajectories_delay_embedding(trajs, 1)
    x_trajs = []
    for traj in ys:
        x_traj = delay_ssm.encode(traj)
        # Apply padding of zeros to the end of the trajectory
        x_traj = x_traj.at[:, -1:].set(jnp.zeros((delay_ssm.SSMDim, 1)))
        x_trajs.append(x_traj)
    x_trajs = jnp.array(x_trajs)

    x_dots_ctrl = trajectories_derivatives(x_trajs, ts)
    x_dots_aut = []
    for traj in x_trajs:
        x_dot_aut = delay_ssm.reduced_dynamics(traj)
        x_dots_aut.append(x_dot_aut)
    x_dots_aut = jnp.array(x_dots_aut)

    delta_x_dots = x_dots_ctrl - x_dots_aut
    delta_x_dots_flat = delta_x_dots.transpose(0, 2, 1).reshape(-1, delay_ssm.SSMDim)
    xs_flat = x_trajs.transpose(0, 2, 1).reshape(-1, delay_ssm.SSMDim)
    us = u_func(ts, N_trajs, rnd_key) if us is None else us  # shape of us is (N_trajs, n_u, len(ts))
    us_flat = us.transpose(0, 2, 1).reshape(-1, us.shape[1])
    return xs_flat, us_flat, delta_x_dots_flat


def generate_ssmr_predictions(ssmr, trajs, ts, u_func=None, rnd_key=None, us=None):
    """
    Generate tip positions as predicted by SSMR model for entire trajectories.
    """
    # Either provide the control inputs or the control function
    if us is None and u_func is None:
        raise ValueError("Either control inputs or control function must be provided.")

    N_trajs = len(trajs)
    us = u_func(ts, N_trajs, rnd_key) if us is None else us
    N_input_states = trajs.shape[1]
    ssmr_predictions = jnp.zeros_like(trajs)
    for i, traj in enumerate(trajs):
        # Assume first 2 observations are known
        ssmr_predictions = ssmr_predictions.at[i, :, :2].set(traj[:, :2])
        y0 = jnp.flip(traj[:, :2], 1).T.flatten()
        x0 = ssmr.delay_ssm.encode(y0)
        xs = ssmr.rollout(x0, us[i, :, 2:].T)[:-1].T  # exclude the last, (N+1)th, state 
        ys = ssmr.delay_ssm.decode(xs)
        ssmr_predictions = ssmr_predictions.at[i, :, 2:].set(ys[:N_input_states, :])  # select the non-delayed predictions
    return ssmr_predictions


def get_ssmr_finite_horizon_accuracy(system, ssmr, N_horizon, N_ics, dt, u_func, rnd_key):
    """
    Generate tip positions as predicted by SSMR model over finite horizon.
    """
    ts_horizon = jnp.arange(0, N_horizon * dt, dt)
    ctrl_trajs = system.generate_controlled_trajs(N_ics, ts_horizon, u_func, rnd_key)
    ctrl_trajs_obs = system.get_observations(ctrl_trajs)
    ssmr_predictions = generate_ssmr_predictions(ssmr, ctrl_trajs_obs, ts_horizon, u_func, rnd_key)
    horizon_ics = ctrl_trajs_obs[:, :, 0]
    rmse_samples = compute_rmse(ctrl_trajs_obs, ssmr_predictions)
    return horizon_ics, rmse_samples
