"""
Reduced order models of controlled systems.
"""

import jax
import jax.numpy as jnp
import copy
from functools import partial
from utils.ssm import DelaySSM, generate_ssm_predictions
from utils.residual import ResidualBr, PolyBr
from utils.misc import trajectories_delay_embedding, trajectories_derivatives, RK4_step, update_parameter, compute_rmse, sample_truncated_normal


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


class SSMR(ReducedOrderModel):
    """
    SSMR model combining a delay SSM with a residual dynamics model.
    """
    def __init__(self, delay_ssm, residual_dynamics, obs_perf_matrix):
        n_x = delay_ssm.SSMDim
        n_u = residual_dynamics.n_u
        n_z, n_y = obs_perf_matrix.shape
        
        super().__init__(n_x, n_u, n_y, n_z)

        # Autonomous dynamics model
        self.delay_ssm = delay_ssm

        # Residual dynamics model
        self.residual_dynamics = residual_dynamics
        
        # Observation-performance matrix maps the observations, y, to the performance variable, z
        self.obs_perf_matrix = obs_perf_matrix

    def continuous_dynamics(self, x, u):
        """
        Continuous dynamics of reduced system.
        """
        return self.delay_ssm.reduced_dynamics(x) + self.residual_dynamics(x, u)

    @partial(jax.jit, static_argnums=(0,))
    def discrete_dynamics(self, x, u, dt=0.01):
        """
        Discrete-time dynamics of reduced system using RK4 integration.
        """
        return RK4_step(self.continuous_dynamics, x, u, dt)

    def dynamics_step(self, x, u_dt):
        """
        Perform a single step of the reduced dynamics.
        """
        u, dt = u_dt[:-1], u_dt[-1]
        return self.discrete_dynamics(x, u, dt), x

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, x0, u, dt=0.01):
        """
        Rollout of the discrete-time dynamics model, with u being an array of length N.
        Note that if u has length N, then the output will have length N+1.
        """
        u_dt = jnp.column_stack([u, jnp.full(u.shape[0], dt)])  # shape of u is (N, n_u)
        final_state, xs = jax.lax.scan(self.dynamics_step, x0, u_dt)  # TODO: use lambda function instead of u_dt, if performance the same
        return jnp.vstack([xs, final_state])

    def performance_mapping(self, x):
        """
        Performance mapping maps the state, x, to the performance output, z, through
        z = C @ y = C @ w(x).
        """
        return self.obs_perf_matrix @ self.delay_ssm.decode(x)
    
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
        return self.delay_ssm.encode(y)
    
    def decode(self, x):
        """
        Decode the reduced state, x, into the observations, y.
        """
        return self.delay_ssm.decode(x)

class ParametricSSMR(ReducedOrderModel):
    """
    Parametric SSMR model.
    """
    def __init__(self, ssmr_models, parameters, bounds: list, N_particles, default_param=None, true_param=None,
                 interp_scheme='inverse_distance', sigma=0.005, param_cov=0.01, obs_cov=0.01, update_interval=10):
        self.obs_perf_matrix = ssmr_models[0].obs_perf_matrix
        
        n_x = ssmr_models[0].delay_ssm.SSMDim
        n_u = ssmr_models[0].residual_dynamics.n_u
        n_z, n_y = self.obs_perf_matrix.shape

        super().__init__(n_x, n_u, n_y, n_z)
        
        self.bounds = bounds
        self.parameters = parameters
        self.N_particles = N_particles
        self.ssmr_models = ssmr_models
        self.interp_scheme = interp_scheme
        self.sigma = sigma
        self.obs_cov = obs_cov
        self.update_interval = update_interval
        self.num_updates = 0

        assert default_param is not None or true_param is not None, "Either a default or true parameter has to be provided for initialization."
        if true_param is not None:
            self.est_param = true_param
        else:
            self.default_param = default_param
            self.est_param = default_param

        self.weights = self._get_weights(self.est_param)

        self._init_particles(param_cov)

    def update_parameter_estimate(self, x0, us, zs):
        """
        The parameter estimate is updated based on the observations using a particle filter.
        """
        # For each particles, construct SSMR model and compute predicted observations
        self.num_updates += 1

        likelihoods = jnp.zeros(self.N_particles)
        for i, sampled_param in enumerate(self.param_particles):
            model_weights = self._get_weights(sampled_param)
            xs_pred = self._interp_rollout(x0, us, model_weights)[:-1]
            zs_pred = self._interp_decode(xs_pred.T, model_weights)[:self.n_z].T
            error = zs_pred - zs  # shape is (update_interval, n_z)
            likelihood = jnp.exp(-0.5 * jnp.sum(error ** 2)/self.obs_cov)
            likelihoods = likelihoods.at[i].set(likelihood)

        # Normalize the likelihoods to get the weights/probabilities
        param_weights = likelihoods / jnp.sum(likelihoods)

        # Update Gaussian distribution of the parameter estimate using the weights
        self.param_mean = jnp.sum(param_weights * self.param_particles)
        self.param_cov = jnp.sum(param_weights * (self.param_particles - self.param_mean) ** 2)

        # Resample particles
        self.param_particles = sample_truncated_normal(self.param_mean, self.param_cov, self.N_particles, self.parameters[0], self.parameters[-1], key_number=self.num_updates)

        # New parameter estimate is the mean of the particles
        self.est_param = self.param_mean
    
    def update_model(self):
        """
        Update the SSMR model based on the new parameter estimate.
        """
        self.weights = self._get_weights(self.est_param)
    
    def _init_particles(self, param_cov):
        """
        Initialize the particles.
        """
        # Initialize the Gaussian distribution for the parameter estimate
        self.param_mean = self.est_param
        self.param_cov = param_cov
        
        # Sample particles from the Gaussian distribution
        self.param_particles = sample_truncated_normal(self.param_mean, self.param_cov, self.N_particles, self.parameters[0], self.parameters[-1])

    def _get_weights(self, est_param, epsilon=1e-6):
        """
        Get the weights of the parametric SSMR models through interpolation.
        """
        # If the parameter is within epsilon of one of the parameters, return the corresponding model
        for i, param in enumerate(self.parameters):
            if abs(param - est_param) < epsilon:
                return jnp.eye(len(self.parameters))[i]

        if self.interp_scheme == 'inverse_distance':
            # Inverse distance weights
            interp_weights = jnp.array([1 / abs(param - est_param) for param in self.parameters])
        elif self.interp_scheme == 'quadratic':
            # Quadratic weights
            interp_weights = jnp.array([(param - est_param) ** 2 for param in self.parameters])
        elif self.interp_scheme == 'gaussian':
            # Gaussian weights
            squared_diffs = jnp.array([(param - est_param) ** 2 for param in self.parameters])
            interp_weights = jnp.exp(-squared_diffs / (2 * self.sigma ** 2))    
        elif self.interp_scheme == 'exponential_decay':
            # Exponential decay weighting
            abs_diffs = jnp.array([abs(param - est_param) for param in self.parameters])
            interp_weights = jnp.exp(-abs_diffs / self.sigma)
        else:
            raise ValueError(f"Interpolation scheme {self.interp_scheme} not recognized.")
        
        # Normalize the weights to sum to 1
        weights = interp_weights / jnp.sum(interp_weights)
        return weights.squeeze()

    def continuous_dynamics(self, x, u):
        """
        Wrapper for continuous dynamics of reduced system using weighted sum.
        """
        return self._continuous_dynamics(x, u, self.weights)

    def _continuous_dynamics(self, x, u, weights):
        """
        Continuous dynamics of reduced system using weighted sum.
        """
        contributions = jnp.array([w * model.continuous_dynamics(x, u) for w, model in zip(weights, self.ssmr_models)])
        return jnp.sum(contributions, axis=0)
        
    def discrete_dynamics(self, x, u, dt=0.01):
        """
        Wrapper of discrete-time dynamics of reduced system using RK4 integration.
        """
        return self._discrete_dynamics(x, u, self.weights, dt)

    @partial(jax.jit, static_argnums=(0,))
    def _discrete_dynamics(self, x, u, weights, dt=0.01):
        """
        Discrete-time dynamics of reduced system using RK4 integration.
        """
        return RK4_step(lambda x, u: self._continuous_dynamics(x, u, weights), x, u, dt)

    def _dynamics_step(self, x, u_dt, weights):
        """
        Perform a single step of the reduced dynamics.
        """
        u, dt = u_dt[:-1], u_dt[-1]
        return self._discrete_dynamics(x, u, weights, dt), x

    def rollout(self, x0, u, dt=0.01):
        """
        Wrapper for rollout of the model with a given control sequence at an initial condition.
        """
        return self._interp_rollout(x0, u, self.weights, dt)

    @partial(jax.jit, static_argnums=(0,))
    def _interp_rollout(self, x0, u, weights, dt=0.01):
        """
        Rollout of the discrete-time dynamics model, with u being an array of length N.
        Note that if u has length N, then the output will have length N+1.
        """
        u_dt = jnp.column_stack([u, jnp.full(u.shape[0], dt)])  # shape of u is (N, n_u)
        final_state, xs = jax.lax.scan(lambda x, u_dt: self._dynamics_step(x, u_dt, weights), x0, u_dt)
        return jnp.vstack([xs, final_state])

    def performance_mapping(self, x):
        """
        Wrapper for performance mapping.
        """
        return self._performance_mapping(x, self.weights)
    
    @partial(jax.jit, static_argnums=(0,))
    def _performance_mapping(self, x, weights):
        """
        Performance mapping maps the state, x, to the performance output, z, through
        z = C @ y = C @ w(x).
        """
        contributions = jnp.array([w * model.delay_ssm.decode(x) for w, model in zip(weights, self.ssmr_models)])
        sum_contributions = jnp.sum(contributions, axis=0)
        return self.obs_perf_matrix @ sum_contributions  
    
    @property
    def H(self):
        """
        Linear mapping from the state, x, to the performance variable, z, through
        z = H @ x.
        """
        raise AttributeError("ParametricSSMR uses a nonlinear performance mapping, hence H is not defined.")

    def get_dynamics_linearizations(self, x, u):
        """
        Wrapper for dynamics linearizations.
        """
        return self._get_interp_dynamics_linearizations(x, u, self.weights)
    
    def get_perf_mapping_linearizations(self, x):
        """
        Wrapper for performance mapping linearizations.
        """
        return self._get_interp_perf_mapping_linearizations(x, self.weights)

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0, 0, None))
    def _get_interp_dynamics_linearizations(self, x, u, weights):
        """
        Obtain the affine dynamics of each point along trajectory in a list.
        """
        f = partial(self._discrete_dynamics, dt=0.01)
        A, B = jax.jacfwd(f, argnums=(0, 1))(x, u, weights)
        d = f(x, u, weights) - A @ x - B @ u
        return A, B, d

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0, None))
    def _get_interp_perf_mapping_linearizations(self, x, weights):
        """
        Obtain the affine performance mappings at each point along trajectory in a list.
        """
        g = self._performance_mapping
        H = jax.jacfwd(g, argnums=0)(x, weights)
        c = g(x, weights) - H @ x
        return H, c
    
    def encode(self, y):
        """
        Encode the observations, y, into the reduced state, x.
        Note that the encoding is done using the first model in the collection, i.e. the first model defines the global coordinates.
        """
        return self.ssmr_models[0].delay_ssm.encode(y)
    
    def decode(self, x):
        """
        Wrapper for the decoder.
        """
        return self._interp_decode(x, self.weights)
    
    @partial(jax.jit, static_argnums=(0,))
    def _interp_decode(self, x, weights):
        """
        Decode the reduced state, x, into the observations, y.
        """
        contributions = jnp.array([w * model.delay_ssm.decode(x) for w, model in zip(weights, self.ssmr_models)])
        return jnp.sum(contributions, axis=0)


def generate_ssmr_model_collection(system, base_config, param_name: str, parameters: list, N_aut: int, N_ctrl: int, ts,
                                   u_func, poly_degree, SSMDim, SSMOrder, ROMOrder, N_delay, N_obs_delay, rnd_key, param_idx=None):
    """
    Model collection for parametric SSMR model.
    """
    SSMR_models = []
    for i, param_value in enumerate(parameters):
        config = update_parameter(copy.deepcopy(base_config), param_name, param_value, param_idx)
        system_instance = system('3-link_pendulum_dynamics', '3-link_pendulum_positions', config=config)
        
        aut_trajs = system_instance.generate_autonomous_trajs(N_aut, ts, rnd_key)
        aut_trajs_obs = system_instance.get_observations(aut_trajs)
        
        if i == 0:
            # We assume the first model is the 'default model' and we use it to define the global reduced coordinates
            delay_ssm = DelaySSM(aut_trajs_obs, SSMDim, SSMOrder, ROMOrder, N_delay, N_obs_delay, ts=ts)
            encoder = {'coefficients': jnp.copy(delay_ssm.encoder_coeff), 'exponents': jnp.copy(delay_ssm.encoder_exp)}
        else:
            # Other models are constructed using the global encoder
            # We do not orthogonalize the reduced coordinates to maintain the same global reduced coordinates
            delay_ssm = DelaySSM(aut_trajs_obs, SSMDim, SSMOrder, ROMOrder, N_delay, N_obs_delay, orthogonalize=False, encoder=encoder, ts=ts)

        aut_trajs_pred = generate_ssm_predictions(delay_ssm, aut_trajs_obs, ts)
        aut_rmse = compute_rmse(aut_trajs_obs, aut_trajs_pred)
        print(f"Mean RMSE is {jnp.mean(aut_rmse)*100:.6f} [cm] for DelaySSM model with parameter being {param_value}.")

        ctrl_trajs = system_instance.generate_controlled_trajs(N_ctrl, ts, u_func, rnd_key)
        ctrl_trajs_obs = system_instance.get_observations(ctrl_trajs)
        
        xs_flat, us_flat, delta_x_dots_flat = get_residual_labels(delay_ssm, ctrl_trajs_obs, ts, u_func, rnd_key)
        
        poly_B_r = PolyBr(SSMDim, 6, poly_degree)
        poly_B_r.fit(xs_flat, us_flat, delta_x_dots_flat)
        residual_B_r = ResidualBr(poly_B_r)
        
        obs_perf_matrix = jnp.zeros((2, delay_ssm.p))
        obs_perf_matrix = obs_perf_matrix.at[:, 0:2].set(jnp.eye(2))

        poly_ssmr = SSMR(delay_ssm, residual_B_r, obs_perf_matrix)

        ctrl_trajs_poly_pred = generate_ssmr_predictions(poly_ssmr, ctrl_trajs_obs, ts, u_func, rnd_key)

        ctrl_poly_rmse = compute_rmse(ctrl_trajs_obs, ctrl_trajs_poly_pred)
        print(f"Mean RMSE is {jnp.mean(ctrl_poly_rmse)*100:.6f} [cm] for SSMR model with parameter being {param_value}.")
        
        SSMR_models.append(poly_ssmr)

    return SSMR_models


def get_residual_labels(delay_ssm, trajs, ts, u_func=None, rnd_key=jax.random.PRNGKey(0), us=None):
    """
    Get labels for B_r learning.
    """
    # Either provide the control inputs or the control function
    if us is None and u_func is None:
        raise ValueError("Either control inputs or control function must be provided.")

    N_trajs = len(trajs)
    ys = trajectories_delay_embedding(trajs, delay_ssm.N_obs_delay)
    x_trajs = []
    for traj in ys:
        x_traj = delay_ssm.encode(traj)
        # Apply padding of zeros to the end of the trajectory
        x_traj = x_traj.at[:, -delay_ssm.N_obs_delay:].set(jnp.zeros((delay_ssm.SSMDim, delay_ssm.N_obs_delay)))
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

    N_obs_delay = ssmr.delay_ssm.N_obs_delay
    N_trajs = len(trajs)
    us = u_func(ts, N_trajs, rnd_key) if us is None else us
    n_z = ssmr.n_z
    ssmr_predictions = jnp.zeros_like(trajs)
    for i, traj in enumerate(trajs):
        # Assume first N_obs_delay+1 observations are known
        ssmr_predictions = ssmr_predictions.at[i, :, :N_obs_delay+1].set(traj[:, :N_obs_delay+1])
        y0 = jnp.flip(traj[:, :(N_obs_delay+1)], 1).T.flatten()
        x0 = ssmr.delay_ssm.encode(y0)
        xs = ssmr.rollout(x0, us[i, :, N_obs_delay+1:].T)[:-1].T  # exclude the last, (N+1)th, state 
        ys = ssmr.delay_ssm.decode(xs)
        ssmr_predictions = ssmr_predictions.at[i, :, N_obs_delay+1:].set(ys[:n_z, :])  # select the non-delayed predictions
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
