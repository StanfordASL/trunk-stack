"""
Custom Spectral Submanifold (SSM) model class.
"""

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from functools import partial
from ssmlearnpy import SSMLearn
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import orth
import numpy as np
import sympy as sp
from utils.misc import trajectories_delay_embedding


class DelaySSM:
    """
    Delay SSM model constructed with SSMLearnPy and inferred using JAX.
    """
    def __init__(self,
                 aut_trajs_obs,             # observed trajectories
                 SSMDim: int,               # dimension of SSM
                 SSMOrder: int,             # expansion order encoding/decoding
                 ROMOrder: int,             # expansion order of reduced dynamics
                 N_delay: int,              # number of delays
                 N_obs_delay: int=None,     # number of observed delays, where None means no reparameterization
                 orthogonalize: bool=True,  # whether to orthogonalize reduced coordinates
                 encoder: dict=None,        # encoder coefficients and exponents
                 ts=None,                   # time array
                 dt=None,                   # time step
                 ):
        self.n_y = aut_trajs_obs.shape[1]

        if N_obs_delay is not None:
            assert N_obs_delay <= N_delay, "Number of observed delays must be less than or equal to total number of delays."
            assert N_obs_delay >= SSMDim//self.n_y -1, "Number of observed delays must be at least SSMDim//n_y - 1."
            self.p = self.n_y * (N_obs_delay + 1)  # total number of observed states
        else:
            self.p = self.n_y * (N_delay + 1)
        assert ts is not None or dt is not None, "Either ts or dt must be provided."
        
        if ts is None:
            steps = aut_trajs_obs.shape[-1]
            ts = np.arange(0, steps * dt, dt)
            print("Total time steps:", len(ts))

        self.SSMDim = SSMDim
        self.SSMOrder = SSMOrder
        self.ROMOrder = ROMOrder
        self.N_delay = N_delay
        self.N_obs_delay = N_obs_delay
        self.orthogonalize = orthogonalize
        
        delayed_trajs_np = self._compute_delayed_trajs(aut_trajs_obs, N_delay)
        if encoder is not None:
            # If encoder is provided, use the encoder to find reduced coordinates
            self.encoder_coeff = jnp.array(encoder['coefficients'])
            self.encoder_exp = jnp.array(encoder['exponents'])
            reduced_delayed_trajs_np = []
            for traj in delayed_trajs_np:
                reduced_delayed_trajs_np.append(self.encode(traj[:self.p, :]))
            reduced_delayed_trajs_np = np.array(reduced_delayed_trajs_np)
        else:
            # If no encoder is provided, find reduced coordinates using SVD
            self.encoder_coeff = None
            self.encoder_exp = None
            reduced_delayed_trajs_np = self._find_reduced_coordinates(delayed_trajs_np, ts)
        self._fit_ssm(delayed_trajs_np, reduced_delayed_trajs_np, ts, SSMOrder, ROMOrder)

    def _compute_delayed_trajs(self, aut_trajs_obs, N_delay):
        """
        Compute delayed trajectories.
        """
        return np.array(trajectories_delay_embedding(aut_trajs_obs, N_delay, skips=0))

    def _find_reduced_coordinates(self, delayed_trajs_np, ts):
        """
        Find reduced coordinates using randomized SVD.
        """
        delayed_trajs_np_flat = np.hstack([delayed_traj_np for delayed_traj_np in delayed_trajs_np])
        V, _, _ = randomized_svd(delayed_trajs_np_flat, n_components=self.SSMDim)
        reduced_delayed_trajs_np_flat = np.dot(delayed_trajs_np_flat.T, V)
        reduced_delayed_trajs_np = reduced_delayed_trajs_np_flat.reshape(len(delayed_trajs_np), len(ts), self.SSMDim).transpose(0, 2, 1)
        return reduced_delayed_trajs_np

    def _fit_ssm(self, delayed_trajs_np, reduced_delayed_trajs_np, ts, SSMOrder, ROMOrder):
        """
        We fit the SSM. Note that we reparameterize observations, i.e. we find encoder and decoder for
        p = n_y*(N_obs_delay+1) observations instead of n_y*(N_delay+1).
        """
        delayed_trajs_obs_np = delayed_trajs_np[:, :self.p, :]

        # Construct parameterization (decoder)
        ssm_paramonly = SSMLearn(
            t = [ts] * len(delayed_trajs_obs_np),
            x = [delayed_traj_obs_np for delayed_traj_obs_np in delayed_trajs_obs_np],
            reduced_coordinates = [reduced_delayed_traj_np for reduced_delayed_traj_np in reduced_delayed_trajs_np],
            ssm_dim = self.SSMDim,
            dynamics_type = 'flow'
        )
        ssm_paramonly.get_parametrization(poly_degree=SSMOrder, alpha=1.0)

        if self.orthogonalize:
            # Calculate tangent space at origin and orthogonalize
            tanspace0_not_orth = ssm_paramonly.decoder.map_info['coefficients'][:self.SSMDim, :self.SSMDim]
            tanspace0 = orth(tanspace0_not_orth)

            # Change reduced coordinates to be orthogonal
            reduced_delayed_trajs_orth_np = np.zeros_like(reduced_delayed_trajs_np)
            for i, traj in enumerate(reduced_delayed_trajs_np):
                reduced_delayed_trajs_orth_np[i] = tanspace0.T @ tanspace0_not_orth @ traj

            # Construct parameterization (decoder) with orthogonalized reduced coordinates
            ssm_paramonly_orth = SSMLearn(
                t = [ts] * len(delayed_trajs_obs_np),
                x = [delayed_traj_obs_np for delayed_traj_obs_np in delayed_trajs_obs_np],
                reduced_coordinates = [reduced_delayed_traj_orth_np for reduced_delayed_traj_orth_np in reduced_delayed_trajs_orth_np],
                ssm_dim = self.SSMDim,
                dynamics_type = 'flow'
            )
            ssm_paramonly_orth.get_parametrization(poly_degree=SSMOrder, alpha=1.0)
            self.decoder_coeff = jnp.array(ssm_paramonly_orth.decoder.map_info['coefficients'])
            self.decoder_exp = jnp.array(ssm_paramonly_orth.decoder.map_info['exponents'])
            
            # Get new dynamics with orthogonalized reduced coordinates
            ssm_paramonly_orth.get_reduced_dynamics(poly_degree=ROMOrder, alpha=50.0)
            self.dynamics_coeff = jnp.array(ssm_paramonly_orth.reduced_dynamics.map_info['coefficients'])
            self.dynamics_exp = jnp.array(ssm_paramonly_orth.reduced_dynamics.map_info['exponents'])

            # Update reduced coordinates for obtaining the encoder
            reduced_delayed_trajs_np = reduced_delayed_trajs_orth_np
        else:
            self.decoder_coeff = jnp.array(ssm_paramonly.decoder.map_info['coefficients'])
            self.decoder_exp = jnp.array(ssm_paramonly.decoder.map_info['exponents'])

            # Get reduced dynamics
            ssm_paramonly.get_reduced_dynamics(poly_degree=ROMOrder, alpha=50.0)
            self.dynamics_coeff = jnp.array(ssm_paramonly.reduced_dynamics.map_info['coefficients'])
            self.dynamics_exp = jnp.array(ssm_paramonly.reduced_dynamics.map_info['exponents'])

        if self.encoder_coeff is None:
            # Construct chart (encoder) with, potentially orthogonalized, reduced coordinates
            # Note that due to reparameterization, this map is not simply linear, but also polynomial
            ssm_chartonly = SSMLearn(
                t = [ts] * len(delayed_trajs_obs_np),
                x = [reduced_delayed_traj_np for reduced_delayed_traj_np in reduced_delayed_trajs_np],
                reduced_coordinates = [delayed_traj_obs_np for delayed_traj_obs_np in delayed_trajs_obs_np],
                ssm_dim = self.p,
                dynamics_type = 'flow'
            )
            ssm_chartonly.get_parametrization(poly_degree=SSMOrder, alpha=1.0)
            self.encoder_coeff = jnp.array(ssm_chartonly.decoder.map_info['coefficients'])
            self.encoder_exp = jnp.array(ssm_chartonly.decoder.map_info['exponents'])

    @partial(jax.jit, static_argnums=(0,))
    def reduced_dynamics(self, x):
        """
        Evaluate the continuous-time dynamics of the reduced system, with batch dimension last.
        """
        if x.ndim == 1:
            x = jnp.expand_dims(x, -1)
            single_input = True
        else:
            single_input = False

        n, n_coeff = self.dynamics_coeff.shape
        results = []

        # Loop over each dimension to compute the derivative
        for dim in range(n):
            polynomial = 0
            for j in range(n_coeff):
                exponents = jnp.expand_dims(self.dynamics_exp[j, :], axis=-1)
                term = self.dynamics_coeff[dim, j] * jnp.prod(x ** exponents, axis=0, keepdims=True)
                polynomial += term
            results.append(polynomial)
        
        x_dot = jnp.concatenate(results, axis=0)

        if single_input:
            return x_dot.squeeze(-1)
        return x_dot
    
    @partial(jax.jit, static_argnums=(0,))
    def simulate_reduced(self, x0, t):
        """
        Simulate the reduced system.
        """
        return odeint(lambda x, _: self.reduced_dynamics(x), x0, t).T

    @partial(jax.jit, static_argnums=(0,))
    def decode(self, x):
        """
        Decode from reduced state to observation, with batch dimension last.
        """
        if x.ndim == 1:
            x = jnp.expand_dims(x, -1)
            single_input = True
        else:
            single_input = False

        p, p_coeff = self.decoder_coeff.shape
        results = []

        # Loop over each dimension to compute the observation
        for obs_dim in range(p):
            polynomial = 0
            for j in range(p_coeff):
                exponents = jnp.expand_dims(self.decoder_exp[j, :], axis=-1)
                term = self.decoder_coeff[obs_dim, j] * jnp.prod(x ** exponents, axis=0, keepdims=True)
                polynomial += term
            results.append(polynomial)

        y = jnp.concatenate(results, axis=0)

        if single_input:
            return y.squeeze(-1)
        return y
 
    @partial(jax.jit, static_argnums=(0,))
    def encode(self, y):
        """
        Encode from observation to reduced state, with batch dimension last.
        """
        if y.ndim == 1:
            y = jnp.expand_dims(y, -1)
            single_input = True
        else:
            single_input = False

        n, n_coeff = self.encoder_coeff.shape
        results = []

        # Loop over each dimension to compute the reduced state
        for dim in range(n):
            polynomial = 0
            for j in range(n_coeff):
                exponents = jnp.expand_dims(self.encoder_exp[j, :], axis=-1)
                term = self.encoder_coeff[dim, j] * jnp.prod(y ** exponents, axis=0, keepdims=True)
                polynomial += term
            results.append(polynomial)

        x = jnp.concatenate(results, axis=0)

        if single_input:
            return x.squeeze(-1)
        return x
    
    def get_symb_reduced_dynamics(self):
        """
        Generate symbolic version of reduced dynamics, which can be used in a
        notebook as:
        for eqn in reduced_dynamics_eqns:
            display(eqn)
        """
        qs = [sp.symbols(f'q_{i}') for i in range(1, self.SSMDim + 1)]
        equations = []
        for dim, dim_coefficients in enumerate(np.array(self.dynamics_coeff), start=1):
            polynomial = 0
            for coeff, exp in zip(dim_coefficients, np.array(self.dynamics_exp)):
                formatted_coeff = sp.N(coeff, 4)
                polynomial += formatted_coeff * sp.prod([qs[j] ** exp[j] for j in range(self.SSMDim)])
            q_dot = sp.symbols(r'\dot{q}_' + str(dim))
            equations.append(sp.Eq(q_dot, polynomial))
        return equations


def generate_ssm_predictions(delay_ssm: DelaySSM, trajs, ts=None, dt=None):
    """
    Generate tip positions as predicted by SSM model.
    """
    if ts is None:
        steps = trajs.shape[-1]
        ts = np.arange(0, steps * dt, dt)

    N_obs_delay = delay_ssm.N_obs_delay
    N_input_states = trajs.shape[1]
    ssm_predictions = jnp.zeros_like(trajs)
    for i, traj in enumerate(trajs):
        # Assume first N_obs_delay+1 observations are known
        ssm_predictions = ssm_predictions.at[i, :, :N_obs_delay+1].set(traj[:, :N_obs_delay+1])
        y0 = jnp.flip(traj[:, :(N_obs_delay+1)], 1).T.flatten()
        x0 = delay_ssm.encode(y0)
        xs = delay_ssm.simulate_reduced(x0, ts[N_obs_delay+1:])
        ys = delay_ssm.decode(xs)
        ssm_predictions = ssm_predictions.at[i, :, N_obs_delay+1:].set(ys[:N_input_states, :])
    return ssm_predictions
