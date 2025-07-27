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
from .misc import trajectories_delay_embedding, trajectories_derivatives, polynomial_features
import cvxpy as cp
import pyomo.environ as pyo
from itertools import combinations_with_replacement


class DelaySSM:
    """
    Delay SSM model constructed with SSMLearnPy and inferred using JAX.
    """
    def __init__(self,
                 aut_trajs_obs=None,             # observed trajectories
                 SSMDim: int=None,               # dimension of SSM
                 SSMOrder: int=None,             # expansion order encoding/decoding
                 ROMOrder: int=None,             # expansion order of reduced dynamics
                 N_delay: int=None,              # number of delays
                 N_obs_delay: int=None,          # number of observed delays, where None means no reparameterization
                 orthogonalize: bool=True,       # whether to orthogonalize reduced coordinates
                 encoder: dict=None,             # encoder coefficients and exponents
                 ts=None,                        # time array
                 dt=None,                        # time step
                 model_data=None):               # model data (load if exists)
        if model_data is not None:
            self.dynamics_coeff = jnp.array(model_data['dynamics_coeff'])
            self.dynamics_exp = jnp.array(model_data['dynamics_exp'])
            self.encoder_coeff = jnp.array(model_data['encoder_coeff'])
            self.encoder_exp = jnp.array(model_data['encoder_exp'])
            self.decoder_coeff = jnp.array(model_data['decoder_coeff'])
            self.decoder_exp = jnp.array(model_data['decoder_exp'])
            self.SSMDim = self.dynamics_coeff.shape[0]
            self.n_y = self.decoder_coeff.shape[0]
        else:
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
                reduced_delayed_trajs_np, self.V = self._find_reduced_coordinates(delayed_trajs_np, ts)
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
        return reduced_delayed_trajs_np, V

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
            ssm_paramonly.get_reduced_dynamics(poly_degree=ROMOrder, alpha=1.0)
            self.dynamics_coeff = jnp.array(ssm_paramonly.reduced_dynamics.map_info['coefficients'])
            self.dynamics_exp = jnp.array(ssm_paramonly.reduced_dynamics.map_info['exponents'])

        if self.encoder_coeff is None and self.N_obs_delay is not None:
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
        elif self.N_obs_delay is None:
            self.encoder_coeff = self.V.T
            self.encoder_exp = jnp.eye(self.p)

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


class OptSSM:
    """
    Delay SSM model with optimal (oblique) projection.
    """
    def __init__(self,
                aut_trajs_obs=None,             # observed trajectories
                t_split=None,                   # split time from transient to on-manifold
                SSMDim: int=None,               # dimension of SSM
                SSMOrder: int=None,             # expansion order encoding/decoding
                ROMOrder: int=None,             # expansion order of reduced dynamics
                N_delay: int=None,              # number of delays
                ts=None,                        # time array
                ipopt_executable=None,          # path to IPOPT executable
                verbose=False):                 # verbosity
        self.SSMDim = SSMDim
        self.SSMOrder = SSMOrder
        self.ROMOrder = ROMOrder
        self.N_delay = N_delay
        self.N_obs_delay = N_delay  # NOTE: we do not reparameterize here
        self.t_split = t_split

        # Split the data
        Y_transient, Y_dot_transient, Y_mani, Y_dot_mani = self._split_data(aut_trajs_obs, ts, t_split)

        # Do SVD on data close to manifold
        self.V_n_svd, _, _ = randomized_svd(np.asarray(Y_mani), n_components=SSMDim)

        # Create optimization model using transient data
        ipopt_model = self._create_optimization_model(np.array(Y_transient), np.array(Y_dot_transient))

        # Solve optimization problem
        self.V_n_opt, _, _ = self._solve_with_ipopt(ipopt_model, executable=ipopt_executable, verbose=verbose)

        # Regress R on data close to manifold
        self.R_opt = self._regress_reduced_dynamics(Y_mani, Y_dot_mani, verbose=verbose)

        # Regress W_nl on data close to manifold
        self.W_nl_opt = self._regress_parameterization_map(Y_mani, verbose=verbose)

    def _split_data(self, aut_trajs_obs, ts, t_split):
        """
        Split data into transient and on-manifold.
        """
        dt = ts[1] - ts[0]
        i_cutoff = int(t_split / dt)

        # Transient data
        ts_transient = ts[:i_cutoff]
        aut_trajs_transient = aut_trajs_obs[:, :, :i_cutoff]
        aut_trajs_transient_delay = np.array(trajectories_delay_embedding(aut_trajs_transient, self.N_delay, skips=0))
        p = aut_trajs_transient_delay.shape[1]
        Y_transient = aut_trajs_transient_delay.transpose(1, 0, 2).reshape(p, -1)  # p x N_traj*len(t) := p x N
        Y_dot_transient = trajectories_derivatives(aut_trajs_transient_delay, ts_transient)
        Y_dot_transient = Y_dot_transient.transpose(1, 0, 2).reshape(p, -1)  # p x N_traj*len(t) := p x N

        # Steady data
        ts_mani = ts[i_cutoff:]
        aut_trajs_mani = aut_trajs_obs[:, :, i_cutoff:]
        aut_trajs_mani_delay = np.array(trajectories_delay_embedding(aut_trajs_mani, self.N_delay, skips=0))
        p = aut_trajs_mani_delay.shape[1]
        Y_mani = aut_trajs_mani_delay.transpose(1, 0, 2).reshape(p, -1)  # p x N_traj*len(t) := p x N
        Y_dot_mani = trajectories_derivatives(aut_trajs_mani_delay, ts_mani)
        Y_dot_mani = Y_dot_mani.transpose(1, 0, 2).reshape(p, -1)  # p x N_traj*len(t) := p x N
        return Y_transient, Y_dot_transient, Y_mani, Y_dot_mani

    def _create_optimization_model(self, Y, Y_dot, reg=1e-6):
        """
        Creates a Pyomo concrete model for the optimization problem.
        """
        p, N = Y.shape
        
        # Calculate polynomial feature indices
        poly_terms = []
        # Add linear terms
        poly_terms.extend([(i,) for i in range(self.SSMDim)])
        # Add higher order terms up to degree n_r
        for degree in range(2, self.ROMOrder + 1):
            poly_terms.extend(combinations_with_replacement(range(self.SSMDim), degree))
        m_r = len(poly_terms)
        
        model = pyo.ConcreteModel()
        
        # Sets
        model.p = pyo.RangeSet(0, p-1)
        model.n = pyo.RangeSet(0, self.SSMDim-1)
        model.N = pyo.RangeSet(0, N-1)
        model.m_r = pyo.RangeSet(0, m_r-1)
        
        # Variables
        model.V_n = pyo.Var(model.p, model.n)
        model.R = pyo.Var(model.n, model.m_r)
        
        # Initialize variables
        V_n_init = self.V_n_svd @ np.linalg.inv(self.V_n_svd.T @ self.V_n_svd)
        for i in model.p:
            for j in model.n:
                model.V_n[i,j] = V_n_init[i,j]
        
        # Affine constraints on V_n (V_n_svd.T @ V_n = I)
        def affine_constraint_rule(model, i, j):
            return sum(self.V_n_svd[k,i] * model.V_n[k,j] for k in model.p) == (1.0 if i==j else 0.0)
        model.affine_constraints = pyo.Constraint(model.n, model.n, rule=affine_constraint_rule)
        
        # Define VnY = V_n^T * Y as helper variables
        model.VnY = pyo.Var(model.n, model.N)
        def VnY_rule(model, i, j):
            return model.VnY[i,j] == sum(model.V_n[k,i] * Y[k,j] for k in model.p)
        model.VnY_constraint = pyo.Constraint(model.n, model.N, rule=VnY_rule)
        
        # Define Vn_dotY = V_n^T * Y_dot as helper variables
        model.Vn_dotY = pyo.Var(model.n, model.N)
        def Vn_dotY_rule(model, i, j):
            return model.Vn_dotY[i,j] == sum(model.V_n[k,i] * Y_dot[k,j] for k in model.p)
        model.Vn_dotY_constraint = pyo.Constraint(model.n, model.N, rule=Vn_dotY_rule)
        
        # Helper variables for polynomial terms
        model.Phi = pyo.Var(model.m_r, model.N)
        def phi_rule(model, k, j):
            return model.Phi[k,j] == pyo.prod(model.VnY[idx,j] for idx in poly_terms[k])
        model.phi_constraint = pyo.Constraint(model.m_r, model.N, rule=phi_rule)
        
        # Objective function using pure Pyomo expressions
        def obj_rule(model):
            residual_term = sum(
                (model.Vn_dotY[i,j] - sum(model.R[i,k] * model.Phi[k,j] for k in model.m_r))**2
                for i in model.n for j in model.N
            )
            reg_term = reg * sum(model.R[i,j]**2 for i in model.n for j in model.m_r)
            return residual_term + reg_term
        
        model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        return model
    
    def _solve_with_ipopt(self, model, executable=None, verbose=False):
        """
        Solves the optimization model using IPOPT.
        """
        if executable is not None:
            solver = pyo.SolverFactory('ipopt', executable=executable)
        else:
            solver = pyo.SolverFactory('ipopt')
        if not solver.available():
            raise RuntimeError(
                "IPOPT solver is not available. Please install it using "
                "`conda install -c conda-forge ipopt` and make sure it's on your PATH."
            )
        solver.options['max_iter'] = 500
        solver.options['tol'] = 1e-6
        results = solver.solve(model, tee=verbose)

        # Extract optimal values
        V_n_opt = jnp.array([[pyo.value(model.V_n[i,j]) for j in model.n] for i in model.p])
        R_opt = jnp.array([[pyo.value(model.R[i,j]) for j in model.m_r] for i in model.n])
        
        return V_n_opt, R_opt, results
    
    def _regress_reduced_dynamics(self, Y, Y_dot, verbose=False):
        """
        Regress reduced dynamics.
        """
        m_r = polynomial_features(jnp.zeros(self.SSMDim), self.ROMOrder, 1).shape[1]
        R = cp.Variable((self.SSMDim, m_r))
        objective = cp.Minimize(cp.sum_squares(self.V_n_opt.T @ Y_dot - R @ polynomial_features(Y.T @ self.V_n_opt, self.ROMOrder, 1).T) + 1e-6 * cp.sum_squares(R))
        problem = cp.Problem(objective)
        problem.solve()
        if verbose:
            print('R optimization status: ', problem.status)
        return jnp.array(R.value)
    
    def _regress_parameterization_map(self, Y, verbose=False):
        """
        Regress parameterization map.
        """
        p = Y.shape[0]
        m_w = polynomial_features(Y.T @ self.V_n_opt, self.SSMOrder, 2).shape[1]
        W_nl = cp.Variable((p, m_w))
        objective = cp.Minimize(cp.sum_squares(Y - self.V_n_svd @ self.V_n_opt.T @ Y - W_nl @ polynomial_features(Y.T @ self.V_n_opt, self.SSMOrder, 2).T) + 1e-8 * cp.sum_squares(W_nl))
        constraints = [self.V_n_opt.T @ W_nl == jnp.zeros((self.SSMDim, m_w))]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if verbose:
            print('W_nl optimization status: ', problem.status)
        return jnp.array(W_nl.value)

    @partial(jax.jit, static_argnums=(0,))
    def reduced_dynamics(self, x):
        """
        Evaluate the continuous-time dynamics of the reduced system, with batch dimension last.
        """
        x_dot = self.R_opt @ polynomial_features(x.T, self.ROMOrder, 1).T
        return x_dot.squeeze()

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
        y = self.V_n_svd @ x + self.W_nl_opt @ polynomial_features(x.T, self.SSMOrder, 2).T.squeeze()
        return y

    @partial(jax.jit, static_argnums=(0,))
    def encode(self, y):
        """
        Encode from observation to reduced state, with batch dimension last.
        """
        x = self.V_n_opt.T @ y
        return x


def get_residual_labels(ssm, trajs, ts, u_func=None, rnd_key=jax.random.PRNGKey(0), us=None):
    """
    Get labels for B_r learning.
    """
    # Either provide the control inputs or the control function
    if us is None and u_func is None:
        raise ValueError("Either control inputs or control function must be provided.")

    N_trajs = len(trajs)
    ys = trajectories_delay_embedding(trajs, ssm.N_obs_delay)
    x_trajs = []
    for traj in ys:
        x_traj = ssm.encode(traj)
        # Apply padding of zeros to the end of the trajectory
        x_traj = x_traj.at[:, -(ssm.N_obs_delay):].set(jnp.zeros((ssm.SSMDim, ssm.N_obs_delay)))
        x_trajs.append(x_traj)
    x_trajs = jnp.array(x_trajs)

    x_dots_ctrl = trajectories_derivatives(x_trajs, ts)
    x_dots_aut = []
    for traj in x_trajs:
        x_dot_aut = ssm.reduced_dynamics(traj)
        x_dots_aut.append(x_dot_aut)
    x_dots_aut = jnp.array(x_dots_aut)

    delta_x_dots = x_dots_ctrl - x_dots_aut
    delta_x_dots_flat = delta_x_dots.transpose(0, 2, 1).reshape(-1, ssm.SSMDim)
    xs_flat = x_trajs.transpose(0, 2, 1).reshape(-1, ssm.SSMDim)
    us = u_func(ts, N_trajs, rnd_key) if us is None else us  # shape of us is (N_trajs, n_u, len(ts))
    us_flat = us.transpose(0, 2, 1).reshape(-1, us.shape[1])
    return xs_flat, us_flat, delta_x_dots_flat


def generate_ssm_predictions(ssm, trajs, ts=None, dt=None):
    """
    Generate tip positions as predicted by SSM model.
    """
    if ts is None:
        steps = trajs.shape[-1]
        ts = np.arange(0, steps * dt, dt)

    N_input_states = trajs.shape[1]
    N_obs_delay = ssm.N_obs_delay
    ssm_predictions = jnp.zeros_like(trajs)
    for i, traj in enumerate(trajs):
        # Assume first (N_obs_delay + 1) observations are known
        ssm_predictions = ssm_predictions.at[i, :, :N_obs_delay+1].set(traj[:, :N_obs_delay+1])
        y0 = jnp.flip(traj[:, :N_obs_delay+1], 1).T.flatten()
        x0 = ssm.encode(y0)
        xs = ssm.simulate_reduced(x0, ts[N_obs_delay+1:])
        ys = ssm.decode(xs)
        ssm_predictions = ssm_predictions.at[i, :, N_obs_delay+1:].set(ys[:N_input_states, :])
    return ssm_predictions


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
    N_obs_delay = ssmr.ssm.N_obs_delay
    for i, traj in enumerate(trajs):
        # Assume first (N_obs_delay + 1) observations are known
        ssmr_predictions = ssmr_predictions.at[i, :, :N_obs_delay+1].set(traj[:, :N_obs_delay+1])
        y0 = jnp.flip(traj[:, :N_obs_delay+1], 1).T.flatten()
        x0 = ssmr.ssm.encode(y0)
        xs = ssmr.rollout(x0, us[i, :, N_obs_delay+1:].T)[:-1].T  # exclude the last, (N+1)th, state 
        ys = ssmr.ssm.decode(xs)
        ssmr_predictions = ssmr_predictions.at[i, :, N_obs_delay+1:].set(ys[:N_input_states, :])  # select the non-delayed predictions
    return ssmr_predictions
