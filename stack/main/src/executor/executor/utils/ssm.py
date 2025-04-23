"""
Custom Spectral Submanifold (SSM) model class.
"""

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from functools import partial
from ssmlearnpy import SSMLearn
from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
from ssmlearnpy.geometry.encode_decode import decode_geometry
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import orth
import numpy as np
import sympy as sp
from .misc import trajectories_delay_embedding
from .rff_kernel import FittedMapping


class Control_Aug_SSM(SSMLearn):
    def __init__(self,
                 reduced_coordinates_aug: list = None,
                 _lambda: jnp.ndarray = None,
                 dim_x=0,
                 SSM_basis: jnp.ndarray = None,
                 specified_params: dict = None,
                 orthogonalize: bool = False,
                 num_obs_delays=0,
                 **kwargs):
        # Initialize the base class with necessary arguments.
        super().__init__(**kwargs)
        # Store additional data. Ensure that data provided as NumPy objects is later converted to JAX arrays.
        self.emb_data['reduced_coordinates_aug'] = reduced_coordinates_aug
        self._lambda = _lambda
        self.dim_x = dim_x
        self.SSM_basis = SSM_basis
        self.orthogonalize = orthogonalize
        self.num_obs_delays = num_obs_delays
        self.specified_params = specified_params

    def encode(self, y):
        return self.SSM_basis.T @ y

    def decode(self, y):
        return (self.decoder.predict(y.T)).T

    def get_parametrization_with_orth(
            self,
            rbf_args=None,
            n_components=1000,
            n_components_orth=1000,
            **regression_args
    ) -> None:
        """
        Learn a reparameterized mapping using JAX with orthogonalization of reduced coordinates.
        The decoder call functions are fully JAX-compatible for differentiability.
        """
        if rbf_args is None:
            rbf_args = {
                'epsilon': 1.0,
                'regularization': 1e-5,
                'epsilon_orth': 1.0,
                'regularization_orth': 1e-5,
            }

        # --- Step 1: Learn initial parametrization ---
        n = self.ssm_dim
        delayed_trajs_obs_np = self.input_data['observables']
        num_obs_states = self.dim_x * (1 + self.num_obs_delays)

        # Convert the numpy arrays to JAX arrays for full JAX compatibility
        delayed_trajs_obs_jax = [jnp.array(traj[:num_obs_states, :]) for traj in delayed_trajs_obs_np]

        # Define the initial parametrization (decoder)
        ssm_paramonly = Control_Aug_SSM(
            t=self.input_data['time'],
            x=delayed_trajs_obs_jax,
            reduced_coordinates_aug=self.emb_data['reduced_coordinates_aug'],
            derive_embdedding=False,
            ssm_dim=self.ssm_dim,
            _lambda=self._lambda,
            dim_x=self.dim_x,
            SSM_basis=self.SSM_basis,
            dynamics_type='flow',
            orthogonalize=self.orthogonalize,
            num_obs_delays=self.num_obs_delays,
            specified_params=self.specified_params
        )
        ssm_paramonly.get_parametrization(
            rbf_args={'epsilon': rbf_args.get('epsilon', 1.0),
                      'regularization': rbf_args.get('regularization', 1.0)},
            n_components=n_components
        )

        # At this point, ssm_paramonly.decoder.map_info['linear_part'] is the full Jacobian (V_lin)
        v_lin = ssm_paramonly.decoder.map_info['linear_part']  # shape: (output_dim, input_dim)

        # --- Step 2: Compute orthonormal basis for the tangent space ---
        v_orth = orth(v_lin)  # shape: (input_dim, r); this gives a basis for the row space.

        v_orth_2 = orth(v_lin.T)

        if v_orth.shape[1] < n:
            raise ValueError(f"Tangent space rank is deficient (got rank {v_orth.shape[1]} < {n})")

        # Compute the transform T = V_orth^T @ V_lin. This maps the original coordinates to the new orthonormal ones.
        transf = v_orth_2.T #v_orth.T @ v_lin  # resulting shape: (n, input_dim)
        assert False, "Correctly implement that part."
        # --- Step 3: Update SSM basis ---
        self.SSM_basis = (transf @ self.SSM_basis.T).T  # Apply the transformation to the SSM basis

        # --- Step 4: Orthogonalize the reduced coordinates ---
        reduced_coords_orth = [transf @ traj for traj in self.emb_data['reduced_coordinates_aug']]

        # --- Step 5: Build a new parametrization (decoder) on the orthogonalized coordinates ---
        ssm_orth = Control_Aug_SSM(
            t=self.input_data['time'],
            x=delayed_trajs_obs_jax,
            reduced_coordinates_aug=reduced_coords_orth,
            derive_embdedding=False,
            ssm_dim=self.ssm_dim,
            _lambda=self._lambda,
            dim_x=self.dim_x,
            SSM_basis=self.SSM_basis,
            dynamics_type='flow',
            orthogonalize=self.orthogonalize,
            num_obs_delays=self.num_obs_delays,
            specified_params=self.specified_params
        )

        ssm_orth.emb_data = self.emb_data.copy()
        ssm_orth.emb_data['reduced_coordinates_aug'] = reduced_coords_orth

        ssm_orth.get_parametrization(
            rbf_args={'epsilon': rbf_args.get('epsilon_orth', 1.0),
                      'regularization': rbf_args.get('regularization_orth', 1.0)},
            n_components=n_components_orth
        )

        # --- Step 6: Overwrite current decoder with new one and update reduced coordinates ---
        self.decoder = ssm_orth.decoder
        self.emb_data['reduced_coordinates_aug'] = reduced_coords_orth

        return

    def get_parametrization(self, rbf_args=None, n_components=1000, **regression_args) -> None:
        """
        Compute the parametrization (decoder) mapping from reduced coordinates to observables
        using an approximate RBF kernel via random Fourier features and closed-form ridge regression,
        implemented in JAX.
        """
        if rbf_args is None:
            rbf_args = {'epsilon': 1.0, 'regularization': 1e-5}

        if self.emb_data['reduced_coordinates_aug'] is not None:
            # Process data.
            X_list = [jnp.array(traj).T for traj in self.emb_data['reduced_coordinates_aug']]
            X_concat = jnp.concatenate(X_list, axis=0)
            Y_list = [jnp.array(traj).T for traj in self.emb_data['observables']]
            Y_concat = jnp.concatenate(Y_list, axis=0)

            if self.emb_data['params'] is not None:
                params_arr = jnp.array(self.emb_data['params'])
                params_list = []
                for i, traj in enumerate(self.emb_data['reduced_coordinates_aug']):
                    T = traj.shape[1]
                    param_repeated = jnp.tile(params_arr[i], (T, 1))
                    params_list.append(param_repeated)
                params_concat = jnp.concatenate(params_list, axis=0)
                X_aug = jnp.hstack((X_concat, params_concat))
            else:
                X_aug = X_concat

            X_aug = jnp.array(X_aug)
            Y_concat = jnp.array(Y_concat)

            epsilon_val = rbf_args.get('epsilon', 1.0)
            gamma = rbf_args.get('gamma', epsilon_val ** 2)
            regularization_val = float(rbf_args.get('regularization', 1e-5))

            self.decoder = FittedMapping.fit(X_aug, Y_concat, n_components, gamma, regularization_val,
                                             **regression_args)
        else:
            self.encoder, self.decoder = self.fit_reduced_coords_and_parametrization(
                self.emb_data['observables'], self.ssm_dim, **regression_args
            )
            self.emb_data['reduced_coordinates'] = [self.encoder.predict(trajectory) for trajectory in
                                                    self.emb_data['observables']]
        return

    def get_reduced_dynamics(self, rbf_args={'epsilon': 1.0, 'regularization': 1e-5}, n_components=1000,
                             **regression_args) -> None:
        """
        Compute the reduced dynamics using an approximate RBF kernel via random Fourier features
        and closed-form ridge regression implemented in JAX.
        """
        # Shift or differentiate the data.
        X, y = shift_or_differentiate(self.emb_data['reduced_coordinates_aug'], self.emb_data['time'],
                                      self.dynamics_type)
        X_concat = jnp.concatenate([jnp.array(Xi).T for Xi in X], axis=0)
        y_concat = jnp.concatenate([jnp.array(yi).T for yi in y], axis=0)

        if self.emb_data['params'] is not None:
            params_arr = jnp.array(self.emb_data['params'])
            params_list = []
            for i, Xi in enumerate(X):
                n_time = Xi.shape[1]
                param_repeated = jnp.tile(params_arr[i], (n_time, 1))
                params_list.append(param_repeated)
            params_concat = jnp.concatenate(params_list, axis=0)
            X_aug = jnp.hstack((X_concat, params_concat))
        else:
            X_aug = X_concat

        X_aug = jnp.array(X_aug)
        y_concat = jnp.array(y_concat)

        epsilon_val = rbf_args.get('epsilon', 1.0)
        gamma = rbf_args.get('gamma', epsilon_val ** 2)
        regularization_val = float(rbf_args.get('regularization', 1e-5))

        self.reduced_dynamics = FittedMapping.fit(X_aug, y_concat, n_components, gamma, regularization_val,
                                                  **regression_args)
        return

    def predict_reduced_dynamics_runge_kutta(
            self,
            t: list,
            x_reduced: list,
            emb_u_desired: list = []
    ) -> dict:
        """
        Predict the evolution of the reduced dynamics using a differentiable ODE solver (odeint)
        from JAX.
        """
        if not t:
            raise AssertionError("Time vector t is empty.")
        else:
            t_predict = jnp.array(t[0])
            x_predict = []

            for traj_idx in range(len(x_reduced)):

                if emb_u_desired:
                    emb_u_desired_array = jnp.array(emb_u_desired[0]).T

                def get_control_input(t_val):
                    return jnp.array([
                        jnp.interp(t_val, t_predict, emb_u_desired_array[:, i])  # Interpolate each control dimension
                        for i in range(emb_u_desired_array.shape[1])
                    ])

                def dynamics(z_fun, t_fun):
                    if not emb_u_desired:
                        current_control_vector = jnp.zeros(self.SSM_basis.shape[0])  # Zero control if no input
                    else:
                        current_control_vector = get_control_input(t_fun)  # Get control input at time t_fun

                    # Predict the reduced dynamics and add the control term
                    z_dot = self.reduced_dynamics.predict(
                        X=z_fun.reshape(1, -1)) + self.SSM_basis.T @ current_control_vector
                    return jnp.ravel(z_dot)  # Flatten to match expected output shape

                print("Type of x_reduced: ", type(x_reduced))
                print("Length of x_reduced: ", len(x_reduced))
                print("Shape of an element in x_reduced: ", x_reduced[0].shape)
                z0 = x_reduced[traj_idx].T
                print("Shape of z0: ", z0.shape)
                sol = odeint(dynamics, z0, t_predict)
                x_predict.append(sol)

            reduced_dynamics_predictions = {
                'time': t_predict,
                'reduced_coordinates': x_predict
            }
            return reduced_dynamics_predictions

    def predict(
            self,
            t: list,
            x: list,               # Initial condition of x
            emb_u_desired: list    # Full array of control variables
    ) -> dict:
        # Convert initial conditions to JAX arrays.
        init_condition_reduced = [self.SSM_basis.T @ jnp.array(init_condition) for init_condition in x]
        reduced_dynamics_predictions = self.predict_reduced_dynamics_runge_kutta(
            t=t,
            x_reduced=init_condition_reduced,
            emb_u_desired=emb_u_desired
        )
        t_predict = reduced_dynamics_predictions['time']
        x_reduced_predict = [traj.T for traj in reduced_dynamics_predictions['reduced_coordinates']]

        # print("Shape of the x_reduced_predict variable: ", x_reduced_predict[0].shape)

        x_predict = decode_geometry(self.decode, x_reduced_predict)

        predictions = {
            'time': t_predict,
            'reduced_coordinates': x_reduced_predict,
            'observables': x_predict
        }
        return predictions


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


def generate_ssm_predictions(delay_ssm: DelaySSM, trajs, ts=None, dt=None):
    """
    Generate tip positions as predicted by SSM model.
    """
    if ts is None:
        steps = trajs.shape[-1]
        ts = np.arange(0, steps * dt, dt)

    N_input_states = trajs.shape[1]
    ssm_predictions = jnp.zeros_like(trajs)
    for i, traj in enumerate(trajs):
        # Assume first 2 observations are known
        ssm_predictions = ssm_predictions.at[i, :, :2].set(traj[:, :2])
        y0 = jnp.flip(traj[:, :2], 1).T.flatten()
        x0 = delay_ssm.encode(y0)
        xs = delay_ssm.simulate_reduced(x0, ts[2:])
        ys = delay_ssm.decode(xs)
        ssm_predictions = ssm_predictions.at[i, :, 2:].set(ys[:N_input_states, :])
    return ssm_predictions
