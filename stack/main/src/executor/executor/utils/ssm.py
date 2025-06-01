"""
Custom Spectral Submanifold (SSM) model class.
"""
from typing import Optional, List

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.ode import odeint
from functools import partial
from ssmlearnpy import SSMLearn
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import orth
from findiff import FinDiff
import numpy as np
import sympy as sp
from .misc import trajectories_delay_embedding
from .mappings.pipeline_mapping_base import FittedMapping


class Control_origin_ssm:
    def __init__(self,
                 t: List[jnp.ndarray] = None,
                 aug_full_states: List[jnp.ndarray] = None,
                 reduced_coordinates_aug: Optional[List[Array]] = None,
                 lam: Optional[Array] = None,
                 dim_x: int = 0,
                 ssm_basis: jnp.ndarray = None,
                 normalize_reduced_coord: bool = True,
                 specified_params: dict = None,
                 orthogonalize: bool = False,
                 num_obs_delays=0,
                 y_eq: jnp.ndarray = None,
                 u_eq: jnp.ndarray = None
                 ):

        self.lam = lam
        self.dim_x = dim_x
        self.orthogonalize = orthogonalize
        self.num_obs_delays = num_obs_delays

        # specified_params contains:
        # "embedding_up_to", "shift_steps", "SSM_dim", "measured_rows", "num_u", "include_velocity"
        self.specified_params = specified_params

        if ssm_basis is None:
            self.SSM_basis = self._get_encoder(aug_full_states, normalize=normalize_reduced_coord)
        else:
            self.SSM_basis = ssm_basis

        assert t is not None, "`t` must not be None"
        assert aug_full_states is not None, "`aug_full_states` must not be None"
        time = jnp.array(t)
        observables = jnp.array(aug_full_states)

        if reduced_coordinates_aug is None:
            reduced_coordinates_aug = self.encode(aug_full_states)

        self.input_data = {
            'time': time,
            'observables': observables,
            'reduced_coordinates_aug': reduced_coordinates_aug,
        }

        # Define the non-linear mappings
        self.decoder = None
        self.reduced_dynamics = None

        self.y_eq = jnp.concatenate([y_eq, u_eq], axis=0)
        self.u_eq = u_eq

    def encode(self, y):
        return jnp.array(self.SSM_basis.T @ y)

    def decode(self, y):
        return (self.decoder.predict(y.T)).T

    def decode_list(self, x_reduced):
        x = []
        for i_elem in range(len(x_reduced)):
            x.append(self.decoder(x_reduced[i_elem]))
        return x

    def get_decoder_with_orth(
            self,
            rbf_args=None,
            n_components=1000,
            n_components_orth=1000
    ) -> None:

        NotImplementedError("get_decoder_with_orth function needs to be adjusted.")
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

        #if not self.orthogonalize:
        #    print("self.orthogonalize is being set to true.")
        #    self.orthogonalize = True

        # --- Step 1: Learn initial parametrization ---
        n = self.specified_params["SSM_dim"]
        delayed_trajs_obs_np = self.input_data['observables']
        num_obs_states = self.dim_x * (1 + self.num_obs_delays)

        NotImplementedError("New observables need to be reduced and then given back to the new class as reduced coordinates.")
        # Convert the numpy arrays to JAX arrays for full JAX compatibility
        delayed_trajs_obs_jax = [jnp.array(traj[:num_obs_states, :]) for traj in delayed_trajs_obs_np]

        # Define the initial parametrization (decoder)
        ssm_paramonly = Control_origin_ssm(
            t=self.input_data['time'],
            aug_full_states=delayed_trajs_obs_jax,
            reduced_coordinates_aug=self.input_data['reduced_coordinates_aug'],
            lam=self.lam,
            dim_x=self.dim_x,
            ssm_basis=self.SSM_basis,
            orthogonalize=self.orthogonalize,
            num_obs_delays=self.num_obs_delays,
            specified_params=self.specified_params
        )
        ssm_paramonly.get_decoder(
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
        reduced_coords_orth = [transf @ traj for traj in self.input_data['reduced_coordinates_aug']]

        # --- Step 5: Build a new parametrization (decoder) on the orthogonalized coordinates ---
        ssm_orth = CustomSSMLearn_mpc(
            t=self.input_data['time'],
            #x=delayed_trajs_obs_jax,
            reduced_coordinates_aug=reduced_coords_orth,
            lam=self.lam,
            dim_x=self.dim_x,
            SSM_basis=self.SSM_basis,
            orthogonalize=self.orthogonalize,
            num_obs_delays=self.num_obs_delays,
            specified_params=self.specified_params
        )

        ssm_orth.emb_data = self.input_data.copy()
        ssm_orth.emb_data['reduced_coordinates_aug'] = reduced_coords_orth

        ssm_orth.get_decoder(
            rbf_args={'epsilon': rbf_args.get('epsilon_orth', 1.0),
                      'regularization': rbf_args.get('regularization_orth', 1.0)},
            n_components=n_components_orth
        )

        # --- Step 6: Overwrite current decoder with new one and update reduced coordinates ---
        self.decoder = ssm_orth.decoder
        self.input_data['reduced_coordinates_aug'] = reduced_coords_orth

        return

    def _get_encoder(self, emb_aug_states, normalize=True):
        """
        Calculate the encoder PCA matrix. Potentially normalize the reduced coordinates.
        """

        u, s, vt = np.linalg.svd(np.concatenate(emb_aug_states, axis=1), full_matrices=False)

        if not normalize:
            ssm_basis = u[:, 0:self.specified_params["SSM_dim"]]
        else:
            r = self.specified_params["SSM_dim"]  # number of modes
            u_r = u[:, :r]  # (m × r)
            s_r = s[:r]
            ssm_basis = u_r * (1.0 / s_r)[None, :]

        print(f"Data with lambda influence was loaded. Its SSM basis has shape: {ssm_basis.shape}")
        return ssm_basis

    def get_decoder(self, fitting_args=None, type_of_fitting='rbf', w=None, b=None) -> None:
        """
        Compute the parametrization (decoder) mapping from reduced coordinates to observables
        using an approximate RBF kernel via random Fourier features and closed-form ridge regression,
        implemented in JAX.
        """
        required = {
            'rbf': {'n_components', 'epsilon', 'regularization'},
            'polynomial': {'degree', 'regularization'},
            'nn': {'num_layers', 'nodes_per_layer', 'regularization'},
        }
        defaults = {
            'rbf': {'n_components': 1000, 'epsilon': 0.5, 'regularization': 1e-5},
            'polynomial': {'degree': 1, 'regularization': 1e-5},
            'nn': {'epsilon': 1.0, 'regularization': 1e-5},
        }

        # if no args given, fill with defaults
        if fitting_args is None:
            try:
                fitting_args = defaults[type_of_fitting].copy()
            except KeyError:
                raise ValueError(f"Unknown fitting type '{type_of_fitting}'. "
                                 f"Valid options are {list(defaults)}.")

        # otherwise, check user‐provided dict contains all required keys
        else:
            if type_of_fitting not in required:
                raise ValueError(f"Unknown fitting type '{type_of_fitting}'. "
                                 f"Valid options are {list(required)}.")
            missing = required[type_of_fitting] - set(fitting_args)
            if missing:
                raise ValueError(
                    f"Missing required fitting_args for type '{type_of_fitting}': "
                    f"{', '.join(sorted(missing))}"
                )

        assert self.input_data['reduced_coordinates_aug'] is not None, "reduced_coordinates_aug must not be None"

        # Process data.
        x_concat = jnp.vstack([traj.T for traj in self.input_data['reduced_coordinates_aug']])
        y_concat = jnp.vstack([traj.T for traj in self.input_data['observables']])

        if type_of_fitting == 'rbf':
            eps = fitting_args.pop('epsilon')
            fitting_args['gamma'] = eps**2
            hyperparams = fitting_args
        elif type_of_fitting == 'polynomial':
            hyperparams = fitting_args
        elif type_of_fitting == 'nn':
            raise NotImplementedError()
        else:
            raise AssertionError("We are not working with any valid mapping type.")

        # New implementation:
        self.decoder = FittedMapping.fit(x_concat, y_concat, type_of_fitting=type_of_fitting,
                                         hyperparams=hyperparams, seed=42, w=w, b=b)

        return

    def get_reduced_dynamics(self, fitting_args=None, type_of_fitting='rbf', w=None, b=None) -> None:
        """
        Compute the reduced dynamics using an approximate RBF kernel via random Fourier features
        and closed-form ridge regression implemented in JAX.
        """
        required = {
            'rbf': {'n_components', 'epsilon', 'regularization'},
            'polynomial': {'degree', 'regularization'},
            'nn': {'num_layers', 'nodes_per_layer', 'regularization'},
        }
        defaults = {
            'rbf': {'n_components': 1000, 'epsilon': 0.5, 'regularization': 1e-5},
            'polynomial': {'degree': 0.5, 'regularization': 1e-5},
            'nn': {'epsilon': 0.5, 'regularization': 1e-5},
        }

        # If no rbf_args given, fill with defaults
        if fitting_args is None:
            try:
                fitting_args = defaults[type_of_fitting].copy()
            except KeyError:
                raise ValueError(f"Unknown fitting type '{type_of_fitting}'. "
                                 f"Valid options are {list(defaults)}.")
        else:
            if type_of_fitting not in required:
                raise ValueError(f"Unknown fitting type '{type_of_fitting}'. "
                                 f"Valid options are {list(required)}.")
            missing = required[type_of_fitting] - set(fitting_args)
            if missing:
                raise ValueError(
                    f"Missing required rbf_args for type '{type_of_fitting}': "
                    f"{', '.join(sorted(missing))}"
                )

        # Shift or differentiate the data.
        x, y = self.shift_or_differentiate(self.input_data['reduced_coordinates_aug'], self.input_data['time'])
        x_concat = jnp.vstack([Xi.T for Xi in x])
        y_concat = jnp.vstack([yi.T for yi in y])

        # Prepare fitting parameters
        if type_of_fitting == 'rbf':
            eps = fitting_args.pop('epsilon')
            fitting_args['gamma'] = eps ** 2
            hyperparams = fitting_args
        elif type_of_fitting == 'polynomial':
            hyperparams = fitting_args
        elif type_of_fitting == 'nn':
            raise NotImplementedError()
        else:
            raise AssertionError("We are not working with any valid mapping type.")

        self.reduced_dynamics = FittedMapping.fit(x_concat, y_concat, type_of_fitting=type_of_fitting,
                                                  hyperparams=hyperparams, seed=42, w=w, b=b)
        return

    def predict_reduced_coordinates(
            self,
            t: jnp.ndarray,
            x_reduced: jnp.ndarray,
            emb_u_desired: jnp.ndarray
    ) -> dict:
        """
        Predict the evolution of the reduced dynamics using a differentiable ODE solver (odeint)
        from JAX.
        """
        if len(t) == 0:
            raise AssertionError("Time vector t is empty.")

        if emb_u_desired is not None:
            emb_u_desired_array = jnp.array(emb_u_desired).T

        def get_control_input(t_val):
            return jnp.array([
                jnp.interp(t_val, t, emb_u_desired_array[:, i])  # Interpolate each control dimension
                for i in range(emb_u_desired_array.shape[1])
            ])

        def dynamics(z_fun, t_fun):
            if emb_u_desired is None:
                current_control_vector = jnp.zeros(self.SSM_basis.shape[0])  # Zero control if no input
            else:
                current_control_vector = get_control_input(t_fun)  # Get control input at time t_fun

            # Predict the reduced dynamics and add the control term
            z_dot = self.reduced_dynamics.predict(
                X=z_fun.reshape(1, -1)) + self.SSM_basis.T @ current_control_vector
            return jnp.ravel(z_dot)  # Flatten to match expected output shape

        z0 = x_reduced.T

        sol = odeint(dynamics, z0, t)
        x_predict = sol

        reduced_dynamics_predictions = {
            'time': t,
            'reduced_coordinates': x_predict
        }
        return reduced_dynamics_predictions

    def predict(
            self,
            t: jnp.ndarray,
            x: jnp.ndarray,                              # Initial condition of x
            emb_u_desired: Optional[jnp.ndarray] = None  # Full array of control variables
    ) -> dict:

        # Convert initial conditions to JAX arrays.
        init_condition_reduced = self.encode(x)
        reduced_dynamics_predictions = self.predict_reduced_coordinates(
            t=t,
            x_reduced=init_condition_reduced,
            emb_u_desired=emb_u_desired
        )
        t_predict = reduced_dynamics_predictions['time']
        x_reduced_predict = reduced_dynamics_predictions['reduced_coordinates'].T

        x_predict = self.decode(x_reduced_predict)

        predictions = {
            'time': t_predict,
            'reduced_coordinates': x_reduced_predict,
            'observables': x_predict
        }
        return predictions

    @staticmethod
    def shift_or_differentiate(x, t, accuracy=4):
        """
        The function prepares the data for regression of the reduced dynamics.
        """
        x_matrix: List[np.ndarray] = []
        y: List[np.ndarray] = []

        for traj, times in zip(x, t):
            traj_np = np.asarray(traj)  # shape (state_dim, T)
            dt = float(times[1] - times[0])  # assume uniform spacing

            # axis=1 → differentiate along the time dimension
            fd = FinDiff(1, dt, 1, acc=accuracy)
            dx_dt_traj = fd(traj_np)  # now works on a 2-D array

            x_matrix.append(traj_np)
            y.append(dx_dt_traj)

        return x_matrix, y

    def save_model_as_dict(self):
        """
        Save the learned quantities of the decoder and reduced dynamics mappings
        into a dictionary.
        """
        assert self.SSM_basis is not None, "SSM_basis must not be None"
        assert self.decoder is not None, "Decoder must not be None"
        assert self.reduced_dynamics is not None, "Reduced dynamics must not be None"

        model_dict = {
            'V': self.SSM_basis,
            'y_eq': self.y_eq,
            'u_eq': self.u_eq,
            'w_coeff': self.decoder.beta,
            'w_intersect': self.decoder.intercept,
            'r_coeff': self.reduced_dynamics.beta,
            'r_intersect': self.reduced_dynamics.intercept
        }

        return model_dict


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
            t=[ts] * len(delayed_trajs_obs_np),
            x=[delayed_traj_obs_np for delayed_traj_obs_np in delayed_trajs_obs_np],
            reduced_coordinates=[reduced_delayed_traj_np for reduced_delayed_traj_np in reduced_delayed_trajs_np],
            ssm_dim=self.SSMDim,
            dynamics_type='flow'
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
                t=[ts] * len(delayed_trajs_obs_np),
                x=[delayed_traj_obs_np for delayed_traj_obs_np in delayed_trajs_obs_np],
                reduced_coordinates=[reduced_delayed_traj_orth_np for reduced_delayed_traj_orth_np
                                     in reduced_delayed_trajs_orth_np],
                ssm_dim=self.SSMDim,
                dynamics_type='flow'
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
                t=[ts] * len(delayed_trajs_obs_np),
                x=[reduced_delayed_traj_np for reduced_delayed_traj_np in reduced_delayed_trajs_np],
                reduced_coordinates=[delayed_traj_obs_np for delayed_traj_obs_np in delayed_trajs_obs_np],
                ssm_dim=self.p,
                dynamics_type='flow'
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
