from typing import Optional, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import numpy as np
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import Array
from scipy.linalg import orth
from findiff import FinDiff

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

        if y_eq is not None and u_eq is not None:
            self.y_eq = jnp.concatenate([y_eq, u_eq], axis=0)
        else:
            self.y_eq = None
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
                x=z_fun.reshape(1, -1)) + self.SSM_basis.T @ current_control_vector
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

    def save_model_as_dict(self, only_decoder=False):
        """
        Save the learned quantities of the decoder and reduced dynamics mappings
        into a dictionary.
        """
        assert self.SSM_basis is not None, "SSM_basis must not be None"
        assert self.decoder is not None, "Decoder must not be None"
        if not only_decoder:
            assert self.reduced_dynamics is not None, "Reduced dynamics must not be None"

        if only_decoder:
            model_dict = {
                'V': self.SSM_basis,
                'y_eq': self.y_eq,
                'u_eq': self.u_eq,
                'w_coeff': self.decoder.beta,
                'w_intersect': self.decoder.intercept
            }
        else:
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
