import jax.numpy as jnp
import numpy as np


class DelayEmbeddedState:
    def __init__(self, state_dim, u_dim, num_delay, also_embedd_u, initial_state=None):
        """
        Initialize the delay-embedded state vector.

        Args:
            state_dim (int): Dimension of the measured state.
            u_dim (int): Dimension of the control input.
            num_delay (int): Number of delay embeddings.
            also_embedd_u (bool): If True, embed the (state, u) pair at each delay step.
                                  Otherwise, embed only the state and append u at the end.
            initial_state (np.array): the last recordings of the state and u
        """
        self.state_dim = state_dim
        self.u_dim = u_dim
        self.num_delay = num_delay
        self.also_embedd_u = also_embedd_u

        if initial_state is None:

            if self.also_embedd_u:
                # The full delay vector has dimension: num_delay * (state_dim + u_dim)
                self.state = jnp.zeros(((num_delay + 1) * (state_dim + u_dim),))
            else:
                # The delay vector has all delayed states (num_delay * state_dim)
                # followed by the most recent control (u_dim)
                self.state = jnp.zeros(((num_delay + 1) * state_dim + u_dim,))

        else:
            assert isinstance(initial_state, tuple) and len(initial_state) == 2, \
                "initial_state must be a tuple (states, controls)"
            states, controls = initial_state

            assert isinstance(states, np.ndarray), "states must be a numpy.ndarray"
            assert states.ndim == 2 and states.shape[1] == state_dim, \
                f"states must have shape (num_delay+1, {state_dim})"
            assert states.shape[0] == num_delay + 1, \
                f"states must contain {num_delay + 1} time‐steps"

            assert isinstance(controls, np.ndarray), "controls must be a numpy.ndarray"
            assert controls.ndim == 2 and controls.shape[1] == u_dim, \
                f"controls must have shape (#steps, {u_dim})"

            if self.also_embedd_u:
                assert controls.shape[0] == num_delay + 1, \
                    f"controls must contain {num_delay + 1} time‐steps when also_embedd_u=True"

                parts = []
                for i in range(num_delay + 1):
                    s_i = jnp.asarray(states[i])  # shape (state_dim,)
                    u_i = jnp.asarray(controls[i])  # shape (u_dim,)
                    parts.append(jnp.concatenate([s_i, u_i]))  # shape (state_dim+u_dim,)
                # final vector: [s0,u0, s1,u1, ..., sN,uN]
                self.state = jnp.concatenate(parts)

            else:
                assert controls.shape[0] == 1, \
                    "controls must have exactly 1 time‐step when also_embedd_u=False"

                flat_states = jnp.asarray(states).reshape(-1)  # shape ((num_delay+1)*state_dim,)
                latest_u = jnp.asarray(controls[0])  # shape (u_dim,)
                # final vector: [s0, s1, ..., sN, u_latest]
                self.state = jnp.concatenate([flat_states, latest_u])

    def update_state(self, current_state, current_u):
        """
        Update the delay-embedded state with the new measurement.
        """
        if self.also_embedd_u:
            block_size = self.state_dim + self.u_dim
            # Roll the vector to discard the oldest block (shift left)
            new_state = jnp.roll(self.state, block_size)
            # Place the new (state,u) pair into the last block
            new_block = jnp.concatenate([current_state, current_u])
            new_state = new_state.at[:block_size].set(new_block)
            self.state = new_state
        else:
            raise NotImplementedError("This function has not been correctly implemented yet.")
            state_hist_size = self.num_delay * self.state_dim
            # Extract the state history portion
            x_history = self.state[:state_hist_size]
            # Roll the state history to discard the oldest state vector (shift left)
            new_x_history = jnp.roll(x_history, -self.state_dim)
            # Set the last block of the state history to the current state
            new_x_history = new_x_history.at[-self.state_dim:].set(current_state)
            # The updated overall vector is the new state history concatenated with current_u
            self.state = jnp.concatenate([new_x_history, current_u])
        return None

    def get_current_state(self):
        """
        Return the current delay-embedded state as one flat vector.

        Returns:
            jnp.ndarray: The delay-embedded state.
                - If also_embedd_u is True, shape is (num_delay*(state_dim+u_dim),)
                - Otherwise, shape is (num_delay*state_dim + u_dim,)
        """
        return self.state
