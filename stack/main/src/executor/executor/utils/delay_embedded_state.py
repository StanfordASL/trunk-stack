import numpy as np
import jax.numpy as jnp


class DelayEmbeddedState:
    def __init__(self, n_u, num_delay, obs_dim=None, n_y=None, also_embedd_u=False):
        """
        Initialize the delay-embedded state vector.

        Args:
            n_u (int): Dimension of the control input
            num_delay (int): Number of delay embeddings
            obs_dim (int): Dimension of the observation vector, e.g. 3 for (x, y, z)
            n_y (int): Dimension of full observation vector, including delayed states
            also_embedd_u (bool): If True, embed the (state, u) pair at each delay step
                                  Otherwise, embed only the state
        """
        if obs_dim is not None:
            self.obs_dim = obs_dim
            self.n_y = obs_dim * (num_delay + 1)
        elif n_y is not None:
            self.obs_dim = n_y // (num_delay + 1)
            self.n_y = n_y
        else:
            raise ValueError("obs_dim or n_y must be provided.")

        self.n_u = n_u
        self.num_delay = num_delay
        self.also_embedd_u = also_embedd_u

        if self.also_embedd_u:
            # The full delay vector has dimension: num_delay * (obs_dim + n_u)
            self.state = jnp.zeros(((num_delay + 1) * (self.obs_dim + n_u),))
        else:
            # The delay vector has all delayed states (num_delay * obs_dim)=(n_y)
            self.state = jnp.zeros(((num_delay + 1) * self.obs_dim,))

    def update_state(self, current_state, current_u):
        """
        Update the delay-embedded state with the new measurement.
        """
        if self.also_embedd_u:
            block_size = self.obs_dim + self.n_u
            # Roll the vector to discard the oldest block
            new_state = jnp.roll(self.state, block_size)
            # Place the new (state,u) pair into the last block
            new_block = jnp.concatenate([current_state, current_u])
            new_state = new_state.at[:block_size].set(new_block)
            self.state = new_state
        else:
            block_size = self.obs_dim
            # Roll the vector to discard the oldest state
            new_state = jnp.roll(self.state, block_size)
            new_block = current_state
            new_state = new_state.at[:block_size].set(new_block)
            self.state = new_state
        return None

    def get_current_state(self):
        """
        Return the current delay-embedded state as one flat vector.

        Returns:
            jnp.ndarray: The delay-embedded state.
                - If also_embedd_u is True, shape is (num_delay*(obs_dim+u_dim),)
                - Otherwise, shape is (n_y,)
        """
        return self.state
