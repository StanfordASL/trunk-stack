from jax.scipy.linalg import expm
import jax.numpy as jnp


class Actuator:
    def __init__(self, num_u, lambda_eigenvalues, current_time=0.0):
        self.current_u = jnp.zeros((num_u,))
        self.lambda_ = jnp.array(lambda_eigenvalues)
        self.current_time = current_time

    def __call__(self, new_time, new_u):
        # assert self.dt == (new_time - self.current_time), "Missmatch in time steps"
        dt = new_time - self.current_time
        self.current_time = new_time

        # Update the current u using the analytical solution:
        # u(t + delta_time) = new_u + expm(lambda * delta_time) dot (u(t) - new_u)
        self.current_u = new_u + jnp.dot(expm(self.lambda_ * dt), (self.current_u - new_u))
        return self.current_u

    def reset(self):
        self.current_time = 0.0
        self.current_u = jnp.zeros_like(self.current_u)
