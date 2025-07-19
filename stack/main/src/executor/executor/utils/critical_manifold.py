import jax
import jax.numpy as jnp
from functools import partial


class CriticalManifold:
    """
    Critical manifold mapping, similar to inverse kinematics.
    """
    def __init__(self, manifold_data):
        self.manifold_data = manifold_data
        self.U_select = jnp.array(manifold_data['U_select'])  # (n_u, M)
        self.W = jnp.array(manifold_data['W'])  # (M, n_s)
        self.epsilon = manifold_data['epsilon']
        self.case_rbf = manifold_data['case_rbf']
        self.n_s = self.W.shape[1]  # number of performance variables
        self.n_u = self.U_select.shape[0]  # number of control inputs
        self.M = self.U_select.shape[1]  # number of RBF centers
        if self.W.shape[0] != self.M:
            raise ValueError('Number of rows in W must match number of centers in U_select')

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, u):
        """
        Evaluate the critical manifold mapping (RBF): s = S(u)
        """
        # Convert 1D input to 2D for consistent processing
        is_1d_input = u.ndim == 1
        if is_1d_input:
            u = u[:, jnp.newaxis]  # (n_u,) -> (n_u, 1)

        # Validate dimensions of u, should be (n_u, N)
        if u.shape[0] != self.n_u:
            raise ValueError('Dimension of u must match dimension of u_select')
        
        # Compute pairwise Euclidean distances between u and U_select
        # dist: M x N matrix, where dist[i,j] is the distance between U_select[:,i] and u[:,j]
        dist = jnp.linalg.norm(self.U_select[:, :, jnp.newaxis] - u[:, jnp.newaxis, :], axis=0)
        
        # Compute Gaussian RBF: phi(r) = exp(-(epsilon * r)^2)
        # A: M x N matrix, where A[i,j] = phi(||U_select[:,i] - u[:,j]||)
        if self.case_rbf:
            A = jnp.exp(-(self.epsilon * dist)**2)
        else:
            A = dist
        
        # Compute RBF interpolation: s = W' * A
        # A: M x N, W: M x n_s -> s: n_s x N
        s = self.W.T @ A

        # If input was 1D, return 1D output
        if is_1d_input:
            s = s[:, 0]  # (n_s, 1) -> (n_s,)
        
        return s
