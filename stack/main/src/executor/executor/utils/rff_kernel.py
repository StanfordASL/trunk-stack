import jax
import jax.numpy as jnp
from typing import Optional
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


def _fit_rbf_pipeline(X_aug, Y, n_components, gamma, regularization, seed=42, **regression_args):
    model = make_pipeline(
        RBFSampler(gamma=gamma, n_components=n_components, random_state=seed),
        Ridge(alpha=regularization, **regression_args)
    )
    model.fit(X_aug, Y)
    W = jnp.array(model.named_steps['rbfsampler'].random_weights_)
    b = jnp.array(model.named_steps['rbfsampler'].random_offset_)
    beta = jnp.array(model.named_steps['ridge'].coef_)
    if beta.ndim == 1:
        beta = beta[jnp.newaxis, :]
    intercept = jnp.array(model.named_steps['ridge'].intercept_)
    return W, b, beta, intercept


def _compute_jacobian(X_aug, W, b, beta):
    x0 = jnp.zeros(X_aug.shape[1])
    arg = jnp.dot(x0, W) + b
    D = arg.shape[0]
    dz_dx = - jnp.sqrt(2.0 / D) * jnp.sin(arg)[:, jnp.newaxis] * W.T
    if beta.ndim == 1:
        beta = beta[jnp.newaxis, :]
    J = jnp.dot(beta, dz_dx)
    return J


# Unified class that can be used for both parametrization and dynamics.
class FittedMapping:
    def __init__(self, W, b, beta, intercept, n_components, gamma, regularization, linear_part):
        """
        Parameters:
          W (jax.numpy.ndarray): Random weights from RBFSampler.
          b (jax.numpy.ndarray): Random offsets from RBFSampler.
          beta (jax.numpy.ndarray): Ridge regression coefficients.
          intercept (jax.numpy.ndarray): Ridge regression intercept.
          n_components (int): Number of random Fourier features.
          gamma (float): Parameter used in the RBFSampler.
          regularization (float): Ridge regression regularization parameter.
          linear_part (jax.numpy.ndarray): Jacobian (linearization) at the origin.
        """
        self.W = W
        self.b = b
        self.beta = beta
        self.intercept = intercept
        self.map_info = {
            'n_components': n_components,
            'gamma': gamma,
            'regularization': regularization,
            'linear_part': linear_part
        }

    def __call__(self, X):
        """
        This method allows the object to be called as a function, directly invoking predict.
        """
        return self.predict(X)

    def predict(self, X):
        """
        Predict using the learned RBF mapping, ridge regression weights and intercept.

        Parameters:
          X (jax.numpy.ndarray): Input data of shape (n_samples, input_dim).

        Returns:
          jax.numpy.ndarray: Predictions computed as z(X) dot beta.T + intercept.
        """
        z = jnp.sqrt(2.0 / self.map_info['n_components']) * jnp.cos(jnp.dot(X, self.W) + self.b)
        return jnp.dot(z, self.beta.T) + self.intercept

    @classmethod
    def fit(cls, X_aug, Y, n_components, gamma, regularization, seed=42, **regression_args):
        """
        Fit the RBFSampler/Ridge model using scikit-learn, extract learned parameters,
        compute the Jacobian at the origin, and return a fitted instance.

        Parameters:
          X_aug (jax.numpy.ndarray): Augmented input data.
          Y (jax.numpy.ndarray): Target data.
          n_components (int): Number of random Fourier features.
          gamma (float): Parameter for the RBFSampler.
          regularization (float): Regularization parameter for Ridge.
          seed (int): Random seed.
          regression_args: Additional keyword arguments for Ridge.

        Returns:
          FittedMapping: A fitted instance.
        """
        W, b, beta, intercept = _fit_rbf_pipeline(X_aug, Y, n_components, gamma, regularization, seed=seed,
                                                  **regression_args)
        linear_part = _compute_jacobian(X_aug, W, b, beta)
        return cls(W, b, beta, intercept, n_components, gamma, regularization, linear_part)


class RFFRegressor:
    def __init__(self, n_components: int, gamma: float, regularization: float, seed: int = 42):
        """
        Initialize the RFF regressor.

        Parameters:
          n_components (int): Number of random Fourier features.
          gamma (float): Parameter for the RBF kernel (typically gamma = epsilon**2).
          regularization (float): Regularization parameter for ridge regression.
          seed (int): Random seed for reproducibility.
        """
        self.n_components = n_components
        self.gamma = gamma
        self.regularization = float(regularization)
        self.seed = seed
        self.W: Optional[jnp.ndarray] = None
        self.b: Optional[jnp.ndarray] = None
        self.beta: Optional[jnp.ndarray] = None

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Transform the input X using the random Fourier features.

        Parameters:
          X (jnp.ndarray): Input data of shape (n_samples, input_dim).

        Returns:
          jnp.ndarray: Transformed features of shape (n_samples, n_components).
        """
        if self.W is None or self.b is None:
            raise ValueError("The model is not fitted yet. Call fit() first.")
        print("Inside transform: X shape:", X.shape)
        print("Inside transform: self.W shape:", self.W.shape)
        print("Inside transform: self.b shape:", self.b.shape)
        return jnp.sqrt(2.0 / self.n_components) * jnp.cos(jnp.dot(X, self.W) + self.b)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """
        Fit the RFF regressor by sampling random weights and biases and solving
        the ridge regression problem.

        Parameters:
          X (jnp.ndarray): Input features of shape (n_samples, input_dim).
          y (jnp.ndarray): Targets of shape (n_samples, output_dim).
        """
        input_dim = X.shape[1]
        key = jax.random.PRNGKey(self.seed)

        # Sample weights: W ~ N(0, 2*gamma)
        self.W = jax.random.normal(key, (input_dim, self.n_components)) * jnp.sqrt(2 * self.gamma)

        # Sample offsets: b ~ Uniform(0, 2*pi)
        self.b = jax.random.uniform(key, (self.n_components,), minval=0.0, maxval=2 * jnp.pi)

        # Compute the RFF transformation.
        Z = self.transform(X)

        # Solve the ridge regression: beta = (Z.T Z + reg*I)^{-1} Z.T y
        I = jnp.eye(int(self.n_components))

        A = jnp.dot(Z.T, Z) + self.regularization * I
        B = jnp.dot(Z.T, y)
        self.beta = jnp.linalg.solve(A, B)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict new outputs for input X using the fitted RFF regressor.

        Parameters:
          X (jnp.ndarray): Input features of shape (n_samples, input_dim).

        Returns:
          jnp.ndarray: Predictions of shape (n_samples, output_dim).
        """
        Z = self.transform(X)
        return jnp.dot(Z, self.beta)
