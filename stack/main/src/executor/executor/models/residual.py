"""
Residual models to add control dependency to models of autonomous systems.
"""

import jax
import jax.numpy as jnp
from utils.nn import MLP
from utils.misc import polynomial_features, fit_linear_regression


class ResidualNN:
    """
    Neural Network based residual dynamics model.
    """
    def __init__(self, state, config):
        """
        Set up the neural network model.
        """
        assert len(config.output_shape) == 1, "Output shape must be a 1D tuple."
        self.state = state
        self.model = MLP(config.hidden_sizes, config.output_shape, config.activation)
        self.n_x = config.output_shape[0]
        self.n_u = config.input_size - self.n_x

    def __call__(self, x, u):
        """
        Evaluate residual dynamics model at state `x` and control `u`.
        """
        inputs = jnp.concatenate([x, u], axis=-1)
        delta_x_dots = self.model.apply({'params': self.state.params}, inputs)
        return delta_x_dots


class ResidualBr:
    """
    B_r based residual dynamics model.
    """
    def __init__(self, learned_B_r):
        self.n_x, self.n_u = learned_B_r.n_x, learned_B_r.n_u
        self.learned_B_r = learned_B_r

    def __call__(self, x, u):
        """
        Evaluate residual dynamics at state `x` and control `u`.
        """
        delta_x_dots = self.learned_B_r(x) @ u
        return delta_x_dots


class LearnedBr:
    """
    Base class for learned B_r(x).
    """
    def __init__(self, n_x, n_u):
        self.n_x, self.n_u = n_x, n_u

    def fit(self, xs, us, delta_x_dots):
        raise NotImplementedError

    def _eval(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self._eval(x)


class NeuralBr(LearnedBr):
    """
    B_r being a neural network, with input x.
    """
    def __init__(self, n_x, n_u, state, config):
        assert (n_x, n_u) == config.output_shape, "The network output shape must be equal to (n_x, n_u)."
        super().__init__(n_x, n_u)
        self.state = state
        self.model = MLP(config.hidden_sizes, config.output_shape, config.activation)

    def _eval(self, x):
        """
        Evaluate B_r(x).
        """
        B_r = self.model.apply({'params': self.state.params}, x)
        return B_r


class PolyBr(LearnedBr):
    """
    B_r being a polynomial function of x.
    """
    def __init__(self, n_x, n_u, poly_degree, lam=0.0):
        super().__init__(n_x, n_u)
        self.poly_degree = poly_degree
        self.lam = lam

    def fit(self, xs, us, delta_x_dots):
        """
        Fit B_r(x) to data using linear regression.
        """
        poly_features = polynomial_features(xs, self.poly_degree)
        self.k = poly_features.shape[1]
        D = self._create_design_matrix(poly_features, us)
        self.B_r_coeff = fit_linear_regression(D, delta_x_dots, lam=self.lam).T

    def _create_design_matrix(self, poly_features, us):
        """
        Construct design matrix D for polynomial B_r model.
        """
        N_data = len(poly_features)
        D = jnp.zeros((N_data, self.n_u * self.k))
        for i in range(self.n_u):
            D = D.at[:, i*self.k:(i+1)*self.k].set(poly_features * us[:, i:i+1])
        return D
    
    def _eval(self, x):
        """
        Evaluate B_r(x).
        """
        B_r = jnp.zeros((self.n_x, self.n_u))
        x_poly = polynomial_features(x, self.poly_degree)
        for i in range(self.n_u):
            B_r = B_r.at[:, i].set((self.B_r_coeff[:, i*self.k:(i+1)*self.k] @ x_poly.T).squeeze())
        return B_r


class PolyBrW(LearnedBr):
    """
    B_r being a polynomial function of x through w(x).
    """
    def __init__(self, n_x, n_u, poly_degree, w):
        super().__init__(n_x, n_u)
        self.poly_degree = poly_degree
        self.w = w

    def fit(self, xs, us, delta_x_dots):
        """
        Fit B_r(x) to data using linear regression.
        """
        ys = self.w(xs.T).T
        poly_features = polynomial_features(ys, self.poly_degree)
        self.k = poly_features.shape[1]
        D = self._create_design_matrix(poly_features, us)
        self.B_r_coeff = fit_linear_regression(D, delta_x_dots).T

    def _create_design_matrix(self, poly_features, us):
        """
        Construct design matrix D for polynomial B_r model.
        """
        N_data = len(poly_features)
        D = jnp.zeros((N_data, self.n_u * self.k))
        for i in range(self.n_u):
            D = D.at[:, i*self.k:(i+1)*self.k].set(poly_features * us[:, i:i+1])
        return D
    
    def _eval(self, x):
        """
        Evaluate B_r(x).
        """
        B_r = jnp.zeros((self.n, self.n_u))
        y_poly = polynomial_features(self.w(x), self.poly_degree)
        for i in range(self.n_u):
            B_r = B_r.at[:, i].set((self.B_r_coeff[:, i*self.k:(i+1)*self.k] @ y_poly.T).squeeze())
        return B_r
