import jax
import jax.numpy as jnp
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Optional, TypeVar, Type
import functools

# scikit-learn bits
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# JAX MLP bits
from jax.example_libraries import stax, optimizers
from jax.example_libraries.stax import Dense, Relu

T = TypeVar("T", bound="BaseFittedMapping")


class BaseFittedMapping(ABC):
    """
    Abstract base class for any “fitted mapping” (RBF, polynomial, NN, …).
    """
    # each subclass must set self.map_info: Dict[str,Any]
    map_info: Dict[str, Any]

    @classmethod
    @abstractmethod
    def fit(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """
        Fit on (X, Y) and return a new instance.
        Must compute and store the Jacobian at zero under self.map_info['linear_part'].
        """
        ...

    @abstractmethod
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Given inputs X (n_samples×input_dim), return model outputs.
        """
        ...

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.predict(x)


class RBFFittedMapping(BaseFittedMapping):
    def __init__(self, w, b, beta, intercept, n_components, gamma, regularization, linear_part):
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
        self.W = w
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

    @staticmethod
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

    @staticmethod
    def _compute_jacobian(X_aug, W, b, beta):
        x0 = jnp.zeros(X_aug.shape[1])
        arg = jnp.dot(x0, W) + b
        D = arg.shape[0]
        dz_dx = - jnp.sqrt(2.0 / D) * jnp.sin(arg)[:, jnp.newaxis] * W.T
        if beta.ndim == 1:
            beta = beta[jnp.newaxis, :]
        J = jnp.dot(beta, dz_dx)

        return J

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, x):
        """
        Predict using the learned RBF mapping, ridge regression weights and intercept.

        Parameters:
          x (jax.numpy.ndarray): Input data of shape (n_samples, input_dim).

        Returns:
          jax.numpy.ndarray: Predictions computed as z(X) dot beta.T + intercept.
        """
        z = jnp.sqrt(2.0 / self.map_info['n_components']) * jnp.cos(jnp.dot(x, self.W) + self.b)
        return jnp.dot(z, self.beta.T) + self.intercept

    @classmethod
    def fit(cls, X_aug, Y, n_components, gamma, regularization, seed=42, w=None, b=None, **regression_args):
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
          W : Already initialized weights
          b : already initialized biases
          regression_args: Additional keyword arguments for Ridge.

        Returns:
          FittedMapping: A fitted instance.
        """

        # 1. If W and b are not provided, fall back to sklearn's RBFSampler
        if w is None or b is None:
            w, b, beta, intercept = cls._fit_rbf_pipeline(X_aug, Y, n_components, gamma, regularization, seed=seed,
                                                          **regression_args)
            linear_part = cls._compute_jacobian(X_aug, w, b, beta)
            return cls(w, b, beta, intercept, n_components, gamma, regularization, linear_part)
        else:
            Z = jnp.sqrt(2.0 / n_components) * jnp.cos(jnp.dot(X_aug, w) + b)

            # 3. Fit Ridge regression on Z (using sklearn)
            ridge = Ridge(alpha=regularization, **regression_args)
            # convert to numpy for sklearn
            ridge.fit(np.array(Z), np.array(Y))

            # 4. Extract weights and intercept back into JAX arrays
            beta = jnp.array(ridge.coef_)
            if beta.ndim == 1:
                beta = beta[jnp.newaxis, :]
            intercept = jnp.array(ridge.intercept_)

            # 5. Compute the Jacobian at the origin
            linear_part = cls._compute_jacobian(X_aug, w, b, beta)

            # 6. Return the fitted mapping
            return cls(w, b, beta, intercept, n_components, gamma, regularization, linear_part)


class PolynomialFittedMapping(BaseFittedMapping):
    def __init__(self,
                 poly:       PolynomialFeatures,
                 powers: jnp.ndarray,
                 beta:       jnp.ndarray,
                 intercept:  jnp.ndarray,
                 degree:     int,
                 linear_part: jnp.ndarray):
        self.poly = poly
        self.powers = powers
        self.beta = beta       # shape (n_output, n_poly_feats)
        self.intercept = intercept  # shape (n_output,)
        self.map_info = {
            'degree':      degree,
            'linear_part': linear_part
        }

    def _poly_transform_jax(self, X: jnp.ndarray) -> jnp.ndarray:
        # X: (n_samples, input_dim)
        # powers: (n_feats, input_dim)
        # we want Xp[n, f] = prod_d X[n,d] ** powers[f,d]
        # → shape (n_samples, n_feats)
        return jnp.prod(
            X[:, None, :] ** self.powers[None, :, :],
            axis=-1
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        # pure JAX, no NumPy anywhere
        Xp = self._poly_transform_jax(x)  # (n, n_feats)
        # now do the matrix multiply
        return Xp @ self.beta.T + self.intercept

    @classmethod
    def fit(cls,
            X_aug: jnp.ndarray,
            Y: jnp.ndarray,
            degree: int,
            regularization: float,
            **ridge_kwargs
            ) -> "PolynomialFittedMapping":
        # 1) build features
        poly = PolynomialFeatures(degree, include_bias=False)
        Xp = poly.fit_transform(np.array(X_aug))
        Y_np = np.array(Y)

        # 2) train ridge
        ridge = Ridge(alpha=regularization, fit_intercept=True, **ridge_kwargs)
        ridge.fit(Xp, Y_np)

        coef = ridge.coef_  # (n_targets, n_feats)
        intercept = ridge.intercept_  # (n_targets,)

        beta = jnp.atleast_2d(jnp.array(coef))
        intercept = jnp.atleast_1d(jnp.array(intercept))

        # get powers once, store as JAX array
        powers = jnp.array(poly.powers_)

        # 3) find pure-linear feature indices
        linear_idx = [i for i, p in enumerate(poly.powers_) if p.sum() == 1]

        # 4) pick out linear coefficients as Jacobian
        linear_part = beta[:, linear_idx]  # (n_targets, input_dim)

        return cls(poly, powers, beta, intercept, degree, linear_part)


class NeuralNetFittedMapping(BaseFittedMapping):
    def __init__(self,
                 init_fun:  Any,
                 apply_fun: Any,
                 params:    Any,
                 map_info:  Dict[str, Any]):
        """
        init_fun, apply_fun = from stax.serial(...)
        params            = trained network parameters
        map_info          = {
            'layers': [...],
            'lr': ...,
            'epochs': ...,
            'linear_part': J,
            'train_losses': [...],
            'val_losses': [...]  # only if validation was provided
        }
        """
        self.init_fun  = init_fun
        self.apply_fun = apply_fun
        self.params    = params
        self.map_info  = map_info

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Pure-JAX forward pass. JIT-compiled for speed.
        """
        return self.apply_fun(self.params, X)

    @classmethod
    def fit(cls,
            x: jnp.ndarray,
            y: jnp.ndarray,
            layers: Sequence[int],
            lr: float = 1e-3,
            epochs: int = 500,
            seed: int = 0,
            x_val: Optional[jnp.ndarray] = None,
            y_val: Optional[jnp.ndarray] = None,
            verbose: bool = False
           ) -> "NeuralNetFittedMapping":
        """
        Trains an MLP on (X, Y), optionally monitors (X_val, Y_val).
        Returns an instance with:
          - .map_info['train_losses'] and ['val_losses']
          - .map_info['linear_part'] = Jacobian at 0
        """

        # 0) data → JAX
        x_train = jnp.array(x)
        y_train = jnp.array(y)
        if x_val is not None and y_val is not None:
            x_val_j, y_val_j = jnp.array(x_val), jnp.array(y_val)
        else:
            x_val_j = y_val_j = None

        # 1) build a small MLP
        init_fun, apply_fun = stax.serial(
            *(sum([[Dense(n), Relu] for n in layers[:-1]], [])),
            Dense(layers[-1])
        )
        rng = jax.random.PRNGKey(seed)
        _, init_params = init_fun(rng, input_shape=(-1, x_train.shape[1]))

        # 2) optimizer
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(init_params)

        # 3) loss + step (both JIT)
        @jax.jit
        def loss_fn(params, x, y):
            preds = apply_fun(params, x)
            return jnp.mean((preds - y)**2)

        @jax.jit
        def step(i, opt_state, x, y):
            params = get_params(opt_state)
            grads = jax.grad(loss_fn)(params, x, y)
            return opt_update(i, grads, opt_state)

        # 4) training loop with metrics
        train_losses = []
        val_losses = [] if x_val_j is not None else None

        for epoch in range(1, epochs+1):
            opt_state = step(epoch, opt_state, x_train, y_train)
            params = get_params(opt_state)

            # record metrics
            train_l = float(loss_fn(params, x_train, y_train))
            train_losses.append(train_l)

            if x_val_j is not None:
                val_l = float(loss_fn(params, x_val_j, y_val_j))
                val_losses.append(val_l)

            if verbose:
                if x_val_j is not None:
                    print(f"[Epoch {epoch}/{epochs}] train_loss={train_l:.6f}, val_loss={val_l:.6f}")
                else:
                    print(f"[Epoch {epoch}/{epochs}] train_loss={train_l:.6f}")

        trained_params = get_params(opt_state)

        # 5) Jacobian at zero
        def f0(x):
            return apply_fun(trained_params, x[None, :])[0]
        J = jax.jacobian(f0)(jnp.zeros(x_train.shape[1]))

        # 6) assemble map_info
        mi: Dict[str, Any] = {
            'layers':       layers,
            'lr':           lr,
            'epochs':       epochs,
            'linear_part':  J,
            'train_losses': train_losses
        }
        if val_losses is not None:
            mi['val_losses'] = val_losses

        # 7) optional plot
        if verbose:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(train_losses, label='train loss')

            if val_losses is not None:
                plt.plot(val_losses, label='val loss')
                plt.xlabel('epoch')
                plt.ylabel('MSE')
                plt.title('Training curves')
                plt.legend()
                plt.show()

        return cls(init_fun, apply_fun, trained_params, mi)
