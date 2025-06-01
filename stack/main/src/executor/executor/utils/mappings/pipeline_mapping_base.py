import jax.numpy as jnp
from typing import Any, Dict
from model_classes.mappings.mapping_modules import RBFFittedMapping, PolynomialFittedMapping, NeuralNetFittedMapping


class FittedMapping:
    @classmethod
    def fit(
            cls,
            X_aug: jnp.ndarray,
            Y: jnp.ndarray,
            *,
            type_of_fitting: str,
            hyperparams: Dict[str, Any],
            seed: int = 42,
            w: jnp.ndarray = None,
            b: jnp.ndarray = None
    ):
        """
        Unified entry point for three fitting strategies.

        Arguments:
          type_of_fitting:  'rbf' | 'polynomial' | 'nn'
          hyperparams:      dict of the needed hyperparameters:

            • if 'rbf':
                - required: 'n_components', 'gamma', 'regularization'
                - optional: any sklearn.Ridge kwargs
            • if 'polynomial':
                - required: 'degree', 'regularization'
            • if 'nn':
                - required: 'layers' (e.g. [64,64,output_dim])
                - optional: 'lr', 'epochs'
          seed: random seed for RBF and NN
          w:    optionally provide mapping
          b:    optionally provide mapping
        """
        fitter_map = {
            'rbf': RBFFittedMapping.fit,
            'polynomial': PolynomialFittedMapping.fit,
            'nn': NeuralNetFittedMapping.fit
        }
        if type_of_fitting not in fitter_map:
            raise ValueError(f"Unknown type_of_fitting={type_of_fitting!r}")

        # figure out which keys are mandatory
        if type_of_fitting == 'rbf':
            required = {'n_components', 'gamma', 'regularization'}
        elif type_of_fitting == 'polynomial':
            required = {'degree', 'regularization'}
        else:  # 'nn'
            required = {'layers'}

        missing = required - hyperparams.keys()
        if missing:
            raise KeyError(f"Missing hyperparams for {type_of_fitting}: {missing}")

        # dispatch to the correct subclass
        if type_of_fitting == 'rbf':
            # pass through any extra Ridge args automatically
            extra = {k: v for k, v in hyperparams.items() if k not in required}
            return RBFFittedMapping.fit(
                X_aug,
                Y,
                n_components=hyperparams['n_components'],
                gamma=hyperparams['gamma'],
                regularization=hyperparams['regularization'],
                seed=seed,
                w=w,
                b=b,
                **extra
            )

        elif type_of_fitting == 'polynomial':
            return PolynomialFittedMapping.fit(
                X_aug,
                Y,
                degree=hyperparams['degree'],
                regularization=hyperparams['regularization']
            )

        else:  # 'nn'
            return NeuralNetFittedMapping.fit(
                X_aug,
                Y,
                layers=hyperparams['layers'],
                lr=hyperparams.get('lr', 1e-3),
                epochs=hyperparams.get('epochs', 500),
                seed=seed
            )
