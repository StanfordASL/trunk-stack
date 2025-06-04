import jax.numpy as jnp
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key
import sympy as sp


class RBFFeatureMap:
    """
    Top‐level class wrapper for RBF‐cosine features.
    Instances are pickleable because RBFFeatureMap is a module‐level class.
    """
    def __init__(self, w: jnp.ndarray, b: jnp.ndarray):
        self.w = w        # shape: (dim, n_components)
        self.b = b        # shape: (n_components,)
        dim, n_components = w.shape
        self.norm = jnp.sqrt(2.0 / n_components)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: either shape (dim,) or (batch, dim)
        returns: (n_components,) or (batch, n_components)
        """
        proj = jnp.dot(x, self.w)      # broadcasts batch‐wise if needed
        return self.norm * jnp.cos(proj + self.b)


class PolyFeatureMap:
    """
    Top‐level class wrapper for polynomial features (degree ≤ order in dim variables).
    """
    def __init__(self, dim: int, order: int):
        # Build sympy monomials once, then store exponents_array as a JAX int array.
        zeta = sp.Matrix(sp.symbols(f'x1:{dim+1}'))
        all_monoms = sorted(
            itermonomials(list(zeta), order),
            key=monomial_key('grevlex', list(reversed(zeta)))
        )
        monoms_no_const = all_monoms[1:]
        exponents_list = []
        for mon in monoms_no_const:
            pd = mon.as_powers_dict()
            exponents_list.append(tuple(int(pd.get(sym, 0)) for sym in zeta))

        self.exponents_array = jnp.array(exponents_list, dtype=jnp.int32)
        # exponents_array.shape == (n_monomials, dim)
        self.dim = dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: either (dim,) or (batch, dim)
        returns: (n_monomials,) or (batch, n_monomials)
        """
        if x.ndim == 1:
            x_expanded = x[jnp.newaxis, :]  # shape (1, dim)
            batched = False
        else:
            if x.ndim != 2 or x.shape[1] != self.dim:
                raise ValueError(f"Expected (dim,) or (batch,dim), got {x.shape}")
            x_expanded = x
            batched = True

        # (batch, dim) ** (1, n_monomials, dim) → (batch, n_monomials, dim)
        power_tensor = x_expanded[:, jnp.newaxis, :] ** self.exponents_array[jnp.newaxis, :, :]
        monom_vals   = jnp.prod(power_tensor, axis=2)  # shape (batch, n_monomials)

        return monom_vals if batched else monom_vals[0]


def rbf_feature_map_factory(w: jnp.ndarray, b: jnp.ndarray) -> RBFFeatureMap:
    """
    Factory function that returns a top‐level RBFFeatureMap instance.
    """
    return RBFFeatureMap(w, b)


def poly_feature_map_factory(dim: int, order: int) -> PolyFeatureMap:
    return PolyFeatureMap(dim, order)
