r"""Base classes for equation of state models."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from .. import utils


class Interpolate_EOS_model(object):
    r"""
    Base class for interpolating equation of state (EOS) data.

    This class provides the fundamental interpolation framework for converting
    tabulated EOS data (density, pressure, energy) into the auxiliary quantities
    needed for neutron star structure calculations using the TOV equations.
    """

    def __init__(self):
        pass

    def interpolate_eos(
        self,
        n: Float[Array, "n_points"],
        p: Float[Array, "n_points"],
        e: Float[Array, "n_points"],
    ):
        r"""
        Convert physical EOS quantities to geometric units and compute auxiliary quantities.

        This method transforms the input EOS data from nuclear physics units to geometric
        units used in general relativity calculations, and computes derived quantities
        needed for the TOV equations.

        Args:
            n (Float[Array, n_points]): Number densities [:math:`\mathrm{fm}^{-3}`]
            p (Float[Array, n_points]): Pressure values [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]
            e (Float[Array, n_points]): Energy densities [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]

        Returns:
            tuple: A tuple containing (all in geometric units):

                - ns: Number densities
                - ps: Pressures
                - hs: Specific enthalpy :math:`h = \int \frac{dp}{\varepsilon + p}`
                - es: Energy densities
                - dloge_dlogps: Logarithmic derivative :math:`\frac{d\ln\varepsilon}{d\ln p}`
        """

        # Save the provided data as attributes, make conversions
        ns = jnp.array(n * utils.fm_inv3_to_geometric)
        ps = jnp.array(p * utils.MeV_fm_inv3_to_geometric)
        es = jnp.array(e * utils.MeV_fm_inv3_to_geometric)

        # rhos = utils.calculate_rest_mass_density(es, ps)

        hs = utils.cumtrapz(ps / (es + ps), jnp.log(ps))  # enthalpy
        dloge_dlogps = jnp.diff(jnp.log(e)) / jnp.diff(jnp.log(p))
        dloge_dlogps = jnp.concatenate(
            (
                jnp.array(
                    [
                        dloge_dlogps.at[0].get(),
                    ]
                ),
                dloge_dlogps,
            )
        )
        return ns, ps, hs, es, dloge_dlogps
