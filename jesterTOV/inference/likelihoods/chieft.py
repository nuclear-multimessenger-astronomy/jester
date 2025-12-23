"""Chiral Effective Field Theory likelihood implementations"""

import jax.numpy as jnp
from jaxtyping import Float
from jesterTOV.inference.base import LikelihoodBase


class ChiEFTLikelihood(LikelihoodBase):
    """
    Chiral Effective Field Theory likelihood for low-density constraints

    Constrains EOS using ChiEFT pressure-density bands at low densities.

    Parameters
    ----------
    n_low : array
        Density grid for low-density band (in units of nsat)
    p_low : array
        Pressure values for low-density band
    n_high : array
        Density grid for high-density band (in units of nsat)
    p_high : array
        Pressure values for high-density band
    nb_n : int, optional
        Number of density points for integration
    """

    def __init__(
        self,
        n_low: jnp.ndarray,
        p_low: jnp.ndarray,
        n_high: jnp.ndarray,
        p_high: jnp.ndarray,
        nb_n: int = 100,
    ):
        super().__init__()
        self.n_low = n_low
        self.p_low = p_low
        self.EFT_low = lambda x: jnp.interp(x, n_low, p_low)

        self.n_high = n_high
        self.p_high = p_high
        self.EFT_high = lambda x: jnp.interp(x, n_high, p_high)

        self.nb_n = nb_n

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # Import here to avoid circular dependency # FIXME: what circular dependency?
        from jesterTOV import utils as jose_utils

        # Get relevant parameters
        n, p = params["n"], params["p"]
        nbreak = params["nbreak"]

        # Convert to nsat for convenience
        nbreak = nbreak / 0.16
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric

        prefactor = 1 / (nbreak - 0.75 * 0.16)

        # Lower limit is at 0.12 fm-3
        this_n_array = jnp.linspace(0.75, nbreak, self.nb_n)
        dn = this_n_array.at[1].get() - this_n_array.at[0].get()
        low_p = self.EFT_low(this_n_array)
        high_p = self.EFT_high(this_n_array)

        # Evaluate the sampled p(n) at the given n
        sample_p = jnp.interp(this_n_array, n, p)

        # Compute f
        def f(sample_p, low_p, high_p):
            beta = 6 / (high_p - low_p)
            return_value = (
                -beta * (sample_p - high_p) * jnp.heaviside(sample_p - high_p, 0)
                + -beta * (low_p - sample_p) * jnp.heaviside(low_p - sample_p, 0)
                + 1
                * jnp.heaviside(sample_p - low_p, 0)
                * jnp.heaviside(high_p - sample_p, 0)
            )
            return return_value

        f_array = f(sample_p, low_p, high_p)
        log_likelihood = prefactor * jnp.sum(f_array) * dn

        return log_likelihood
