"""
Chiral Effective Field Theory likelihood implementations

TODO: need to generalize to other ways to use chiEFT information in the likelihood. For now only works with the data from Koehn et al., Phys.Rev.X 15 (2025) 2, 021014.
"""

import jax.numpy as jnp
import numpy as np
from pathlib import Path
from jaxtyping import Float
from jesterTOV.inference.base import LikelihoodBase

class ChiEFTLikelihood(LikelihoodBase):
    """
    Chiral Effective Field Theory likelihood for low-density constraints

    Constrains EOS using ChiEFT pressure-density bands at low densities.

    Parameters
    ----------
    low_filename : str or Path, optional
        Path to file containing lower bound of chiEFT band.
        Defaults to data/chiEFT/2402.04172/low.dat
    high_filename : str or Path, optional
        Path to file containing upper bound of chiEFT band.
        Defaults to data/chiEFT/2402.04172/high.dat
    nb_n : int, optional
        Number of density points for integration (default: 100)
    """

    def __init__(
        self,
        low_filename: str | Path | None = None,
        high_filename: str | Path | None = None,
        nb_n: int = 100,
    ):
        super().__init__()

        # Set default paths if not provided
        if low_filename is None:
            data_dir = Path(__file__).parent.parent / "data" / "chiEFT" / "2402.04172"
            low_filename = data_dir / "low.dat"
        if high_filename is None:
            data_dir = Path(__file__).parent.parent / "data" / "chiEFT" / "2402.04172"
            high_filename = data_dir / "high.dat"

        # Load data files
        # File format: 3 columns (density [fm^-3], pressure [MeV/fm^3], energy density [MeV/fm^3])
        # We only use the first two columns
        low_data = np.loadtxt(low_filename)
        high_data = np.loadtxt(high_filename)

        # Extract density and pressure columns
        # Convert density to nsat units (nsat = 0.16 fm^-3)
        n_low = jnp.array(low_data[:, 0]) / 0.16
        p_low = jnp.array(low_data[:, 1])

        n_high = jnp.array(high_data[:, 0]) / 0.16
        p_high = jnp.array(high_data[:, 1])

        # Store data and create interpolation functions
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
        prefactor = 1 / (nbreak - 0.75 * 0.16)
        log_likelihood = prefactor * jnp.sum(f_array) * dn

        return log_likelihood
