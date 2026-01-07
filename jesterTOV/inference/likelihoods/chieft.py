r"""
Chiral Effective Field Theory constraints for low-density nuclear matter.

This module implements likelihood functions based on chiral effective field
theory (chiEFT) predictions for the nuclear equation of state at low densities
(below ~2 saturation density). ChiEFT provides rigorous theoretical constraints
on the pressure-density relationship derived from fundamental interactions,
serving as a complementary constraint to high-density astrophysical observations.

The current implementation uses the pressure bands from [chieft1]_, which
provide upper and lower bounds on the allowed pressure at each density.

References
----------
.. [chieft1] Koehn et al., "Equation of state constraints from multi-messenger
   observations of neutron stars," Phys. Rev. X 15, 021014 (2025).

Notes
-----
Future extensions may include other chiEFT formulations and additional
low-density constraints beyond the current pressure band implementation.
"""

from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.inference.base import LikelihoodBase


class ChiEFTLikelihood(LikelihoodBase):
    """Likelihood function enforcing chiral EFT constraints on the nuclear EOS.

    This likelihood evaluates how well a candidate equation of state agrees with
    theoretical predictions from chiral effective field theory in the low-density
    regime (0.75 - 2.0 n_sat). The chiEFT calculations provide a band of allowed
    pressures at each density; EOSs within the band receive higher likelihood,
    while those outside are penalized proportional to their deviation.

    The likelihood is computed as an integral over density of a penalty function
    that assigns:
    - Weight 1.0 for pressures within the chiEFT band
    - Exponential penalty for pressures outside the band (slope β = 6/(p_high - p_low))

    This formulation smoothly incorporates theoretical uncertainties while strongly
    disfavoring unphysical EOSs.

    Parameters
    ----------
    low_filename : str | Path | None, optional
        Path to data file containing the lower boundary of the chiEFT allowed band.
        The file should have three columns: density [fm⁻³], pressure [MeV/fm³],
        energy density [MeV/fm³] (only first two are used).
        If None, defaults to the Koehn et al. (2025) low band in the package data.
    high_filename : str | Path | None, optional
        Path to data file containing the upper boundary of the chiEFT allowed band.
        Same format as low_filename.
        If None, defaults to the Koehn et al. (2025) high band in the package data.
    nb_n : int, optional
        Number of density points for numerical integration of the penalty function.
        More points provide better accuracy but increase computation time.
        Default is 100, which provides good balance for typical applications.

    Attributes
    ----------
    n_low : Float[Array, " n_points"]
        Density grid for lower bound in units of n_sat (saturation density = 0.16 fm⁻³)
    p_low : Float[Array, " n_points"]
        Pressure values for lower bound in MeV/fm³
    n_high : Float[Array, " n_points"]
        Density grid for upper bound in units of n_sat
    p_high : Float[Array, " n_points"]
        Pressure values for upper bound in MeV/fm³
    EFT_low : Callable[[Float | Float[Array, "..."]], Float | Float[Array, "..."]]
        Interpolation function returning lower bound pressure at given density
    EFT_high : Callable[[Float | Float[Array, "..."]], Float | Float[Array, "..."]]
        Interpolation function returning upper bound pressure at given density
    nb_n : int
        Number of density points used for integration

    Notes
    -----
    The penalty function f(p_sample, p_low, p_high) is defined as:

    .. math::
        f(p) = \\begin{cases}
            1 - \\beta(p - p_{high}) & \\text{if } p > p_{high} \\\\
            1 & \\text{if } p_{low} \\leq p \\leq p_{high} \\\\
            1 - \\beta(p_{low} - p) & \\text{if } p < p_{low}
        \\end{cases}

    where β = 6/(p_high - p_low) controls the penalty strength.

    The integration is performed from 0.75 n_sat (lower limit of chiEFT validity)
    to nbreak (where the CSE extension begins, if present).

    See Also
    --------
    REXLikelihood : Nuclear radius constraints from PREX/CREX experiments

    Examples
    --------
    Create a chiEFT likelihood with default data:

    >>> from jesterTOV.inference.likelihoods import ChiEFTLikelihood
    >>> likelihood = ChiEFTLikelihood(nb_n=100)
    >>> log_like = likelihood.evaluate(params, data={})
    """

    n_low: Float[Array, " n_points"]
    p_low: Float[Array, " n_points"]
    n_high: Float[Array, " n_points"]
    p_high: Float[Array, " n_points"]
    EFT_low: Callable[[Float | Float[Array, "..."]], Float | Float[Array, "..."]]
    EFT_high: Callable[[Float | Float[Array, "..."]], Float | Float[Array, "..."]]
    nb_n: int

    def __init__(
        self,
        low_filename: str | Path | None = None,
        high_filename: str | Path | None = None,
        nb_n: int = 100,
    ) -> None:
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

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """Evaluate the log-likelihood for chiEFT constraints.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Dictionary containing EOS quantities from the transform. Required keys:
            - "n" : Baryon number density grid (geometric units)
            - "p" : Pressure values on density grid (geometric units)
            - "nbreak" : Breaking density where CSE begins (fm⁻³)

        Returns
        -------
        Float
            Natural logarithm of the likelihood. Higher values indicate better
            agreement with chiEFT predictions. The value is normalized by the
            integration range so that perfect agreement gives log L ≈ 0.

        Notes
        -----
        The integration is performed from 0.75 n_sat to nbreak using nb_n
        equally spaced points. The EOS pressure is interpolated onto this grid
        and compared against the chiEFT band at each point.

        Unit conversions are applied automatically:
        - Input densities converted from geometric to fm⁻³ units
        - Input pressures converted from geometric to MeV/fm³ units
        """
        # Get relevant parameters
        n, p = params["n"], params["p"]
        nbreak = params["nbreak"]

        # Convert to nsat for convenience
        nbreak = nbreak / 0.16
        n = n / utils.fm_inv3_to_geometric / 0.16
        p = p / utils.MeV_fm_inv3_to_geometric

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
