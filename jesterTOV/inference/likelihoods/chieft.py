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
    n_max_nsat: float

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
        # Maximum density covered by the chiEFT data (in nsat units).
        # Integration must not exceed this to avoid unphysical flat-line extrapolation.
        self.n_max_nsat = float(jnp.minimum(n_low[-1], n_high[-1]))

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """Evaluate the log-likelihood for chiEFT constraints.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Dictionary containing EOS quantities from the transform. Required keys:
            - "n" : Baryon number density grid (geometric units, i.e. fm⁻³ × ``utils.fm_inv3_to_geometric``)
            - "p" : Pressure values on density grid (geometric units, i.e. MeV/fm³ × ``utils.MeV_fm_inv3_to_geometric``)
            - "nbreak" : Breaking density where CSE begins (fm⁻³, physical units)

        Returns
        -------
        Float
            Natural logarithm of the likelihood. Higher values indicate better
            agreement with chiEFT predictions. The value is normalized by the
            integration range so that perfect agreement gives log L ≈ 0.

        Notes
        -----
        All density arithmetic inside this method is performed in units of
        n_sat (saturation density = 0.16 fm⁻³).  Input quantities are
        converted at the start:

        - ``nbreak`` [fm⁻³] → ``nbreak / 0.16`` [n_sat]
        - ``n`` [geometric] → ``n / fm_inv3_to_geometric / 0.16`` [n_sat]
        - ``p`` [geometric] → ``p / MeV_fm_inv3_to_geometric`` [MeV/fm³]

        The integration runs from 0.75 n_sat (lower limit of chiEFT validity)
        to ``min(nbreak, n_max_nsat)`` [n_sat] using ``nb_n`` equally spaced
        points. The upper limit is capped at ``n_max_nsat`` (2.0 n_sat for the
        default Koehn et al. 2025 bands) to avoid integrating over the flat
        constant extrapolation that ``jnp.interp`` produces beyond the data
        range, which has no physical meaning. If ``nbreak`` falls below
        0.75 n_sat the upper limit is clamped to ``0.75 + 1e-8`` n_sat to
        prevent a degenerate integration range.
        """
        # Get relevant parameters
        n, p = params["n"], params["p"]
        nbreak = params["nbreak"]

        # Convert all densities to n_sat units (n_sat = 0.16 fm^-3).
        # nbreak arrives in fm^-3 (physical); n arrives in geometric units.
        # Pressures are converted from geometric to MeV/fm^3.
        nbreak = nbreak / 0.16  # fm^-3 → n_sat
        n = n / utils.fm_inv3_to_geometric / 0.16  # geometric → n_sat
        p = p / utils.MeV_fm_inv3_to_geometric  # geometric → MeV/fm^3

        # Cap the integration at the maximum density covered by the chiEFT data.
        # Beyond n_max_nsat the data ends and jnp.interp returns a flat
        # (constant) extrapolation that has no physical meaning.
        # Both nbreak and n_max_nsat are in n_sat units at this point.
        n_upper = jnp.minimum(nbreak, self.n_max_nsat)

        # Guard against a degenerate integration range when nbreak < 0.75 n_sat.
        n_upper = jnp.maximum(n_upper, 0.75 + 1e-8)

        # Build density grid in n_sat units: lower limit is 0.75 n_sat.
        this_n_array = jnp.linspace(0.75, n_upper, self.nb_n)
        dn = this_n_array.at[1].get() - this_n_array.at[0].get()
        low_p = self.EFT_low(this_n_array)
        high_p = self.EFT_high(this_n_array)

        # Evaluate the sampled p(n) at the given n
        sample_p = jnp.interp(this_n_array, n, p)

        # Compute f
        def f(sample_p, low_p, high_p):
            beta = 6 / (high_p - low_p)
            return_value = -beta * (sample_p - high_p) * jnp.heaviside(
                sample_p - high_p, 0
            ) + -beta * (low_p - sample_p) * jnp.heaviside(low_p - sample_p, 0)
            return return_value

        f_array = f(sample_p, low_p, high_p)
        # Normalise by the integration width (in n_sat units).
        # n_upper and 0.75 are both in n_sat units here.
        prefactor = 1 / (n_upper - 0.75)
        log_likelihood = prefactor * jnp.sum(f_array) * dn

        return log_likelihood
