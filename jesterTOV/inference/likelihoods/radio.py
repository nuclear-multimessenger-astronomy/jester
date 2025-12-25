"""Radio pulsar timing likelihood implementations"""

from typing import Any

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from jesterTOV.inference.base import LikelihoodBase


class RadioTimingLikelihood(LikelihoodBase):
    r"""
    Radio pulsar timing likelihood for maximum mass constraints.

    This likelihood constrains the maximum TOV mass (M_max) based on observed
    pulsar masses from radio timing observations. It implements a marginalization
    over possible true masses below M_max, assuming a Gaussian measurement uncertainty.

    The likelihood is computed as:

    TODO: double-check the math just to be sure

    .. math::
        \mathcal{L}(M_{\text{max}} | M_{\text{obs}}, \sigma) =
        \frac{1}{M_{\text{max}}} \int_0^{M_{\text{max}}}
        \mathcal{N}(m | M_{\text{obs}}, \sigma) dm

    where the 1/M_max factor represents a uniform prior on the true mass.

    Parameters
    ----------
    psr_name : str
        Pulsar name (e.g., "J1614-2230", "J0740+6620")
    mean : float
        Observed mass mean in solar masses (M_☉)
    std : float
        Observed mass standard deviation in solar masses (M_☉)
    nb_masses : int, optional
        Number of mass points for numerical integration. Default: 500
    m_min : float, optional
        Minimum mass for integration in solar masses. Default: 0.1

    Attributes
    ----------
    psr_name : str
        Pulsar name
    mean : float
        Observed mass mean
    std : float
        Observed mass standard deviation
    nb_masses : int
        Number of integration points
    m_min : float
        Minimum integration mass

    Notes
    -----
    The integration is performed using a discrete sum with logsumexp for
    numerical stability. The mass grid spans from m_min M_☉ to M_max.

    Examples
    --------
    >>> # PSR J1614-2230: 1.94 ± 0.06 M_☉
    >>> likelihood = RadioTimingLikelihood("J1614", 1.94, 0.06)
    >>> log_prob = likelihood.evaluate(params, {})
    """

    psr_name: str
    mean: float
    std: float
    nb_masses: int
    m_min: float

    def __init__(
        self,
        psr_name: str,
        mean: float,
        std: float,
        nb_masses: int = 500,
        m_min: float = 0.1,  # TODO: determine if needs tuning later on
    ) -> None:
        super().__init__()
        self.psr_name = psr_name
        self.mean = mean
        self.std = std
        self.nb_masses = nb_masses
        self.m_min = m_min  # Minimum mass for integration (M_☉)

    def evaluate(self, params: dict[str, Float | Array], data: dict[str, Any]) -> Float:
        """
        Evaluate the log likelihood.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Dictionary containing 'masses_EOS' key with sampled mass points
        data : dict[str, Any]
            Unused (data is encapsulated in the likelihood object)

        Returns
        -------
        Float
            Log likelihood value
        """
        # Extract maximum TOV mass from the EOS
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        # Create mass grid for integration from m_min to M_max
        # Per Eq. (X): P(θ_EOS | d_radio) ∝ (1/M_TOV) ∫₀^M_TOV P(M | d_radio) dM
        # Note: Start at m_min (default 0.1 M_☉) instead of 0 to avoid numerical issues
        m = jnp.linspace(self.m_min, mtov, self.nb_masses)

        # Gaussian log likelihood with proper normalization constant
        # P(M | d_radio) = N(M | mean, std)
        gauss_norm = -0.5 * jnp.log(2 * jnp.pi * self.std**2)
        log_likelihood_array = gauss_norm + (-0.5 * ((m - self.mean) / self.std) ** 2)

        # Numerical integration using logsumexp (for stability)
        # integral ≈ dx * sum(exp(log_likelihood_array))
        # log(integral) = log(dx) + logsumexp(log_likelihood_array)
        dx = (mtov - self.m_min) / self.nb_masses
        log_likelihood = logsumexp(log_likelihood_array) + jnp.log(dx)

        # Apply 1/M_max normalization factor (uniform prior on true mass in [0, M_TOV])
        log_likelihood -= jnp.log(mtov)

        # Penalty for unphysical masses (mtov <= m_min)
        # If M_TOV is below the minimum integration mass, the integral is invalid
        penalty_mtov = jnp.where(mtov <= self.m_min, -1e10, 0.0)

        return log_likelihood + penalty_mtov
