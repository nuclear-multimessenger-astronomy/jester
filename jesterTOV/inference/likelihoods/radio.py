r"""
Radio pulsar mass measurements from timing observations.

This module implements likelihood functions for constraining the equation of
state using precisely measured masses of radio pulsars. Pulsar timing provides
some of the most accurate mass measurements in astrophysics, with uncertainties
as low as 1-2% for the best-measured systems.

These measurements constrain the maximum gravitational mass that can be supported
by neutron star matter, providing a crucial lower bound on the stiffness of the
equation of state. The most massive precisely measured pulsars (e.g., PSR J0740+6620
at ~2.1 solar masses) rule out many soft EOS models that predict lower maximum masses.

The likelihood marginalizes over the true pulsar mass (unknown but bounded by the
maximum TOV mass) assuming a Gaussian measurement uncertainty and a uniform prior
on the true mass.

References
----------
Demorest et al., "A two-solar-mass neutron star measured using Shapiro delay,"
Nature 467, 1081-1083 (2010).

Fonseca et al., "Refined Mass and Geometric Measurements of the High-Mass
PSR J0740+6620," ApJL 915, L12 (2021).
"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from jesterTOV.inference.base import LikelihoodBase


class RadioTimingLikelihood(LikelihoodBase):
    r"""Likelihood for radio pulsar mass measurements constraining maximum NS mass.

    This likelihood evaluates how well an equation of state's maximum TOV mass
    (M_TOV) is consistent with an observed pulsar mass measurement. Since we
    observe a specific pulsar (with some measurement uncertainty) but don't know
    its true mass relative to the theoretical maximum, we must marginalize over
    all possible true masses between some minimum value and M_TOV.

    The marginalization assumes:
    - The measured mass follows a Gaussian distribution: :math:`M_{\text{obs}} \sim \mathcal{N}(M_{\text{true}}, \sigma)`
    - The true mass has a uniform prior: :math:`P(M_{\text{true}} | M_{\text{TOV}}) = 1/M_{\text{TOV}}` for :math:`M_{\text{true}} \in [0, M_{\text{TOV}}]`

    This gives the marginal likelihood: # TODO: check equations here

    .. math::
        \mathcal{L}(M_{\text{TOV}} | M_{\text{obs}}, \sigma) =
        \frac{1}{M_{\text{TOV}}} \int_{m_{\text{min}}}^{M_{\text{TOV}}}
        \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac{(m - M_{\text{obs}})^2}{2\sigma^2}\right] dm

    where the integration is performed numerically using a discrete grid and
    logsumexp for numerical stability.

    Parameters
    ----------
    psr_name : str
        Pulsar designation for identification (e.g., "J1614-2230", "J0740+6620").
        Used for logging and tracking which pulsar constraint is being applied.
    mean : float
        Measured pulsar mass in solar masses (:math:`M_{\odot}`). This is typically the
        reported value from timing analysis.
    std : float
        Measurement uncertainty (:math:`1\sigma`) in solar masses. This combines statistical
        and systematic uncertainties from the timing analysis.
    nb_masses : int, optional
        Number of grid points for numerical integration of the marginalization
        integral. More points provide better accuracy but increase computation time.
        Default is 500, which provides good accuracy for typical uncertainties.
    m_min : float, optional
        Minimum mass for the integration grid in solar masses. This should be
        well below any physical neutron star mass to avoid truncation effects.
        Default is 0.1 :math:`M_{\odot}`.

    Attributes
    ----------
    psr_name : str
        Pulsar designation
    mean : float
        Observed mass mean in solar masses
    std : float
        Observed mass uncertainty in solar masses
    nb_masses : int
        Number of mass grid points for integration
    m_min : float
        Minimum mass for integration (solar masses)

    Notes
    -----
    Invalid TOV solutions (M_TOV ≤ m_min) receive a large negative log-likelihood
    penalty (-1e10) to effectively exclude them from the posterior.

    The implementation uses log-space arithmetic throughout to avoid numerical
    underflow when combining with other log-likelihoods.

    See Also
    --------
    GWLikelihood : Gravitational wave constraints on mass and tidal deformability
    NICERLikelihood : X-ray timing constraints on mass and radius

    Examples
    --------
    Create a likelihood for PSR J0740+6620 (Fonseca et al. 2021: 2.08 ± 0.07 :math:`M_{\odot}`):

    >>> from jesterTOV.inference.likelihoods import RadioTimingLikelihood
    >>> likelihood = RadioTimingLikelihood("J0740+6620", mean=2.08, std=0.07)
    >>> params = {"masses_EOS": jnp.array([1.0, 1.5, 2.0, 2.2])}  # Example TOV masses
    >>> log_like = likelihood.evaluate(params, data={})

    Create a more precise likelihood for PSR J1614-2230 (narrower uncertainty):

    >>> likelihood_j1614 = RadioTimingLikelihood("J1614-2230", mean=1.94, std=0.06)
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
        self.m_min = m_min  # Minimum mass for integration (solar masses)

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """Evaluate the marginalized log-likelihood for the pulsar mass measurement.

        This method computes the marginal likelihood by:
        1. Extracting the maximum TOV mass from the EOS
        2. Creating a mass grid from m_min to M_TOV
        3. Evaluating the Gaussian likelihood at each grid point
        4. Numerically integrating using logsumexp for stability
        5. Applying the 1/M_TOV normalization factor

        Parameters
        ----------
        params : dict[str, Float | Array]
            Dictionary containing TOV solution outputs from the transform.
            Required keys:

            - "masses_EOS" : Array of neutron star masses (solar masses) at
              different central pressures. The maximum value is taken as M_TOV.

        Returns
        -------
        Float
            Natural logarithm of the marginalized likelihood. Returns a large
            negative penalty (-1e10) for invalid EOSs (M_TOV ≤ m_min), which
            indicates TOV integration failure or unphysical EOS.

        Notes
        -----
        The numerical integration uses nb_masses grid points and evaluates
        the Gaussian likelihood in log-space for numerical stability. The
        logsumexp operation prevents underflow when summing small probabilities.

        Any NaN or infinity values in the result are replaced with -1e10 to
        ensure stable MCMC sampling.
        """
        # Extract maximum TOV mass from the EOS
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        # Check for invalid M_TOV before computing (avoids NaN from log(mtov))
        # Invalid cases: mtov <= m_min (unphysical masses from TOV failures)
        invalid_mtov = mtov <= self.m_min

        # Create mass grid for integration from m_min to M_max
        # Per Eq. (X): P(θ_EOS | d_radio) ∝ (1/M_TOV) ∫₀^M_TOV P(M | d_radio) dM
        # Note: Start at m_min (default 0.1 solar masses) instead of 0 to avoid numerical issues
        m = jnp.linspace(self.m_min, mtov, self.nb_masses)

        # Gaussian log likelihood with proper normalization constant
        # P(M | d_radio) = N(M | mean, std)
        gauss_norm = -0.5 * jnp.log(2 * jnp.pi * self.std**2)
        log_likelihood_array = gauss_norm + (-0.5 * ((m - self.mean) / self.std) ** 2)

        # Numerical integration using logsumexp (for stability)
        # integral ≈ dx * sum(exp(log_likelihood_array))
        # log(integral) = log(dx) + logsumexp(log_likelihood_array)
        dx = (mtov - self.m_min) / self.nb_masses
        log_integral = logsumexp(log_likelihood_array) + jnp.log(dx)

        # Apply 1/M_max normalization factor (uniform prior on true mass in [0, M_TOV])
        # Use jnp.where to avoid computing log(mtov) when mtov is invalid
        log_likelihood = jnp.where(
            invalid_mtov,
            -1e10,  # Large negative penalty for invalid masses
            log_integral - jnp.log(mtov),  # Normal calculation for valid masses
        )

        # Safety net: replace any remaining NaN/inf with large negative value
        # This catches edge cases not covered by the mtov check
        log_likelihood = jnp.nan_to_num(
            log_likelihood, nan=-1e10, posinf=-1e10, neginf=-1e10
        )

        return log_likelihood
