"""Radio pulsar timing likelihood implementations"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float
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
        Number of mass points for numerical integration. Default: 100

    Notes
    -----
    The integration is performed using a discrete sum with logsumexp for
    numerical stability. The mass grid spans from 1.0 M_☉ to M_max.

    Examples
    --------
    >>> # PSR J1614-2230: 1.94 ± 0.06 M_☉
    >>> likelihood = RadioTimingLikelihood("J1614", 1.94, 0.06)
    >>> log_prob = likelihood.evaluate(params, {})
    """

    def __init__(
        self,
        psr_name: str,
        mean: float,
        std: float,
        nb_masses: int = 100,
    ):
        super().__init__()
        self.psr_name = psr_name
        self.mean = mean
        self.std = std
        self.nb_masses = nb_masses

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """
        Evaluate the log likelihood.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary containing 'masses_EOS' key with sampled mass points
        data : dict
            Unused (data is encapsulated in the likelihood object)

        Returns
        -------
        Float
            Log likelihood value
        """
        # Extract maximum TOV mass from the EOS
        masses_EOS = params["masses_EOS"]
        mtov = jnp.max(masses_EOS)

        # FIXME: This will fail if mtov < 1.0 M_☉ ! Watch out!
        # Create mass grid for integration from 1 M_☉ to M_max
        m = jnp.linspace(1.0, mtov, self.nb_masses)

        # Gaussian log likelihood for each mass point
        log_likelihood_array = -0.5 * ((m - self.mean) / self.std) ** 2

        # Numerical integration using logsumexp (for stability)
        # log(sum(exp(x))) - log(N) = log(mean(exp(x)))
        log_likelihood = logsumexp(log_likelihood_array) - jnp.log(self.nb_masses)

        # Apply 1/M_max normalization factor (uniform prior on true mass)
        log_likelihood -= jnp.log(mtov)

        return log_likelihood
