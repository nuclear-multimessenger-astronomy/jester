"""Radio pulsar timing likelihood implementations"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float
from jesterTOV.inference.base import LikelihoodBase


class RadioTimingLikelihood(LikelihoodBase):
    """
    Radio pulsar timing likelihood for maximum mass constraints

    Constrains the maximum TOV mass based on observed pulsar masses.

    Parameters
    ----------
    psr_name : str
        Pulsar name
    mean : float
        Mean of observed mass (solar masses)
    std : float
        Standard deviation of observed mass (solar masses)
    nb_masses : int, optional
        Number of mass points for integration
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
        self.nb_masses = nb_masses
        self.mean = mean
        self.std = std

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # Log likelihood is a Gaussian with given mean and std, evaluate it on the masses:
        masses_EOS = params["masses_EOS"]
        mtov = jnp.max(masses_EOS)
        m = jnp.linspace(1.0, mtov, self.nb_masses)

        log_likelihood_array = -0.5 * (m - self.mean) ** 2 / self.std**2
        # Do integration with discrete sum
        log_likelihood = logsumexp(log_likelihood_array) - jnp.log(
            len(log_likelihood_array)
        )
        log_likelihood -= mtov

        return log_likelihood
