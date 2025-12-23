"""NICER X-ray timing likelihood implementations"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float
from jesterTOV.inference.base import LikelihoodBase


class NICERLikelihood(LikelihoodBase):
    """
    NICER likelihood marginalizing over mass using M-R interpolation

    Parameters
    ----------
    psr_name : str
        Pulsar name (e.g., "J0030", "J0740")
    amsterdam_posterior : gaussian_kde
        KDE of Amsterdam group posterior
    maryland_posterior : gaussian_kde
        KDE of Maryland group posterior
    m_min : float, optional
        Minimum mass for integration grid
    m_max : float, optional
        Maximum mass for integration grid
    nb_masses : int, optional
        Number of mass points for integration
    use_NF : bool, optional
        Whether to use normalizing flow posteriors (not yet implemented)
    """

    def __init__(
        self,
        psr_name: str,
        amsterdam_posterior,
        maryland_posterior,
        m_min: float = 1.0,
        m_max: float = 2.5,
        nb_masses: int = 100,
        use_NF: bool = False,
    ):
        super().__init__()
        self.psr_name = psr_name
        self.counter = 0
        self.nb_masses = nb_masses
        self.m_min = m_min
        self.m_max = m_max
        self.masses = jnp.linspace(m_min, m_max, nb_masses)
        self.dm = self.masses[1] - self.masses[0]
        self.use_NF = use_NF

        self.amsterdam_posterior = amsterdam_posterior
        self.maryland_posterior = maryland_posterior

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        masses_EOS, radii_EOS = params["masses_EOS"], params["radii_EOS"]

        m = jnp.linspace(1.0, jnp.max(masses_EOS), self.nb_masses)
        r = jnp.interp(m, masses_EOS, radii_EOS)

        mr_grid = jnp.vstack([m, r])
        if self.use_NF:
            logy_maryland = self.maryland_posterior.log_prob(mr_grid.T)
            logy_amsterdam = self.amsterdam_posterior.log_prob(mr_grid.T)
        else:
            logy_maryland = self.maryland_posterior.logpdf(mr_grid)
            logy_amsterdam = self.amsterdam_posterior.logpdf(mr_grid)

        logL_maryland = logsumexp(logy_maryland) - jnp.log(len(logy_maryland))
        logL_amsterdam = logsumexp(logy_amsterdam) - jnp.log(len(logy_amsterdam))

        L_maryland = jnp.exp(logL_maryland)
        L_amsterdam = jnp.exp(logL_amsterdam)
        L = 1 / 2 * (L_maryland + L_amsterdam)
        log_likelihood = jnp.log(L)

        return log_likelihood


class NICERLikelihood_with_masses(LikelihoodBase):
    """
    NICER likelihood with mass as a sampled parameter (no marginalization)

    Parameters
    ----------
    psr_name : str
        Pulsar name (e.g., "J0030", "J0740")
    amsterdam_posterior : gaussian_kde
        KDE of Amsterdam group posterior
    maryland_posterior : gaussian_kde
        KDE of Maryland group posterior
    """

    def __init__(
        self,
        psr_name: str,
        amsterdam_posterior,
        maryland_posterior,
    ):
        super().__init__()
        self.psr_name = psr_name

        self.amsterdam_posterior = amsterdam_posterior
        self.maryland_posterior = maryland_posterior

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        masses_EOS, radii_EOS = params["masses_EOS"], params["radii_EOS"]
        mass = params[f"mass_{self.psr_name}"]
        radius = jnp.interp(mass, masses_EOS, radii_EOS, left=0, right=0)

        mr_grid = jnp.vstack([mass, radius])
        logL_maryland = self.maryland_posterior.logpdf(mr_grid)
        logL_amsterdam = self.amsterdam_posterior.logpdf(mr_grid)

        logL_array = jnp.array([logL_maryland, logL_amsterdam])
        log_likelihood = logsumexp(logL_array) - jnp.log(2)

        return log_likelihood
