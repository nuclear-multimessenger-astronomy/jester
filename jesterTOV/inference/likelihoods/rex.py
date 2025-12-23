"""PREX/CREX lead radius experiment likelihood implementations"""

import jax.numpy as jnp
from jaxtyping import Float
from jesterTOV.inference.base import LikelihoodBase


class REXLikelihood(LikelihoodBase):
    """
    PREX/CREX likelihood for nuclear symmetry energy constraints

    Uses posterior KDE from lead radius measurements to constrain
    E_sym and L_sym parameters.

    Parameters
    ----------
    experiment_name : str
        Experiment name ("PREX" or "CREX")
    posterior : gaussian_kde
        KDE of experiment posterior in (E_sym, L_sym) space
    """

    def __init__(
        self,
        experiment_name: str,
        posterior,
    ):
        super().__init__()
        assert experiment_name in [
            "PREX",
            "CREX",
        ], "Only PREX and CREX are supported as experiment name arguments."
        self.experiment_name = experiment_name
        self.counter = 0
        self.posterior = posterior

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        log_likelihood_array = self.posterior.logpdf(
            jnp.array([params["E_sym"], params["L_sym"]])
        )
        log_likelihood = log_likelihood_array.at[0].get()
        return log_likelihood
