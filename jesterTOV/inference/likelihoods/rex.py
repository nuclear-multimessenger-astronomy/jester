"""PREX/CREX lead radius experiment likelihood implementations"""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float

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
    posterior : Any
        KDE of experiment posterior in (E_sym, L_sym) space
        (typically gaussian_kde from jax.scipy.stats or scipy.stats)

    Attributes
    ----------
    experiment_name : str
        Experiment name
    counter : int
        Evaluation counter (for debugging/monitoring)
    posterior : Any
        KDE posterior object
    """

    experiment_name: str
    counter: int
    posterior: Any  # gaussian_kde type

    def __init__(
        self,
        experiment_name: str,
        posterior: Any,
    ) -> None:
        super().__init__()
        assert experiment_name in [
            "PREX",
            "CREX",
        ], "Only PREX and CREX are supported as experiment name arguments."
        self.experiment_name = experiment_name
        self.counter = 0
        self.posterior = posterior

    def evaluate(self, params: dict[str, Float | Array], data: dict[str, Any]) -> Float:
        log_likelihood_array = self.posterior.logpdf(
            jnp.array([params["E_sym"], params["L_sym"]])
        )
        log_likelihood = log_likelihood_array.at[0].get()
        return log_likelihood
