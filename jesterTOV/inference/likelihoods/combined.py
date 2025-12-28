r"""Combined and utility likelihood classes"""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.inference.base import LikelihoodBase


class CombinedLikelihood(LikelihoodBase):
    """
    Combine multiple likelihoods into a single log-likelihood sum

    Parameters
    ----------
    likelihoods_list : list[LikelihoodBase]
        List of likelihood objects to combine

    Attributes
    ----------
    likelihoods_list : list[LikelihoodBase]
        Stored list of likelihoods
    counter : int
        Evaluation counter (for debugging/monitoring)
    """

    likelihoods_list: list[LikelihoodBase]
    counter: int

    def __init__(self, likelihoods_list: list[LikelihoodBase]) -> None:
        super().__init__()
        self.likelihoods_list = likelihoods_list
        self.counter = 0

    def evaluate(self, params: dict[str, Float | Array], data: dict[str, Any]) -> Float:
        """
        Evaluate combined log-likelihood

        Parameters
        ----------
        params : dict[str, Float | Array]
            Parameter dictionary passed to all likelihoods
        data : dict[str, Any]
            Data dictionary passed to all likelihoods

        Returns
        -------
        Float
            Sum of all log-likelihoods
        """
        all_log_likelihoods: Float[Array, " n_likelihoods"] = jnp.array(
            [likelihood.evaluate(params, data) for likelihood in self.likelihoods_list]
        )
        return jnp.sum(all_log_likelihoods)


class ZeroLikelihood(LikelihoodBase):
    """
    Placeholder likelihood that always returns 0 (for testing/debugging)

    Attributes
    ----------
    counter : int
        Evaluation counter (for debugging/monitoring)
    """

    counter: int

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    def evaluate(self, params: dict[str, Float | Array], data: dict[str, Any]) -> Float:
        """
        Evaluate zero log-likelihood

        Parameters
        ----------
        params : dict[str, Float | Array]
            Parameter dictionary (ignored)
        data : dict[str, Any]
            Data dictionary (ignored)

        Returns
        -------
        Float
            Always returns 0.0
        """
        return 0.0
