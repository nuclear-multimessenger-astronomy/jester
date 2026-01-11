r"""Likelihood base class for JESTER inference system.

This module contains the LikelihoodBase class, which was originally from Jim (jimgw v0.2.0).
It is copied here to remove the dependency on jimgw.
"""

from abc import ABC, abstractmethod
from typing import Any

from jaxtyping import Float


class LikelihoodBase(ABC):
    """
    Base class for likelihoods.

    This class is designed for likelihoods where data is encapsulated within the
    likelihood object during initialization. This differs from other frameworks
    (e.g., jimgw) that pass data at evaluation time.

    The likelihood encapsulates:
    - The data: observations or measurements (stored internally)
    - The model: theoretical predictions (computed from parameters)

    It evaluates the log-likelihood for a given set of parameters, using the
    internally stored data.
    """

    _model: Any
    _data: Any

    @property
    def model(self) -> Any:
        """
        The model for the likelihood.

        Returns
        -------
        Any
            The model object. Specific type depends on the likelihood implementation.
        """
        return self._model

    @property
    def data(self) -> Any:
        """
        The data for the likelihood.

        Returns
        -------
        Any
            The data object. Specific type depends on the likelihood implementation.
        """
        return self._data

    @abstractmethod
    def evaluate(self, params: dict[str, Float]) -> Float:
        """
        Evaluate the log-likelihood for a given set of parameters.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary of parameter names to values.

        Returns
        -------
        log_likelihood : Float
            The log-likelihood value.
        """
        raise NotImplementedError
