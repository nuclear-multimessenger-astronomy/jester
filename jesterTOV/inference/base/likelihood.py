"""Likelihood base class for JESTER inference system.

This module contains the LikelihoodBase class, which was originally from Jim (jimgw v0.2.0).
It is copied here to remove the dependency on jimgw.
"""

from abc import ABC, abstractmethod
from typing import Any

from jaxtyping import Float


class LikelihoodBase(ABC):
    """
    Base class for likelihoods.

    Note: This class follows the jimgw architecture. It is designed to work
    for a general class of problems where the likelihood depends on data and a model.

    This class handles two main components of a likelihood:
    - The data: observations or measurements
    - The model: theoretical predictions

    It should be able to take the data and model and evaluate the likelihood for
    a given set of parameters.
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
    def evaluate(self, params: dict[str, Float], data: dict[str, Any]) -> Float:
        """
        Evaluate the log-likelihood for a given set of parameters.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary of parameter names to values.
        data : dict
            Dictionary containing the data. In JESTER's case, this is often
            an empty dict {} since data is encapsulated in the likelihood object.

        Returns
        -------
        log_likelihood : Float
            The log-likelihood value.
        """
        raise NotImplementedError
