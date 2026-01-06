r"""
Base sampler implementation for jesterTOV.

This module provides a lightweight, modular base class for sampling.
Backend-specific implementations (e.g., flowMC, Jim, NumPyro) should
inherit from JesterSampler and implement the sampler initialization.
"""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jesterTOV.inference.base import (
    LikelihoodBase,
    Prior,
    BijectiveTransform,
    NtoMTransform,
)
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class JesterSampler:
    """
    Lightweight base class for JESTER samplers.

    This class provides a modular interface for Bayesian inference with different
    sampling backends (flowMC, Jim, NumPyro, etc.). It handles:
    - Parameter transforms (sample and likelihood transforms)
    - Posterior evaluation with Jacobian corrections
    - Parameter name propagation
    - Generic sampling interface

    Backend-specific implementations should inherit from this class and:
    1. Call super().__init__() to set up common attributes
    2. Create self.sampler (the backend sampler instance)
    3. Optionally override methods for backend-specific behavior

    Critical features:
    - Uses jnp.inf instead of jnp.nan for initialization
    - Preserves parameter ordering when converting dict to array

    Parameters
    ----------
    likelihood : LikelihoodBase
        Likelihood object with evaluate(params, data) method
    prior : Prior
        Prior object with sample() and log_prob() methods
    sample_transforms : list[BijectiveTransform] | None, optional
        Bijective transforms applied during sampling (with Jacobians)
    likelihood_transforms : list[NtoMTransform] | None, optional
        N-to-M transforms applied before likelihood evaluation

    Attributes
    ----------
    likelihood : LikelihoodBase
        Likelihood object
    prior : Prior
        Prior object
    sample_transforms : list[BijectiveTransform]
        Transforms applied during sampling
    likelihood_transforms : list[NtoMTransform]
        Transforms applied before likelihood evaluation
    parameter_names : list[str]
        Names of parameters (propagated through sample transforms)
    sampler : Any | None
        Backend sampler instance (created by subclasses)

    Notes
    -----
    Subclasses must create self.sampler in their __init__ method.
    The sampler should have a .sample() method and support get_sampler_state().
    """

    likelihood: LikelihoodBase
    prior: Prior
    sample_transforms: list[BijectiveTransform]
    likelihood_transforms: list[NtoMTransform]
    parameter_names: list[str]
    sampler: Any | None

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform] | None = None,
        likelihood_transforms: list[NtoMTransform] | None = None,
    ) -> None:
        # Handle None defaults
        if sample_transforms is None:
            sample_transforms = []
        if likelihood_transforms is None:
            likelihood_transforms = []

        self.likelihood = likelihood
        self.prior = prior

        self.sample_transforms = sample_transforms
        self.likelihood_transforms = likelihood_transforms
        self.parameter_names = prior.parameter_names

        if len(sample_transforms) == 0:
            logger.debug(
                "No sample transforms provided. Using prior parameters as sampling parameters"
            )
        else:
            logger.debug("Using sample transforms")
            for transform in sample_transforms:
                self.parameter_names = transform.propagate_name(self.parameter_names)

        if len(likelihood_transforms) == 0:
            logger.debug(
                "No likelihood transforms provided. Using prior parameters as likelihood parameters"
            )

        # Backend sampler instance - must be created by subclasses
        self.sampler = None

    def add_name(self, x: Float[Array, " n_dim"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary.

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dim,).

        Returns
        -------
        dict[str, Float]
            Dictionary mapping parameter names to values
        """
        return dict(zip(self.parameter_names, x))

    def posterior_from_dict(
        self, named_params: dict[str, Float], data: dict[str, Any]
    ) -> Float:
        """
        Evaluate posterior log probability from parameter dict.

        Parameters
        ----------
        named_params : dict
            Parameter dictionary
        data : dict
            Data dictionary (unused in JESTER, pass {})

        Returns
        -------
        Float
            Log posterior probability
        """
        transform_jacobian = 0.0
        for transform in reversed(self.sample_transforms):
            named_params, jacobian = transform.inverse(named_params)
            transform_jacobian += jacobian

        prior = self.prior.log_prob(named_params) + transform_jacobian

        # Apply likelihood transforms
        for transform in self.likelihood_transforms:
            named_params = transform.forward(named_params)

        return self.likelihood.evaluate(named_params) + prior

    def posterior(self, params: Float[Array, " n_dim"], data: dict[str, Any]) -> Float:
        """
        Evaluate posterior log probability from flat array.

        Parameters
        ----------
        params : Array
            Parameter array in sampling space
        data : dict
            Data dictionary (unused in JESTER, pass {})

        Returns
        -------
        Float
            Log posterior probability
        """
        named_params = self.add_name(params)
        return self.posterior_from_dict(named_params, data)

    def sample(
        self, key: PRNGKeyArray, initial_position: Array = jnp.array([])
    ) -> None:
        """
        Run MCMC sampling.

        This method must be implemented by backend-specific subclasses.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key
        initial_position : Array, optional
            Initial positions for chains. If not provided, implementation
            should sample from prior.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError(
            "sample() must be implemented by backend-specific subclass"
        )

    def print_summary(self, transform: bool = True) -> None:
        """
        Print summary of sampling run.

        This method must be implemented by backend-specific subclasses.

        Parameters
        ----------
        transform : bool, optional
            Whether to apply inverse sample transforms to results (default: True)

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError(
            "print_summary() must be implemented by backend-specific subclass"
        )

    def get_samples(self, training: bool = False) -> dict:
        """
        Get samples from the sampler.

        This method must be implemented by backend-specific subclasses.

        Parameters
        ----------
        training : bool, optional
            Whether to get training or production samples (default: False)
            Note: Only FlowMC has train/production split. Other samplers ignore this.

        Returns
        -------
        dict
            Dictionary of samples with parameter names as keys

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError(
            "get_samples() must be implemented by backend-specific subclass"
        )

    def get_log_prob(self, training: bool = False) -> Array:
        """
        Get log probabilities for samples.

        This method must be implemented by backend-specific subclasses.

        Parameters
        ----------
        training : bool, optional
            Whether to get training or production log probs (default: False)
            Note: Only FlowMC has train/production split. Other samplers ignore this.

        Returns
        -------
        Array
            Log probability values (1D array)

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses

        Notes
        -----
        - FlowMC: Returns log posterior from sampler state
        - Nested Sampling: Returns log likelihood (use weights separately)
        - SMC: Returns log posterior (uniform weights at λ=1)
        """
        raise NotImplementedError(
            "get_log_prob() must be implemented by backend-specific subclass"
        )

    def get_n_samples(self, training: bool = False) -> int:
        """
        Get number of samples.

        This method must be implemented by backend-specific subclasses.

        Parameters
        ----------
        training : bool, optional
            Whether to count training or production samples (default: False)
            Note: Only FlowMC has train/production split. Other samplers ignore this.

        Returns
        -------
        int
            Number of samples

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError(
            "get_n_samples() must be implemented by backend-specific subclass"
        )

    # TODO: Future optimization - implement transform caching
    # Current issue: JAX tracing limitations prevent caching inside compiled functions
    # Potential solution: Cache transforms outside JAX trace (sampler-specific implementation)
    # - FlowMC: Cache during production phase
    # - BlackJAX SMC: Cache final temperature samples
    # - BlackJAX NS-AW: Cache all samples
    # Would eliminate redundant TOV solver calls when generating EOS samples
