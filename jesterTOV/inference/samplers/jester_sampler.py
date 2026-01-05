r"""
Base sampler implementation for jesterTOV.

This module provides a lightweight, modular base class for sampling.
Backend-specific implementations (e.g., flowMC, Jim, NumPyro) should
inherit from JesterSampler and implement the sampler initialization.
"""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jesterTOV.inference.base import LikelihoodBase, Prior, BijectiveTransform, NtoMTransform
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

        # Transform caching for efficiency (avoid redundant TOV solver calls)
        self._cache_transforms = False
        self._transform_cache: dict[bytes, dict[str, Any]] = {}  # Cache keyed by param hash

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

    def posterior_from_dict(self, named_params: dict[str, Float], data: dict[str, Any]) -> Float:
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

        # Cache transforms if enabled
        if self._cache_transforms and len(self.likelihood_transforms) > 0:
            original_params = named_params.copy()

            # Apply likelihood transforms
            for transform in self.likelihood_transforms:
                named_params = transform.forward(named_params)

            # Cache transformed output
            param_key = self._hash_params(original_params)
            self._transform_cache[param_key] = named_params.copy()
        else:
            # Just apply transforms without caching
            for transform in self.likelihood_transforms:
                named_params = transform.forward(named_params)

        return self.likelihood.evaluate(named_params, data) + prior

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

    def sample(self, key: PRNGKeyArray, initial_position: Array = jnp.array([])) -> None:
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

    # Transform caching methods

    def _hash_params(self, params: dict[str, Any]) -> bytes:
        """Create hash key for parameter dict (for caching).

        Parameters
        ----------
        params : dict
            Parameter dictionary

        Returns
        -------
        bytes
            Hash of sorted parameter items
        """
        import hashlib
        import json
        # Convert to JSON string with sorted keys for consistent hashing
        # Convert JAX arrays to lists for JSON serialization
        param_list = []
        for k in sorted(params.keys()):
            v = params[k]
            # Convert JAX/NumPy arrays to Python lists
            if hasattr(v, 'tolist'):
                param_list.append((k, v.tolist()))
            else:
                param_list.append((k, float(v)))
        param_str = json.dumps(param_list)
        return hashlib.md5(param_str.encode()).digest()

    def enable_transform_caching(self) -> None:
        """Enable caching of likelihood transform outputs.

        This caches the output of likelihood_transforms (e.g., TOV solver results)
        to avoid redundant computation when generating final EOS samples.

        Call this BEFORE sampling to enable caching during the sampling process.
        After sampling, use get_cached_transforms() to retrieve cached outputs.
        """
        self._cache_transforms = True
        self._transform_cache = {}
        logger.info("Transform caching enabled - will cache likelihood transform outputs")

    def disable_transform_caching(self) -> None:
        """Disable transform caching."""
        self._cache_transforms = False
        logger.debug("Transform caching disabled")

    def clear_transform_cache(self) -> None:
        """Clear the transform cache."""
        self._transform_cache = {}
        logger.debug(f"Cleared transform cache")

    def get_cached_transforms(self, param_samples: dict[str, Any]) -> dict[str, Any] | None:
        """Retrieve cached transform outputs for given parameter samples.

        Parameters
        ----------
        param_samples : dict
            Dictionary of parameter arrays, shape (n_samples, ...)

        Returns
        -------
        dict or None
            Dictionary of cached transformed outputs if all samples found in cache,
            None if any samples are missing from cache

        Notes
        -----
        This method looks up each parameter sample in the cache and returns the
        corresponding transformed outputs. If any sample is not found in the cache,
        returns None (indicating fallback to recomputation is needed).
        """
        import numpy as np

        if not self._transform_cache:
            logger.debug("Transform cache is empty, fallback to recomputation")
            return None

        # Get number of samples
        n_samples = len(next(iter(param_samples.values())))

        # Try to retrieve cached transforms for each sample
        cached_outputs = None
        n_found = 0

        for i in range(n_samples):
            # Extract i-th sample
            sample_dict = {k: v[i] for k, v in param_samples.items()}

            # Look up in cache
            param_key = self._hash_params(sample_dict)
            if param_key in self._transform_cache:
                n_found += 1
                cached_output = self._transform_cache[param_key]

                # Initialize output dict on first hit
                if cached_outputs is None:
                    cached_outputs = {k: [] for k in cached_output.keys()}

                # Append this sample's output
                for k, v in cached_output.items():
                    cached_outputs[k].append(v)
            else:
                # Cache miss - abort and fallback to recomputation
                logger.warning(
                    f"Cache miss for sample {i}/{n_samples}, "
                    f"found {n_found}/{i+1} so far. Falling back to recomputation."
                )
                return None

        if cached_outputs is not None:
            # Convert lists to arrays
            cached_outputs = {k: np.array(v) for k, v in cached_outputs.items()}
            logger.info(f"Successfully retrieved {n_samples} cached transforms")
            return cached_outputs

        return None
