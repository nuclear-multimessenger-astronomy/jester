r"""
Base sampler implementation for jesterTOV.

This module provides a lightweight, modular base class for sampling.
Backend-specific implementations (e.g., BlackJAX SMC, nested sampling) should
inherit from JesterSampler and implement the sampler initialization.
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
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

EOS_OUTPUT_KEYS = (
    "masses_EOS",
    "radii_EOS",
    "Lambdas_EOS",
    "n",
    "p",
    "e",
    "cs2",
    "n_TOV",
)
"""Outputs of the likelihood transforms that are stored with the posterior."""


class BatchEvaluation(NamedTuple):
    """Result of :meth:`JesterSampler.evaluate_in_batches`."""

    log_prob: Array
    log_likelihood: Array
    eos: dict[str, Array]


@dataclass
class SamplerOutput:
    """
    Standardized output from JESTER samplers.

    This dataclass provides a uniform interface for accessing samples,
    log probabilities, and sampler-specific metadata across different
    sampling backends (SMC, NS-AW).

    Attributes
    ----------
    samples : dict[str, Array]
        Dictionary of parameter samples. Keys are parameter names,
        values are JAX arrays of shape (n_samples,) or (n_samples, n_dim).
        Only contains actual parameters, not metadata fields.
    log_prob : Array
        Log probability for each sample. Interpretation depends on sampler:
        - SMC: log posterior probability
        - NS-AW: log likelihood (nested sampling uses likelihood)
        Shape: (n_samples,)
    metadata : dict[str, Any]
        Sampler-specific metadata. Common fields:
        - SMC: {"weights": Array, "ess": float}
        - NS-AW: {"weights": Array, "logL": Array, "logL_birth": Array}

    Notes
    -----
    The log_prob field has different semantics for NS-AW (log likelihood)
    versus SMC (log posterior). Consumers should check the sampler
    type when interpreting this field.
    """

    samples: dict[str, Array]
    log_prob: Array
    metadata: dict[str, Any] = field(default_factory=dict)


class JesterSampler:
    """
    Lightweight base class for JESTER samplers.

    This class provides a modular interface for Bayesian inference with different
    sampling backends (BlackJAX SMC, nested sampling, etc.). It handles:
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

        # One compiled function for every batched evaluation outside the sampling loop, so the
        # startup check, the final log_prob and the EOS outputs share a single compilation.
        self._batch_eval = jax.jit(jax.vmap(self._prior_space_posterior_and_eos))

    def _prior_space_posterior_and_eos(
        self, named_params: dict[str, Float]
    ) -> tuple[Float, Float, dict[str, Array]]:
        """Log posterior and log likelihood of a point in prior space, and the EOS outputs of the likelihood transforms."""
        log_prior = self.prior.log_prob(named_params)
        for transform in self.likelihood_transforms:
            named_params = transform.forward(named_params)
        eos = {key: named_params[key] for key in EOS_OUTPUT_KEYS if key in named_params}
        log_likelihood = self.likelihood.evaluate(named_params)
        return log_likelihood + log_prior, log_likelihood, eos

    def evaluate_in_batches(
        self, samples: dict[str, Array], batch_size: int
    ) -> BatchEvaluation:
        """Evaluate the log posterior, log likelihood and EOS outputs of samples in prior space.

        Every batch has exactly ``batch_size`` samples, the last one padded with repeats of the first
        sample, so the function compiles once for any number of samples.

        Parameters
        ----------
        samples : dict[str, Array]
            Samples in prior space (no sample transforms), one array of length N per parameter.
        batch_size : int
            Number of samples per compiled call.

        Returns
        -------
        BatchEvaluation
            Log posterior and log likelihood, each of shape (N,), and the EOS outputs listed in
            ``EOS_OUTPUT_KEYS``.
        """
        n = len(next(iter(samples.values())))
        n_batches = -(-n // batch_size)
        pad = n_batches * batch_size - n
        padded = {
            k: jnp.concatenate([jnp.asarray(v), jnp.broadcast_to(jnp.asarray(v)[:1], (pad,))])
            for k, v in samples.items()
        }
        log_probs, log_likelihoods, eos_batches = [], [], []
        for b in range(n_batches):
            batch = {k: v[b * batch_size : (b + 1) * batch_size] for k, v in padded.items()}
            log_prob, log_likelihood, eos = self._batch_eval(batch)
            log_probs.append(log_prob)
            log_likelihoods.append(log_likelihood)
            eos_batches.append(eos)
        eos = {k: jnp.concatenate([e[k] for e in eos_batches])[:n] for k in eos_batches[0]}
        return BatchEvaluation(
            jnp.concatenate(log_probs)[:n], jnp.concatenate(log_likelihoods)[:n], eos
        )

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

    def sample(self, key: PRNGKeyArray) -> None:
        """
        Run sampling.

        This method must be implemented by backend-specific subclasses.
        Initial positions are sampled from the prior internally by each
        sampler implementation.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key

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

        This method can be implemented by backend-specific subclasses.

        Parameters
        ----------
        transform : bool, optional
            Whether to apply inverse sample transforms to results (default: True)
        """
        # Optionally overridden by subclasses, but if not, then just do nothing
        pass

    def get_samples(self) -> dict:
        """
        Get production samples from the sampler.

        This method must be implemented by backend-specific subclasses.
        Always returns production/final samples (not training samples).

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

    def get_log_prob(self) -> Array:
        """
        Get log probabilities for production samples.

        This method must be implemented by backend-specific subclasses.
        Always returns production/final samples (not training samples).

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
        - SMC: Returns log posterior (uniform weights at λ=1)
        - Nested Sampling: Returns log likelihood (use weights separately)
        """
        raise NotImplementedError(
            "get_log_prob() must be implemented by backend-specific subclass"
        )

    def get_n_samples(self) -> int:
        """
        Get number of production/final samples.

        This method must be implemented by backend-specific subclasses.
        Always returns the number of production/final samples (not training samples).

        Returns
        -------
        int
            Number of production/final samples

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError(
            "get_n_samples() must be implemented by backend-specific subclass"
        )

    def get_sampler_output(self) -> SamplerOutput:
        """
        Get standardized sampler output with samples, log probabilities, and metadata.

        This is the preferred method for accessing sampler results. It returns
        a SamplerOutput dataclass containing all samples, log probabilities,
        and sampler-specific metadata in a standardized format.

        Always returns production/final samples (not training samples).

        Returns
        -------
        SamplerOutput
            Standardized output containing:
            - samples: Dict of parameter arrays (no metadata fields)
            - log_prob: Log probability array (posterior or likelihood)
            - metadata: Sampler-specific fields (weights, ESS, etc.)

        Raises
        ------
        NotImplementedError
            If called on base class (must be implemented by backend-specific subclass)
        RuntimeError
            If sampling has not been run yet (no results available)

        Notes
        -----
        This method should be used instead of the older get_samples() and
        get_log_prob() methods, which are now considered legacy.

        For NS-AW, log_prob contains log likelihood (not log posterior),
        as nested sampling works in likelihood space.
        """
        raise NotImplementedError(
            "get_sampler_output() must be implemented by backend-specific subclass"
        )
