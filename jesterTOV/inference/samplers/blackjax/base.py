"""Base class for BlackJAX samplers with shared transform logic.

This module provides BlackjaxSampler, which handles parameter space transformations
in a way that can be shared across different BlackJAX sampling algorithms (SMC, NS, etc.).
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import flatten_util
from jax.tree_util import tree_map
from jaxtyping import Array

from jesterTOV.inference.base import (
    LikelihoodBase,
    Prior,
    BijectiveTransform,
    NtoMTransform,
)
from jesterTOV.inference.samplers.jester_sampler import JesterSampler
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class BlackjaxSampler(JesterSampler):
    """Base class for BlackJAX samplers with shared transform logic.

    This class provides common functionality for all BlackJAX-based samplers:
    - Creating dict-based log prior functions (with inverse transforms + Jacobian)
    - Creating dict-based log likelihood functions (with inverse + likelihood transforms)

    Different BlackJAX algorithms have different API requirements:
    - SMC requires flat array functions → subclass wraps these dict functions
    - NS-AW requires dict functions → subclass uses these directly

    This design maximizes code reuse while respecting each algorithm's API.

    Parameters
    ----------
    likelihood : LikelihoodBase
        Likelihood object with evaluate(params, data) method
    prior : Prior
        Prior object
    sample_transforms : list[BijectiveTransform] | None, optional
        Bijective transforms applied during sampling (with Jacobians)
    likelihood_transforms : list[NtoMTransform] | None, optional
        N-to-M transforms applied before likelihood evaluation

    Notes
    -----
    Subclasses must implement:
    - sample(): Run the sampling algorithm
    - get_samples(): Return samples in dict format
    - get_log_prob(): Return log probabilities
    - get_n_samples(): Return number of samples
    - get_sampler_output(): Return standardized SamplerOutput
    """

    _unflatten_fn: Any  # Callable[[Array], dict]
    _flatten_fn: Any  # Callable[[dict], Array]

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform] | None = None,
        likelihood_transforms: list[NtoMTransform] | None = None,
    ) -> None:
        """Initialize BlackJAX sampler base class."""
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)
        self._unflatten_fn = None
        self._flatten_fn = None

    def _create_flatten_unflatten_utilities(
        self, initial_position_dict: dict[str, Array]
    ) -> None:
        """Create flatten/unflatten functions for a flat-array API (e.g. SMC).

        Parameters
        ----------
        initial_position_dict : dict[str, Array]
            Dictionary of initial particle positions (each value is array of shape (n_particles,))
        """
        # Extract single sample to determine structure
        single_sample_dict = tree_map(lambda x: x[0], initial_position_dict)

        # Create unflatten function using ravel_pytree (alphabetical ordering)
        _, self._unflatten_fn = flatten_util.ravel_pytree(single_sample_dict)

        # Create flatten function
        self._flatten_fn = lambda x: flatten_util.ravel_pytree(x)[0]

    def _wrap_dict_fn_for_flat_arrays(
        self, dict_fn: Callable[[dict], float]
    ) -> Callable[[Array], float]:
        """Wrap a dict-based function to work with flat arrays.

        This is the bridge between the dict functions above and a flat array
        API (e.g. SMC).

        Parameters
        ----------
        dict_fn : Callable[[dict], float]
            Function that takes parameter dict and returns float

        Returns
        -------
        Callable[[Array], float]
            Function that takes flat array and returns float

        Examples
        --------
        >>> logprior_dict = self._create_logprior_fn_from_dict()
        >>> logprior_flat = self._wrap_dict_fn_for_flat_arrays(logprior_dict)
        >>> # Now logprior_flat can be passed to BlackJAX SMC
        """

        def flat_fn(x_flat: Array) -> float:
            """Convert flat array to dict, evaluate function."""
            x_flat = jnp.atleast_1d(x_flat)
            x_dict = self._unflatten_fn(x_flat)
            return dict_fn(x_dict)

        return flat_fn

    def _create_logprior_fn_from_dict(self) -> Callable[[dict[str, Any]], float]:
        """Create log prior function that works with parameter dicts.

        This function:
        1. Applies inverse sample transforms (sampling space → prior space)
        2. Adds Jacobian corrections from transforms
        3. Evaluates prior log probability

        Both SMC and NS can use this - SMC will wrap it for flat arrays.

        Returns
        -------
        Callable[[dict[str, Any]], float]
            JIT-compiled log prior function for single sample dict

        Examples
        --------
        >>> logprior_fn = self._create_logprior_fn_from_dict()
        >>> params = {"K_sat": 0.5, "L_sym": 0.3}  # In sampling space (e.g., unit cube)
        >>> log_p = logprior_fn(params)  # Returns log prior in prior space + Jacobian
        """

        def logprior_fn(params_dict: dict[str, Any]) -> float:
            """Evaluate log prior with transforms and Jacobian corrections."""
            transform_jacobian = 0.0
            named_params = params_dict.copy()

            # Apply inverse sample transforms (sampling space → prior space)
            for transform in reversed(self.sample_transforms):
                named_params, jacobian = transform.inverse(named_params)
                transform_jacobian += jacobian

            # Evaluate prior + Jacobian
            return self.prior.log_prob(named_params) + transform_jacobian

        # JIT compile for performance
        return jax.jit(logprior_fn)

    def _create_loglikelihood_fn_from_dict(self) -> Callable[[dict[str, Any]], float]:
        """Create log likelihood function that works with parameter dicts.

        This function:
        1. Applies inverse sample transforms (sampling space → prior space)
        2. Applies forward likelihood transforms (prior → likelihood params)
        3. Evaluates likelihood

        Both SMC and NS can use this - SMC will wrap it for flat arrays.

        Returns
        -------
        Callable[[dict[str, Any]], float]
            JIT-compiled log likelihood function for single sample dict

        Examples
        --------
        >>> loglikelihood_fn = self._create_loglikelihood_fn_from_dict()
        >>> params = {"K_sat": 0.5, "L_sym": 0.3}  # In sampling space (e.g., unit cube)
        >>> log_l = loglikelihood_fn(params)  # Returns log likelihood
        """

        def loglikelihood_fn(params_dict: dict[str, Any]) -> float:
            """Evaluate log likelihood with transforms."""
            named_params = params_dict.copy()

            # Apply inverse sample transforms (sampling space → prior space)
            for transform in reversed(self.sample_transforms):
                named_params, _ = transform.inverse(named_params)

            # Apply likelihood transforms (prior → likelihood params)
            for transform in self.likelihood_transforms:
                named_params = transform.forward(named_params)

            # Evaluate likelihood
            return self.likelihood.evaluate(named_params)

        # JIT compile for performance
        return jax.jit(loglikelihood_fn)
