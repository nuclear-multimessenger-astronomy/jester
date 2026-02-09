"""Base class for BlackJAX samplers with shared transform logic.

This module provides BlackjaxSampler, which handles parameter space transformations
in a way that can be shared across different BlackJAX sampling algorithms (SMC, NS, etc.).
"""

from typing import Any, Callable

import jax
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

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform] | None = None,
        likelihood_transforms: list[NtoMTransform] | None = None,
    ) -> None:
        """Initialize BlackJAX sampler base class."""
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)

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
