r"""Unnormalized log-target builders for neural variational inference.

Two flavors of target, matching the two training modes of
:func:`jesterTOV.inference.variational.trainer.train_vi`:

- ``make_prior_likelihood_target``: the ordinary target
  :math:`\log P(\lambda) = \log \mathrm{prior}(\lambda) + \log \mathrm{likelihood}(\lambda)`.
- ``make_flow_prior_target``: the sequential-update target, where a previously
  trained flow :math:`Q_{m-1}` stands in for the prior,
  :math:`\log P_m(\lambda) = \log Q_{m-1}(\lambda) + \log \mathrm{likelihood}_m(\lambda)`.

Both operate on a *batch* of samples, shape ``(batch_size, dim)``, and reuse jester's
existing ``Prior.log_prob`` / ``NtoMTransform.forward`` / ``LikelihoodBase.evaluate``
unmodified, following the same single-sample calling convention as
``JesterSampler.posterior_from_dict`` (``jesterTOV/inference/samplers/jester_sampler.py``),
vmapped over the batch dimension.
"""

from typing import Any, Callable, Sequence

import jax
from jax import Array
from jaxtyping import Float

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.base.prior import Prior
from jesterTOV.inference.base.transform import NtoMTransform

LogTargetFn = Callable[[Array], Float[Array, " batch"]]


def _vmapped_log_likelihood(
    likelihood: LikelihoodBase,
    likelihood_transforms: Sequence[NtoMTransform],
    parameter_names: Sequence[str],
) -> Callable[[dict[str, Array]], Float[Array, " batch"]]:
    """vmap a single-sample ``transform -> likelihood.evaluate`` pipeline over a batch."""

    def log_likelihood_single(named_params: dict[str, Float]) -> Float:
        transformed = named_params
        for transform in likelihood_transforms:
            transformed = transform.forward(transformed)
        return likelihood.evaluate(transformed)

    return jax.vmap(log_likelihood_single)


def make_prior_likelihood_target(
    prior: Prior,
    likelihood: LikelihoodBase,
    likelihood_transforms: Sequence[NtoMTransform],
    parameter_names: Sequence[str],
) -> LogTargetFn:
    """Build ``log_target(samples) = prior.log_prob(samples) + likelihood(samples)``."""
    log_likelihood_fn = _vmapped_log_likelihood(
        likelihood, likelihood_transforms, parameter_names
    )
    log_prior_fn = jax.vmap(prior.log_prob)

    def log_target(samples: Array) -> Float[Array, " batch"]:
        named_params = dict(zip(parameter_names, samples.T))
        return log_prior_fn(named_params) + log_likelihood_fn(named_params)

    return log_target


def make_flow_prior_target(
    prev_flow: Any,
    likelihood: LikelihoodBase,
    likelihood_transforms: Sequence[NtoMTransform],
    parameter_names: Sequence[str],
) -> LogTargetFn:
    """Build the sequential-update target ``log_target(samples) = prev_flow.log_prob(samples)
    + likelihood(samples)``, where ``prev_flow`` (jester's ``Flow`` wrapper, or a raw flowjax
    ``AbstractDistribution``) plays the role of the new prior. Both accept a batched
    ``(batch_size, dim)`` array directly for ``.log_prob``, so no dispatch is needed there.
    """
    log_likelihood_fn = _vmapped_log_likelihood(
        likelihood, likelihood_transforms, parameter_names
    )

    def log_target(samples: Array) -> Float[Array, " batch"]:
        named_params = dict(zip(parameter_names, samples.T))
        return prev_flow.log_prob(samples) + log_likelihood_fn(named_params)

    return log_target
