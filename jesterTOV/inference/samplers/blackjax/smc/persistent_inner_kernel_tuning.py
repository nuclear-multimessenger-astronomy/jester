"""Inner-kernel tuning wrapper that persists the previous parameter override.

BlackJAX's own ``blackjax.smc.inner_kernel_tuning`` calls
``mcmc_parameter_update_fn(key, state, info)`` where ``state`` is the *post-step* SMC state
(particles/weights only) -- it never hands back the parameter values that were actually used to
take that step. That is enough for parameters that are fully recomputed from scratch at every
annealing step (e.g. a proposal covariance re-estimated from the current particle spread), but it
makes genuine recursive adaptation (e.g. Robbins-Monro step-size tuning, dual averaging) impossible
to implement correctly: any attempt to track a running value via a mutated Python closure is a
no-op under ``jax.lax.while_loop`` tracing, since the closure is only ever mutated once, at trace
time, not on each actual loop iteration.

This module is a drop-in, minimal-diff replacement for
``blackjax.smc.inner_kernel_tuning.build_kernel`` / ``as_top_level_api`` that passes the previous
step's ``parameter_override`` into ``mcmc_parameter_update_fn`` as an extra argument, so kernels can
implement real persistent adaptation. Everything else (state type, resampling, tempering) is
unchanged -- we reuse BlackJAX's ``StateWithParameterOverride`` and ``init`` directly.
"""

from typing import Callable, Dict, Tuple

import jax

from blackjax.base import SamplingAlgorithm
from blackjax.smc.base import SMCInfo, SMCState
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride, init
from blackjax.types import ArrayTree, PRNGKey

__all__ = ["persistent_inner_kernel_tuning"]


def _build_kernel(
    smc_algorithm,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    mcmc_parameter_update_fn: Callable[
        [PRNGKey, Dict[str, ArrayTree], SMCState, SMCInfo], Dict[str, ArrayTree]
    ],
    num_mcmc_steps: int = 10,
    **extra_parameters,
) -> Callable:
    """Build the SMC step, forwarding the previous parameter override to the update fn.

    Same contract as ``blackjax.smc.inner_kernel_tuning.build_kernel``, except
    ``mcmc_parameter_update_fn`` is called as
    ``mcmc_parameter_update_fn(key, previous_parameter_override, new_state, info)``.
    """

    def kernel(
        rng_key: PRNGKey, state: StateWithParameterOverride, **extra_step_parameters
    ) -> Tuple[StateWithParameterOverride, SMCInfo]:
        step_fn = smc_algorithm(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            mcmc_parameters=state.parameter_override,
            resampling_fn=resampling_fn,
            num_mcmc_steps=num_mcmc_steps,
            **extra_parameters,
        ).step
        parameter_update_key, step_key = jax.random.split(rng_key, 2)
        new_state, info = step_fn(step_key, state.sampler_state, **extra_step_parameters)
        new_parameter_override = mcmc_parameter_update_fn(
            parameter_update_key, state.parameter_override, new_state, info
        )
        return StateWithParameterOverride(new_state, new_parameter_override), info

    return kernel


def persistent_inner_kernel_tuning(
    smc_algorithm,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    mcmc_parameter_update_fn: Callable[
        [PRNGKey, Dict[str, ArrayTree], SMCState, SMCInfo], Dict[str, ArrayTree]
    ],
    initial_parameter_value: ArrayTree,
    num_mcmc_steps: int = 10,
    **extra_parameters,
) -> SamplingAlgorithm:
    """Top-level API mirroring ``blackjax.inner_kernel_tuning`` with parameter persistence.

    Parameters
    ----------
    mcmc_parameter_update_fn : Callable
        Called as ``(key, previous_parameter_override, new_state, info) -> new_parameter_override``,
        i.e. with the parameter values used for the step that was just taken, enabling recursive
        adaptation (Robbins-Monro, dual averaging, etc.) across annealing steps.

    See Also
    --------
    blackjax.smc.inner_kernel_tuning.as_top_level_api
        The upstream BlackJAX function this mirrors (without parameter persistence).
    """
    kernel = _build_kernel(
        smc_algorithm,
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        mcmc_parameter_update_fn,
        num_mcmc_steps,
        **extra_parameters,
    )

    def init_fn(position, rng_key=None):
        del rng_key
        return init(smc_algorithm.init, position, initial_parameter_value)

    def step_fn(
        rng_key: PRNGKey, state, **extra_step_parameters
    ) -> Tuple[StateWithParameterOverride, SMCInfo]:
        return kernel(rng_key, state, **extra_step_parameters)

    return SamplingAlgorithm(init_fn, step_fn)
