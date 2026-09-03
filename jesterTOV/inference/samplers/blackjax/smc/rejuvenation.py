"""Final rejuvenation move for terminal SMC resample-to-storage steps.

BlackJAX's tempered SMC (``blackjax.smc.tempered``) builds each step's MCMC
"move" to target the *previous* step's distribution, not the new one the
step's reweighting represents (``tempered_logposterior_fn`` is built from
``state.lmbda``, the pre-increment tempering parameter -- see
``blackjax/smc/tempered.py``). This means the particles produced by
:meth:`~jesterTOV.inference.samplers.blackjax.smc.base.BlackjaxSMCSampler._run_tempering`
are only ever *reweighted* -- never actually MCMC-moved -- to represent the
true final target (lambda=1). The terminal "resample to uniform weights so
saved particles are i.i.d. posterior draws" step that ``_run_tempering``
performs before returning correctly fixes the *weighting* (turns
importance-weighted particles into an unweighted empirical sample), but does
nothing about the fact that the *positions* being resampled were last
actually explored under the second-to-last target, not the final one.
Systematic resampling from weights with effective sample size ESS <
n_particles necessarily produces some number of exact-duplicate rows
(empirically ~5-10% of n_particles for typical production runs, matching
n_particles - ESS almost exactly), and because there is no subsequent move
step, those duplicates survive unchanged into the stored posterior and every
downstream credible-band/statistic computed from it.

`rejuvenate_particles` fixes this with a small number of *plain* MCMC steps
targeting the fixed final log-posterior, run once, after the existing
resample-to-uniform-weights call (never instead of it -- removing that call
would leave the stored samples non-uniformly weighted, which every
downstream consumer in this pipeline silently assumes they are not). Since
the target is fixed and the kernel is invariant with respect to it, this
cannot introduce bias -- it only diversifies whatever exact duplicates the
terminal resample created.

Used by ``samplers/blackjax/smc/ibis.py``'s ``BlackJAXIBISSampler`` as the
final step after its last SMC batch, targeting the fully-assimilated
closed-form posterior (prior x all likelihoods for every event).
"""

from __future__ import annotations

from typing import Callable

import jax
from blackjax.smc.base import update_and_take_last
from blackjax.smc.from_mcmc import unshared_parameters_and_step_fn
from jaxtyping import Array, PRNGKeyArray

__all__ = ["rejuvenate_particles"]


def rejuvenate_particles(
    rng_key: PRNGKeyArray,
    particles: Array,
    logposterior_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    n_steps: int,
) -> Array:
    """Run ``n_steps`` of ``mcmc_step_fn`` targeting ``logposterior_fn`` on
    each particle independently (vmapped internally), returning the final
    positions.

    Parameters
    ----------
    rng_key
        Fresh PRNG key for this rejuvenation pass.
    particles
        Equal-weight particles, shape ``(n_particles, ...)`` -- must already
        be the *output* of the terminal resample-to-uniform-weights step,
        not the pre-resample importance-weighted state.
    logposterior_fn
        The fixed final target log-density (e.g. ``lambda x: logprior_fn(x)
        + loglikelihood_fn(x)`` at lambda=1, or the fully-assimilated
        posterior). Must be the *same* target the particles were just
        resampled to represent -- this function only rejuvenates within that
        target, it does not reweight or change it.
    mcmc_step_fn, mcmc_init_fn
        The same MCMC kernel functions used during annealing/assimilation
        (from ``_setup_mcmc_kernel``), so the rejuvenation move uses an
        identical proposal family to whatever already explored this
        posterior.
    mcmc_parameters
        The tuned kernel parameters to reuse (e.g. the final adapted
        random-walk covariance/scale), following the same shared/unshared
        convention as the rest of the SMC-RW/NUTS machinery -- entries with
        leading dimension 1 are shared across all particles, everything
        else is treated as one value per particle (see
        ``unshared_parameters_and_step_fn``'s docstring).
    n_steps
        Number of MCMC steps to run. ``n_steps <= 0`` is a no-op (returns
        ``particles`` unchanged).

    Returns
    -------
    Array
        The rejuvenated particle positions, same shape as ``particles``.
    """
    if n_steps <= 0:
        return particles

    n_particles = particles.shape[0]
    unshared_mcmc_parameters, shared_mcmc_step_fn = unshared_parameters_and_step_fn(
        mcmc_parameters, mcmc_step_fn
    )
    mcmc_kernel, _ = update_and_take_last(
        mcmc_init_fn,
        logposterior_fn,
        shared_mcmc_step_fn,
        num_mcmc_steps=n_steps,
        n_particles=n_particles,
    )
    keys = jax.random.split(rng_key, n_particles)
    new_particles, _ = mcmc_kernel(keys, particles, unshared_mcmc_parameters)
    return new_particles
