r"""SMC on the "path of partial posteriors" (data tempering / IBIS).

TODO: prune this docstring once we are ready to have it merged

This module implements a sampler that tempers by *number of GW events
included* rather than by the usual inverse-temperature :math:`\lambda` used
by :class:`~jesterTOV.inference.samplers.blackjax.smc.base.BlackjaxSMCSampler`.
Each configured GW event's likelihood term is turned on one at a time,
exposing more of the combined likelihood as sampling progresses. This is
Chopin's Iterated Batch Importance Sampling (IBIS, 2002), presented as the
"path of partial posteriors" in Dai, Heng, Jacob & Whiteley, "An invitation
to sequential Monte Carlo samplers" (arXiv:2007.11936).

This is a genuinely new sampling loop, not a variant of the existing
adaptive-tempering loop in ``smc/base.py`` -- it does not use
``blackjax.adaptive_tempered_smc`` or ``blackjax.inner_kernel_tuning`` at
all, instead driving ``blackjax.smc.partial_posteriors_path`` directly. It
is implemented as a new class in a new module so that the existing
``smc/base.py``, ``smc/random_walk.py`` and ``smc/nuts.py`` files remain
completely untouched; the RW kernel setup and flatten/unflatten utilities
are reused via subclassing :class:`BlackJAXSMCRandomWalkSampler`.

Correctness note (see ``_build_jitted_substep_fn``'s docstring below for
the full derivation, and ``dev/biased_evidence/biased_evidence.tex`` in the
parent development workspace for the toy-model measurements): a naive port
of ``blackjax.smc.partial_posteriors_path.build_kernel`` -- move the
particles with an MCMC kernel targeting the *new* mask, then reweight the
post-move particles by the new/old ratio -- is only a good approximation
of the incremental log-evidence for small target-to-target jumps, and is
measurably biased for a whole-event jump. ``_build_jitted_substep_fn``
instead builds the MCMC move to target the *old* mask (the one the
incoming weights already represent) -- the "resample-move" construction of
Gilks & Berzuini (2001), also used (via a different code path) by
blackjax's own ``smc.tempered.build_kernel`` and by Chopin's reference
``particles`` package's ``IBIS``/``Tempering`` classes -- which makes the
per-substep log-evidence increment exact at *any* mask-jump size, not just
asymptotically as the jump shrinks.

Each event is nonetheless still ramped in over several small fractional
mask increments (matching the source paper's own suggestion of "a
geometric path between successive partial posteriors",
``papers/2007.11936/smcsamplers.tex:409-411``), using an ESS-targeting
bisection search (``blackjax.smc.ess.ess_solver`` /
``blackjax.smc.solver.dichotomy``) -- the same machinery the base
sampler's adaptive :math:`\lambda` schedule uses -- applied to the mask
fraction instead of :math:`\lambda`. This is now purely an MCMC-mixing /
proposal-adaptation concern (a single enormous jump would still leave the
random-walk proposal poorly scaled for the new target), not a correctness
requirement for the evidence. It works unmodified because the
mask-weighted logposterior is linear in the fraction of the single event
currently being ramped in (all other terms cancel in the successive-target
log-weight difference), exactly the structure ``ess_solver`` assumes. The
number of sub-steps per event is uncapped: the search runs until the mask
fraction reaches 1.0, however many sub-steps that takes.

Configuration is split into two levels (see
``SMCPartialPosteriorsRandomWalkSamplerConfig``): the top level only
orchestrates which events are assimilated, in what order, and warm-start
bookkeeping; ``config.inner`` (an ``InnerSMCRandomWalkConfig``) fully
specifies the adaptive SMC-RW loop used to ramp in each event. Each
event's ramp-in also gets its own SMC-diagnostics-style plot (mask
fraction / ESS / acceptance vs. sub-step, mirroring
``BlackjaxSMCSampler.plot_diagnostics``), in addition to the overall
partial-posteriors-path plot across events -- see ``plot_diagnostics``
and ``_plot_substep_diagnostics`` below.
"""

from typing import Any, Callable, cast
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random
from jaxtyping import Array, PRNGKeyArray

from jesterTOV.inference.base import LikelihoodBase
from jesterTOV.inference.config.schema import (
    InferenceConfig,
    SMCPartialPosteriorsRandomWalkSamplerConfig,
)
from jesterTOV.inference.likelihoods.combined import CombinedLikelihood, ZeroLikelihood
from jesterTOV.inference.likelihoods.gw import (
    GWLikelihood,
    GWLikelihoodResampled,
    StackedGWLikelihood,
)
from jesterTOV.inference.samplers.blackjax.smc.base import (
    format_elapsed,
    make_progress_bar,
)
from jesterTOV.inference.samplers.blackjax.smc.rejuvenation import rejuvenate_particles
from jesterTOV.inference.samplers.blackjax.smc.random_walk import (
    BlackJAXSMCRandomWalkSampler,
)
from jesterTOV.logging_config import get_logger

from jax.scipy.special import logsumexp

from blackjax.smc.ess import log_ess
from blackjax.smc.resampling import systematic
from blackjax.smc.solver import dichotomy
from blackjax.smc.from_mcmc import build_kernel as smc_from_mcmc_build_kernel
from blackjax.smc.partial_posteriors_path import (
    init as pp_init,
    PartialPosteriorsSMCState,
)

logger = get_logger("jester")

_GW_EVENT_LIKELIHOOD_TYPES = (GWLikelihood, GWLikelihoodResampled, StackedGWLikelihood)

# Fallback minimal fractional mask increment used only if the ESS-targeting
# bisection has no admissible root (particles already below target_ess even
# at delta=0) -- keeps the run progressing on the safe (small-jump) side
# instead of stalling.
_MIN_FALLBACK_FRACTION = 1e-3

_GW_LIKELIHOOD_TYPES_IN_CONFIG = ("gw", "gw_resampled")


def _extract_gw_event_order(likelihoods: list[dict[str, Any]]) -> list[str]:
    """GW event names configured in a saved run's likelihoods list, in order.

    Parameters
    ----------
    likelihoods : list[dict[str, Any]]
        A run's ``InferenceConfig.likelihoods`` section, as a plain dict
        (e.g. from ``InferenceResult.config_dict["likelihoods"]``).

    Returns
    -------
    list[str]
        Names of every event under any enabled ``gw``/``gw_resampled``
        likelihood block, concatenated in the order the blocks and their
        ``events`` lists appear.
    """
    event_order: list[str] = []
    for lik in likelihoods:
        if (
            lik.get("enabled", True)
            and lik.get("type") in _GW_LIKELIHOOD_TYPES_IN_CONFIG
        ):
            event_order.extend(ev["name"] for ev in lik.get("events", []))
    return event_order


def _canonical_always_on_signature(
    likelihoods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Canonical, order-independent form of the "always-on" likelihoods.

    "Always-on" means every enabled likelihood that isn't a GW event term
    (GW events are the ones this sampler tempers over; everything else --
    ChiEFT, NICER, radio, EOS/TOV constraints, ... -- is applied at full
    weight throughout the whole run). Two configs with the same canonical
    signature are guaranteed to specify identical always-on likelihoods
    (same types, same kwargs -- e.g. the same ChiEFT data path), which is
    the invariant the cumulative logZ bookkeeping across a warm-started run
    depends on.

    Parameters
    ----------
    likelihoods : list[dict[str, Any]]
        A run's ``InferenceConfig.likelihoods`` section, as a plain dict.

    Returns
    -------
    list[dict[str, Any]]
        Enabled, non-GW-event likelihood configs, sorted into a
        deterministic order so two configs can be compared with plain
        equality regardless of how the likelihoods happened to be listed.
    """
    always_on = [
        lik
        for lik in likelihoods
        if lik.get("enabled", True)
        and lik.get("type") not in _GW_LIKELIHOOD_TYPES_IN_CONFIG
    ]
    return sorted(always_on, key=lambda lik: json.dumps(lik, sort_keys=True))


def _build_jitted_substep_fn(
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    num_mcmc_steps: int,
    partial_logposterior_factory: Callable[[Array], Callable[[Array], Array]],
    ordered_event_vals: Callable[[Array], Array],
) -> Callable[
    [PRNGKeyArray, PartialPosteriorsSMCState, Array, Any],
    tuple[PartialPosteriorsSMCState, Any],
]:
    """Build a single ``jax.jit``-compiled sub-step function, compiled once
    and reused for every sub-step of every event.

    Reimplements ``blackjax.smc.partial_posteriors_path.build_kernel``'s
    ``step()`` (rather than calling it directly), because that function
    bakes ``mcmc_parameters`` into a Python closure at construction time --
    fine for blackjax's own top-level API (which doesn't update MCMC
    parameters between steps), but wrong here: jester's random-walk kernel
    re-adapts its proposal covariance from the current particles after
    *every* sub-step (see ``_setup_mcmc_kernel``'s
    ``mcmc_parameter_update_fn`` in ``smc/random_walk.py``). Closing over a
    changing ``mcmc_parameters`` forced ``partial_posteriors.py`` to rebuild
    this kernel -- a brand new Python closure -- on every sub-step, which
    defeats JAX's compilation cache and forces a full retrace + XLA
    recompile of the whole EOS/TOV/GW forward model each time (measured:
    ~7-12s per sub-step on a *lightweight* test config, dominated by
    compile time -- see ``eos_bayesian_updates/dev_notes/slow_jit/``).

    The fix is to use ``blackjax.smc.from_mcmc.build_kernel`` (the
    ``delegate`` that ``partial_posteriors_path.build_kernel`` wraps)
    directly: its returned ``step()`` already takes ``mcmc_parameters`` as
    a *runtime* argument, not a closure constant. Building this once and
    wrapping it in a single ``jax.jit`` -- with ``state``, ``data_mask``
    and ``mcmc_parameters`` all traced arguments -- means the whole
    sampling run (every sub-step of every event) reuses one compiled
    executable, since none of the array shapes/dtypes involved change
    across sub-steps.

    Correctness: the MCMC move is deliberately built to target
    ``previous_logposterior_fn`` (the *old*, pre-substep mask), not
    ``logposterior_fn`` (the new one) -- the opposite of what looks like
    the "obvious" choice. This is the resample-move construction of Gilks
    & Berzuini (2001) that Chopin's own reference SMC implementation
    (``particles``, see ``smc_samplers.py``'s ``IBIS``/``Tempering``
    classes -- and, by the same accidental construction, blackjax's own
    ``smc.tempered.build_kernel``, which builds its tempered log-posterior
    from ``state.tempering_param``, the pre-increment value) uses for
    exactly this reason: a move that leaves the *old* target invariant
    cannot change the particles' marginal distribution at all, however
    large the proposal step, so reweighting the post-move particles by the
    plain incremental ratio ``logposterior_fn(x) - previous_logposterior_fn(x)``
    is an exact, unbiased estimate of the incremental log-evidence with no
    reversal-kernel or small-step assumption anywhere. Building the move
    around ``logposterior_fn`` (the new mask) instead -- what an earlier
    version of this function did, and what
    ``blackjax.smc.partial_posteriors_path.build_kernel`` still does -- is
    the AIS-style construction, which only recovers exactness in the
    small-jump limit and is measurably biased for a whole-event jump; see
    ``dev/biased_evidence/biased_evidence.tex`` (Sections 2 and 5) in the
    parent development workspace for the full derivation and the toy-model
    measurements of both regimes.

    ``ordered_event_vals`` lets ``log_weights_fn`` compute the mask-diff
    likelihood ratio directly (``sum((data_mask - state.data_mask) *
    ordered_event_vals(x))``) instead of building two separate
    ``partial_logposterior_factory(...)`` closures and subtracting their
    outputs -- both would otherwise independently recompute
    ``ordered_event_vals(x)`` (the shared, potentially expensive forward-model
    likelihood array) for the same particle, which XLA's common-subexpression
    elimination does not reliably fold away (measured ~1.5x slowdown in
    isolation for a synthetic analogue -- see the optimization plan's Finding
    2). ``previous_logposterior_fn`` still needs its own
    ``ordered_event_vals`` evaluation since it is also the MCMC move's
    target, evaluated many times during the inner MCMC steps, not just once
    for the reweight.
    """
    delegate = smc_from_mcmc_build_kernel(mcmc_step_fn, mcmc_init_fn, systematic)

    def substep(
        key: PRNGKeyArray,
        state: PartialPosteriorsSMCState,
        data_mask: Array,
        mcmc_parameters: Any,
    ) -> tuple[PartialPosteriorsSMCState, Any]:
        previous_logposterior_fn = partial_logposterior_factory(state.data_mask)
        mask_diff = data_mask - state.data_mask

        def log_weights_fn(x: Array) -> Array:
            return jnp.sum(mask_diff * ordered_event_vals(x))

        # Move targets the OLD mask's posterior (resample-move exactness,
        # see docstring above) -- only the move's target changed relative
        # to a naive port of partial_posteriors_path.build_kernel; the
        # weighting function, resample scheme and SMCInfo bookkeeping are
        # untouched.
        new_state, info = delegate(
            key,
            state,
            num_mcmc_steps,
            mcmc_parameters,
            previous_logposterior_fn,
            log_weights_fn,
        )
        return (
            PartialPosteriorsSMCState(
                new_state.particles, new_state.weights, data_mask
            ),
            info,
        )

    return jax.jit(substep)


def _build_jitted_ess_check_fn(
    ordered_event_vals: Callable[[Array], Array],
) -> Callable[[Array, Array, Array, float | Array, float | Array], tuple[Array, Array]]:
    """Build a single ``jax.jit``-compiled, fixed-shape ESS lookahead,
    compiled once and reused for every candidate-mask ESS check of the run.

    Two call sites need exactly this "if I jumped the mask by up to
    ``max_delta`` along ``candidate_mask``, what would ESS be" question,
    answered without any resample or MCMC move: the per-sub-step delta
    bisection (mirroring ``blackjax.smc.ess.ess_solver``/
    ``blackjax.smc.solver.dichotomy``, but jitted -- see the optimization
    plan's Finding 1: calling ``ess_solver`` bare, unjitted, rebuilds a fresh
    Python closure and retraces/recompiles the bisection on *every*
    sub-step of *every* run, measured ~250x slower than a pre-jitted
    version) and the ``cadence="auto"`` full-jump ESS predictor (replacing
    the old ``_predict_full_jump_ess``, which built a variable-length
    ``jnp.array(group_indices)`` gather inside a fresh Python closure on
    every call -- retracing/recompiling for every distinct candidate-queue
    length, up to ~T times per run).

    The fix for both is the same: represent "which events are in play" as a
    fixed-length ``(T,)`` mask (multiplied against the *already*
    fixed-length ``ordered_event_vals(x)`` array), not a Python list of
    indices gathered out of it. A candidate mask changes *values*, never
    *shape*, as a queue grows or a bisection narrows -- so this compiles
    exactly once for the whole run, unlike either replaced call site.

    Parameters
    ----------
    ordered_event_vals : Callable[[Array], Array]
        Per-particle, per-event log-likelihood array, in mask-index order
        (see ``sample()``) -- closed over once, fixed for the whole run.

    Returns
    -------
    Callable
        ``(particles, base_log_weights, candidate_mask, target_ess,
        max_delta) -> (delta, ess_at_max_delta_fraction)``.
        ``base_log_weights`` (shape ``(n_particles,)``) is added to the
        candidate log-likelihood ratio before the ESS is evaluated -- pass
        all zeros to recover ``ess_solver``'s own implicit assumption that
        the incoming particles are already uniformly weighted (the case
        right after any real resample+move), or ``jnp.log(state.weights)``
        to correctly account for weights already carrying an un-resampled
        importance-sampling ratio (the case while
        ``skip_move_when_ess_ok`` is deferring a move -- see
        ``_reweight_only_step``). ``delta`` is the bisected mask-fraction
        increment (identical, to floating-point tolerance, to
        ``ess_solver``/``dichotomy``'s own output when ``base_log_weights``
        is all zeros). ``ess_at_max_delta_fraction`` is the ESS (as a
        fraction of ``n_particles``) if the *entire* ``max_delta`` jump were
        taken -- exactly the one-shot full-jump ESS
        ``_predict_full_jump_ess`` used to compute standalone, and (for the
        skip design) the quantity that decides whether the whole remaining
        jump can be taken as a cheap reweight-only step instead of a real
        move -- available here for free, since it is a strict subset of
        what the bisection already evaluates internally.
    """

    def _check(
        particles: Array,
        base_log_weights: Array,
        candidate_mask: Array,
        target_ess: float | Array,
        max_delta: float | Array,
    ) -> tuple[Array, Array]:
        def group_loglik(x: Array) -> Array:
            return jnp.sum(candidate_mask * ordered_event_vals(x))

        logprob = jax.vmap(group_loglik)(particles)
        n_particles = logprob.shape[0]
        target_val = jnp.log(n_particles * target_ess)

        def fun_to_solve(delta: Array) -> Array:
            log_weights = jnp.nan_to_num(base_log_weights + delta * logprob)
            return log_ess(log_weights) - target_val

        delta = dichotomy(fun_to_solve, 0.0, max_delta)
        log_ess_at_max = log_ess(jnp.nan_to_num(base_log_weights + max_delta * logprob))
        ess_at_max_fraction = jnp.exp(log_ess_at_max) / n_particles
        return delta, ess_at_max_fraction

    return jax.jit(_check)


def _build_jitted_reweight_only_step_fn(
    ordered_event_vals: Callable[[Array], Array],
) -> Callable[
    [PartialPosteriorsSMCState, Array], tuple[PartialPosteriorsSMCState, Array]
]:
    """Build a jitted, resample-free, move-free reweighting step -- the
    "skip" branch of ``skip_move_when_ess_ok`` (Chopin's IBIS Algorithm
    17.2; see the optimization plan's "New cheap reweight-only step").

    Multiplies the running importance weights by the new/old likelihood
    ratio at the *current*, frozen particle positions (no resample, no
    MCMC) -- ordinary un-resampled importance sampling, unbiased for the
    same reason a single IS step always is, regardless of how large the
    mask jump is (see the module docstring's "Correctness invariant this
    must preserve").

    Crucially, this keeps ``state.data_mask`` in sync with whatever
    ``state.weights`` currently represent -- exactly the invariant
    ``_build_jitted_substep_fn``'s resample-move step already relies on
    every time it runs (going into any call, ``state.weights`` must
    represent the importance ratio of ``state.data_mask`` relative to the
    actual, already-resampled-and-moved particle population; resample-move
    exactness follows from the subsequent MCMC move being invariant to
    exactly that mask). Because this function updates ``data_mask`` and
    ``weights`` together on every call (never advancing one without the
    other), no separate "pending mask" vs. "last-moved mask" bookkeeping is
    needed: whenever a real move eventually fires (because the next
    candidate jump would drop ESS below target), the ordinary
    ``jitted_substep_fn`` can be called completely unmodified on whatever
    ``state`` this function last returned -- it transparently moves
    relative to however many (zero or more) cheap reweight-only steps
    preceded it, and reweights across the *whole* deferred span in one
    exact resample-move step.

    Parameters
    ----------
    ordered_event_vals : Callable[[Array], Array]
        Per-particle, per-event log-likelihood array (see ``sample()``),
        closed over once, fixed for the whole run.

    Returns
    -------
    Callable
        ``(state, new_mask) -> (new_state, log_evidence_increment)``.
        ``log_evidence_increment`` is this step's unbiased contribution to
        the cumulative log evidence, on the same footing as
        ``info.log_likelihood_increment`` from a real move -- summing it
        alongside real moves' increments telescopes correctly to the
        overall log-evidence estimate regardless of how the run splits
        between cheap and real steps (see the module docstring's
        correctness invariant).
    """

    def reweight_only_step(
        state: PartialPosteriorsSMCState, new_mask: Array
    ) -> tuple[PartialPosteriorsSMCState, Array]:
        mask_diff = new_mask - state.data_mask

        def log_ratio(x: Array) -> Array:
            return jnp.sum(mask_diff * ordered_event_vals(x))

        log_ratio_vals = jax.vmap(log_ratio)(cast(Array, state.particles))
        log_weights = jnp.log(state.weights) + log_ratio_vals
        # state.weights already sums to 1 (softmax-normalized, both right
        # after a real move and after any prior reweight-only step), so the
        # incremental log-evidence contribution needs no further
        # normalization -- see the module docstring's derivation.
        log_evidence_increment = logsumexp(log_weights)
        weights = jnp.exp(log_weights - log_evidence_increment)
        new_state = state._replace(weights=weights, data_mask=new_mask)
        return new_state, log_evidence_increment

    return jax.jit(reweight_only_step)


class BlackJAXPartialPosteriorsRandomWalkSampler(BlackJAXSMCRandomWalkSampler):
    """SMC on the path of partial posteriors, with a Random Walk kernel.

    Tempers by number of GW events included rather than by :math:`\\lambda`:
    each configured GW event's likelihood term is turned on one at a time
    (ramped in over an adaptive, ESS-targeting sequence of fractional mask
    sub-steps -- see module docstring), with MCMC rejuvenation after each
    sub-step. Non-GW likelihoods (ChiEFT, NICER, radio, EOS/TOV
    constraints, ...) are always on, exactly as in the base combined
    likelihood.

    Reuses ``_setup_mcmc_kernel`` (Random Walk proposal + covariance
    adaptation), the flatten/unflatten utilities, and the sample-retrieval
    methods (``get_samples``, ``get_log_prob``, ``get_n_samples``,
    ``get_sampler_output``) from :class:`BlackJAXSMCRandomWalkSampler`
    unchanged. Only ``sample()``, ``_get_kernel_name()`` and
    ``plot_diagnostics()`` are overridden.

    Parameters
    ----------
    likelihood : LikelihoodBase
        Combined likelihood. Must contain at least one GW event likelihood
        (``GWLikelihood`` or ``GWLikelihoodResampled``) to temper over.
    prior : Prior
        Prior object
    sample_transforms : list[BijectiveTransform]
        Sample transforms (typically empty for SMC)
    likelihood_transforms : list[NtoMTransform]
        Likelihood transforms
    config : SMCPartialPosteriorsRandomWalkSamplerConfig
        Partial-posteriors SMC configuration
    seed : int, optional
        Random seed (default: 0)

    Raises
    ------
    ValueError
        If ``likelihood`` contains no GW event likelihoods, or if
        ``config.event_order`` doesn't match the set of GW event names
        found in ``likelihood``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # NOTE: not overriding the inherited `config` attribute's type
        # annotation on purpose -- `BlackjaxSMCSampler.config` is typed as
        # `SMCRandomWalkSamplerConfig | SMCNUTSSamplerConfig`, and narrowing
        # a mutable attribute in a subclass is an invariant-type error for
        # pyright. Cast locally instead, matching the existing pattern in
        # `smc/random_walk.py::_setup_mcmc_kernel`.
        config = cast(SMCPartialPosteriorsRandomWalkSamplerConfig, self.config)

        self._event_likelihoods, self._event_names, self._always_on_likelihood = (
            self._partition_likelihood(self.likelihood)
        )
        if len(self._event_likelihoods) == 0:
            raise ValueError(
                "BlackJAXPartialPosteriorsRandomWalkSampler requires at least "
                "one GW event likelihood (GWLikelihood or GWLikelihoodResampled) "
                "in the combined likelihood to temper over -- found none."
            )

        configured_order = config.event_order
        if configured_order is None:
            self._event_order = list(self._event_names)
        else:
            if set(configured_order) != set(self._event_names):
                raise ValueError(
                    "config.event_order does not match the GW event names found "
                    f"in the likelihood. Configured: {sorted(configured_order)}, "
                    f"found: {sorted(self._event_names)}."
                )
            self._event_order = list(configured_order)

        logger.info(
            f"Partial-posteriors SMC: {len(self._event_order)} GW event(s) will be "
            f"assimilated in order: {self._event_order}"
        )

        # Populated by sample(): per-event sub-step diagnostics (mask
        # fraction / ESS / acceptance / cumulative logZ), keyed by event
        # name. Used by plot_diagnostics() to produce one SMC-diagnostics
        # style plot per event, in addition to the overall
        # partial-posteriors-path plot.
        self._substep_diagnostics: dict[str, dict[str, list[float] | list[bool]]] = {}

        # Populated by configure_intermediate_saving(), called externally
        # (run_inference.py) because this sampler doesn't otherwise have
        # access to the full InferenceConfig (only its own sampler
        # sub-config) or the output directory. Only consumed if
        # config.save_intermediate_results is True -- see
        # _save_intermediate_result.
        self._intermediate_save_config: InferenceConfig | None = None
        self._intermediate_save_outdir: Path | None = None
        self._intermediate_save_fixed_params: dict[str, float] = {}

    @staticmethod
    def _partition_likelihood(
        likelihood: LikelihoodBase,
    ) -> tuple[list[LikelihoodBase], list[str], LikelihoodBase]:
        """Split a combined likelihood into per-event GW terms and the rest.

        Parameters
        ----------
        likelihood : LikelihoodBase
            Either a ``CombinedLikelihood`` wrapping a flat list of
            likelihood terms, or a single bare likelihood (the shape
            ``create_combined_likelihood`` returns when there is exactly
            one likelihood total).

        Returns
        -------
        tuple[list[LikelihoodBase], list[str], LikelihoodBase]
            (event_likelihoods, event_names, always_on_likelihood).
            ``event_likelihoods`` is a list of *sources*, not necessarily one
            per event: a ``StackedGWLikelihood`` is a single source
            contributing several event names at once (see
            ``_event_names_for``). The always-on likelihood wraps everything
            that isn't a GW event source; ``ZeroLikelihood()`` if there is
            nothing else.
        """
        if isinstance(likelihood, CombinedLikelihood):
            all_likelihoods = likelihood.likelihoods_list
        else:
            all_likelihoods = [likelihood]

        event_likelihoods: list[LikelihoodBase] = []
        event_names: list[str] = []
        always_on: list[LikelihoodBase] = []
        for lik in all_likelihoods:
            if isinstance(lik, _GW_EVENT_LIKELIHOOD_TYPES):
                event_likelihoods.append(lik)
                event_names.extend(
                    BlackJAXPartialPosteriorsRandomWalkSampler._event_names_for(lik)
                )
            else:
                always_on.append(lik)

        if len(always_on) == 0:
            always_on_likelihood: LikelihoodBase = ZeroLikelihood()
        elif len(always_on) == 1:
            always_on_likelihood = always_on[0]
        else:
            always_on_likelihood = CombinedLikelihood(always_on)

        return event_likelihoods, event_names, always_on_likelihood

    @staticmethod
    def _event_names_for(lik: LikelihoodBase) -> list[str]:
        """Event name(s) contributed by one GW event source.

        A ``StackedGWLikelihood`` batches several events behind one object
        (see its module docstring), so it contributes its whole
        ``event_names`` list; a plain ``GWLikelihood``/``GWLikelihoodResampled``
        contributes its single ``event_name``.
        """
        if isinstance(lik, StackedGWLikelihood):
            return list(lik.event_names)
        return [lik.event_name]  # type: ignore[attr-defined]

    def _get_kernel_name(self) -> str:
        return "partial_posteriors_random_walk"

    def configure_intermediate_saving(
        self,
        full_config: InferenceConfig,
        outdir: str | Path,
        fixed_params: dict[str, float] | None = None,
    ) -> None:
        """Wire up context needed to save per-step intermediate results.

        This sampler only receives its own sampler sub-config (``self.config``,
        a ``SMCPartialPosteriorsRandomWalkSamplerConfig``), not the full
        ``InferenceConfig`` or the run's output directory -- both are needed
        to build and save an ``InferenceResult`` snapshot after each
        data-tempering step (see ``_save_intermediate_result``). Call this
        (from ``run_inference.py``, right after ``create_sampler``) before
        ``sample()`` if ``config.sampler.save_intermediate_results`` is
        ``True``; harmless no-op setup otherwise.

        Parameters
        ----------
        full_config : InferenceConfig
            The complete run configuration (serialized into each
            intermediate result's metadata, exactly as for the final
            ``results.h5``).
        outdir : str or Path
            Run output directory. Intermediate results are saved under
            ``outdir/substep_results/``.
        fixed_params : dict[str, float] | None, optional
            Parameters pinned to constant values during inference, stored
            in each intermediate result's metadata like the final result.
        """
        self._intermediate_save_config = full_config
        self._intermediate_save_outdir = Path(outdir)
        self._intermediate_save_fixed_params = fixed_params or {}

    def _build_metadata(
        self,
        config: SMCPartialPosteriorsRandomWalkSamplerConfig,
        n_particles: int,
        event_order_so_far: list[str],
        event_groups_names_so_far: list[list[str]],
        n_prev_events: int,
        n_prev_batches: int,
        batch_to_sources_so_far: dict[str, list[str]],
        final_ess_fraction: float,
        ess_history_so_far: list[float],
        acceptance_history_so_far: list[float],
        log_evidence_history_so_far: list[float],
        n_substeps_history_so_far: list[int],
        auto_cadence_ess_checks_so_far: list[tuple[float, bool]],
        log_evidence_so_far: float,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        """Build the sampler metadata dict, for either an intermediate or the
        final snapshot of the run.

        Factored out of ``sample()`` so that ``_save_intermediate_result``
        can build a metadata dict describing "the run as if it stopped
        after this data-tempering step" using the exact same shape/keys as
        the real final metadata -- required for it to be accepted by
        ``InferenceResult.from_sampler`` (which reads several
        ``blackjax_smc_partial_posteriors_rw``-specific keys, see
        ``result.py``).

        Parameters
        ----------
        config : SMCPartialPosteriorsRandomWalkSamplerConfig
            Sampler configuration.
        n_particles : int
            Number of particles.
        event_order_so_far : list[str]
            GW events assimilated up to and including this point (a prefix
            of ``self._event_order``; equals the full list for the final
            call).
        event_groups_names_so_far : list[list[str]]
            Data-tempering groups processed up to and including this point.
        n_prev_events : int
            Number of events skipped via warm-starting (see ``sample()``).
        n_prev_batches : int
            Number of data-tempering batches already used up by a
            warm-start source run (0 if not warm-starting) -- this run's
            own batches are numbered starting from ``n_prev_batches + 1``,
            so ``batch_<NN>`` labels/filenames never collide with the
            warm-start source's when both land under the same ``outdir``.
        batch_to_sources_so_far : dict[str, list[str]]
            Maps each processed ``batch_<NN>`` key to the list of event
            names it contains, up to and including this point.
        final_ess_fraction : float
            ESS as a fraction of ``n_particles`` (i.e. already divided by
            ``n_particles``) at this point in the run.
        ess_history_so_far, acceptance_history_so_far, log_evidence_history_so_far, n_substeps_history_so_far : list
            Per-step diagnostic histories up to and including this point.
        auto_cadence_ess_checks_so_far : list[tuple[float, bool]]
            Auto-cadence full-jump ESS predictor checks made so far, one
            per not-yet-assimilated event considered (``(ess, triggered)``
            pairs, in the order added to the pending queue) -- empty when
            ``cadence`` isn't ``"auto"``/``"automatic"``. See
            ``_plot_auto_cadence_ess``.
        log_evidence_so_far : float
            Cumulative log evidence up to and including this point.
        elapsed_seconds : float
            Wall-clock time elapsed since sampling started.

        Returns
        -------
        dict[str, Any]
            Metadata dict matching the shape stored in ``self.metadata`` by
            a completed ``sample()`` call.
        """
        mean_ess = float(jnp.mean(jnp.array(ess_history_so_far)))
        min_ess = float(jnp.min(jnp.array(ess_history_so_far)))
        # nanmean, not mean: a data-tempering step that finishes via
        # reweight-only skips alone (skip_move_when_ess_ok=True, no real
        # move at all that step) records NaN acceptance for that step (see
        # sample()'s group loop) -- identical to jnp.mean when no NaNs are
        # present, i.e. always, when skip_move_when_ess_ok is False.
        mean_acceptance = float(jnp.nanmean(jnp.array(acceptance_history_so_far)))
        return {
            "sampler": f"blackjax_smc_{self._get_kernel_name()}",
            "kernel_type": self._get_kernel_name(),
            "n_particles": n_particles,
            "n_mcmc_steps": config.n_mcmc_steps,
            "target_ess": config.target_ess,
            "event_order": event_order_so_far,
            "n_events": len(event_order_so_far),
            "cadence": config.cadence,
            "auto_ess_threshold": (
                config.resolved_auto_ess_threshold if config.is_auto_cadence else 0.0
            ),
            "skip_move_when_ess_ok": config.skip_move_when_ess_ok,
            "event_groups": event_groups_names_so_far,
            "batch_to_sources": dict(batch_to_sources_so_far),
            "n_batches": n_prev_batches + len(event_groups_names_so_far),
            "auto_cadence_ess_history": [
                ess for ess, _ in auto_cadence_ess_checks_so_far
            ],
            "auto_cadence_triggered_history": [
                triggered for _, triggered in auto_cadence_ess_checks_so_far
            ],
            "warm_start_from": config.warm_start_from or "",
            "n_events_replayed": n_prev_events,
            "final_ess": float(final_ess_fraction * n_particles),
            "final_ess_percent": float(final_ess_fraction * 100),
            "mean_ess": mean_ess,
            "min_ess": min_ess,
            "mean_acceptance": mean_acceptance,
            "logZ": float(log_evidence_so_far),
            "logZ_err": 0.0,  # FIXME: same placeholder as base.py
            "sampling_time_seconds": elapsed_seconds,
            "ess_history": ess_history_so_far,
            "acceptance_history": acceptance_history_so_far,
            "log_evidence_history": log_evidence_history_so_far,
            "n_substeps_history": n_substeps_history_so_far,
        }

    def _save_intermediate_result(
        self,
        key: PRNGKeyArray,
        state: PartialPosteriorsSMCState,
        config: SMCPartialPosteriorsRandomWalkSamplerConfig,
        n_particles: int,
        logposterior_fn: Callable[[Array], Array],
        mcmc_step_fn: Callable,
        mcmc_init_fn: Callable,
        mcmc_parameters: Any,
        event_order_so_far: list[str],
        event_groups_names_so_far: list[list[str]],
        n_prev_events: int,
        n_prev_batches: int,
        batch_to_sources_so_far: dict[str, list[str]],
        ess_history_so_far: list[float],
        acceptance_history_so_far: list[float],
        log_evidence_history_so_far: list[float],
        n_substeps_history_so_far: list[int],
        auto_cadence_ess_checks_so_far: list[tuple[float, bool]],
        log_evidence_so_far: float,
        elapsed_seconds: float,
        batch_key: str,
    ) -> None:
        """Save an ``InferenceResult`` snapshot of the posterior after a
        data-tempering step, including derived EOS quantities from the TOV
        solver -- exactly mirroring what ``run_inference.py`` does for the
        final result, just run mid-loop on the current particles instead of
        the converged ones.

        Requires ``configure_intermediate_saving()`` to have been called;
        otherwise this logs a warning and does nothing (rather than
        raising, since ``config.save_intermediate_results`` is read from
        the sampler's own config and could be enabled without the caller
        having wired up the extra context this needs).

        Parameters
        ----------
        key : PRNGKeyArray
            Random key used only for the equal-weight resampling below.
        state : PartialPosteriorsSMCState
            Current SMC state (particles/weights) after the just-completed
            data-tempering step.
        logposterior_fn : Callable
            The target this step's ``state.data_mask`` represents (i.e.
            ``partial_logposterior_factory(state.data_mask)``, computed by
            the caller). Used only for the optional final-rejuvenation
            move below (see ``config.n_final_rejuvenation_steps``) -- not
            for resampling or reweighting, which use ``state.weights`` as
            already computed by the just-completed sub-step.
        mcmc_step_fn, mcmc_init_fn, mcmc_parameters :
            The same MCMC kernel/parameters used for this batch's
            sub-steps (``mcmc_params`` in ``sample()``'s loop), reused for
            the optional final-rejuvenation move so it uses an identical
            proposal family to whatever already explored this checkpoint's
            posterior.
        config, n_particles, event_order_so_far, event_groups_names_so_far, n_prev_events, n_prev_batches, batch_to_sources_so_far, ess_history_so_far, acceptance_history_so_far, log_evidence_history_so_far, n_substeps_history_so_far, auto_cadence_ess_checks_so_far, log_evidence_so_far, elapsed_seconds :
            See ``_build_metadata`` -- forwarded directly.
        batch_key : str
            This step's sequential ``batch_<NN>`` key (see ``sample()``),
            used as the intermediate result's filename
            (``results_<batch_key>.h5``) -- matches the naming used for
            ``substep_diagnostics/<batch_key>.png`` and
            ``metadata["batch_to_sources"]``, and stays gap-free/collision-free
            across warm-started runs since ``batch_key`` already accounts for
            ``n_prev_batches``.
        """
        if self._intermediate_save_config is None:
            logger.warning(
                "config.save_intermediate_results is True but "
                "configure_intermediate_saving() was never called on this "
                "sampler -- skipping intermediate result saving. "
                "(run_inference.py should call it automatically; this is "
                "expected only if the sampler is being driven manually.)"
            )
            return

        logger.info(
            "Postprocessing and saving intermediate run results "
            f"({batch_key}: events {event_order_so_far})..."
        )

        weights = state.weights
        final_ess_fraction = float(
            jnp.sum(weights) ** 2 / jnp.sum(weights**2) / n_particles
        )
        resample_idx = systematic(key, weights, n_particles)
        particles_flat = cast(Array, state.particles)[resample_idx]
        uniform_weights = jnp.ones(n_particles) / n_particles

        # Optional final rejuvenation for this checkpoint: the resample
        # above corrects the *weighting* but the resampled particles were
        # last actually MCMC-moved under the previous sub-step's mask, not
        # this one -- see rejuvenation.py's module docstring (same
        # mechanism as base.py's terminal resample, just once per batch
        # checkpoint here). Disabled by default (n_final_rejuvenation_steps=0).
        if config.n_final_rejuvenation_steps > 0:
            key, rejuvenation_key = jax.random.split(key)
            particles_flat = rejuvenate_particles(
                rejuvenation_key,
                particles_flat,
                logposterior_fn,
                mcmc_step_fn,
                mcmc_init_fn,
                mcmc_parameters,
                config.n_final_rejuvenation_steps,
            )

        intermediate_metadata = self._build_metadata(
            config=config,
            n_particles=n_particles,
            event_order_so_far=event_order_so_far,
            event_groups_names_so_far=event_groups_names_so_far,
            n_prev_events=n_prev_events,
            n_prev_batches=n_prev_batches,
            batch_to_sources_so_far=batch_to_sources_so_far,
            final_ess_fraction=final_ess_fraction,
            ess_history_so_far=ess_history_so_far,
            acceptance_history_so_far=acceptance_history_so_far,
            log_evidence_history_so_far=log_evidence_history_so_far,
            n_substeps_history_so_far=n_substeps_history_so_far,
            auto_cadence_ess_checks_so_far=auto_cadence_ess_checks_so_far,
            log_evidence_so_far=log_evidence_so_far,
            elapsed_seconds=elapsed_seconds,
        )

        # Temporarily swap in this step's particles/weights/metadata so
        # get_samples()/get_log_prob()/get_sampler_output() -- and thus
        # InferenceResult.from_sampler, which is driven entirely through
        # those methods -- report this step's state. Restored in `finally`
        # so the real end-of-run assignment in sample() (and any other
        # caller inspecting these attributes) is unaffected; at this point
        # in the loop they are still None (unset), matching __init__.
        prev_particles_flat = self._particles_flat
        prev_weights = self._weights
        prev_final_state = self.final_state
        prev_metadata = self.metadata
        self._particles_flat = particles_flat
        self._weights = uniform_weights
        self.final_state = state
        self.metadata = intermediate_metadata
        try:
            from jesterTOV.inference.result import InferenceResult

            result = InferenceResult.from_sampler(
                sampler=self,
                config=self._intermediate_save_config,
                runtime=elapsed_seconds,
                fixed_params=self._intermediate_save_fixed_params,
            )
            result.add_eos_from_transform(
                transform=self.likelihood_transforms[0],
                n_eos_samples=config.n_eos_samples,
                batch_size=config.log_prob_batch_size,
            )
            assert self._intermediate_save_outdir is not None
            substep_outdir = self._intermediate_save_outdir / "substep_results"
            substep_outdir.mkdir(parents=True, exist_ok=True)
            result_path = substep_outdir / f"results_{batch_key}.h5"
            result.save(result_path)
        finally:
            self._particles_flat = prev_particles_flat
            self._weights = prev_weights
            self.final_state = prev_final_state
            self.metadata = prev_metadata

    def _compute_event_groups(self, n_prev_events: int) -> list[list[int]]:
        """Chunk the not-yet-assimilated events into cadence groups.

        Parameters
        ----------
        n_prev_events : int
            Number of events already assimilated (skipped entirely, e.g. by
            a warm start) -- see :meth:`_load_warm_start`. Grouping only
            applies to ``self._event_order[n_prev_events:]``.

        Returns
        -------
        list[list[int]]
            Each element is a list of indices into ``self._event_order``
            (all >= ``n_prev_events``), the events to be ramped in jointly
            during one data-tempering step. Concatenating all groups
            recovers ``range(n_prev_events, len(self._event_order))``.

        Raises
        ------
        ValueError
            If ``config.cadence`` is a list whose entries don't sum to
            exactly the number of new events (``len(self._event_order) -
            n_prev_events``).
        """
        config = cast(SMCPartialPosteriorsRandomWalkSamplerConfig, self.config)
        new_event_indices = list(range(n_prev_events, len(self._event_order)))
        n_new_events = len(new_event_indices)

        cadence = config.cadence
        if isinstance(cadence, str):
            # Only reached if a caller invokes this fixed-schedule grouping
            # directly with cadence="auto"/"automatic" -- sample() itself
            # never does (it builds groups dynamically instead, see
            # _next_auto_group).
            raise ValueError(
                f"_compute_event_groups does not support cadence={cadence!r}; "
                "auto-cadence groups are built dynamically in sample()."
            )
        elif isinstance(cadence, int):
            group_sizes = [cadence] * (n_new_events // cadence)
            remainder = n_new_events - sum(group_sizes)
            if remainder > 0:
                group_sizes.append(remainder)
        else:
            if sum(cadence) != n_new_events:
                raise ValueError(
                    f"config.cadence={cadence} sums to {sum(cadence)}, but "
                    f"there are {n_new_events} new event(s) to assimilate "
                    f"this run ({new_event_indices and self._event_order[n_prev_events:]}). "
                    "The cadence list must sum to exactly the number of new "
                    "events (total configured events minus any already "
                    "covered by warm_start_from)."
                )
            group_sizes = list(cadence)

        groups: list[list[int]] = []
        cursor = 0
        for size in group_sizes:
            groups.append(new_event_indices[cursor : cursor + size])
            cursor += size
        return groups

    def _check_always_on_likelihoods_match(
        self, path: str, prev_likelihoods: list[dict[str, Any]]
    ) -> None:
        """Verify the warm-start source's always-on likelihoods match this run's.

        The cumulative logZ bookkeeping across a warm-started run only
        makes sense if the previous run's always-on (non-GW) likelihoods
        -- ChiEFT, NICER, radio, EOS/TOV constraints, ... -- are the exact
        same types with the exact same kwargs as this run's (e.g. the same
        ChiEFT data path). A mismatch would silently corrupt the evidence
        without this check.

        Parameters
        ----------
        path : str
            The ``warm_start_from`` path, used only for the error message.
        prev_likelihoods : list[dict[str, Any]]
            The previous run's saved ``likelihoods`` config section (see
            ``InferenceResult.config_dict``).

        Raises
        ------
        ValueError
            If the previous run's always-on likelihoods don't exactly
            match this run's.
        """
        if self.likelihood_configs is None:
            logger.warning(
                "Cannot verify that warm_start_from's always-on (non-GW) "
                "likelihoods match this run's -- this sampler wasn't given "
                "the full likelihood config (create_sampler's "
                "likelihood_configs argument). Proceeding without this "
                "check; a mismatch (e.g. a different ChiEFT data path) "
                "would silently corrupt the cumulative logZ bookkeeping."
            )
            return

        current_likelihoods = [lik.model_dump() for lik in self.likelihood_configs]
        prev_signature = _canonical_always_on_signature(prev_likelihoods)
        current_signature = _canonical_always_on_signature(current_likelihoods)
        if prev_signature != current_signature:
            raise ValueError(
                f"warm_start_from={path!r}'s always-on (non-GW) likelihoods "
                "do not match this run's -- the cumulative logZ bookkeeping "
                "assumes they are identical (same types, same kwargs, e.g. "
                f"the same ChiEFT data path).\nPrevious: {prev_signature}\n"
                f"Current: {current_signature}"
            )

    def _load_warm_start(
        self, path: str, n_particles: int, key: PRNGKeyArray
    ) -> tuple[dict[str, Array], int, float, int]:
        """Load initial particles from a previous run's converged posterior.

        The previous run does not have to have used this sampler: any saved
        ``InferenceResult`` (``smc-rw``, ``smc-nuts``, ``flowmc``, or this
        sampler itself) works, as long as its posterior is converged on a
        combined likelihood whose GW events are a strict prefix of this
        run's ``self._event_order`` and whose always-on (non-GW)
        likelihoods exactly match this run's. Both are derived from the
        previous run's *saved config* (``InferenceResult.config_dict``),
        not from sampler-specific metadata -- see :func:`_extract_gw_event_order`
        and :func:`_canonical_always_on_signature`. The previous run may
        have had *no* GW events at all (e.g. a radio- or ChiEFT-only run):
        the empty list is trivially a strict prefix, so every configured
        GW event is then assimilated from scratch, only the initial
        particles come from that run's posterior instead of the prior.

        Parameters
        ----------
        path : str
            Path to a previous ``InferenceResult`` HDF5 file.
        n_particles : int
            Number of particles this run wants to start with. Resampled
            (with replacement, uniform weights) from the previous run's
            particles if the counts differ.
        key : PRNGKeyArray
            JAX random key used for resampling.

        Returns
        -------
        tuple[dict[str, Array], int, float, int]
            (initial_position_dict, n_prev_events, prev_log_evidence,
            n_prev_batches): particle positions in prior-parameter space
            (same shape as ``self.prior.sample(...)``), the number of
            events already assimilated by the previous run, its final
            ``logZ`` (used to keep the cumulative evidence history
            meaningful across the warm-started run), and the number of
            data-tempering batches already used up by the previous run
            (read from its saved ``n_batches`` metadata, default 0 for
            older results that predate it) -- so this run's own batches
            continue numbering from there (``batch_<n_prev_batches+1>``,
            ...) instead of colliding with the previous run's ``batch_01``,
            ``batch_02``, ... filenames when both land under the same
            ``outdir``.

        Raises
        ------
        ValueError
            If the previous run's configured GW events are not a strict
            prefix of this run's ``self._event_order``, if its always-on
            (non-GW) likelihoods don't match this run's, or if its
            posterior is missing a parameter required by the current prior.
        """
        from jesterTOV.inference.result import InferenceResult

        logger.info(f"Warm-starting partial-posteriors SMC from: {path}")
        prev_result = InferenceResult.load(path)
        prev_likelihoods = prev_result.config_dict.get("likelihoods", [])

        # An empty prev_event_order is valid and useful: it means the
        # warm-start source had no GW events at all (e.g. a radio/ChiEFT
        # -only run), and this run's initial particles are resampled from
        # that converged posterior instead of the prior, with every
        # configured GW event still assimilated from scratch (n_prev_events
        # stays 0). It is trivially a "strict prefix" of self._event_order.
        prev_event_order = _extract_gw_event_order(prev_likelihoods)
        if (
            len(prev_event_order) >= len(self._event_order)
            or self._event_order[: len(prev_event_order)] != prev_event_order
        ):
            raise ValueError(
                "warm_start_from's configured GW events must be a strict "
                "prefix of this run's event_order to keep mask indices "
                f"aligned. Previous: {prev_event_order}, current: "
                f"{self._event_order}."
            )

        self._check_always_on_likelihoods_match(path, prev_likelihoods)

        prev_posterior = prev_result.posterior
        missing = [
            name for name in self.prior.parameter_names if name not in prev_posterior
        ]
        if missing:
            raise ValueError(
                f"warm_start_from={path!r} posterior is missing "
                f"parameter(s) {missing} required by the current prior."
            )

        n_prev_particles = len(prev_posterior[self.prior.parameter_names[0]])
        resample_idx = systematic(
            key, jnp.ones(n_prev_particles) / n_prev_particles, n_particles
        )
        initial_position_dict = {
            name: jnp.asarray(prev_posterior[name])[resample_idx]
            for name in self.prior.parameter_names
        }

        prev_log_evidence = float(prev_result.metadata.get("logZ", 0.0))
        n_prev_batches = int(prev_result.metadata.get("n_batches", 0))
        logger.info(
            f"Warm start: {len(prev_event_order)} event(s) already "
            f"assimilated ({prev_event_order}), resampled {n_particles} "
            f"particles from {n_prev_particles} previous particles, "
            f"prev logZ={prev_log_evidence:.3f}, prev n_batches={n_prev_batches}"
        )
        return (
            initial_position_dict,
            len(prev_event_order),
            prev_log_evidence,
            n_prev_batches,
        )

    def _make_loglikelihood_dict_fn(
        self, single_likelihood: LikelihoodBase
    ) -> Callable[[dict[str, Any]], float]:
        """Dict-based log-likelihood function for one likelihood term.

        Mirrors the transform-application logic of
        ``BlackjaxSampler._create_loglikelihood_fn_from_dict``
        (``jesterTOV/inference/samplers/blackjax/base.py``), which is
        hardcoded to ``self.likelihood``. Duplicated here (rather than
        adding a parameter to that shared method) to keep the existing
        SMC/BlackJAX base files completely untouched, per this sampler's
        design constraint of being purely additive.
        """

        def loglikelihood_fn(params_dict: dict[str, Any]) -> float:
            named_params = params_dict.copy()
            for transform in reversed(self.sample_transforms):
                named_params, _ = transform.inverse(named_params)
            for transform in self.likelihood_transforms:
                named_params = transform.forward(named_params)
            return single_likelihood.evaluate(named_params)

        return jax.jit(loglikelihood_fn)

    def _make_event_source_array_fn(
        self, source: LikelihoodBase
    ) -> Callable[[dict[str, Any]], Array]:
        """Dict-based, per-event-array log-likelihood function for one GW
        event source.

        Like ``_make_loglikelihood_dict_fn``, but returns an array (one
        entry per event contributed by ``source``, in
        ``_event_names_for(source)`` order) instead of a summed scalar --
        needed to mask individual events in and out. A plain
        ``GWLikelihood``/``GWLikelihoodResampled`` source contributes a
        length-1 array; a ``StackedGWLikelihood`` contributes one entry per
        event it batches (via its ``evaluate_per_event``).
        """

        def loglikelihood_array_fn(params_dict: dict[str, Any]) -> Array:
            named_params = params_dict.copy()
            for transform in reversed(self.sample_transforms):
                named_params, _ = transform.inverse(named_params)
            for transform in self.likelihood_transforms:
                named_params = transform.forward(named_params)
            if isinstance(source, StackedGWLikelihood):
                return source.evaluate_per_event(named_params)
            return jnp.reshape(source.evaluate(named_params), (1,))

        return jax.jit(loglikelihood_array_fn)

    def sample(self, key: PRNGKeyArray) -> None:
        """Run SMC on the path of partial posteriors, one event at a time.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key

        Notes
        -----
        Initial particles are sampled from the prior internally, unless
        ``config.warm_start_from`` is set, in which case they are resampled
        from a previous run's converged posterior instead (see
        :meth:`_load_warm_start`). Each event in ``self._event_order`` that
        hasn't already been assimilated by the warm-started run is
        assimilated by ramping its mask entry from 0 to 1 over an uncapped
        ESS-targeting sequence of sub-steps (see module docstring), with
        ``config.inner.n_mcmc_steps`` rejuvenation moves per sub-step.
        Already-covered events are not replayed: their mask entries start
        (and stay) at 1.
        """
        config = cast(SMCPartialPosteriorsRandomWalkSamplerConfig, self.config)

        logger.info(
            f"Starting partial-posteriors SMC sampling with {self._get_kernel_name()} "
            "kernel..."
        )
        start_time = time.time()

        key, subkey = jax.random.split(key)
        n_prev_events = 0
        warm_start_log_evidence = 0.0
        n_prev_batches = 0
        if config.warm_start_from is not None:
            (
                initial_position_dict,
                n_prev_events,
                warm_start_log_evidence,
                n_prev_batches,
            ) = self._load_warm_start(
                config.warm_start_from, config.n_particles, subkey
            )
        else:
            initial_position_dict = self.prior.sample(subkey, config.n_particles)

        for transform in self.sample_transforms:
            initial_position_list = []
            for i in range(config.n_particles):
                particle_dict = {
                    name: initial_position_dict[name][i]
                    for name in self.prior.parameter_names
                }
                transformed_dict, _ = transform.transform(particle_dict)
                initial_position_list.append(transformed_dict)
            initial_position_dict = {
                name: jnp.array([p[name] for p in initial_position_list])
                for name in initial_position_list[0].keys()
            }

        self._create_flatten_unflatten_utilities(initial_position_dict)
        initial_position_flat = jax.vmap(self._flatten_fn)(initial_position_dict)
        if not jnp.issubdtype(initial_position_flat.dtype, jnp.floating):
            logger.warning(
                f"Converting initial_position_flat from {initial_position_flat.dtype} "
                "to float64"
            )
            initial_position_flat = initial_position_flat.astype(jnp.float64)

        logprior_dict = self._create_logprior_fn_from_dict()
        logprior_fn = self._wrap_dict_fn_for_flat_arrays(logprior_dict)

        always_on_loglik_fn = self._wrap_dict_fn_for_flat_arrays(
            self._make_loglikelihood_dict_fn(self._always_on_likelihood)
        )
        # Each source contributes an array of per-event log-likelihoods (one
        # entry per event for a plain GWLikelihood/GWLikelihoodResampled,
        # several for a StackedGWLikelihood -- see _event_names_for). These
        # arrays are concatenated in source order ("natural order"), then
        # permuted once (a static, Python-level index list, not something
        # traced) into self._event_order so the mask lines up regardless of
        # how sources/events happen to be grouped or configured.
        source_array_fns = [
            self._wrap_dict_fn_for_flat_arrays(
                cast(
                    Callable[[dict[str, Any]], float],
                    self._make_event_source_array_fn(lik),
                )
            )
            for lik in self._event_likelihoods
        ]
        natural_order_names = [
            name
            for lik in self._event_likelihoods
            for name in self._event_names_for(lik)
        ]
        name_to_natural_index = {name: i for i, name in enumerate(natural_order_names)}
        event_order_perm = jnp.array(
            [name_to_natural_index[name] for name in self._event_order]
        )

        def ordered_event_vals(x_flat: Array) -> Array:
            """Per-event log-likelihoods for one particle, in
            ``self._event_order`` (i.e. mask-index) order."""
            event_vals_natural = jnp.concatenate([f(x_flat) for f in source_array_fns])
            return event_vals_natural[event_order_perm]

        def partial_logposterior_factory(data_mask: Array) -> Callable[[Array], Array]:
            def logpost(x_flat: Array) -> Array:
                event_vals = ordered_event_vals(x_flat)
                return (
                    logprior_fn(x_flat)
                    + always_on_loglik_fn(x_flat)
                    + jnp.sum(data_mask * event_vals)
                )

            return logpost

        n_events = len(self._event_order)
        full_logposterior_fn = partial_logposterior_factory(jnp.ones(n_events))

        mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn = (
            self._setup_mcmc_kernel(
                logprior_fn,
                always_on_loglik_fn,
                full_logposterior_fn,
                initial_position_flat,
            )
        )

        state: PartialPosteriorsSMCState = pp_init(initial_position_flat, n_events)
        mcmc_params = init_params
        mask = jnp.zeros(n_events)
        if n_prev_events > 0:
            # Already-covered events start (and stay) fully on -- they are
            # not replayed, per the warm-start design (see _load_warm_start).
            mask = mask.at[:n_prev_events].set(1.0)
            state = state._replace(data_mask=mask)

        jitted_substep_fn = _build_jitted_substep_fn(
            mcmc_step_fn,
            mcmc_init_fn,
            config.n_mcmc_steps,
            partial_logposterior_factory,
            ordered_event_vals,
        )
        # Compiled once, fixed-(n_events,)-shape ESS lookahead, shared by
        # the per-sub-step bisection below (Finding 1: a bare, unjitted
        # ess_solver() call retraces/recompiles on every sub-step) and the
        # auto-cadence full-jump predictor inside _next_auto_group
        # (previously _predict_full_jump_ess, which retraced/recompiled for
        # every distinct pending-queue length) -- see
        # _build_jitted_ess_check_fn's docstring.
        jitted_ess_check_fn = _build_jitted_ess_check_fn(ordered_event_vals)
        # Compiled once, resample-free/move-free reweighting step -- the
        # "skip" branch used when config.skip_move_when_ess_ok is True (see
        # _build_jitted_reweight_only_step_fn's docstring). Built
        # unconditionally (cheap) so the flag can be toggled without any
        # other change to this method.
        jitted_reweight_only_step_fn = _build_jitted_reweight_only_step_fn(
            ordered_event_vals
        )

        n_particles = config.n_particles
        ess_history = []
        acceptance_history = []
        log_evidence_history = []
        n_substeps_history = []
        log_evidence = warm_start_log_evidence

        # Per-group sub-step diagnostics (mask fraction / ESS / acceptance /
        # cumulative logZ at each sub-step), for the per-group SMC-diagnostics
        # style plot produced in plot_diagnostics() -- see
        # _plot_substep_diagnostics. Keyed by batch_key ("batch_<NN>", NN
        # a sequential 1-indexed counter of data-tempering batches processed
        # so far -- across warm-started runs too, via n_prev_batches -- so
        # batch numbers always increase by exactly 1 (batch_01, batch_02,
        # ...) regardless of how many events land in each batch, rather than
        # jumping by each batch's event count (previously NN was the
        # absolute index of the group's first event in event_order, e.g.
        # batch_00, batch_01, batch_04, batch_08, ... once batches started
        # holding more than one event). Kept short rather than the full
        # "+"-joined event-name string, which can get arbitrarily long (and
        # blow past OS filename limits) once many events are batched into
        # one group; batch_to_sources (below) recovers the full mapping.
        substep_diagnostics: dict[str, dict[str, list[float] | list[bool]]] = {}
        # Maps each batch_key to the list of event names it contains --
        # the same information as event_groups_names, just keyed by the
        # short batch label for easy lookup/reporting.
        batch_to_sources: dict[str, list[str]] = {}
        # One-shot full-jump ESS predictor checks made by _next_auto_group
        # (auto-cadence only): one entry per not-yet-assimilated event
        # considered, in the order it was added to the pending queue,
        # recording the resulting ESS and whether it dropped below
        # auto_ess_threshold (triggering that batch to be assimilated) --
        # see _plot_auto_cadence_ess.
        auto_cadence_ess_checks: list[tuple[float, bool]] = []

        auto_cadence = config.is_auto_cadence
        n_new_events = n_events - n_prev_events
        # auto_ess_threshold is only read by _next_auto_group(), which is
        # only ever called below when auto_cadence is True -- the 0.0
        # placeholder in the non-auto branch is never consulted.
        if auto_cadence:
            event_groups: list[list[int]] | None = None
            auto_ess_threshold: float = config.resolved_auto_ess_threshold
            new_event_indices = list(range(n_prev_events, n_events))
        else:
            event_groups = self._compute_event_groups(n_prev_events)
            auto_ess_threshold = 0.0
            new_event_indices = []
        event_groups_names: list[list[str]] = []

        logger.info("=" * 70)
        logger.info("STARTING DATA TEMPERING (PATH OF PARTIAL POSTERIORS)")
        logger.info("=" * 70)
        logger.info(f"Kernel: {self._get_kernel_name().upper()}")
        logger.info(f"Particles: {n_particles}")
        logger.info(f"Events: {n_events} ({self._event_order})")
        if auto_cadence:
            logger.info(
                f"Cadence: auto (full-jump ESS threshold={auto_ess_threshold}) "
                f"-> data-tempering steps determined dynamically over "
                f"{n_new_events} new event(s), batches assigned as processed"
            )
        else:
            assert event_groups is not None
            logger.info(
                f"Cadence: {config.cadence} -> {len(event_groups)} data-tempering "
                f"step(s) (batches) over {n_new_events} new event(s)"
            )
        if n_prev_events > 0:
            logger.info(
                f"Warm start: skipping {n_prev_events} already-covered "
                f"event(s), prev logZ={warm_start_log_evidence:.3f}"
            )
        logger.info(f"Target ESS per sub-step: {config.target_ess}")
        logger.info(f"MCMC steps per sub-step: {config.n_mcmc_steps}")
        logger.info("=" * 70)

        zero_log_weights = jnp.zeros(n_particles)

        def _next_auto_group(cursor: int) -> tuple[list[int], int]:
            """Grow the pending queue of not-yet-assimilated events one at a
            time, checking the full-jump ESS (via ``jitted_ess_check_fn``,
            see its docstring) of the queue against the *current* particle
            cloud (``state``, read from the enclosing scope -- reflects
            whatever the most recently completed data-tempering step left
            it as) after each addition. Returns as soon as that ESS drops
            below ``auto_ess_threshold`` (an "update now" decision) or the
            last configured event is reached (nothing left to defer to).
            Every check made (whether or not it triggers) is appended to
            the enclosing ``auto_cadence_ess_checks`` list, for the
            ``_plot_auto_cadence_ess`` diagnostic plot.

            ``queue_mask`` is a fixed-``(n_events,)``-shape indicator
            array, flipped ``0 -> 1`` as events are added to the queue --
            not a variable-length ``jnp.array(pending)`` gather -- so this
            reuses the same compiled executable regardless of how long the
            pending queue grows (see ``_build_jitted_ess_check_fn``).
            """
            pending: list[int] = []
            queue_mask = jnp.zeros(n_events)
            while cursor < len(new_event_indices):
                idx = new_event_indices[cursor]
                pending.append(idx)
                queue_mask = queue_mask.at[idx].set(1.0)
                cursor += 1
                _, ess_hat_arr = jitted_ess_check_fn(
                    cast(Array, state.particles),
                    zero_log_weights,
                    queue_mask,
                    config.target_ess,
                    1.0,
                )
                ess_hat = float(ess_hat_arr)
                triggered = ess_hat < auto_ess_threshold
                auto_cadence_ess_checks.append((ess_hat, triggered))
                if triggered:
                    break
            return pending, cursor

        auto_cursor = 0
        group_num = -1
        # Tracks whether state.weights currently represents a genuinely
        # accumulated, un-resampled importance ratio from one or more prior
        # cheap reweight-only (skip) steps -- as opposed to the fresh,
        # single-sub-step ratio a real resample+move just produced (which
        # jitted_substep_fn's own next call would discard via resampling
        # regardless of its magnitude). Only the former needs to be fed into
        # the next sub-step's ESS bisection as a non-trivial baseline -- see
        # the base_log_weights computation below.
        weights_are_pending_reweight_only = False
        while True:
            group_num += 1
            if auto_cadence:
                group, auto_cursor = _next_auto_group(auto_cursor)
                if len(group) == 0:
                    break
            else:
                assert event_groups is not None
                if group_num >= len(event_groups):
                    break
                group = event_groups[group_num]
            group_names = [self._event_order[i] for i in group]
            group_name = "+".join(group_names)
            batch_number = n_prev_batches + group_num + 1
            batch_key = f"batch_{batch_number:02d}"
            logger.info(f"{batch_key}: {group_name}")
            event_groups_names.append(group_names)
            batch_to_sources[batch_key] = group_names
            group_indices = jnp.array(group)
            group_one_hot = jnp.zeros(n_events).at[group_indices].set(1.0)
            group_log_evidence = 0.0
            info = None
            n_substeps_taken = 0
            n_skips_taken = 0

            mask_fraction_history: list[float] = []
            substep_ess_history: list[float] = []
            substep_acceptance_history: list[float] = []
            substep_log_evidence_history: list[float] = []
            skipped_history: list[bool] = []

            while float(mask[group[0]]) < 1.0:
                current_frac = float(mask[group[0]])
                max_delta = 1.0 - current_frac

                # base_log_weights=0 recovers ess_solver's own implicit
                # assumption that incoming particles are already uniformly
                # weighted -- always true right after a real resample+move
                # (every substep resamples+moves when skip_move_when_ess_ok
                # is False; and even when the flag is on, blackjax's own
                # weight_fn recomputes state.weights from scratch out of
                # that one substep's incremental ratio -- see
                # _build_jitted_substep_fn's docstring and
                # blackjax.smc.base.step -- so whatever ESS it landed at is
                # about to be discarded by the *next* real move's resample
                # step regardless of magnitude, and must not be fed forward
                # here as extra baseline degeneracy). Only when the *last*
                # sub-step actually taken was itself a cheap reweight-only
                # skip does state.weights hold a genuinely un-resampled,
                # accumulating importance ratio that this bisection needs to
                # account for -- see _build_jitted_ess_check_fn and
                # _build_jitted_reweight_only_step_fn.
                base_log_weights = (
                    jnp.log(state.weights)
                    if (config.skip_move_when_ess_ok and weights_are_pending_reweight_only)
                    else zero_log_weights
                )
                raw_delta, ess_at_max_delta = jitted_ess_check_fn(
                    cast(Array, state.particles),
                    base_log_weights,
                    group_one_hot,
                    config.target_ess,
                    max_delta,
                )

                if (
                    config.skip_move_when_ess_ok
                    and float(ess_at_max_delta) >= config.target_ess
                ):
                    # The entire remaining jump for this group is cheap and
                    # safe -- accumulate the reweighting only (Chopin's
                    # IBIS Algorithm 17.2 skip branch), deferring the real
                    # resample+move. state.data_mask stays in sync with
                    # what state.weights represent (see
                    # _build_jitted_reweight_only_step_fn), so if a real
                    # move fires later (in this group or a subsequent one --
                    # the outer group loop just continues, naturally
                    # extending the deferral across event boundaries too),
                    # jitted_substep_fn below runs completely unmodified.
                    new_frac = 1.0
                    new_mask = mask.at[group_indices].set(new_frac)
                    state, dlogZ_arr = jitted_reweight_only_step_fn(state, new_mask)
                    substep_log_evidence_increment = float(dlogZ_arr)
                    mask = new_mask
                    group_log_evidence += substep_log_evidence_increment
                    n_substeps_taken += 1
                    n_skips_taken += 1
                    weights_are_pending_reweight_only = True

                    substep_ess = float(
                        jnp.sum(state.weights) ** 2
                        / jnp.sum(state.weights**2)
                        / n_particles
                    )
                    bar = make_progress_bar(new_frac)
                    logger.info(
                        f"    -> [{batch_key}] sub-step {n_substeps_taken:2d} | "
                        f"mask={new_frac:.5f} | ESS={substep_ess * 100:5.1f}% | "
                        f"SKIPPED (reweight-only, no move) | "
                        f"dlogZ={substep_log_evidence_increment:8.3f} | {bar}"
                    )

                    mask_fraction_history.append(new_frac)
                    substep_ess_history.append(substep_ess)
                    substep_acceptance_history.append(float("nan"))
                    substep_log_evidence_history.append(
                        log_evidence + group_log_evidence
                    )
                    skipped_history.append(True)
                    continue

                if not jnp.isfinite(raw_delta):
                    # Particles are already below target_ess even at
                    # delta=0 (dichotomy has no admissible root) --
                    # fall back to a minimal fractional increment rather
                    # than stalling, to stay on the safe (small-jump) side.
                    logger.warning(
                        f"{batch_key}: ESS already below "
                        "target_ess before this sub-step -- falling "
                        "back to a minimal fractional increment."
                    )
                    delta = min(max_delta, _MIN_FALLBACK_FRACTION)
                else:
                    delta = float(jnp.clip(raw_delta, 0.0, max_delta))
                new_frac = current_frac + delta

                new_mask = mask.at[group_indices].set(new_frac)
                key, subkey, update_key = jax.random.split(key, 3)
                substep_start_time = time.time()
                state, info = jitted_substep_fn(subkey, state, new_mask, mcmc_params)
                mcmc_params = mcmc_parameter_update_fn(
                    update_key, mcmc_params, state, info
                )
                substep_elapsed = time.time() - substep_start_time
                mask = new_mask
                substep_log_evidence_increment = float(info.log_likelihood_increment)
                group_log_evidence += substep_log_evidence_increment
                n_substeps_taken += 1
                weights_are_pending_reweight_only = False

                substep_ess = float(
                    jnp.sum(state.weights) ** 2
                    / jnp.sum(state.weights**2)
                    / n_particles
                )
                substep_acceptance = float(info.update_info.acceptance_rate.mean())  # type: ignore[attr-defined]
                bar = make_progress_bar(new_frac)
                logger.info(
                    f"    -> [{batch_key}] sub-step {n_substeps_taken:2d} | "
                    f"mask={new_frac:.5f} | ESS={substep_ess * 100:5.1f}% | "
                    f"Accept={substep_acceptance * 100:5.1f}% | "
                    f"dlogZ={substep_log_evidence_increment:8.3f} | "
                    f"t={substep_elapsed:6.2f}s | {bar}"
                )

                mask_fraction_history.append(new_frac)
                substep_ess_history.append(substep_ess)
                substep_acceptance_history.append(substep_acceptance)
                substep_log_evidence_history.append(log_evidence + group_log_evidence)
                skipped_history.append(False)

            substep_diagnostics[batch_key] = {
                "mask_fraction_history": mask_fraction_history,
                "ess_history": substep_ess_history,
                "acceptance_history": substep_acceptance_history,
                "log_evidence_history": substep_log_evidence_history,
                "skipped_history": skipped_history,
            }

            weights = state.weights
            ess_value = float(jnp.sum(weights) ** 2 / jnp.sum(weights**2) / n_particles)
            # A group can finish via reweight-only skips alone (no real move
            # at all, so info stays None) -- acceptance rate is undefined in
            # that case, not zero; recorded as NaN and ignored by
            # _build_metadata's nanmean.
            acceptance_rate = (
                float(info.update_info.acceptance_rate.mean())  # type: ignore[attr-defined]
                if info is not None
                else float("nan")
            )
            log_evidence += group_log_evidence

            ess_history.append(ess_value)
            acceptance_history.append(acceptance_rate)
            log_evidence_history.append(log_evidence)
            n_substeps_history.append(n_substeps_taken)

            elapsed_str = format_elapsed(time.time() - start_time)
            events_done = group[-1] + 1 - n_prev_events
            progress_fraction = events_done / n_new_events if n_new_events > 0 else 1.0
            bar = make_progress_bar(progress_fraction)
            logger.info(
                f"Step {events_done:3d}/{n_new_events} [{batch_key}] | "
                f"ESS={ess_value * 100:5.1f}% | Accept={acceptance_rate * 100:5.1f}% | "
                f"logZ={log_evidence:8.3f} | substeps={n_substeps_taken:3d} "
                f"(skipped={n_skips_taken}) | t={elapsed_str} | {bar}"
            )

            if config.save_intermediate_results:
                key, save_key = jax.random.split(key)
                self._save_intermediate_result(
                    key=save_key,
                    state=state,
                    config=config,
                    n_particles=n_particles,
                    logposterior_fn=partial_logposterior_factory(state.data_mask),
                    mcmc_step_fn=mcmc_step_fn,
                    mcmc_init_fn=mcmc_init_fn,
                    mcmc_parameters=mcmc_params,
                    event_order_so_far=self._event_order[: group[-1] + 1],
                    event_groups_names_so_far=event_groups_names,
                    n_prev_events=n_prev_events,
                    n_prev_batches=n_prev_batches,
                    batch_to_sources_so_far=batch_to_sources,
                    ess_history_so_far=ess_history,
                    acceptance_history_so_far=acceptance_history,
                    log_evidence_history_so_far=log_evidence_history,
                    n_substeps_history_so_far=n_substeps_history,
                    auto_cadence_ess_checks_so_far=auto_cadence_ess_checks,
                    log_evidence_so_far=log_evidence,
                    elapsed_seconds=time.time() - start_time,
                    batch_key=batch_key,
                )

        end_time = time.time()

        self._particles_flat = cast(Array, state.particles)
        self._weights = state.weights
        self.final_state = state

        assert self._weights is not None
        final_ess_fraction = float(
            jnp.sum(self._weights) ** 2 / jnp.sum(self._weights**2) / n_particles
        )

        key, resample_key = jax.random.split(key)
        resample_idx = systematic(resample_key, self._weights, n_particles)
        self._particles_flat = self._particles_flat[resample_idx]
        self._weights = jnp.ones(n_particles) / n_particles

        # Optional final rejuvenation, mirroring smc/base.py's terminal
        # step -- see rejuvenation.py's module docstring. By this point
        # every configured event's mask has reached 1.0, so
        # full_logposterior_fn (built earlier in this method) is exactly
        # the fixed final target. Disabled by default
        # (n_final_rejuvenation_steps=0).
        if config.n_final_rejuvenation_steps > 0:
            key, rejuvenation_key = jax.random.split(key)
            self._particles_flat = rejuvenate_particles(
                rejuvenation_key,
                self._particles_flat,
                full_logposterior_fn,
                mcmc_step_fn,
                mcmc_init_fn,
                mcmc_params,
                config.n_final_rejuvenation_steps,
            )

        self.metadata = self._build_metadata(
            config=config,
            n_particles=n_particles,
            event_order_so_far=self._event_order,
            event_groups_names_so_far=event_groups_names,
            n_prev_events=n_prev_events,
            n_prev_batches=n_prev_batches,
            batch_to_sources_so_far=batch_to_sources,
            final_ess_fraction=final_ess_fraction,
            ess_history_so_far=ess_history,
            acceptance_history_so_far=acceptance_history,
            log_evidence_history_so_far=log_evidence_history,
            n_substeps_history_so_far=n_substeps_history,
            auto_cadence_ess_checks_so_far=auto_cadence_ess_checks,
            log_evidence_so_far=log_evidence,
            elapsed_seconds=end_time - start_time,
        )
        self._substep_diagnostics = substep_diagnostics

    def plot_diagnostics(
        self, outdir: str | Path = ".", filename: str = "smc_diagnostics.png"
    ) -> None:
        """Generate diagnostic plots for partial-posteriors SMC.

        Produces two kinds of plots:

        1. The overall partial-posteriors-path plot (``filename``): a
           3-panel figure showing, per data-tempering step (one or more
           events assimilated jointly, per ``config.cadence``) instead of
           per individual event, effective sample size, acceptance rate,
           and cumulative log evidence -- the "full picture" of how the
           run progressed across steps.
        2. One per-step sub-step diagnostics plot (see
           :meth:`_plot_substep_diagnostics`), mirroring
           ``BlackjaxSMCSampler.plot_diagnostics``'s 3-panel style
           (tempering/mask-fraction schedule, ESS, acceptance) but for the
           adaptive ramp-in of that step's event group, saved under
           ``outdir/substep_diagnostics/``.

        Parameters
        ----------
        outdir : str or Path, optional
            Output directory for saving the plot (default: current directory)
        filename : str, optional
            Filename for the overall diagnostic plot (default: "smc_diagnostics.png")
        """
        if self.final_state is None:
            logger.warning("No samples yet - run sample() first")
            return

        # event_groups already covers only the steps actually taken by this
        # run's sample() call (warm-started/already-covered events are
        # skipped entirely by _compute_event_groups, not included here), so
        # no replay slicing is needed -- unlike event_order which is the
        # full configured list.
        event_groups = self.metadata["event_groups"]
        ess_history = self.metadata["ess_history"]
        acceptance_history = self.metadata["acceptance_history"]
        log_evidence_history = self.metadata["log_evidence_history"]
        n_steps = len(event_groups)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        kernel_name = self._get_kernel_name().upper()
        fig.suptitle(
            f"Partial-Posteriors SMC Diagnostics ({kernel_name} kernel)",
            fontsize=14,
            fontweight="bold",
        )

        x = range(1, n_steps + 1)

        ess_percent = [ess * 100 for ess in ess_history]
        axes[0].plot(x, ess_percent, "g-o", linewidth=2)
        axes[0].set_ylabel("ESS (%)", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)

        acceptance_percent = [acc * 100 for acc in acceptance_history]
        axes[1].plot(
            x, acceptance_percent, "orange", linestyle="-", marker="o", linewidth=2
        )
        axes[1].set_ylabel("Acceptance Rate (%)", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 105)

        axes[2].plot(x, log_evidence_history, "b-o", linewidth=2)
        axes[2].set_ylabel(r"Cumulative $\log Z$", fontsize=12)
        axes[2].set_xlabel("Data-tempering step", fontsize=12)
        axes[2].grid(True, alpha=0.3)

        group_labels = list(self.metadata["batch_to_sources"].keys())
        axes[2].set_xticks(list(x))
        axes[2].set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)

        plt.tight_layout()

        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        output_path = outdir_path / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved diagnostic plot to {output_path}")
        plt.close(fig)

        self._plot_substep_diagnostics(outdir_path)
        self._plot_auto_cadence_ess(outdir_path)

    def _plot_substep_diagnostics(self, outdir_path: Path) -> None:
        """Plot one SMC-diagnostics-style figure per data-tempering step.

        Mirrors ``BlackjaxSMCSampler.plot_diagnostics`` (3 panels: tempering
        schedule, ESS, acceptance rate), but replaces the inverse
        temperature :math:`\\lambda` with the event group's shared mask
        fraction (0 to 1) and the x-axis annealing-step index with the
        sub-step index within that group's ramp-in. This gives the full
        per-step picture to complement the across-steps overview plot.

        Parameters
        ----------
        outdir_path : Path
            Directory under which a ``substep_diagnostics/`` subdirectory
            is created to hold one PNG per data-tempering step.
        """
        if not self._substep_diagnostics:
            return

        substep_outdir = outdir_path / "substep_diagnostics"
        substep_outdir.mkdir(parents=True, exist_ok=True)

        target_ess = cast(
            SMCPartialPosteriorsRandomWalkSamplerConfig, self.config
        ).target_ess
        kernel_name = self._get_kernel_name().upper()

        # batch_to_sources is keyed by the same sequential batch_key used to
        # populate self._substep_diagnostics (see sample()), so iterating it
        # directly recovers both the key and its event names without any
        # extra lookup -- unlike the pre-sequential-numbering scheme, a
        # batch's number no longer depends on its first event's absolute
        # position in event_order.
        for batch_key in self.metadata["batch_to_sources"]:
            diagnostics = self._substep_diagnostics.get(batch_key)
            if diagnostics is None:
                # Should not happen -- event_groups only lists groups
                # actually processed this run (see _compute_event_groups).
                continue

            mask_fraction_history = diagnostics["mask_fraction_history"]
            ess_history = diagnostics["ess_history"]
            acceptance_history = diagnostics["acceptance_history"]
            skipped_history = diagnostics.get("skipped_history", [])
            n_substeps = len(mask_fraction_history)

            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            fig.suptitle(
                f"[{batch_key}] Sub-step Diagnostics ({kernel_name} kernel)",
                fontsize=14,
                fontweight="bold",
            )

            x = range(1, n_substeps + 1)

            axes[0].plot(x, mask_fraction_history, "b-o", linewidth=2)
            # skip_move_when_ess_ok only: mark sub-steps that took the cheap
            # reweight-only branch (no resample, no MCMC move) with a
            # distinct red 'x', so a real run's realized skip rate is
            # visible at a glance -- see the ESS-gated-skip optimization
            # plan's Phase 2 diagnostics ask.
            skip_x = [xi for xi, s in zip(x, skipped_history) if s]
            skip_y = [yi for yi, s in zip(mask_fraction_history, skipped_history) if s]
            if skip_x:
                axes[0].plot(
                    skip_x,
                    skip_y,
                    "rx",
                    markersize=10,
                    markeredgewidth=2,
                    zorder=3,
                    label="skipped (no move)",
                )
                axes[0].legend(loc="best", fontsize=9)
            axes[0].set_ylabel("Mask fraction", fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(-0.05, 1.05)
            axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
            axes[0].axhline(y=1, color="black", linestyle="--", alpha=0.3, linewidth=1)

            ess_percent = [ess * 100 for ess in ess_history]
            axes[1].plot(x, ess_percent, "g-o", linewidth=2)
            axes[1].axhline(
                y=target_ess * 100,
                color="black",
                linestyle="--",
                alpha=0.5,
                linewidth=1.5,
                label=f"Target ({target_ess * 100:.0f}%)",
            )
            axes[1].set_ylabel("ESS (%)", fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="best", fontsize=10)
            axes[1].set_ylim(0, 105)

            acceptance_percent = [acc * 100 for acc in acceptance_history]
            axes[2].plot(
                x,
                acceptance_percent,
                "orange",
                linestyle="-",
                marker="o",
                linewidth=2,
            )
            axes[2].set_ylabel("Acceptance Rate (%)", fontsize=12)
            axes[2].set_xlabel("Sub-step", fontsize=12)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim(0, 105)

            plt.tight_layout()

            output_path = substep_outdir / f"{batch_key}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved sub-step diagnostic plot to {output_path}")
            plt.close(fig)

    def _plot_auto_cadence_ess(self, outdir_path: Path) -> None:
        """Plot the auto-cadence full-jump ESS predictor's decision trace.

        One point per not-yet-assimilated event considered by
        ``_next_auto_group`` during ``sample()``, in the order it was added
        to the pending queue: the resulting one-shot full-jump ESS
        (``_predict_full_jump_ess``) against the current particle cloud,
        and whether it dropped below ``auto_ess_threshold`` and thereby
        triggered that queue to be assimilated (vs. staying queued/deferred
        for the next event to be added). No-op when ``cadence`` isn't
        ``"auto"``/``"automatic"`` (``auto_cadence_ess_history`` is empty).

        Parameters
        ----------
        outdir_path : Path
            Directory the plot (``auto_cadence_ess.png``) is saved under.
        """
        ess_history = self.metadata.get("auto_cadence_ess_history", [])
        if not ess_history:
            return
        triggered_history = self.metadata["auto_cadence_triggered_history"]
        threshold = self.metadata["auto_ess_threshold"]

        x = list(range(1, len(ess_history) + 1))
        deferred_x = [xi for xi, t in zip(x, triggered_history) if not t]
        deferred_y = [yi for yi, t in zip(ess_history, triggered_history) if not t]
        triggered_x = [xi for xi, t in zip(x, triggered_history) if t]
        triggered_y = [yi for yi, t in zip(ess_history, triggered_history) if t]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x, ess_history, "-", color="tab:blue", linewidth=1.5, zorder=1)
        ax.plot(
            deferred_x,
            deferred_y,
            "o",
            color="tab:blue",
            zorder=2,
            label="deferred (queued)",
        )
        ax.plot(
            triggered_x,
            triggered_y,
            "v",
            color="red",
            markersize=10,
            zorder=3,
            label="triggered update",
        )
        ax.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"threshold={threshold:g}",
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("event added to queue")
        ax.set_ylabel("queue one-shot ESS")
        ax.set_title(f"auto-cadence threshold = {threshold:g}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)

        plt.tight_layout()
        output_path = outdir_path / "auto_cadence_ess.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved auto-cadence ESS plot to {output_path}")
        plt.close(fig)
