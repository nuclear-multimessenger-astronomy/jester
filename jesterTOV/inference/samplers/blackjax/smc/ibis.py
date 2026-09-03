r"""IBIS (Iterated Batch Importance Sampling, Chopin 2002) hybridized with
likelihood tempering -- informally "partial-posteriors SMC" (YAML
``type: "smc-pp"``, config class ``SMCPartialPosteriorsSamplerConfig``).

GW events are assimilated one at a time into the posterior. From the current
(unweighted, i.i.d.) particle set, each new event's log-likelihood is added
to a running per-particle importance weight -- cheap, since it only needs
that one event's likelihood, evaluated at flat, catalog-size-independent
cost via :meth:`~jesterTOV.inference.likelihoods.gw.StackedGWLikelihood.evaluate_single_event`
(see :meth:`BlackJAXIBISSampler._cheap_reweight_walk_incremental`).
As long as the resulting effective sample size stays at or above
``ess_threshold * n_particles``, this reweighting is accepted and the next
event is queued the same way, with no resampling and no particle movement.
Once ESS would drop below threshold (or the event list is exhausted), the
entire queued span of events -- from the last real (moved) particle set --
is assimilated in one shot via a full adaptive-tempered-SMC batch (lambda:
0 -> 1), reusing :meth:`~jesterTOV.inference.samplers.blackjax.smc.base.BlackjaxSMCSampler._run_tempering`
generalized to start from an arbitrary particle set and an arbitrary
(logprior, loglikelihood) pair rather than always the prior. This is *not*
vanilla IBIS (which resample-moves after every single datapoint) -- the
ESS-triggered skip/batch hybrid described here is per
``chapter3_data_analysis.tex`` (PhD-thesis, "Data tempering" section)
[CITATION NEEDED -- exact reference to be filled in]. It also
supersedes an earlier, more complex implementation of this idea (data
tempering via a fractional per-event mask ramped in through
``blackjax.smc.partial_posteriors_path``, with a nested ESS-bisection
schedule within each event) -- that machinery is not reused here; this
module instead anneals whole batches of events at once via ordinary
whole-batch adaptive-lambda tempering, which is both simpler and (per the
ESS-triggered batching itself) already only invoked when a single-shot
reweight would not have been reliable anyway.

Scope (v1): random-walk inner kernel only (no NUTS). Only the batched
``StackedGWLikelihood`` (``type: "gw"``) path is supported as an event
source -- ``GWLikelihoodResampled`` (``type: "gw_resampled"``) is not
supported and is rejected with a clear error (see
:func:`split_event_and_background_likelihoods`); that likelihood resamples
its mass grid fresh on every call, which is incompatible with the
per-event, dynamically-indexed evaluation this module relies on
(:meth:`~jesterTOV.inference.likelihoods.gw.StackedGWLikelihood.evaluate_single_event`,
:meth:`~jesterTOV.inference.likelihoods.gw.StackedGWLikelihood.subset`). No
warm-starting/resuming a run from a saved state -- distinct from the
per-batch ``InferenceResult`` snapshots written for inspection when
``config.save_intermediate_results`` is ``True`` (the default; see
:meth:`BlackJAXIBISSampler.configure_intermediate_saving`), which are not
themselves resumable.

Scaling note (resolved): earlier versions of this module (a) precomputed a
full ``n_particles x n_events`` per-event log-likelihood matrix, over
*every* configured event, every time the particle set changed, for cheap
reweighting, and (b) built each real SMC batch's tempered target by
evaluating *every* configured event's likelihood and only slicing out the
needed span afterwards. Both meant cost scaled with the full catalog size
regardless of how few events a given check/batch actually needed -- fine for
tens of events, badly wrong for O(1000)-event catalogs (ET/CE), where it
dominates runtime (see ``debug/new_jester_pp_smc/slow_pp_smc/FINDINGS.md`` in
the parent project for the diagnosis). Fixed by scoping every likelihood
evaluation to only the events actually needed:
:meth:`BlackJAXIBISSampler._cheap_reweight_walk_incremental` evaluates one
event at a time via ``evaluate_single_event`` (flat, O(1 event) cost, no
retracing across events -- see that method's docstring), and
:meth:`BlackJAXIBISSampler._make_batch_logprior_fn`/
:meth:`~BlackJAXIBISSampler._make_batch_loglikelihood_fn` build their
absorbed-events/new-batch-events sums via ``StackedGWLikelihood.subset()``,
so a batch's cost scales with ``absorbed_upto`` and ``stop - start``, not
with the total number of configured events.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from jesterTOV.inference.base import (
    LikelihoodBase,
    Prior,
    BijectiveTransform,
    NtoMTransform,
)
from jesterTOV.inference.config.schema import (
    InferenceConfig,
    SMCPartialPosteriorsSamplerConfig,
)
from jesterTOV.inference.likelihoods.combined import CombinedLikelihood, ZeroLikelihood
from jesterTOV.inference.likelihoods.gw import (
    GWLikelihoodResampled,
    StackedGWLikelihood,
)
from jesterTOV.inference.samplers.jester_sampler import SamplerOutput
from jesterTOV.inference.samplers.blackjax.base import BlackjaxSampler
from jesterTOV.inference.samplers.blackjax.smc.base import TemperingResult
from jesterTOV.inference.samplers.blackjax.smc.random_walk import (
    BlackJAXSMCRandomWalkSampler,
)
from jesterTOV.inference.samplers.blackjax.smc.rejuvenation import rejuvenate_particles
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

__all__ = [
    "BlackJAXIBISSampler",
    "split_event_and_background_likelihoods",
]


def split_event_and_background_likelihoods(
    likelihood: LikelihoodBase,
) -> tuple[LikelihoodBase, StackedGWLikelihood]:
    """Split a (possibly Combined) likelihood into its non-GW "always-on"
    background piece and its single ``StackedGWLikelihood`` piece.

    Parameters
    ----------
    likelihood : LikelihoodBase
        Whatever ``create_combined_likelihood(config.likelihoods)`` produced
        -- either a bare ``StackedGWLikelihood`` (if it was the only enabled
        likelihood), or a ``CombinedLikelihood`` whose ``likelihoods_list``
        contains exactly one ``StackedGWLikelihood`` plus zero or more other
        likelihoods.

    Returns
    -------
    background : LikelihoodBase
        ``CombinedLikelihood`` over every non-GW constituent (``ZeroLikelihood``
        if none), i.e. everything except the ``StackedGWLikelihood``.
    stacked_gw : StackedGWLikelihood
        The single GW likelihood, exposing ``evaluate_per_event`` and
        ``event_names`` (assimilation order, fixed at construction time to
        config order).

    Raises
    ------
    ValueError
        If no ``StackedGWLikelihood`` is present, if more than one is
        present, or if a ``GWLikelihoodResampled`` instance is present
        anywhere (smc-pp/IBIS only supports the ``type: "gw"`` /
        ``StackedGWLikelihood`` path -- ``type: "gw_resampled"`` is
        unsupported).
    """
    constituents: list[LikelihoodBase] = (
        list(likelihood.likelihoods_list)
        if isinstance(likelihood, CombinedLikelihood)
        else [likelihood]
    )

    resampled = [c for c in constituents if isinstance(c, GWLikelihoodResampled)]
    if resampled:
        raise ValueError(
            "smc-pp (IBIS) does not support GWLikelihoodResampled "
            "(type: 'gw_resampled'). Use type: 'gw' (StackedGWLikelihood) "
            "for all GW events, or use a different sampler."
        )

    stacked = [c for c in constituents if isinstance(c, StackedGWLikelihood)]
    if len(stacked) == 0:
        raise ValueError(
            "smc-pp (IBIS) requires exactly one GW likelihood block "
            "(type: 'gw') to assimilate as events; none found in config."
        )
    if len(stacked) > 1:
        raise ValueError(
            "smc-pp (IBIS) requires exactly one 'gw' likelihood block; "
            f"found {len(stacked)}. Combine all events into a single "
            "type: 'gw' block's events list."
        )
    stacked_gw = stacked[0]

    background_list = [c for c in constituents if c is not stacked_gw]
    background: LikelihoodBase
    if len(background_list) > 1:
        background = CombinedLikelihood(background_list)
    elif len(background_list) == 1:
        background = background_list[0]
    else:
        background = ZeroLikelihood()
    return background, stacked_gw


def _np_logsumexp(x: np.ndarray) -> float:
    """Plain-NumPy logsumexp (max-shifted for numerical stability), avoiding
    a scipy dependency for this small, hot inner-loop computation."""
    x_max = np.max(x)
    return float(x_max + np.log(np.sum(np.exp(x - x_max))))


def _cheap_reweight_step(
    log_w: np.ndarray, event_loglik: np.ndarray, alpha: float, n_particles: int
) -> tuple[bool, float, np.ndarray, float | None]:
    """One event of the cheap-reweight walk: given the current running
    log-weights and one event's per-particle log-likelihood, compute the
    candidate ESS and decide accept/stop.

    Factored out of :func:`_cheap_reweight_walk` so that function (dense
    matrix, unit-tested in isolation in ``test_ibis_bookkeeping.py`` against
    analytic/brute-force ground truth) and
    :meth:`BlackJAXIBISSampler._cheap_reweight_walk_incremental` (evaluates
    one event's log-likelihood at a time, at flat cost, instead of a
    precomputed matrix -- see that method's docstring) share the exact same
    accept/stop rule rather than risking the two drifting apart.

    Parameters
    ----------
    log_w : np.ndarray
        Current running (un-normalized) log-weights, shape ``(n_particles,)``.
    event_loglik : np.ndarray
        This event's per-particle log-likelihood, shape ``(n_particles,)``.
    alpha : float
        ESS threshold fraction.
    n_particles : int
        Number of particles (``len(log_w)``, passed explicitly to avoid
        recomputing it every call).

    Returns
    -------
    accepted : bool
        Whether ``ess_fraction >= alpha`` -- i.e. whether this event can be
        cheaply absorbed.
    ess_fraction : float
        ESS fraction after tentatively adding this event.
    log_w_candidate : np.ndarray
        Running log-weights after tentatively adding this event (the caller
        should only carry this forward as the new ``log_w`` if ``accepted``).
    logZ_increment : float or None
        Cheap-reweight log-evidence increment if accepted, else ``None``.
    """
    log_w_candidate = log_w + event_loglik
    log_ess = 2 * _np_logsumexp(log_w_candidate) - _np_logsumexp(2 * log_w_candidate)
    ess_fraction = float(np.exp(log_ess)) / n_particles
    if ess_fraction < alpha:
        return False, ess_fraction, log_w_candidate, None
    logZ_increment = _np_logsumexp(log_w_candidate) - _np_logsumexp(log_w)
    return True, ess_fraction, log_w_candidate, logZ_increment


def _cheap_reweight_walk(
    event_matrix: np.ndarray, alpha: float
) -> tuple[int, np.ndarray, list[float], list[float]]:
    """Plain-NumPy walk deciding how many of ``event_matrix``'s columns
    (events, in assimilation order starting at column 0) can be cheaply
    absorbed by importance reweighting alone before ESS drops below
    ``alpha * n_particles``.

    Factored out as a standalone pure function (no JAX tracing, no sampler
    state) so it is unit-testable in isolation against analytic/brute-force
    ground truth.

    Parameters
    ----------
    event_matrix : np.ndarray
        Shape ``(n_particles, n_events_remaining)`` -- per-particle,
        per-event log-likelihoods for the events still to be assimilated, in
        assimilation order, for the current (unmoved) particle set.
    alpha : float
        ESS threshold fraction (the ``ess_threshold`` config field).

    Returns
    -------
    m : int
        Number of columns checked before stopping (see Notes) -- the batch
        the caller runs must absorb all ``m`` of them.
    log_w_final : np.ndarray
        Un-normalized running log-weights after accepting the checked
        columns that passed (excludes the failing column, if any -- see
        Notes). Shape ``(n_particles,)``. Diagnostic/testing use only; not
        used to weight the sampler's actual stored particles (see
        :meth:`BlackJAXIBISSampler.sample`'s docstring for why).
    ess_trace : list[float]
        ESS fraction computed for each of the ``m`` checked columns, in
        order.
    logZ_increment_trace : list[float]
        Cheap-reweight log-evidence increment for each *accepted*
        (ESS-passing) column among the ``m`` checked -- diagnostic only.

    Notes
    -----
    Walks columns 0, 1, 2, ... of ``event_matrix``. After tentatively adding
    each column to the running log-weights, checks ESS. Two ways to stop:

    - ESS drops below ``alpha * n_particles`` on column ``j``: stop, having
      accepted columns ``0..j-1`` (``j`` of them) plus checked (but not
      accepted) column ``j`` itself -- ``m = j + 1`` total columns checked;
      the caller's batch must cover all ``m`` of them (this is the event
      that "triggered" the batch).
    - Every column clears the threshold (no drop): ``m =
      event_matrix.shape[1]`` (all columns checked and accepted); the
      caller's batch must still cover all of them, per the algorithm's "run
      one final batch over the remaining tail" rule -- stored particles must
      always be real (moved), never raw importance-weighted ones.

    In both cases the caller runs exactly one real SMC batch over all ``m``
    checked columns -- this function only decides *how many* events belong
    in that batch; it never itself constitutes the final assimilation of any
    event.
    """
    n_particles, n_remaining = event_matrix.shape
    log_w = np.zeros(n_particles)
    ess_trace: list[float] = []
    logZ_increment_trace: list[float] = []

    for j in range(n_remaining):
        accepted, ess_fraction, log_w_candidate, logZ_increment = _cheap_reweight_step(
            log_w, event_matrix[:, j], alpha, n_particles
        )
        ess_trace.append(ess_fraction)

        if not accepted:
            return j + 1, log_w, ess_trace, logZ_increment_trace

        assert logZ_increment is not None  # accepted implies a real increment
        logZ_increment_trace.append(logZ_increment)
        log_w = log_w_candidate

    return n_remaining, log_w, ess_trace, logZ_increment_trace


class BlackJAXIBISSampler(BlackjaxSampler):
    """IBIS (Chopin 2002) hybridized with likelihood tempering: assimilate GW
    events one at a time via cheap importance reweighting, falling back to a
    full adaptive-tempered-SMC batch whenever ESS would drop below
    ``ess_threshold * n_particles``. See the module docstring for the full
    algorithm and its relationship to vanilla IBIS / the superseded
    fractional-mask partial-posteriors-path implementation.

    Holds one inner :class:`BlackJAXSMCRandomWalkSampler` purely to reuse its
    ``_setup_mcmc_kernel``/``_run_tempering`` machinery (composition, not
    inheritance -- this sampler's own ``sample()`` loop is structurally
    nothing like a single lambda-anneal, so it does not subclass
    ``BlackjaxSMCSampler``).

    Parameters
    ----------
    likelihood : LikelihoodBase
        The full combined likelihood, as usual -- this sampler re-derives
        (background, stacked GW likelihood) from it via
        :func:`split_event_and_background_likelihoods`.
    prior : Prior
        Prior object.
    sample_transforms : list[BijectiveTransform]
        Should be empty (works in prior space, like plain SMC).
    likelihood_transforms : list[NtoMTransform]
        N-to-M transforms applied before likelihood evaluation.
    config : SMCPartialPosteriorsSamplerConfig
        IBIS configuration.
    seed : int, optional
        Random seed (default: 0).
    """

    config: SMCPartialPosteriorsSamplerConfig
    background: LikelihoodBase
    stacked_gw: StackedGWLikelihood
    event_names: list[str]
    metadata: dict
    final_state: Any | None
    _particles_flat: Array | None
    _weights: Array | None

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform],
        likelihood_transforms: list[NtoMTransform],
        config: SMCPartialPosteriorsSamplerConfig,
        seed: int = 0,
    ) -> None:
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)
        if len(sample_transforms) > 0:
            logger.warning(
                "IBIS sampler received sample transforms. IBIS typically "
                "works best without sample transforms (in prior space). "
                "Proceeding anyway."
            )

        self.config = config
        self.background, self.stacked_gw = split_event_and_background_likelihoods(
            likelihood
        )
        self.event_names = self.stacked_gw.event_names
        self._seed = seed

        self._inner = BlackJAXSMCRandomWalkSampler(
            likelihood=self.background,
            prior=prior,
            sample_transforms=sample_transforms,
            likelihood_transforms=likelihood_transforms,
            config=config.inner,
            seed=seed,
        )

        self.metadata = {}
        self.final_state = None
        self._particles_flat = None
        self._weights = None
        self._jitted_single_event_particle_fn: Any = None

        # Populated by configure_intermediate_saving(), called externally
        # (run_inference.py) because this sampler doesn't otherwise have
        # access to the full InferenceConfig (only its own sampler
        # sub-config) or the output directory. Only consumed if
        # config.save_intermediate_results is True -- see
        # _save_intermediate_result.
        self._intermediate_save_config: InferenceConfig | None = None
        self._intermediate_save_outdir: Path | None = None
        self._intermediate_save_fixed_params: dict[str, float] = {}

        logger.info(
            f"Initializing BlackJAX IBIS sampler: {len(self.event_names)} GW "
            f"events to assimilate, ess_threshold={config.ess_threshold}, "
            f"{config.inner.n_particles} particles"
        )

    def _transform_to_likelihood_space(self, named_params: dict) -> dict:
        """Apply inverse sample transforms then forward likelihood
        transforms, mirroring ``_create_loglikelihood_fn_from_dict``'s exact
        convention (``blackjax/base.py``) -- needed here since IBIS calls
        ``self.background``/``self.stacked_gw`` directly rather than going
        through ``self.likelihood``."""
        named_params = dict(named_params)
        for transform in reversed(self.sample_transforms):
            named_params, _ = transform.inverse(named_params)
        for transform in self.likelihood_transforms:
            named_params = transform.forward(named_params)
        return named_params

    def _get_single_event_particle_fn(self) -> Callable[[Array, Array | int], Array]:
        """Jitted, particle-batched evaluator for ONE event's GW
        log-likelihood, selected by a dynamic index.

        Built once (lazily, on first call) and cached. Because
        ``event_index`` is passed as an ordinary (traced) argument rather
        than baked into the function at trace time, this compiles ONCE for
        the whole run and can then be called with any event index -- early
        in the catalog or late -- without retracing (see
        ``StackedGWLikelihood.evaluate_single_event``'s docstring). Used by
        :meth:`_cheap_reweight_walk_incremental` to check one event's ESS
        impact at a time, at flat, catalog-size-independent cost -- this
        replaces the previous ``_evaluate_particles`` approach, which
        computed a dense ``(n_particles, n_events)`` matrix over *every*
        stacked event via ``evaluate_per_event`` up front, even though the
        cheap-reweight walk usually only needs to check a handful of events
        before hitting the ESS threshold (see the module-level scaling note
        above, now resolved by this method).
        """
        if self._jitted_single_event_particle_fn is None:

            def per_particle_single_event_loglik(
                x_flat: Array, event_index: Array | int
            ) -> Float:
                named_params = self._unflatten_fn(x_flat)
                transformed = self._transform_to_likelihood_space(named_params)
                return self.stacked_gw.evaluate_single_event(transformed, event_index)

            def fn(particles_flat: Array, event_index: Array | int) -> Array:
                return jax.lax.map(
                    lambda x_flat: per_particle_single_event_loglik(
                        x_flat, event_index
                    ),
                    particles_flat,
                    batch_size=self.config.particle_batch_size,
                )

            self._jitted_single_event_particle_fn = jax.jit(fn)
        return self._jitted_single_event_particle_fn

    def _cheap_reweight_walk_incremental(
        self,
        particles_flat: Float[Array, "n_particles n_dim"],
        start: int,
        alpha: float,
    ) -> tuple[int, list[float], list[float]]:
        """Event-by-event version of :func:`_cheap_reweight_walk`.

        Evaluates one event's per-particle log-likelihood at a time (via
        :meth:`_get_single_event_particle_fn`, flat O(1 event) cost)
        instead of precomputing a dense matrix over every remaining event,
        so a walk that stops after a handful of events never pays for the
        rest of the catalog -- and genuinely never computes an event past
        the one that triggers the stop. Uses the exact same accept/stop
        rule as :func:`_cheap_reweight_walk` (via the shared
        :func:`_cheap_reweight_step` helper), so behaviour (which/how many
        events get absorbed) is unchanged; only the cost profile is. Also
        logs ESS and this single step's wall-clock runtime after each event,
        via the jester logger, for live monitoring of long-running catalogs.

        Parameters
        ----------
        particles_flat : Array
            Current (unmoved) particle set, shape ``(n_particles, n_dim)``.
        start : int
            Index into ``self.event_names`` of the first not-yet-assimilated
            event to check.
        alpha : float
            ESS threshold fraction (the ``ess_threshold`` config field).

        Returns
        -------
        m : int
            Number of events checked (relative to ``start``) -- same
            semantics as :func:`_cheap_reweight_walk`'s ``m``.
        ess_trace : list[float]
            ESS fraction for each checked event, in order.
        logZ_increment_trace : list[float]
            Cheap-reweight log-evidence increment for each *accepted* event.
        """
        single_event_fn = self._get_single_event_particle_fn()
        n_particles = particles_flat.shape[0]
        n_total_events = len(self.event_names)
        n_to_check = n_total_events - start
        log_w = np.zeros(n_particles)
        ess_trace: list[float] = []
        logZ_increment_trace: list[float] = []

        j = start
        while j < n_total_events:
            step_start = time.time()
            event_loglik = np.asarray(single_event_fn(particles_flat, j))
            accepted, ess_fraction, log_w_candidate, logZ_increment = (
                _cheap_reweight_step(log_w, event_loglik, alpha, n_particles)
            )
            step_time = time.time() - step_start
            logger.info(
                f"IBIS cheap-reweight event {j - start + 1}/{n_to_check} "
                f"[{self.event_names[j]}]: ESS={ess_fraction * 100:5.1f}% | "
                f"t={step_time:.3f}s"
            )
            ess_trace.append(ess_fraction)

            if not accepted:
                return j - start + 1, ess_trace, logZ_increment_trace

            assert logZ_increment is not None  # accepted implies a real increment
            logZ_increment_trace.append(logZ_increment)
            log_w = log_w_candidate
            j += 1

        return j - start, ess_trace, logZ_increment_trace

    def _make_batch_logprior_fn(self, absorbed_upto: int) -> Callable[[Array], float]:
        """``prior(theta) + jacobian + L_always_on(theta) + sum_{j<absorbed_upto} log L_j(theta)``,
        evaluated at arbitrary flat ``theta`` (not just the current
        particles) -- required since ``_run_tempering``'s inner MCMC kernel
        proposes new theta during annealing and must evaluate this there
        too. ``absorbed_upto`` is a static Python int, safely closed over
        per batch.

        Builds ``self.stacked_gw.subset(event_names[:absorbed_upto])`` once
        here (i.e. once per batch, not once per tempering step/MCMC step --
        this closure itself is only constructed once per batch by
        ``sample()``), so the returned ``flat_fn`` costs O(absorbed_upto)
        events per call instead of O(len(self.event_names)): it never
        evaluates the not-yet-absorbed events this batch doesn't need. See
        ``StackedGWLikelihood.subset()`` and the module-level scaling note
        above for why this matters."""
        logprior_dict = self._create_logprior_fn_from_dict()
        absorbed_gw = (
            self.stacked_gw.subset(self.event_names[:absorbed_upto])
            if absorbed_upto > 0
            else None
        )

        def flat_fn(x_flat: Array) -> float:
            named_params = self._unflatten_fn(x_flat)
            prior_val = logprior_dict(named_params)
            transformed = self._transform_to_likelihood_space(named_params)
            background_val = self.background.evaluate(transformed)
            if absorbed_gw is not None:
                absorbed_val = absorbed_gw.evaluate(transformed)
            else:
                absorbed_val = 0.0
            return prior_val + background_val + absorbed_val

        return flat_fn

    def _make_batch_loglikelihood_fn(
        self, start: int, stop: int
    ) -> Callable[[Array], float]:
        """``sum_{start <= j < stop} log L_j(theta)``, evaluated at arbitrary
        flat ``theta``. ``start``/``stop`` are static Python ints, safely
        closed over per batch.

        Builds ``self.stacked_gw.subset(event_names[start:stop])`` once here
        (once per batch -- same reasoning as ``_make_batch_logprior_fn``),
        so the returned ``flat_fn`` costs O(stop - start) events per call
        instead of O(len(self.event_names))."""
        batch_gw = self.stacked_gw.subset(self.event_names[start:stop])

        def flat_fn(x_flat: Array) -> float:
            named_params = self._unflatten_fn(x_flat)
            transformed = self._transform_to_likelihood_space(named_params)
            return batch_gw.evaluate(transformed)  # type: ignore[return-value]

        return flat_fn

    def configure_intermediate_saving(
        self,
        full_config: InferenceConfig,
        outdir: str | Path,
        fixed_params: dict[str, float] | None = None,
    ) -> None:
        """Wire up context needed to save per-batch intermediate results.

        This sampler only receives its own sampler sub-config (``self.config``,
        a ``SMCPartialPosteriorsSamplerConfig``), not the full
        ``InferenceConfig`` or the run's output directory -- both are needed
        to build and save an ``InferenceResult`` snapshot after each IBIS
        batch (see ``_save_intermediate_result``). Call this (from
        ``run_inference.py``, right after ``create_sampler``) before
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
        n_particles: int,
        n_events_so_far: int,
        n_batches_so_far: int,
        batch_boundaries_so_far: list[int],
        alpha: float,
        logZ_so_far: float,
        per_event_ess_trace_so_far: list[float],
        per_event_logZ_increment_trace_so_far: list[float],
        cumulative_logZ_history_so_far: list[float],
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        """Build the sampler metadata dict, for either an intermediate or the
        final snapshot of the run.

        Factored out of ``sample()`` so that ``_save_intermediate_result``
        can build a metadata dict describing "the run as if it stopped after
        this batch" using the exact same shape/keys as the real final
        metadata -- required for it to be accepted by
        ``InferenceResult.from_sampler`` (which reads several
        ``blackjax_smc_ibis``-specific keys, see ``result.py``).
        """
        return {
            "sampler": "blackjax_smc_ibis",
            "n_particles": n_particles,
            "n_events": n_events_so_far,
            "n_batches": n_batches_so_far,
            "batch_boundaries": list(batch_boundaries_so_far),
            "ess_threshold": alpha,
            "logZ": float(logZ_so_far),
            "logZ_err": 0.0,
            "cheap_reweight_ess_history": list(per_event_ess_trace_so_far),
            "cheap_reweight_logZ_increment_history": list(
                per_event_logZ_increment_trace_so_far
            ),
            "cumulative_logZ_history": list(cumulative_logZ_history_so_far),
            "sampling_time_seconds": elapsed_seconds,
        }

    def _save_intermediate_result(
        self,
        particles_flat: Array,
        n_particles: int,
        batch_key: str,
        event_names_so_far: list[str],
        batch_result: TemperingResult,
        metadata_so_far: dict[str, Any],
        elapsed_seconds: float,
    ) -> None:
        """Save an ``InferenceResult`` snapshot of the posterior after an
        IBIS batch, including derived EOS quantities from the TOV solver --
        exactly mirroring what ``run_inference.py`` does for the final
        result, just run mid-loop on the current particles instead of the
        fully-assimilated ones.

        Unlike the superseded fractional-mask partial-posteriors sampler,
        no separate resampling step is needed here: ``particles_flat`` is
        already ``batch_result.particles_flat``, i.e. uniformly weighted
        (post terminal resample-to-uniform-weights) straight out of
        ``_run_tempering`` -- see ``TemperingResult``'s docstring.

        Requires ``configure_intermediate_saving()`` to have been called;
        otherwise this logs a warning and does nothing (rather than
        raising, since ``config.save_intermediate_results`` is read from
        the sampler's own config and could be enabled without the caller
        having wired up the extra context this needs).

        Parameters
        ----------
        particles_flat : Array
            This batch's final (moved, uniformly-weighted) particle set.
        n_particles : int
            Number of particles.
        batch_key : str
            This batch's sequential ``batch_<NN>`` key (see ``sample()``),
            used as the intermediate result's filename
            (``results_<batch_key>.h5``).
        event_names_so_far : list[str]
            GW events assimilated up to and including this batch.
        batch_result : TemperingResult
            This batch's ``_run_tempering`` output, reused as
            ``self.final_state`` for the snapshot.
        metadata_so_far : dict[str, Any]
            Metadata dict from ``_build_metadata``, describing the run as if
            it stopped after this batch.
        elapsed_seconds : float
            Wall-clock time elapsed since sampling started.
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
            f"({batch_key}: events {event_names_so_far})..."
        )

        # Temporarily swap in this batch's particles/weights/metadata so
        # get_samples()/get_log_prob()/get_sampler_output() -- and thus
        # InferenceResult.from_sampler, which is driven entirely through
        # those methods -- report this batch's state. Restored in `finally`
        # so the real end-of-run assignment in sample() (and any other
        # caller inspecting these attributes) is unaffected.
        prev_particles_flat = self._particles_flat
        prev_weights = self._weights
        prev_final_state = self.final_state
        prev_metadata = self.metadata
        self._particles_flat = particles_flat
        self._weights = jnp.ones(n_particles) / n_particles
        self.final_state = batch_result
        self.metadata = metadata_so_far
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
                n_eos_samples=self.config.n_eos_samples,
                batch_size=self.config.log_prob_batch_size,
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

    def sample(self, key: PRNGKeyArray) -> None:
        """Run the IBIS outer loop until all GW events are assimilated.

        Evidence bookkeeping note: only the initial prior->background fold-in
        (step 1 below) and each real SMC batch's own ``logZ`` contribute to
        the official running total (``self.metadata["logZ"]``). The
        cheap-reweight walk's own per-event log-evidence increments
        (``_cheap_reweight_walk``'s ``logZ_increment_trace``, stored in
        ``self.metadata["cheap_reweight_logZ_increment_history"]``) are
        diagnostic-only and are deliberately *not* summed into the total:
        every queued span of cheap-reweighted events is always eventually
        redone via a real batch (triggered either by an ESS drop or by
        reaching the end of the event list -- see the module docstring),
        which recomputes an unbiased evidence increment for that exact same
        span via properly annealed resample-move steps. Since both are valid
        unbiased estimators of the *same* ratio, summing both would double
        count; the batch's estimate is used because it is the one actually
        backed by moved (not just reweighted) particles.
        """
        logger.info("Starting BlackJAX IBIS sampling...")
        start_time = time.time()

        # 1. Establish p_0 = prior x L_always_on via the inner SMC-RW
        # sampler's own unmodified sample() -- this IS one full tempered
        # anneal (prior -> prior*background), so no bespoke code is needed
        # for "folding in" the always-on likelihood.
        key, subkey = jax.random.split(key)
        self._inner.sample(subkey)
        particles_flat = self._inner._particles_flat
        assert particles_flat is not None
        self._unflatten_fn = self._inner._unflatten_fn
        self._flatten_fn = self._inner._flatten_fn
        logZ_cumulative = float(self._inner.metadata["logZ"])

        n = 0
        n_total_events = len(self.event_names)
        n_particles = self.config.inner.n_particles
        alpha = self.config.ess_threshold

        n_batches = 0
        batch_boundaries: list[int] = [0]
        per_event_ess_trace: list[float] = []
        per_event_logZ_increment_trace: list[float] = []
        cumulative_logZ_history: list[float] = []
        last_batch_result: TemperingResult | None = None

        while n < n_total_events:
            m_batch, ess_trace, logZ_increment_trace = (
                self._cheap_reweight_walk_incremental(particles_flat, n, alpha)
            )
            per_event_ess_trace.extend(ess_trace)
            per_event_logZ_increment_trace.extend(logZ_increment_trace)
            # Pad to one diagnostic entry per checked event: the (at most
            # one) event that triggered the batch has no cheap-reweight
            # increment, since it was never accepted (see
            # _cheap_reweight_walk's docstring).
            while len(per_event_logZ_increment_trace) < len(per_event_ess_trace):
                per_event_logZ_increment_trace.append(float("nan"))

            n_next = n + m_batch
            logger.info(
                f"IBIS batch {n_batches + 1}: assimilating events "
                f"{n}..{n_next - 1} ({m_batch} events) "
                f"[{', '.join(self.event_names[n:n_next])}]"
            )

            logprior_fn_batch = self._make_batch_logprior_fn(absorbed_upto=n)
            loglikelihood_fn_batch = self._make_batch_loglikelihood_fn(n, n_next)

            key, batch_key = jax.random.split(key)
            batch_result = self._inner._run_tempering(
                batch_key, particles_flat, logprior_fn_batch, loglikelihood_fn_batch
            )
            particles_flat = batch_result.particles_flat
            last_batch_result = batch_result

            batch_logZ = float(batch_result.metadata["logZ"])
            base_cumulative = logZ_cumulative
            logZ_cumulative += batch_logZ
            # Display-only per-event cumulative logZ: split this batch's
            # total increment evenly across its m_batch events (the only
            # exact per-event granularity is unavailable once a whole batch
            # anneals jointly). metadata["logZ"] stays exact.
            per_event_increment = batch_logZ / m_batch
            for k in range(1, m_batch + 1):
                cumulative_logZ_history.append(
                    base_cumulative + per_event_increment * k
                )

            n_batches += 1
            batch_boundaries.append(n_next)

            if self.config.save_intermediate_results:
                batch_label = f"batch_{n_batches:02d}"
                metadata_so_far = self._build_metadata(
                    n_particles=n_particles,
                    n_events_so_far=n_next,
                    n_batches_so_far=n_batches,
                    batch_boundaries_so_far=batch_boundaries,
                    alpha=alpha,
                    logZ_so_far=logZ_cumulative,
                    per_event_ess_trace_so_far=per_event_ess_trace,
                    per_event_logZ_increment_trace_so_far=per_event_logZ_increment_trace,
                    cumulative_logZ_history_so_far=cumulative_logZ_history,
                    elapsed_seconds=time.time() - start_time,
                )
                self._save_intermediate_result(
                    particles_flat=particles_flat,
                    n_particles=n_particles,
                    batch_key=batch_label,
                    event_names_so_far=self.event_names[:n_next],
                    batch_result=batch_result,
                    metadata_so_far=metadata_so_far,
                    elapsed_seconds=time.time() - start_time,
                )

            n = n_next

        # 2. Final rejuvenation pass: fixed-target MCMC on the
        # fully-assimilated closed-form posterior (no further loglikelihood
        # needed -- everything is already folded into the logprior closure).
        assert last_batch_result is not None
        final_logprior_fn = self._make_batch_logprior_fn(absorbed_upto=n_total_events)
        key, rejuv_key = jax.random.split(key)
        particles_flat = rejuvenate_particles(
            rejuv_key,
            particles_flat,
            logposterior_fn=final_logprior_fn,
            mcmc_step_fn=last_batch_result.mcmc_step_fn,
            mcmc_init_fn=last_batch_result.mcmc_init_fn,
            mcmc_parameters=last_batch_result.final_mcmc_parameters,
            n_steps=self.config.n_final_rejuvenation_steps,
        )

        self._particles_flat = particles_flat
        self._weights = jnp.ones(n_particles) / n_particles
        self.final_state = last_batch_result
        self.metadata = self._build_metadata(
            n_particles=n_particles,
            n_events_so_far=n_total_events,
            n_batches_so_far=n_batches,
            batch_boundaries_so_far=batch_boundaries,
            alpha=alpha,
            logZ_so_far=logZ_cumulative,
            per_event_ess_trace_so_far=per_event_ess_trace,
            per_event_logZ_increment_trace_so_far=per_event_logZ_increment_trace,
            cumulative_logZ_history_so_far=cumulative_logZ_history,
            elapsed_seconds=time.time() - start_time,
        )

    def plot_diagnostics(
        self, outdir: str | Path = ".", filename: str = "ibis_diagnostics.png"
    ) -> None:
        """Generate diagnostic plots for an IBIS sampling run.

        Creates a 2-panel figure showing cheap-reweight ESS vs. event index
        (with the ``ess_threshold`` line and batch-trigger points marked),
        and cumulative log Z vs. event index.

        Parameters
        ----------
        outdir : str or Path, optional
            Output directory for saving the plot (default: current directory)
        filename : str, optional
            Filename for the diagnostic plot (default: "ibis_diagnostics.png")
        """
        if self._particles_flat is None:
            logger.warning("No samples yet - run sample() first")
            return

        ess_history = self.metadata.get("cheap_reweight_ess_history", [])
        logZ_history = self.metadata.get("cumulative_logZ_history", [])
        batch_boundaries = self.metadata.get("batch_boundaries", [])
        alpha = self.metadata.get("ess_threshold", self.config.ess_threshold)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle("IBIS Diagnostics", fontsize=14, fontweight="bold")

        event_idx = range(len(ess_history))
        ess_percent = [e * 100 for e in ess_history]
        axes[0].plot(event_idx, ess_percent, "g-o", linewidth=2, markersize=3)
        axes[0].axhline(
            y=alpha * 100,
            color="black",
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label=f"Threshold ({alpha * 100:.0f}%)",
        )
        for b in batch_boundaries[1:]:
            axes[0].axvline(x=b - 1, color="red", linestyle=":", alpha=0.6)
        axes[0].set_ylabel("Cheap-reweight ESS (%)", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=10)
        axes[0].set_ylim(0, 105)

        axes[1].plot(
            range(len(logZ_history)), logZ_history, "b-o", linewidth=2, markersize=3
        )
        for b in batch_boundaries[1:]:
            axes[1].axvline(x=b - 1, color="red", linestyle=":", alpha=0.6)
        axes[1].set_ylabel("Cumulative log Z", fontsize=12)
        axes[1].set_xlabel("Event index", fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        output_path = outdir_path / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved diagnostic plot to {output_path}")
        plt.close(fig)

    def get_samples(self) -> dict:
        """Return final particle positions.

        Returns
        -------
        dict
            Dictionary with parameter samples (transformed back to prior
            space) plus 'weights' and 'ess'.
        """
        if self._particles_flat is None or self._weights is None:
            raise RuntimeError("No samples available - run sample() first")

        particles_dict = jax.vmap(self._unflatten_fn)(self._particles_flat)

        for transform in reversed(self.sample_transforms):
            particles_list = []
            n_particles = len(self._particles_flat)
            for i in range(n_particles):
                particle_dict = {
                    name: particles_dict[name][i] for name in particles_dict.keys()
                }
                transformed_dict, _ = transform.inverse(particle_dict)
                particles_list.append(transformed_dict)
            particles_dict = {
                name: jnp.array([p[name] for p in particles_list])
                for name in particles_list[0].keys()
            }

        ess = float(jnp.sum(self._weights) ** 2 / jnp.sum(self._weights**2))
        particles_dict["weights"] = self._weights
        particles_dict["ess"] = ess  # type: ignore[assignment]

        return particles_dict

    def get_log_prob(self) -> Array:
        """Get log posterior probabilities (evaluated against the full,
        un-split ``self.likelihood`` -- background + all GW events).

        Returns
        -------
        Array
            Log posterior probability values (1D array)
        """
        if self._particles_flat is None:
            raise RuntimeError("No samples available - run sample() first")

        def compute_log_prob(particle_flat):
            x_dict = self._unflatten_fn(particle_flat)
            return self.posterior_from_dict(x_dict, {})

        log_probs = jax.lax.map(
            compute_log_prob,
            self._particles_flat,
            batch_size=self.config.log_prob_batch_size,
        )
        logger.info(f"Computed {len(log_probs)} log probability values")

        return log_probs

    def get_n_samples(self) -> int:
        """Get number of particles.

        Returns
        -------
        int
            Number of particles
        """
        if self._particles_flat is None:
            return 0
        return len(self._particles_flat)

    def get_sampler_output(self) -> SamplerOutput:
        """Get standardized sampler output.

        Returns
        -------
        SamplerOutput
            - samples: Parameter samples (dict of arrays, no weights/ess)
            - log_prob: Log posterior (full, un-split likelihood)
            - metadata: {"weights": Array, "ess": float}

        Raises
        ------
        RuntimeError
            If sampling has not been run yet.
        """
        if self._particles_flat is None:
            raise RuntimeError("No samples available. Run sample() first.")

        all_data = self.get_samples()

        samples: dict[str, Array] = {}
        metadata: dict[str, Any] = {}

        metadata_keys = {"weights", "ess"}
        for key, value in all_data.items():
            if key in metadata_keys:
                metadata[key] = value
            else:
                samples[key] = value

        log_prob = self.get_log_prob()

        return SamplerOutput(samples=samples, log_prob=log_prob, metadata=metadata)
