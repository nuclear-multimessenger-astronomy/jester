r"""IBIS (Iterated Batch Importance Sampling, Chopin 2002) hybridized with
likelihood tempering -- informally "partial-posteriors SMC" (YAML
``type: "smc-pp"``, config class ``SMCPartialPosteriorsSamplerConfig``).

GW events are assimilated one at a time into the posterior. From the current
(unweighted, i.i.d.) particle set, each new event's log-likelihood is added
to a running per-particle importance weight -- cheap, since it only needs
that one event's likelihood (:meth:`~jesterTOV.inference.likelihoods.gw.StackedGWLikelihood.evaluate_per_event`).
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
its mass grid fresh on every call, which is incompatible with precomputing
a per-particle, per-event log-likelihood matrix once per particle set. No
warm-starting/checkpointing.

Scaling note: :meth:`BlackJAXIBISSampler._evaluate_particles` computes the
full ``n_particles x n_events`` per-event log-likelihood matrix up front
every time the particle set changes, which assumes this comfortably fits in
memory -- true for realistic present-day GW catalogs (tens of events).
Revisit (e.g. capping the lookahead window) if this is ever pointed at
O(1000)-event catalogs (ET/CE).
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
from jesterTOV.inference.config.schema import SMCPartialPosteriorsSamplerConfig
from jesterTOV.inference.likelihoods.combined import CombinedLikelihood, ZeroLikelihood
from jesterTOV.inference.likelihoods.gw import GWLikelihoodResampled, StackedGWLikelihood
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
        log_w_candidate = log_w + event_matrix[:, j]
        log_ess = 2 * _np_logsumexp(log_w_candidate) - _np_logsumexp(
            2 * log_w_candidate
        )
        ess_fraction = float(np.exp(log_ess)) / n_particles
        ess_trace.append(ess_fraction)

        if ess_fraction < alpha:
            return j + 1, log_w, ess_trace, logZ_increment_trace

        logZ_increment = _np_logsumexp(log_w_candidate) - _np_logsumexp(log_w)
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
        self._jitted_event_loglik_map: Any = None

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

    def _evaluate_particles(
        self, particles_flat: Float[Array, "n_particles n_dim"]
    ) -> Float[Array, "n_particles n_events"]:
        """Evaluate per-event GW log-likelihoods for every particle in one
        batched pass.

        Called once whenever the particle SET changes (real positions
        moved): after the initial prior draw and after every SMC batch's
        terminal resample -- NOT per cheap-reweight step, since those only
        read columns of the matrix this returns (particle positions don't
        move during cheap reweighting).

        The jitted map is built once (lazily, on first call, once
        ``self._unflatten_fn`` exists) and cached, so repeated calls within
        one run reuse the same compiled executable rather than retracing.
        """
        if self._jitted_event_loglik_map is None:

            def per_particle_event_loglik(x_flat: Array) -> Float[Array, " n_events"]:
                named_params = self._unflatten_fn(x_flat)
                transformed = self._transform_to_likelihood_space(named_params)
                return self.stacked_gw.evaluate_per_event(transformed)

            self._jitted_event_loglik_map = jax.jit(
                lambda p: jax.lax.map(
                    per_particle_event_loglik,
                    p,
                    batch_size=self.config.particle_batch_size,
                )
            )
        return self._jitted_event_loglik_map(particles_flat)

    def _make_batch_logprior_fn(self, absorbed_upto: int) -> Callable[[Array], float]:
        """``prior(theta) + jacobian + L_always_on(theta) + sum_{j<absorbed_upto} log L_j(theta)``,
        evaluated at arbitrary flat ``theta`` (not just the current
        particles) -- required since ``_run_tempering``'s inner MCMC kernel
        proposes new theta during annealing and must evaluate this there
        too. ``absorbed_upto`` is a static Python int, safely closed over
        per batch."""
        logprior_dict = self._create_logprior_fn_from_dict()

        def flat_fn(x_flat: Array) -> float:
            named_params = self._unflatten_fn(x_flat)
            prior_val = logprior_dict(named_params)
            transformed = self._transform_to_likelihood_space(named_params)
            background_val = self.background.evaluate(transformed)
            if absorbed_upto > 0:
                event_vals = self.stacked_gw.evaluate_per_event(transformed)
                absorbed_val = jnp.sum(event_vals[:absorbed_upto])
            else:
                absorbed_val = 0.0
            return prior_val + background_val + absorbed_val

        return flat_fn

    def _make_batch_loglikelihood_fn(
        self, start: int, stop: int
    ) -> Callable[[Array], float]:
        """``sum_{start <= j < stop} log L_j(theta)``, evaluated at arbitrary
        flat ``theta``. ``start``/``stop`` are static Python ints, safely
        closed over per batch."""

        def flat_fn(x_flat: Array) -> float:
            named_params = self._unflatten_fn(x_flat)
            transformed = self._transform_to_likelihood_space(named_params)
            event_vals = self.stacked_gw.evaluate_per_event(transformed)
            return jnp.sum(event_vals[start:stop])  # type: ignore[return-value]

        return flat_fn

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
            event_matrix = self._evaluate_particles(particles_flat)
            event_matrix_np = np.asarray(event_matrix[:, n:])

            m_batch, _, ess_trace, logZ_increment_trace = _cheap_reweight_walk(
                event_matrix_np, alpha
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
                cumulative_logZ_history.append(base_cumulative + per_event_increment * k)

            n_batches += 1
            batch_boundaries.append(n_next)
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
        self.metadata = {
            "sampler": "blackjax_smc_ibis",
            "n_particles": n_particles,
            "n_events": n_total_events,
            "n_batches": n_batches,
            "batch_boundaries": batch_boundaries,
            "ess_threshold": alpha,
            "logZ": float(logZ_cumulative),
            "logZ_err": 0.0,
            "cheap_reweight_ess_history": per_event_ess_trace,
            "cheap_reweight_logZ_increment_history": per_event_logZ_increment_trace,
            "cumulative_logZ_history": cumulative_logZ_history,
            "sampling_time_seconds": time.time() - start_time,
        }

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
