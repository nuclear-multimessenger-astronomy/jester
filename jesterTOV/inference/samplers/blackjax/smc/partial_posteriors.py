r"""SMC on the "path of partial posteriors" (data tempering / IBIS).

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

Correctness note (see the "path of partial posteriors" section of this
module for detail, and ``eos_bayesian_updates/scripts/ibis_sanity_check.py``
for the empirical validation): turning an event's mask on in a single SMC
step is measurably biased for informative events. The underlying
``blackjax.smc.base.step`` reweights *after* the MCMC rejuvenation move,
which is only a good approximation for small target-to-target jumps (true
for the existing sampler's adaptive :math:`\lambda` bisection, false for a
whole-event jump). Each event is therefore ramped in over several small
fractional mask increments, matching the source paper's own suggestion of
"a geometric path between successive partial posteriors"
(``papers/2007.11936/smcsamplers.tex:409-411``). Two schedules are
supported (``config.substep_schedule``):

- ``"fixed"``: a fixed count of log-spaced fractions (see
  ``_fixed_event_fractions``), denser near mask=0.
- ``"adaptive"``: an ESS-targeting bisection search reusing
  ``blackjax.smc.ess.ess_solver`` / ``blackjax.smc.solver.dichotomy`` --
  the same machinery the base sampler's adaptive :math:`\lambda` schedule
  uses -- applied to the mask fraction instead of :math:`\lambda`. This
  works unmodified because the mask-weighted logposterior is linear in the
  fraction of the single event currently being ramped in (all other terms
  cancel in the successive-target log-weight difference), exactly the
  structure ``ess_solver`` assumes.
"""

from typing import Any, Callable, cast
import time
from pathlib import Path
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random
from jaxtyping import Array, PRNGKeyArray

from jesterTOV.inference.base import LikelihoodBase
from jesterTOV.inference.config.schema import SMCPartialPosteriorsRandomWalkSamplerConfig
from jesterTOV.inference.likelihoods.combined import CombinedLikelihood, ZeroLikelihood
from jesterTOV.inference.likelihoods.gw import GWLikelihood, GWLikelihoodResampled
from jesterTOV.inference.samplers.blackjax.smc.random_walk import (
    BlackJAXSMCRandomWalkSampler,
)
from jesterTOV.logging_config import get_logger

from blackjax.smc import extend_params
from blackjax.smc.ess import ess_solver
from blackjax.smc.resampling import systematic
from blackjax.smc.solver import dichotomy
from blackjax.smc.partial_posteriors_path import (
    init as pp_init,
    build_kernel as pp_build_kernel,
    PartialPosteriorsSMCState,
)

logger = get_logger("jester")

_GW_EVENT_LIKELIHOOD_TYPES = (GWLikelihood, GWLikelihoodResampled)

# Minimum fraction of an event's fixed log-spaced schedule (see
# `_fixed_event_fractions`). Not 0 -- geomspace requires a positive start,
# and this also sets how fine the very first sub-step is.
_FIXED_SCHEDULE_MIN_FRACTION = 1e-3


def _fixed_event_fractions(n_substeps: int) -> Array:
    """Log-spaced fractional mask schedule for ramping in one event.

    Denser near mask=0 than mask=1: the empirically-validated bias fix
    (see module docstring) is most sensitive to the *first* increment out
    of an event's "off" state (mask 0 -> epsilon is a much bigger relative
    jump in the target distribution than mask ~0.9 -> 1), so a geometric
    (log-spaced) schedule -- as the source paper recommends -- resolves
    that regime much more finely than a uniform linear grid for the same
    number of sub-steps.
    """
    fractions = jnp.geomspace(_FIXED_SCHEDULE_MIN_FRACTION, 1.0, n_substeps)
    return fractions.at[-1].set(1.0)


class BlackJAXPartialPosteriorsRandomWalkSampler(BlackJAXSMCRandomWalkSampler):
    """SMC on the path of partial posteriors, with a Random Walk kernel.

    Tempers by number of GW events included rather than by :math:`\\lambda`:
    each configured GW event's likelihood term is turned on one at a time
    (ramped in over several fractional mask sub-steps -- see module
    docstring), with MCMC rejuvenation after each sub-step. Non-GW
    likelihoods (ChiEFT, NICER, radio, EOS/TOV constraints, ...) are always
    on, exactly as in the base combined likelihood.

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
            (event_likelihoods, event_names, always_on_likelihood). The
            always-on likelihood wraps everything that isn't a GW event
            likelihood; ``ZeroLikelihood()`` if there is nothing else.
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
                event_names.append(lik.event_name)
            else:
                always_on.append(lik)

        if len(always_on) == 0:
            always_on_likelihood: LikelihoodBase = ZeroLikelihood()
        elif len(always_on) == 1:
            always_on_likelihood = always_on[0]
        else:
            always_on_likelihood = CombinedLikelihood(always_on)

        return event_likelihoods, event_names, always_on_likelihood

    def _get_kernel_name(self) -> str:
        return "partial_posteriors_random_walk"

    def _load_warm_start(
        self, path: str, n_particles: int, key: PRNGKeyArray
    ) -> tuple[dict[str, Array], int, float]:
        """Load initial particles from a previous run's converged posterior.

        Parameters
        ----------
        path : str
            Path to a previous ``InferenceResult`` HDF5 file, produced by
            this same sampler (``config.warm_start_from``).
        n_particles : int
            Number of particles this run wants to start with. Resampled
            (with replacement, uniform weights) from the previous run's
            particles if the counts differ.
        key : PRNGKeyArray
            JAX random key used for resampling.

        Returns
        -------
        tuple[dict[str, Array], int, float]
            (initial_position_dict, n_prev_events, prev_log_evidence):
            particle positions in prior-parameter space (same shape as
            ``self.prior.sample(...)``), the number of events already
            assimilated by the previous run, and its final ``logZ`` (used
            to keep the cumulative evidence history meaningful across the
            warm-started run).

        Raises
        ------
        ValueError
            If the previous run's ``event_order`` is not a strict prefix of
            this run's ``self._event_order``, or its posterior is missing a
            parameter required by the current prior.
        """
        from jesterTOV.inference.result import InferenceResult

        logger.info(f"Warm-starting partial-posteriors SMC from: {path}")
        prev_result = InferenceResult.load(path)

        prev_event_order = list(prev_result.metadata.get("event_order", []))
        if len(prev_event_order) == 0:
            raise ValueError(
                f"warm_start_from={path!r} has no 'event_order' in its "
                "metadata -- it does not look like a partial-posteriors "
                "SMC result."
            )
        if (
            len(prev_event_order) >= len(self._event_order)
            or self._event_order[: len(prev_event_order)] != prev_event_order
        ):
            raise ValueError(
                "warm_start_from's event_order must be a strict prefix of "
                f"this run's event_order to keep mask indices aligned. "
                f"Previous: {prev_event_order}, current: {self._event_order}."
            )

        prev_posterior = prev_result.posterior
        missing = [
            name
            for name in self.prior.parameter_names
            if name not in prev_posterior
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
        logger.info(
            f"Warm start: {len(prev_event_order)} event(s) already "
            f"assimilated ({prev_event_order}), resampled {n_particles} "
            f"particles from {n_prev_particles} previous particles, "
            f"prev logZ={prev_log_evidence:.3f}"
        )
        return initial_position_dict, len(prev_event_order), prev_log_evidence

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
        assimilated by ramping its mask entry from 0 to 1 over
        ``config.n_substeps_per_event`` fractional sub-steps (see module
        docstring for why this is necessary), with ``config.n_mcmc_steps``
        rejuvenation moves per sub-step. Already-covered events are not
        replayed: their mask entries start (and stay) at 1.
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
        if config.warm_start_from is not None:
            initial_position_dict, n_prev_events, warm_start_log_evidence = (
                self._load_warm_start(config.warm_start_from, config.n_particles, subkey)
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
        event_loglik_fns = [
            self._wrap_dict_fn_for_flat_arrays(self._make_loglikelihood_dict_fn(lik))
            for lik in self._event_likelihoods
        ]
        # Preserve event ordering: self._event_likelihoods may not already be
        # sorted according to self._event_order (which can be overridden via
        # config), so re-order the per-event flat-array functions accordingly.
        name_to_fn = dict(zip(self._event_names, event_loglik_fns))
        ordered_event_loglik_fns = [name_to_fn[name] for name in self._event_order]

        def partial_logposterior_factory(data_mask: Array) -> Callable[[Array], Array]:
            def logpost(x_flat: Array) -> Array:
                event_vals = jnp.stack([f(x_flat) for f in ordered_event_loglik_fns])
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
                logprior_fn, always_on_loglik_fn, full_logposterior_fn,
                initial_position_flat,
            )
        )

        state: PartialPosteriorsSMCState = pp_init(initial_position_flat, n_events)
        mcmc_params = extend_params(init_params)  # type: ignore[arg-type]
        mask = jnp.zeros(n_events)
        if n_prev_events > 0:
            # Already-covered events start (and stay) fully on -- they are
            # not replayed, per the warm-start design (see _load_warm_start).
            mask = mask.at[:n_prev_events].set(1.0)
            state = state._replace(data_mask=mask)
        fixed_fractions = _fixed_event_fractions(config.n_substeps_per_event)

        n_particles = config.n_particles
        ess_history = []
        acceptance_history = []
        log_evidence_history = []
        n_substeps_history = []
        log_evidence = warm_start_log_evidence

        logger.info("=" * 70)
        logger.info("STARTING DATA TEMPERING (PATH OF PARTIAL POSTERIORS)")
        logger.info("=" * 70)
        logger.info(f"Kernel: {self._get_kernel_name().upper()}")
        logger.info(f"Particles: {n_particles}")
        logger.info(f"Events: {n_events} ({self._event_order})")
        if n_prev_events > 0:
            logger.info(
                f"Warm start: skipping {n_prev_events} already-covered "
                f"event(s), prev logZ={warm_start_log_evidence:.3f}"
            )
        logger.info(f"Sub-step schedule: {config.substep_schedule}")
        logger.info(f"Sub-steps per event (fixed count / adaptive cap): {config.n_substeps_per_event}")
        logger.info(f"MCMC steps per sub-step: {config.n_mcmc_steps}")
        logger.info("=" * 70)

        for event_idx in range(n_prev_events, n_events):
            event_name = self._event_order[event_idx]
            event_log_evidence = 0.0
            info = None
            n_substeps_taken = 0
            batched_event_loglik_fn = None
            if config.substep_schedule == "adaptive":
                batched_event_loglik_fn = jax.vmap(ordered_event_loglik_fns[event_idx])

            while float(mask[event_idx]) < 1.0:
                current_frac = float(mask[event_idx])
                if config.substep_schedule == "fixed":
                    new_frac = float(fixed_fractions[n_substeps_taken])
                else:
                    assert batched_event_loglik_fn is not None
                    max_delta = 1.0 - current_frac
                    if n_substeps_taken >= config.n_substeps_per_event:
                        logger.warning(
                            f"Event {event_name}: adaptive sub-step cap "
                            f"({config.n_substeps_per_event}) reached before "
                            "the mask converged to 1.0 -- forcing the final "
                            "step. Consider raising n_substeps_per_event or "
                            "lowering target_ess."
                        )
                        new_frac = 1.0
                    else:
                        raw_delta = ess_solver(
                            batched_event_loglik_fn, state.particles,
                            config.target_ess, max_delta, dichotomy,
                        )
                        if not jnp.isfinite(raw_delta):
                            # Particles are already below target_ess even at
                            # delta=0 (dichotomy has no admissible root) --
                            # fall back to the fixed schedule's minimum step
                            # rather than jumping straight to the cap, to
                            # stay on the safe (small-jump) side.
                            logger.warning(
                                f"Event {event_name}: ESS already below "
                                "target_ess before this sub-step -- falling "
                                "back to a minimal fractional increment."
                            )
                            delta = min(max_delta, _FIXED_SCHEDULE_MIN_FRACTION)
                        else:
                            delta = float(jnp.clip(raw_delta, 0.0, max_delta))
                        new_frac = current_frac + delta

                new_mask = mask.at[event_idx].set(new_frac)
                # `pp_build_kernel` closes over `mcmc_params` at construction
                # time (unlike `inner_kernel_tuning`, it does not accept
                # updated MCMC parameters per call) -- must be rebuilt every
                # sub-step for the RW covariance adaptation to take effect.
                step_fn = pp_build_kernel(
                    mcmc_step_fn, mcmc_init_fn, systematic,
                    config.n_mcmc_steps, mcmc_params,
                    partial_logposterior_factory,
                )
                key, subkey, update_key = jax.random.split(key, 3)
                state, info = step_fn(subkey, state, new_mask)
                mcmc_params = mcmc_parameter_update_fn(update_key, state, info)
                mask = new_mask
                event_log_evidence += float(info.log_likelihood_increment)
                n_substeps_taken += 1

            assert info is not None
            weights = state.weights
            ess_value = float(
                jnp.sum(weights) ** 2 / jnp.sum(weights**2) / n_particles
            )
            acceptance_rate = float(info.update_info.acceptance_rate.mean())  # type: ignore[attr-defined]
            log_evidence += event_log_evidence

            ess_history.append(ess_value)
            acceptance_history.append(acceptance_rate)
            log_evidence_history.append(log_evidence)
            n_substeps_history.append(n_substeps_taken)

            elapsed = time.time() - start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            bar_length = 30
            filled = int((event_idx + 1) / n_events * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            logger.info(
                f"Event {event_idx + 1:3d}/{n_events} [{event_name}] | "
                f"ESS={ess_value * 100:5.1f}% | Accept={acceptance_rate * 100:5.1f}% | "
                f"logZ={log_evidence:8.3f} | substeps={n_substeps_taken:3d} | "
                f"t={elapsed_str} | {bar}"
            )

        end_time = time.time()

        self._particles_flat = cast(Array, state.particles)
        self._weights = state.weights
        self.final_state = state

        assert self._weights is not None
        ess = jnp.sum(self._weights) ** 2 / jnp.sum(self._weights**2)

        key, resample_key = jax.random.split(key)
        resample_idx = systematic(resample_key, self._weights, n_particles)
        self._particles_flat = self._particles_flat[resample_idx]
        self._weights = jnp.ones(n_particles) / n_particles

        mean_ess = float(jnp.mean(jnp.array(ess_history)))
        min_ess = float(jnp.min(jnp.array(ess_history)))
        mean_acceptance = float(jnp.mean(jnp.array(acceptance_history)))
        log_evidence_err = 0.0  # FIXME: same placeholder as base.py

        self.metadata = {
            "sampler": f"blackjax_smc_{self._get_kernel_name()}",
            "kernel_type": self._get_kernel_name(),
            "n_particles": n_particles,
            "n_mcmc_steps": config.n_mcmc_steps,
            "substep_schedule": config.substep_schedule,
            "n_substeps_per_event": config.n_substeps_per_event,
            "event_order": self._event_order,
            "n_events": n_events,
            "warm_start_from": config.warm_start_from or "",
            "n_events_replayed": n_prev_events,
            "final_ess": float(ess),
            "final_ess_percent": float(ess / n_particles * 100),
            "mean_ess": mean_ess,
            "min_ess": min_ess,
            "mean_acceptance": mean_acceptance,
            "logZ": float(log_evidence),
            "logZ_err": float(log_evidence_err),
            "sampling_time_seconds": end_time - start_time,
            "ess_history": ess_history,
            "acceptance_history": acceptance_history,
            "log_evidence_history": log_evidence_history,
            "n_substeps_history": n_substeps_history,
        }

    def plot_diagnostics(
        self, outdir: str | Path = ".", filename: str = "smc_diagnostics.png"
    ) -> None:
        """Generate diagnostic plots for partial-posteriors SMC.

        Creates a 3-panel figure showing, per event instead of per
        tempering step: effective sample size, acceptance rate, and
        cumulative log evidence.

        Parameters
        ----------
        outdir : str or Path, optional
            Output directory for saving the plot (default: current directory)
        filename : str, optional
            Filename for the diagnostic plot (default: "smc_diagnostics.png")
        """
        if self.final_state is None:
            logger.warning("No samples yet - run sample() first")
            return

        event_order = self.metadata["event_order"]
        ess_history = self.metadata["ess_history"]
        acceptance_history = self.metadata["acceptance_history"]
        log_evidence_history = self.metadata["log_evidence_history"]
        n_events = len(event_order)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        kernel_name = self._get_kernel_name().upper()
        fig.suptitle(
            f"Partial-Posteriors SMC Diagnostics ({kernel_name} kernel)",
            fontsize=14,
            fontweight="bold",
        )

        x = range(1, n_events + 1)

        ess_percent = [ess * 100 for ess in ess_history]
        axes[0].plot(x, ess_percent, "g-o", linewidth=2)
        axes[0].set_ylabel("ESS (%)", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)

        acceptance_percent = [acc * 100 for acc in acceptance_history]
        axes[1].plot(x, acceptance_percent, "orange", linestyle="-", marker="o", linewidth=2)
        axes[1].set_ylabel("Acceptance Rate (%)", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 105)

        axes[2].plot(x, log_evidence_history, "b-o", linewidth=2)
        axes[2].set_ylabel(r"Cumulative $\log Z$", fontsize=12)
        axes[2].set_xlabel("Event index", fontsize=12)
        axes[2].grid(True, alpha=0.3)

        axes[2].set_xticks(list(x))
        axes[2].set_xticklabels(event_order, rotation=45, ha="right", fontsize=8)

        plt.tight_layout()

        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        output_path = outdir_path / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved diagnostic plot to {output_path}")
        plt.close(fig)
