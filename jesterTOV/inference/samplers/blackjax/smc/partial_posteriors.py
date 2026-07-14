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
(``papers/2007.11936/smcsamplers.tex:409-411``), using an ESS-targeting
bisection search (``blackjax.smc.ess.ess_solver`` /
``blackjax.smc.solver.dichotomy``) -- the same machinery the base
sampler's adaptive :math:`\lambda` schedule uses -- applied to the mask
fraction instead of :math:`\lambda`. This works unmodified because the
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
    SMCPartialPosteriorsRandomWalkSamplerConfig,
)
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
from blackjax.smc.from_mcmc import build_kernel as smc_from_mcmc_build_kernel
from blackjax.smc.partial_posteriors_path import (
    init as pp_init,
    PartialPosteriorsSMCState,
)

logger = get_logger("jester")

_GW_EVENT_LIKELIHOOD_TYPES = (GWLikelihood, GWLikelihoodResampled)

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
    likelihoods: list[dict[str, Any]]
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
    """
    delegate = smc_from_mcmc_build_kernel(mcmc_step_fn, mcmc_init_fn, systematic)

    def substep(
        key: PRNGKeyArray,
        state: PartialPosteriorsSMCState,
        data_mask: Array,
        mcmc_parameters: Any,
    ) -> tuple[PartialPosteriorsSMCState, Any]:
        logposterior_fn = partial_logposterior_factory(data_mask)
        previous_logposterior_fn = partial_logposterior_factory(state.data_mask)

        def log_weights_fn(x: Array) -> Array:
            return logposterior_fn(x) - previous_logposterior_fn(x)

        new_state, info = delegate(
            key, state, num_mcmc_steps, mcmc_parameters, logposterior_fn, log_weights_fn
        )
        return (
            PartialPosteriorsSMCState(
                new_state.particles, new_state.weights, data_mask
            ),
            info,
        )

    return jax.jit(substep)


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
        self._substep_diagnostics: dict[str, dict[str, list[float]]] = {}

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
    ) -> tuple[dict[str, Array], int, float]:
        """Load initial particles from a previous run's converged posterior.

        The previous run does not have to have used this sampler: any saved
        ``InferenceResult`` (``smc-rw``, ``smc-nuts``, ``flowmc``, or this
        sampler itself) works, as long as its posterior is converged on a
        combined likelihood whose GW events are a strict prefix of this
        run's ``self._event_order`` and whose always-on (non-GW)
        likelihoods exactly match this run's. Both are derived from the
        previous run's *saved config* (``InferenceResult.config_dict``),
        not from sampler-specific metadata -- see :func:`_extract_gw_event_order`
        and :func:`_canonical_always_on_signature`.

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
            If the previous run's configured GW events are not a strict
            prefix of this run's ``self._event_order``, if its always-on
            (non-GW) likelihoods don't match this run's, or if its
            posterior is missing a parameter required by the current prior.
        """
        from jesterTOV.inference.result import InferenceResult

        logger.info(f"Warm-starting partial-posteriors SMC from: {path}")
        prev_result = InferenceResult.load(path)
        prev_likelihoods = prev_result.config_dict.get("likelihoods", [])

        prev_event_order = _extract_gw_event_order(prev_likelihoods)
        if len(prev_event_order) == 0:
            raise ValueError(
                f"warm_start_from={path!r}'s saved config has no enabled GW "
                "likelihood events -- nothing to warm-start from."
            )
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
        if config.warm_start_from is not None:
            initial_position_dict, n_prev_events, warm_start_log_evidence = (
                self._load_warm_start(
                    config.warm_start_from, config.n_particles, subkey
                )
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
                logprior_fn,
                always_on_loglik_fn,
                full_logposterior_fn,
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

        jitted_substep_fn = _build_jitted_substep_fn(
            mcmc_step_fn,
            mcmc_init_fn,
            config.n_mcmc_steps,
            partial_logposterior_factory,
        )

        n_particles = config.n_particles
        ess_history = []
        acceptance_history = []
        log_evidence_history = []
        n_substeps_history = []
        log_evidence = warm_start_log_evidence

        # Per-event sub-step diagnostics (mask fraction / ESS / acceptance /
        # cumulative logZ at each sub-step), for the per-event SMC-diagnostics
        # style plot produced in plot_diagnostics() -- see
        # _plot_substep_diagnostics.
        substep_diagnostics: dict[str, dict[str, list[float]]] = {}

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
        logger.info(f"Target ESS per sub-step: {config.target_ess}")
        logger.info(f"MCMC steps per sub-step: {config.n_mcmc_steps}")
        logger.info("=" * 70)

        for event_idx in range(n_prev_events, n_events):
            event_name = self._event_order[event_idx]
            event_log_evidence = 0.0
            info = None
            n_substeps_taken = 0
            batched_event_loglik_fn = jax.vmap(ordered_event_loglik_fns[event_idx])
            mask_fraction_history: list[float] = []
            substep_ess_history: list[float] = []
            substep_acceptance_history: list[float] = []
            substep_log_evidence_history: list[float] = []

            while float(mask[event_idx]) < 1.0:
                current_frac = float(mask[event_idx])
                max_delta = 1.0 - current_frac
                raw_delta = ess_solver(
                    batched_event_loglik_fn,
                    state.particles,
                    config.target_ess,
                    max_delta,
                    dichotomy,
                )
                if not jnp.isfinite(raw_delta):
                    # Particles are already below target_ess even at
                    # delta=0 (dichotomy has no admissible root) --
                    # fall back to a minimal fractional increment rather
                    # than stalling, to stay on the safe (small-jump) side.
                    logger.warning(
                        f"Event {event_name}: ESS already below "
                        "target_ess before this sub-step -- falling "
                        "back to a minimal fractional increment."
                    )
                    delta = min(max_delta, _MIN_FALLBACK_FRACTION)
                else:
                    delta = float(jnp.clip(raw_delta, 0.0, max_delta))
                new_frac = current_frac + delta

                new_mask = mask.at[event_idx].set(new_frac)
                key, subkey, update_key = jax.random.split(key, 3)
                substep_start_time = time.time()
                state, info = jitted_substep_fn(subkey, state, new_mask, mcmc_params)
                mcmc_params = mcmc_parameter_update_fn(update_key, state, info)
                substep_elapsed = time.time() - substep_start_time
                mask = new_mask
                substep_log_evidence_increment = float(info.log_likelihood_increment)
                event_log_evidence += substep_log_evidence_increment
                n_substeps_taken += 1

                substep_ess = float(
                    jnp.sum(state.weights) ** 2
                    / jnp.sum(state.weights**2)
                    / n_particles
                )
                substep_acceptance = float(info.update_info.acceptance_rate.mean())  # type: ignore[attr-defined]
                logger.info(
                    f"    -> [{event_name}] sub-step {n_substeps_taken:2d} | "
                    f"mask={new_frac:.5f} | ESS={substep_ess * 100:5.1f}% | "
                    f"Accept={substep_acceptance * 100:5.1f}% | "
                    f"dlogZ={substep_log_evidence_increment:8.3f} | "
                    f"t={substep_elapsed:6.2f}s"
                )

                mask_fraction_history.append(new_frac)
                substep_ess_history.append(substep_ess)
                substep_acceptance_history.append(substep_acceptance)
                substep_log_evidence_history.append(log_evidence + event_log_evidence)

            substep_diagnostics[event_name] = {
                "mask_fraction_history": mask_fraction_history,
                "ess_history": substep_ess_history,
                "acceptance_history": substep_acceptance_history,
                "log_evidence_history": substep_log_evidence_history,
            }

            assert info is not None
            weights = state.weights
            ess_value = float(jnp.sum(weights) ** 2 / jnp.sum(weights**2) / n_particles)
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
            "target_ess": config.target_ess,
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
        self._substep_diagnostics = substep_diagnostics

    def plot_diagnostics(
        self, outdir: str | Path = ".", filename: str = "smc_diagnostics.png"
    ) -> None:
        """Generate diagnostic plots for partial-posteriors SMC.

        Produces two kinds of plots:

        1. The overall partial-posteriors-path plot (``filename``): a
           3-panel figure showing, per event instead of per tempering step,
           effective sample size, acceptance rate, and cumulative log
           evidence -- the "full picture" of how the run progressed across
           events.
        2. One per-event sub-step diagnostics plot (see
           :meth:`_plot_substep_diagnostics`), mirroring
           ``BlackjaxSMCSampler.plot_diagnostics``'s 3-panel style
           (tempering/mask-fraction schedule, ESS, acceptance) but for the
           adaptive ramp-in of that single event, saved under
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
        axes[1].plot(
            x, acceptance_percent, "orange", linestyle="-", marker="o", linewidth=2
        )
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

        self._plot_substep_diagnostics(outdir_path)

    def _plot_substep_diagnostics(self, outdir_path: Path) -> None:
        """Plot one SMC-diagnostics-style figure per assimilated event.

        Mirrors ``BlackjaxSMCSampler.plot_diagnostics`` (3 panels: tempering
        schedule, ESS, acceptance rate), but replaces the inverse
        temperature :math:`\\lambda` with the event's mask fraction (0 to
        1) and the x-axis annealing-step index with the sub-step index
        within that event's ramp-in. This gives the full per-event picture
        to complement the across-events overview plot.

        Parameters
        ----------
        outdir_path : Path
            Directory under which a ``substep_diagnostics/`` subdirectory
            is created to hold one PNG per event.
        """
        if not self._substep_diagnostics:
            return

        substep_outdir = outdir_path / "substep_diagnostics"
        substep_outdir.mkdir(parents=True, exist_ok=True)

        target_ess = cast(
            SMCPartialPosteriorsRandomWalkSamplerConfig, self.config
        ).target_ess
        kernel_name = self._get_kernel_name().upper()

        for event_idx, event_name in enumerate(self.metadata["event_order"]):
            diagnostics = self._substep_diagnostics.get(event_name)
            if diagnostics is None:
                # Already-covered event from a warm start -- not replayed,
                # so it has no sub-step history to plot.
                continue

            mask_fraction_history = diagnostics["mask_fraction_history"]
            ess_history = diagnostics["ess_history"]
            acceptance_history = diagnostics["acceptance_history"]
            n_substeps = len(mask_fraction_history)

            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            fig.suptitle(
                f"Event {event_idx + 1} [{event_name}] Sub-step Diagnostics "
                f"({kernel_name} kernel)",
                fontsize=14,
                fontweight="bold",
            )

            x = range(1, n_substeps + 1)

            axes[0].plot(x, mask_fraction_history, "b-o", linewidth=2)
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

            output_path = substep_outdir / f"{event_idx:02d}_{event_name}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved sub-step diagnostic plot to {output_path}")
            plt.close(fig)
