"""Phase 3.2 unit tests (see PLAN.md, "ESS-gated skip of resample+move in
jester's partial-posteriors SMC sampler") -- port of the Phase 1 toy-model
closed-form validation (``debug/optimizing_pp_smc/ess_skip_toy_validation.py``
in the parent development workspace) into CI, exercising the *actual*
production ``_build_jitted_reweight_only_step_fn``/``_build_jitted_ess_check_fn``
helpers from ``partial_posteriors.py`` directly -- not a standalone
reimplementation.

Toy model: conjugate Gaussian, ``theta ~ N(0, tau0^2)``,
``y_i | theta ~ N(theta, sigma^2)`` i.i.d. across ``T`` events, closed-form
evidence and posterior mean via the standard 1D conjugate-normal formulas.

Unlike the Phase 1 script (which uses real MCMC particles and therefore has
genuine Monte Carlo variance), these tests represent "particles" as a fixed
quadrature grid over theta with quadrature weights approximating the prior
density. Since the reweight-only step is *exact* importance sampling (no
resampling, no proposal), a fine-enough grid reproduces the closed-form
evidence/posterior mean to float64 precision -- giving a deterministic,
non-flaky numerical-equivalence test instead of a statistical one.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jaxtyping import Array
from scipy.stats import multivariate_normal

from blackjax.smc.partial_posteriors_path import PartialPosteriorsSMCState

from jesterTOV.inference.samplers.blackjax.smc.partial_posteriors import (
    _build_jitted_ess_check_fn,
    _build_jitted_reweight_only_step_fn,
)

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------
# Toy model helpers
# ----------------------------------------------------------------------

SIGMA = 1.5
TAU0 = 5.0
THETA_TRUE = 2.0
N_EVENTS = 6

# Quadrature grid resolution: fine enough that the grid-based importance
# estimate matches the closed form to ~1e-14 (verified against these exact
# settings before picking the tolerances below).
N_GRID = 2001
N_SIGMA_RANGE = 8.0


def _make_data(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(THETA_TRUE, SIGMA, size=N_EVENTS)


def _closed_form_log_evidence(y: np.ndarray) -> float:
    cov = SIGMA**2 * np.eye(N_EVENTS) + TAU0**2 * np.ones((N_EVENTS, N_EVENTS))
    # scipy's stubs declare `cov` as scalar-only; the runtime API accepts a
    # full covariance matrix (used the same way in the Phase 1 toy script).
    return float(multivariate_normal.logpdf(y, mean=np.zeros(N_EVENTS), cov=cov))  # type: ignore[arg-type]


def _closed_form_posterior_mean(y: np.ndarray) -> float:
    post_var = 1.0 / (N_EVENTS / SIGMA**2 + 1.0 / TAU0**2)
    return post_var * float(np.sum(y)) / SIGMA**2


def _quadrature_prior_particles() -> tuple[jnp.ndarray, jnp.ndarray]:
    """A fixed grid of "particles" over theta, with quadrature weights
    approximating the prior density -- deterministic stand-in for the
    N-particle cloud a real SMC run would carry, with no Monte Carlo
    variance."""
    theta_grid = np.linspace(-N_SIGMA_RANGE * TAU0, N_SIGMA_RANGE * TAU0, N_GRID)
    dtheta = theta_grid[1] - theta_grid[0]
    log_prior = -0.5 * np.log(2 * np.pi * TAU0**2) - 0.5 * (theta_grid / TAU0) ** 2
    weights = np.exp(log_prior) * dtheta
    weights = weights / weights.sum()
    return jnp.asarray(theta_grid), jnp.asarray(weights)


def _make_ordered_event_vals(y: jnp.ndarray):
    """Mirrors the production ``ordered_event_vals`` contract: given one
    particle (here, a scalar theta), returns the per-event log-likelihood
    array in mask-index order."""
    norm_const = -0.5 * jnp.log(2 * jnp.pi * SIGMA**2)

    def ordered_event_vals(theta: jnp.ndarray) -> jnp.ndarray:
        return norm_const - 0.5 * ((y - theta) / SIGMA) ** 2

    return ordered_event_vals


@pytest.fixture(scope="module")
def toy_setup():
    y_np = _make_data(seed=0)
    y = jnp.asarray(y_np)
    particles, weights0 = _quadrature_prior_particles()
    ordered_event_vals = _make_ordered_event_vals(y)
    return {
        "y_np": y_np,
        "particles": particles,
        "weights0": weights0,
        "ordered_event_vals": ordered_event_vals,
        "true_log_evidence": _closed_form_log_evidence(y_np),
        "true_post_mean": _closed_form_posterior_mean(y_np),
    }


class TestReweightOnlyStepAgainstClosedForm:
    """Exercises ``_build_jitted_reweight_only_step_fn`` (the
    ``skip_move_when_ess_ok`` skip branch) directly against the toy model's
    known closed-form evidence and posterior mean."""

    def test_one_shot_full_jump_matches_closed_form(self, toy_setup):
        state = PartialPosteriorsSMCState(
            toy_setup["particles"], toy_setup["weights0"], jnp.zeros(N_EVENTS)
        )
        reweight_only_step = _build_jitted_reweight_only_step_fn(
            toy_setup["ordered_event_vals"]
        )

        new_state, log_evidence_increment = reweight_only_step(
            state, jnp.ones(N_EVENTS)
        )

        assert float(log_evidence_increment) == pytest.approx(
            toy_setup["true_log_evidence"], abs=1e-8
        )
        post_mean = float(jnp.sum(new_state.weights * toy_setup["particles"]))
        assert post_mean == pytest.approx(toy_setup["true_post_mean"], abs=1e-8)
        # Weights normalize to 1 -- reweight_only_step never resamples.
        assert float(jnp.sum(new_state.weights)) == pytest.approx(1.0, abs=1e-12)

    def test_sequential_steps_telescope_to_one_shot_result(self, toy_setup):
        """The "composing multiple cheap reweight-only steps" open question
        in PLAN.md: accumulating one event at a time (relative to the
        running ``pending_mask``, as production code does) must equal a
        single jump straight to the full mask -- both are the same quantity
        mathematically (telescoping), only floating-point roundoff paths
        differ."""
        reweight_only_step = _build_jitted_reweight_only_step_fn(
            toy_setup["ordered_event_vals"]
        )
        particles, weights0 = toy_setup["particles"], toy_setup["weights0"]

        one_shot_state, one_shot_log_evidence = reweight_only_step(
            PartialPosteriorsSMCState(particles, weights0, jnp.zeros(N_EVENTS)),
            jnp.ones(N_EVENTS),
        )

        sequential_state = PartialPosteriorsSMCState(
            particles, weights0, jnp.zeros(N_EVENTS)
        )
        sequential_log_evidence = 0.0
        for i in range(N_EVENTS):
            new_mask = sequential_state.data_mask.at[i].set(1.0)
            sequential_state, increment = reweight_only_step(sequential_state, new_mask)
            sequential_log_evidence += float(increment)

        assert sequential_log_evidence == pytest.approx(
            float(one_shot_log_evidence), abs=1e-8
        )
        np.testing.assert_allclose(
            np.asarray(sequential_state.weights),
            np.asarray(one_shot_state.weights),
            atol=1e-10,
        )

    def test_ess_check_prediction_matches_realized_ess(self, toy_setup):
        """The skip decision relies on ``_build_jitted_ess_check_fn``'s
        cheap, move-free ``ess_at_max_delta_fraction`` correctly predicting
        the ESS a subsequent ``_reweight_only_step`` call would actually
        realize -- verify the two agree."""
        state = PartialPosteriorsSMCState(
            toy_setup["particles"], toy_setup["weights0"], jnp.zeros(N_EVENTS)
        )
        ess_check_fn = _build_jitted_ess_check_fn(toy_setup["ordered_event_vals"])
        reweight_only_step = _build_jitted_reweight_only_step_fn(
            toy_setup["ordered_event_vals"]
        )

        delta_unused, predicted_ess_fraction = ess_check_fn(
            cast(Array, state.particles),
            jnp.log(state.weights),
            jnp.ones(N_EVENTS),
            0.9,
            1.0,
        )
        del delta_unused  # only ess_at_max_delta_fraction is under test here
        new_state, log_evidence_unused = reweight_only_step(state, jnp.ones(N_EVENTS))
        del log_evidence_unused  # covered by the closed-form tests above
        n_particles = cast(Array, new_state.particles).shape[0]
        realized_ess_fraction = (
            float(jnp.sum(new_state.weights) ** 2 / jnp.sum(new_state.weights**2))
            / n_particles
        )

        assert float(predicted_ess_fraction) == pytest.approx(
            realized_ess_fraction, rel=1e-6
        )


def _run_substep_loop(
    ordered_event_vals,
    particles: jnp.ndarray,
    weights0: jnp.ndarray,
    *,
    always_carry_weights_forward: bool,
    target_ess: float = 0.9,
    min_fallback_fraction: float = 1e-3,
) -> dict[str, int]:
    """Reimplements ``sample()``'s per-sub-step loop (``partial_posteriors.py``,
    the ``while float(mask[group[0]]) < 1.0`` loop) against a single group
    covering every event, using the *real* production
    ``_build_jitted_ess_check_fn``/``_build_jitted_reweight_only_step_fn``
    helpers -- with a hand-rolled stand-in for a real MCMC move's *weight*
    output, since this test has no MCMC kernel: a fresh
    ``softmax(mask_diff * ordered_event_vals(x))``, exactly matching what
    ``blackjax.smc.base.step`` actually computes (it discards whatever
    ``state.weights`` were on the way in -- see ``base.py:169-172`` --
    consuming them only to pick resampling indices, which is a no-op here
    since particles never move in this stand-in and is irrelevant to the
    weight/ESS bookkeeping under test).

    ``always_carry_weights_forward=True`` reproduces the bug this test
    guards against: feeding ``jnp.log(state.weights)`` into the next
    sub-step's ESS bisection *unconditionally* whenever the skip flag is on,
    even right after a real move (whose output weights are a fresh,
    single-sub-step quantity, not a genuinely accumulated pending ratio).
    ``False`` reproduces the fix: zero baseline after a real move, the
    running weights only when the immediately preceding sub-step was itself
    a skip.
    """
    n_events = int(ordered_event_vals(particles[0]).shape[0])
    group_mask = jnp.ones(n_events)
    ess_check_fn = _build_jitted_ess_check_fn(ordered_event_vals)
    reweight_only_fn = _build_jitted_reweight_only_step_fn(ordered_event_vals)
    zero_log_weights = jnp.zeros(particles.shape[0])

    state = PartialPosteriorsSMCState(particles, weights0, jnp.zeros(n_events))
    mask_frac = 0.0
    pending_reweight_only = False
    n_real_moves = 0
    n_skips = 0
    n_warnings = 0

    while mask_frac < 1.0:
        max_delta = 1.0 - mask_frac
        if always_carry_weights_forward:
            base_log_weights = jnp.log(state.weights)
        else:
            base_log_weights = (
                jnp.log(state.weights) if pending_reweight_only else zero_log_weights
            )

        raw_delta, ess_at_max = ess_check_fn(
            cast(Array, state.particles), base_log_weights, group_mask, target_ess, max_delta
        )

        if float(ess_at_max) >= target_ess:
            new_mask = jnp.full(n_events, 1.0)
            state, _ = reweight_only_fn(state, new_mask)
            n_skips += 1
            pending_reweight_only = True
            mask_frac = 1.0
            continue

        if not jnp.isfinite(raw_delta):
            n_warnings += 1
            delta = min(max_delta, min_fallback_fraction)
        else:
            delta = float(jnp.clip(raw_delta, 0.0, max_delta))
        new_frac = mask_frac + delta

        mask_diff = new_frac - mask_frac
        log_w = mask_diff * jax.vmap(ordered_event_vals)(state.particles).sum(axis=1)
        new_weights = jax.nn.softmax(log_w)
        state = state._replace(
            weights=new_weights, data_mask=jnp.full(n_events, new_frac)
        )
        n_real_moves += 1
        pending_reweight_only = False
        mask_frac = new_frac

    return {"n_real_moves": n_real_moves, "n_skips": n_skips, "n_warnings": n_warnings}


class TestSubstepLoopBaseLogWeightsAfterRealMove:
    """Regression test for the ``base_log_weights`` bug found by comparing
    real Snellius partial-posteriors runs with ``skip_move_when_ess_ok``
    on vs. off (see PLAN.md and the session that diagnosed it): gating
    ``base_log_weights`` purely on the static ``skip_move_when_ess_ok``
    flag -- rather than on whether the *immediately preceding* sub-step was
    itself a skip -- fed a real move's fresh, single-sub-step output
    weights back into the *next* sub-step's ESS bisection as if they were a
    genuinely accumulated, un-resampled ratio. Because that output already
    sits at (or, with real MCMC/finite-particle noise, sometimes just
    under) ``target_ess``, this makes the very next bisection see almost no
    headroom, forcing either the explicit "ESS already below target_ess"
    ``_MIN_FALLBACK_FRACTION`` fallback or, more often, a legitimate but
    near-zero delta -- both collapsing into a pathological
    jump-then-reset-to-100%-ESS pair every one or two sub-steps, roughly
    *doubling* the number of real moves needed to assimilate the same
    events (observed directly on Snellius: 23 sub-steps with the bug vs. 15
    without it, on an otherwise-identical single-event batch) while the
    skip branch itself almost never fires.
    """

    @pytest.fixture(scope="class")
    def uniform_particles(self):
        """A real particle cloud right after ``pp_init``: i.i.d. prior
        draws with UNIFORM weights -- unlike ``toy_setup``'s fixed
        quadrature grid (whose weights approximate the prior *density* and
        are deliberately non-uniform, the right stand-in for exact-evidence
        checks but the wrong one here, since it would contaminate the very
        first sub-step's baseline with a non-uniform quantity no real run
        ever starts from).
        """
        rng = np.random.default_rng(1)
        n_particles = 5000
        particles = jnp.asarray(rng.normal(0.0, TAU0, size=n_particles))
        weights = jnp.ones(n_particles) / n_particles
        return particles, weights

    def test_fixed_logic_does_not_double_real_move_count(self, uniform_particles):
        particles, weights0 = uniform_particles
        y = jnp.asarray(_make_data(seed=0))
        ordered_event_vals = _make_ordered_event_vals(y)

        buggy = _run_substep_loop(
            ordered_event_vals,
            particles,
            weights0,
            always_carry_weights_forward=True,
        )
        fixed = _run_substep_loop(
            ordered_event_vals,
            particles,
            weights0,
            always_carry_weights_forward=False,
        )

        # The bug's signature: real-move count roughly doubles because
        # every "real" jump gets paired with a near-zero corrective move.
        # The fix must break that pairing, not just reduce it slightly.
        assert fixed["n_real_moves"] < 0.7 * buggy["n_real_moves"], (
            f"expected the fixed base_log_weights logic to need "
            f"substantially fewer real moves than always-carry-forward "
            f"(bug: {buggy['n_real_moves']}, fixed: {fixed['n_real_moves']})"
        )
        # Both variants must still assimilate all events exactly once via
        # the terminal full-remaining-jump skip -- this test is about
        # sub-step *count*, not about breaking the skip branch itself.
        assert fixed["n_skips"] >= 1
        assert buggy["n_skips"] >= 1
