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
