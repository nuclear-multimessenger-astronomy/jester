"""Tests for `_cheap_reweight_walk`
(`jesterTOV/inference/samplers/blackjax/smc/ibis.py`), the plain-NumPy
ESS-threshold walk that decides how many queued GW events IBIS's cheap
reweighting can absorb before falling back to a real SMC batch.

Uses the same toy conjugate Gaussian model as
`test_smc_tempering_from_nonprior.py` (Gaussian prior, Gaussian "event"
likelihoods as a function of theta) so ESS and log-evidence can be checked
against closed-form/manual references.
"""

import numpy as np

from jesterTOV.inference.samplers.blackjax.smc.ibis import (
    _cheap_reweight_walk,
    _np_logsumexp,
)


def _conjugate_update(
    m: float, v: float, mu: float, sigma: float
) -> tuple[float, float]:
    v_new = 1.0 / (1.0 / v + 1.0 / sigma**2)
    m_new = v_new * (m / v + mu / sigma**2)
    return m_new, v_new


def _conjugate_logZ_increment(m: float, v: float, mu: float, sigma: float) -> float:
    return float(
        -0.5 * np.log(2 * np.pi * (v + sigma**2)) - 0.5 * (m - mu) ** 2 / (v + sigma**2)
    )


def _manual_ess_fraction(log_w: np.ndarray) -> float:
    """Reference ESS computation, independent of _np_logsumexp, for
    cross-checking _cheap_reweight_walk's internal formula."""
    w = np.exp(log_w - log_w.max())
    return float((w.sum() ** 2) / np.sum(w**2)) / len(log_w)


def test_np_logsumexp_matches_naive_computation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50) * 3.0
    naive = float(np.log(np.sum(np.exp(x))))
    assert np.isclose(_np_logsumexp(x), naive, atol=1e-8)


def test_walk_ess_matches_manual_computation():
    """Deterministic, hand-crafted event matrix: cross-check the walk's
    internal ESS formula against an independent manual computation."""
    event_matrix = np.array(
        [
            [0.0, -1.0],
            [-0.5, -2.0],
            [-1.0, 0.5],
            [-2.0, -0.5],
        ]
    )
    m, log_w_final, ess_trace, logZ_trace = _cheap_reweight_walk(
        event_matrix, alpha=0.0
    )
    assert m == 2
    assert len(ess_trace) == 2

    # Reproduce the walk's own accept/ESS sequence manually.
    log_w = np.zeros(4)
    for j in range(2):
        log_w_candidate = log_w + event_matrix[:, j]
        expected_ess = _manual_ess_fraction(log_w_candidate)
        assert np.isclose(ess_trace[j], expected_ess, atol=1e-10)
        log_w = log_w_candidate
    assert np.allclose(log_w_final, log_w)


def test_walk_accepts_all_events_and_telescopes_to_analytic_evidence():
    """With alpha low enough that ESS never drops, the walk must accept
    every event, and the sum of its logZ increments (a one-shot importance-
    sampling estimate of the full combined evidence, since particles are
    unweighted prior draws) must match the analytic evidence within Monte
    Carlo tolerance."""
    rng_key_seed = 0
    n_particles = 20_000
    m0, s0 = 0.0, 2.0
    events = [(1.0, 1.0), (-0.5, 1.5), (0.3, 1.2)]

    rng = np.random.default_rng(rng_key_seed)
    particles = rng.normal(loc=m0, scale=s0, size=n_particles)

    event_matrix = np.stack(
        [
            -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (particles - mu) ** 2 / sigma**2
            for mu, sigma in events
        ],
        axis=1,
    )

    m, log_w_final, ess_trace, logZ_increment_trace = _cheap_reweight_walk(
        event_matrix, alpha=0.01
    )

    assert m == len(events)
    assert len(logZ_increment_trace) == len(events)
    assert all(e >= 0.01 for e in ess_trace)

    # Analytic ground truth: sequential conjugate updates give the exact
    # combined evidence log Z = int prior(theta) * prod_i L_i(theta) dtheta.
    m_state, v_state = m0, s0**2
    logZ_analytic = 0.0
    for mu, sigma in events:
        logZ_analytic += _conjugate_logZ_increment(m_state, v_state, mu, sigma)
        m_state, v_state = _conjugate_update(m_state, v_state, mu, sigma)

    logZ_walk_total = float(np.sum(logZ_increment_trace))
    assert abs(logZ_walk_total - logZ_analytic) < 0.5, (logZ_walk_total, logZ_analytic)


def test_walk_stops_on_ess_drop():
    """A highly informative, off-center event should collapse ESS on its
    own column, so the walk must stop there (m == 1) rather than continuing
    to absorb further events."""
    n_particles = 5000
    rng = np.random.default_rng(1)
    # Particles distributed as N(0, 2^2) (the "current particle set").
    particles = rng.normal(loc=0.0, scale=2.0, size=n_particles)

    # Event 0: narrow and far from the particle cloud -> collapses ESS.
    mu0, sigma0 = 8.0, 0.3
    # Event 1: broad and centered -> would be easily absorbed on its own.
    mu1, sigma1 = 0.0, 3.0

    event_matrix = np.stack(
        [
            -0.5 * np.log(2 * np.pi * sigma0**2)
            - 0.5 * (particles - mu0) ** 2 / sigma0**2,
            -0.5 * np.log(2 * np.pi * sigma1**2)
            - 0.5 * (particles - mu1) ** 2 / sigma1**2,
        ],
        axis=1,
    )

    m, log_w_final, ess_trace, logZ_increment_trace = _cheap_reweight_walk(
        event_matrix, alpha=0.5
    )

    assert m == 1
    assert len(ess_trace) == 1
    assert ess_trace[0] < 0.5
    # The failing event is never "accepted" -- no logZ increment recorded
    # for it, and log_w_final stays at the pre-walk (zero) state.
    assert len(logZ_increment_trace) == 0
    assert np.allclose(log_w_final, np.zeros(n_particles))


def test_walk_runs_off_end_of_event_list():
    """When every column clears the threshold, m equals the full width of
    event_matrix (the whole tail must still become one real batch, per the
    algorithm -- this function only reports that all of them were checked
    and accepted)."""
    n_particles = 2000
    rng = np.random.default_rng(2)
    particles = rng.normal(loc=0.0, scale=2.0, size=n_particles)
    events = [(0.2, 2.0), (-0.1, 2.5)]
    event_matrix = np.stack(
        [
            -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (particles - mu) ** 2 / sigma**2
            for mu, sigma in events
        ],
        axis=1,
    )

    m, log_w_final, ess_trace, logZ_increment_trace = _cheap_reweight_walk(
        event_matrix, alpha=0.001
    )
    assert m == event_matrix.shape[1]
    assert len(ess_trace) == event_matrix.shape[1]
    assert len(logZ_increment_trace) == event_matrix.shape[1]
