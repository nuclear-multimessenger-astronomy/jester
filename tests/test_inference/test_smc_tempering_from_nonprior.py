"""Core correctness test for the `_run_tempering` refactor
(`BlackjaxSMCSampler._run_tempering`, `jesterTOV/inference/samplers/blackjax/smc/base.py`).

This validates the mathematical premise the whole IBIS/partial-posteriors
sampler's batch-fallback path rests on: feeding `_run_tempering` particles
already drawn from a non-prior distribution p_n = prior * L1, together with
`logprior_fn = prior*L1` and `loglikelihood_fn = L2`, must recover the same
posterior and the same (telescoping) evidence increment as running
`_run_tempering` from the raw prior with the full likelihood L1+L2.

Uses a toy conjugate Gaussian model (Gaussian prior, Gaussian "event"
likelihoods as a function of theta) so the true posterior mean/variance and
the true log-evidence at every stage are known in closed form -- no data
simulation needed, no JAX required for the ground truth.
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from jesterTOV.inference.base import LikelihoodBase, MultivariateGaussianPrior
from jesterTOV.inference.likelihoods.combined import CombinedLikelihood
from jesterTOV.inference.config.schema import SMCRandomWalkSamplerConfig
from jesterTOV.inference.samplers.blackjax.smc.random_walk import (
    BlackJAXSMCRandomWalkSampler,
)


class _GaussianEventLikelihood(LikelihoodBase):
    """Toy "event" likelihood: log N(theta; mu, sigma^2), i.e. a Gaussian
    factor in theta-space (conjugate to a Gaussian prior), letting the true
    posterior and log-evidence be computed in closed form."""

    def __init__(self, mu: float, sigma: float, param_name: str = "x"):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.param_name = param_name

    def evaluate(self, params):
        theta = params[self.param_name]
        return norm.logpdf(theta, loc=self.mu, scale=self.sigma)


def _conjugate_update(
    m: float, v: float, mu: float, sigma: float
) -> tuple[float, float]:
    """Fold Gaussian factor N(theta; mu, sigma^2) into current N(theta; m, v)
    (v = variance, not std), returning the new (mean, variance)."""
    v_new = 1.0 / (1.0 / v + 1.0 / sigma**2)
    m_new = v_new * (m / v + mu / sigma**2)
    return m_new, v_new


def _conjugate_logZ_increment(m: float, v: float, mu: float, sigma: float) -> float:
    """log evidence increment for folding N(theta; mu, sigma^2) into the
    current N(theta; m, v) state: log Z_incr = log N(m; mu, v + sigma^2)."""
    return float(norm.logpdf(m, loc=mu, scale=jnp.sqrt(v + sigma**2)))


class _Float32EventLikelihood(LikelihoodBase):
    """Toy "event" likelihood whose ``evaluate()`` genuinely returns a
    float32 array -- mimicking ``StackedGWLikelihood.evaluate_per_event``'s
    ``use_float32=True`` fast path (``jesterTOV/inference/likelihoods/gw.py``),
    which deliberately evaluates GW flows in float32 for speed.

    Regression test for the bug this reproduces: when *all* of a batch's
    events are float32-typed (no float64-typed likelihood term contributing
    to that batch's ``loglikelihood_fn``), blackjax's own per-step
    importance-weight update (``blackjax.smc.base.step``:
    ``weights = jnp.exp(log_weights - logsum_weights)``) is derived *purely*
    from ``loglikelihood_fn``'s output, with no admixture of the previous
    float64 ``state.weights`` to force an upcast -- so the SMC particle
    weights silently become float32 partway through the run, and
    ``jax.lax.while_loop`` (``BlackjaxSMCSampler._run_tempering``) then
    raises a carry dtype mismatch (weights float64 in, float32 out).
    """

    def __init__(self, mu: float, sigma: float, param_name: str = "x"):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.param_name = param_name

    def evaluate(self, params):
        theta = params[self.param_name]
        return norm.logpdf(theta, loc=self.mu, scale=self.sigma).astype(jnp.float32)


def test_run_tempering_with_float32_loglikelihood_does_not_crash():
    """`_run_tempering` must not crash with a `jax.lax.while_loop` carry
    dtype mismatch when `loglikelihood_fn` is entirely float32-typed (the
    IBIS sampler's per-batch `loglikelihood_fn`, built from
    `StackedGWLikelihood.evaluate_per_event` with `use_float32=True`, is
    exactly this shape when every event in the batch uses float32) while the
    particles/prior are the usual float64 -- see
    `_Float32EventLikelihood`'s docstring for the exact mechanism."""
    seed = 0
    n_particles = 500

    m0, s0 = 0.0, 2.0
    mu, sigma = 1.0, 1.0

    prior = MultivariateGaussianPrior(
        ["x"], mean=jnp.array([m0]), cov=jnp.array([[s0**2]])
    )
    L_float32 = _Float32EventLikelihood(mu, sigma)

    config = SMCRandomWalkSamplerConfig(
        n_particles=n_particles,
        n_mcmc_steps=5,
        target_ess=0.9,
        random_walk_sigma=1.0,
    )

    sampler = BlackJAXSMCRandomWalkSampler(
        likelihood=L_float32,
        prior=prior,
        sample_transforms=[],
        likelihood_transforms=[],
        config=config,
        seed=seed,
    )

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    initial_particles = m0 + s0 * jax.random.normal(subkey, (n_particles,))
    initial_particles_flat = initial_particles[:, None].astype(jnp.float64)
    sampler._create_flatten_unflatten_utilities({"x": initial_particles})

    def logprior_fn_dict(params) -> float:
        return prior.log_prob(params)  # type: ignore[return-value]

    def loglik_fn_dict(params) -> float:
        return L_float32.evaluate(params)  # type: ignore[return-value]

    logprior_fn = sampler._wrap_dict_fn_for_flat_arrays(logprior_fn_dict)
    loglik_fn = sampler._wrap_dict_fn_for_flat_arrays(loglik_fn_dict)

    # Sanity check that the toy likelihood actually reproduces the float32
    # output shape the bug depends on before trusting the rest of the test.
    assert jnp.asarray(loglik_fn(initial_particles_flat[0])).dtype == jnp.float32

    key, subkey = jax.random.split(key)
    result = sampler._run_tempering(
        subkey, initial_particles_flat, logprior_fn, loglik_fn
    )

    assert result.particles_flat.dtype == jnp.float64
    assert result.weights.dtype == jnp.float64


def test_run_tempering_from_nonprior_matches_full_batch_from_prior():
    """The core correctness property `_run_tempering` (and hence the whole
    IBIS batch-fallback mechanism) relies on."""
    seed = 0
    n_particles = 2000

    # Toy conjugate Gaussian model.
    m0, s0 = 0.0, 2.0
    mu1, sigma1 = 1.0, 1.0
    mu2, sigma2 = -0.5, 1.5

    prior = MultivariateGaussianPrior(
        ["x"], mean=jnp.array([m0]), cov=jnp.array([[s0**2]])
    )
    L1 = _GaussianEventLikelihood(mu1, sigma1)
    L2 = _GaussianEventLikelihood(mu2, sigma2)

    config = SMCRandomWalkSamplerConfig(
        n_particles=n_particles,
        n_mcmc_steps=10,
        target_ess=0.9,
        random_walk_sigma=1.0,
    )

    # Analytic p_1 = prior * L1.
    m1, v1 = _conjugate_update(m0, s0**2, mu1, sigma1)

    key = jax.random.PRNGKey(seed)

    # --- (a) _run_tempering from particles already drawn from p_1, folding
    # in only L2 -----------------------------------------------------------
    key, subkey = jax.random.split(key)
    particles_p1 = m1 + jnp.sqrt(v1) * jax.random.normal(subkey, (n_particles,))
    particles_p1_flat = particles_p1[:, None]

    sampler_a = BlackJAXSMCRandomWalkSampler(
        likelihood=L2,
        prior=prior,
        sample_transforms=[],
        likelihood_transforms=[],
        config=config,
        seed=seed,
    )
    sampler_a._create_flatten_unflatten_utilities({"x": particles_p1})

    def logprior_fn_a_dict(params) -> float:
        return prior.log_prob(params) + L1.evaluate(params)  # type: ignore[return-value]

    def loglik_fn_a_dict(params) -> float:
        return L2.evaluate(params)  # type: ignore[return-value]

    logprior_fn_a = sampler_a._wrap_dict_fn_for_flat_arrays(logprior_fn_a_dict)
    loglik_fn_a = sampler_a._wrap_dict_fn_for_flat_arrays(loglik_fn_a_dict)

    key, subkey = jax.random.split(key)
    result_a = sampler_a._run_tempering(
        subkey, particles_p1_flat, logprior_fn_a, loglik_fn_a
    )

    # --- (b) a single full-batch _run_tempering from the raw prior with the
    # full likelihood L1 + L2 (via sample(), i.e. the standard smc-rw path) -
    sampler_b = BlackJAXSMCRandomWalkSampler(
        likelihood=CombinedLikelihood([L1, L2]),
        prior=prior,
        sample_transforms=[],
        likelihood_transforms=[],
        config=config,
        seed=seed + 1,
    )
    key, subkey = jax.random.split(key)
    sampler_b.sample(subkey)

    # --- (c) analytic ground truth for the full posterior (prior*L1*L2) ---
    m_full, v_full = _conjugate_update(m1, v1, mu2, sigma2)

    mean_a = float(jnp.mean(result_a.particles_flat))
    mean_b = float(jnp.mean(sampler_b._particles_flat))  # type: ignore[arg-type]

    mc_tol = 6.0 * jnp.sqrt(v_full / n_particles)
    assert abs(mean_a - m_full) < mc_tol, (mean_a, m_full, mc_tol)
    assert abs(mean_b - m_full) < mc_tol, (mean_b, m_full, mc_tol)

    std_a = float(jnp.std(result_a.particles_flat))
    std_b = float(jnp.std(sampler_b._particles_flat))  # type: ignore[arg-type]
    assert abs(std_a - jnp.sqrt(v_full)) < 0.15 * jnp.sqrt(v_full)
    assert abs(std_b - jnp.sqrt(v_full)) < 0.15 * jnp.sqrt(v_full)

    # --- (d) log-evidence telescoping ---------------------------------
    logZ_p1_analytic = _conjugate_logZ_increment(m0, s0**2, mu1, sigma1)
    logZ_a_analytic = _conjugate_logZ_increment(m1, v1, mu2, sigma2)
    logZ_b_analytic = logZ_p1_analytic + logZ_a_analytic

    logZ_atol = 0.3  # generous: finite-N SMC evidence estimator has real variance
    assert abs(result_a.metadata["logZ"] - logZ_a_analytic) < logZ_atol, (
        result_a.metadata["logZ"],
        logZ_a_analytic,
    )
    assert abs(sampler_b.metadata["logZ"] - logZ_b_analytic) < logZ_atol, (
        sampler_b.metadata["logZ"],
        logZ_b_analytic,
    )
