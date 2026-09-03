"""End-to-end tests for the smc-pp (IBIS / partial-posteriors) sampler.

Uses a chiEFT "always-on" likelihood (reusing the smc-rw chiEFT fixture
pattern) plus 2 mock GW events (cheap, untrained toy normalizing flows --
see conftest.py::mock_gw_events_model_dirs) to exercise the full
config -> sampler.sample() -> SamplerOutput pipeline, and compares against
plain smc-rw run on the exact same combined likelihood.
"""

import pytest
import jax
import jax.numpy as jnp

from jesterTOV.inference.config.schema import InferenceConfig
from jesterTOV.inference.run_inference import (
    setup_prior,
    setup_transform,
    setup_likelihood,
    determine_keep_names,
)
from jesterTOV.inference.samplers import create_sampler

from .conftest import validate_sampler_output, NEP_PARAMS


def _run(config_dict):
    config = InferenceConfig(**config_dict)
    prior, _fixed_params = setup_prior(config)
    keep_names = determine_keep_names(config, prior)
    transform = setup_transform(config, prior=prior, keep_names=keep_names)
    likelihood = setup_likelihood(config, transform)
    sampler = create_sampler(
        config=config.sampler,
        prior=prior,
        likelihood=likelihood,
        likelihood_transforms=[transform],
        seed=config.seed,
    )
    sampler.sample(jax.random.PRNGKey(config.seed))
    return sampler


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.e2e
class TestSMCIBISE2E:
    """End-to-end tests for the smc-pp (IBIS) sampler."""

    @pytest.mark.timeout(480)
    def test_smc_pp_full_pipeline(self, smc_pp_config, e2e_temp_dir):
        """Smoke test: config -> sampler.sample() -> SamplerOutput, with 2
        mock GW events assimilated on top of the chiEFT always-on likelihood."""
        sampler = _run(smc_pp_config)
        output = sampler.get_sampler_output()

        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=50)

        assert "weights" in output.metadata
        weights = output.metadata["weights"]
        assert jnp.isclose(jnp.sum(weights), 1.0, atol=0.01)
        # IBIS always ends with uniformly-weighted, rejuvenated particles.
        assert jnp.allclose(weights, weights[0], atol=1e-8)

        # Bookkeeping sanity, per the module's algorithm: every event gets
        # individually checked, and the run always ends via at least one
        # real SMC batch (never returns raw importance-weighted particles).
        metadata = sampler.metadata
        assert metadata["n_events"] == 2
        assert metadata["n_batches"] >= 1
        assert metadata["batch_boundaries"][-1] == 2
        assert len(metadata["cheap_reweight_ess_history"]) == 2
        assert len(metadata["cumulative_logZ_history"]) == 2
        assert jnp.isfinite(metadata["logZ"])

        # Samples within prior bounds.
        assert jnp.all(output.samples["K_sat"] >= 150.0)
        assert jnp.all(output.samples["K_sat"] <= 300.0)

    @pytest.mark.timeout(480)
    def test_smc_pp_saves_intermediate_results_by_default(
        self, smc_pp_config, e2e_temp_dir
    ):
        """``save_intermediate_results`` defaults to ``True``: once
        ``configure_intermediate_saving`` is wired up (as ``run_inference.py``
        does right after ``create_sampler``), ``sample()`` must write one
        ``substep_results/results_batch_<NN>.h5`` per IBIS batch, each
        independently loadable and carrying EOS-derived quantities."""
        from jesterTOV.inference.result import InferenceResult

        config = InferenceConfig(**smc_pp_config)
        assert config.sampler.save_intermediate_results is True  # type: ignore[union-attr]

        prior, fixed_params = setup_prior(config)
        keep_names = determine_keep_names(config, prior)
        transform = setup_transform(config, prior=prior, keep_names=keep_names)
        likelihood = setup_likelihood(config, transform)
        sampler = create_sampler(
            config=config.sampler,
            prior=prior,
            likelihood=likelihood,
            likelihood_transforms=[transform],
            seed=config.seed,
        )
        sampler.configure_intermediate_saving(
            full_config=config, outdir=e2e_temp_dir, fixed_params=fixed_params
        )
        sampler.sample(jax.random.PRNGKey(config.seed))

        n_batches = sampler.metadata["n_batches"]
        assert n_batches >= 1

        saved_files = sorted((e2e_temp_dir / "substep_results").glob("results_batch_*.h5"))
        assert len(saved_files) == n_batches

        for i, path in enumerate(saved_files, start=1):
            assert path.name == f"results_batch_{i:02d}.h5"
            result = InferenceResult.load(path)
            assert "K_sat" in result.posterior
            assert "masses_EOS" in result.posterior
            assert result.metadata["n_batches"] == i

    @pytest.mark.timeout(900)
    def test_smc_pp_matches_smc_rw_on_combined_likelihood(
        self, smc_pp_config, smc_rw_gw_config, e2e_temp_dir
    ):
        """smc-pp (splitting always-on vs. per-event GW likelihoods and
        assimilating events one at a time) must recover the same posterior
        and a comparable log-evidence as plain smc-rw run directly on the
        exact same combined likelihood (background + both GW events summed).
        """
        sampler_pp = _run(smc_pp_config)
        sampler_rw = _run(smc_rw_gw_config)

        output_pp = sampler_pp.get_sampler_output()
        output_rw = sampler_rw.get_sampler_output()

        for param in ["K_sat", "L_sym"]:
            samples_pp = output_pp.samples[param]
            samples_rw = output_rw.samples[param]

            mean_pp, mean_rw = float(jnp.mean(samples_pp)), float(jnp.mean(samples_rw))
            std_pp, std_rw = float(jnp.std(samples_pp)), float(jnp.std(samples_rw))
            n_pp, n_rw = len(samples_pp), len(samples_rw)

            # Two-sample standard-error-based tolerance: generous multiplier
            # since particle counts here are deliberately tiny for speed.
            se = (std_pp**2 / n_pp + std_rw**2 / n_rw) ** 0.5
            tol = max(8.0 * se, 1e-6)

            assert abs(mean_pp - mean_rw) < tol, (
                f"{param}: smc-pp mean={mean_pp:.3f}, smc-rw mean={mean_rw:.3f}, "
                f"tol={tol:.3f}"
            )

        # Evidence estimates should be in the same ballpark (loose tolerance:
        # this compares two independent, noisy, small-N SMC evidence
        # estimators on a 9-dim posterior, not a tight numerical match).
        logZ_pp = sampler_pp.metadata["logZ"]
        logZ_rw = sampler_rw.metadata["logZ"]
        assert jnp.isfinite(logZ_pp) and jnp.isfinite(logZ_rw)
        assert abs(logZ_pp - logZ_rw) < 5.0, (logZ_pp, logZ_rw)
