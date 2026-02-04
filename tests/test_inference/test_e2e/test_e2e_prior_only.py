"""Fast prior-only E2E tests - these should run quickly for CI validation.

These tests use ultra-light hyperparameters to complete in <30 seconds each.
They verify basic pipeline functionality without expensive likelihood computations.
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
from jesterTOV.inference.samplers.factory import create_sampler


@pytest.mark.integration
@pytest.mark.e2e
class TestPriorOnlyFast:
    """Fast prior-only tests that should complete quickly.

    These tests are NOT marked @slow so they run in regular CI.
    """

    def test_smc_rw_prior_only_minimal(self, smc_rw_prior_config, e2e_temp_dir):
        """Minimal SMC-RW test - should complete very quickly.

        Uses ultra-light parameters for fast CI validation.
        """
        # Make even lighter for fast CI
        smc_rw_prior_config["sampler"]["n_particles"] = 50
        smc_rw_prior_config["sampler"]["n_mcmc_steps"] = 2

        config = InferenceConfig(**smc_rw_prior_config)

        prior = setup_prior(config)
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

        key = jax.random.PRNGKey(config.seed)
        sampler.sample(key)

        output = sampler.get_sampler_output()

        # Basic checks
        assert len(output.samples) > 0
        assert len(output.log_prob) > 0
        assert not jnp.isnan(output.log_prob).any()

    def test_flowmc_prior_only_minimal(self, flowmc_prior_config, e2e_temp_dir):
        """Minimal FlowMC test - should complete very quickly.

        Uses ultra-light parameters for fast CI validation.
        """
        # Make even lighter for fast CI
        flowmc_prior_config["sampler"]["n_chains"] = 20
        flowmc_prior_config["sampler"]["n_loop_training"] = 2
        flowmc_prior_config["sampler"]["n_loop_production"] = 2
        flowmc_prior_config["sampler"]["n_local_steps"] = 5
        flowmc_prior_config["sampler"]["n_global_steps"] = 5

        config = InferenceConfig(**flowmc_prior_config)

        prior = setup_prior(config)
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

        key = jax.random.PRNGKey(config.seed)
        sampler.sample(key)

        output = sampler.get_sampler_output()

        # Basic checks
        assert len(output.samples) > 0
        assert len(output.log_prob) > 0
        assert not jnp.isnan(output.log_prob).any()

    def test_blackjax_ns_aw_prior_only_minimal(
        self, blackjax_ns_aw_prior_config, e2e_temp_dir
    ):
        """Minimal NS-AW test - should complete very quickly.

        Uses ultra-light parameters for fast CI validation.
        This test verifies the NamedTuple bug fix is working.
        """
        # Make even lighter for fast CI
        blackjax_ns_aw_prior_config["sampler"]["n_live"] = 50
        blackjax_ns_aw_prior_config["sampler"]["n_target"] = 10
        blackjax_ns_aw_prior_config["sampler"]["max_mcmc"] = 200
        blackjax_ns_aw_prior_config["sampler"]["termination_dlogz"] = 1.0

        config = InferenceConfig(**blackjax_ns_aw_prior_config)

        prior = setup_prior(config)
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

        key = jax.random.PRNGKey(config.seed)
        sampler.sample(key)

        output = sampler.get_sampler_output()

        # Basic checks
        assert len(output.samples) > 0
        assert len(output.log_prob) > 0
        assert not jnp.isnan(output.log_prob).any()


@pytest.mark.integration
@pytest.mark.e2e
class TestSamplerFactorySmoke:
    """Smoke tests for sampler factory - verify all samplers can be instantiated."""

    def test_smc_rw_can_be_created(self, smc_rw_prior_config, e2e_temp_dir):
        """Test that SMC-RW sampler can be created from config."""
        config = InferenceConfig(**smc_rw_prior_config)

        prior = setup_prior(config)
        transform = setup_transform(config, prior=prior)
        likelihood = setup_likelihood(config, transform)

        sampler = create_sampler(
            config=config.sampler,
            prior=prior,
            likelihood=likelihood,
            likelihood_transforms=[transform],
            seed=config.seed,
        )

        from jesterTOV.inference.samplers.blackjax_smc import (
            BlackJAXSMCRandomWalkSampler,
        )

        assert isinstance(sampler, BlackJAXSMCRandomWalkSampler)

    def test_flowmc_can_be_created(self, flowmc_prior_config, e2e_temp_dir):
        """Test that FlowMC sampler can be created from config."""
        config = InferenceConfig(**flowmc_prior_config)

        prior = setup_prior(config)
        transform = setup_transform(config, prior=prior)
        likelihood = setup_likelihood(config, transform)

        sampler = create_sampler(
            config=config.sampler,
            prior=prior,
            likelihood=likelihood,
            likelihood_transforms=[transform],
            seed=config.seed,
        )

        from jesterTOV.inference.samplers.flowmc import FlowMCSampler

        assert isinstance(sampler, FlowMCSampler)

    def test_blackjax_ns_aw_can_be_created(
        self, blackjax_ns_aw_prior_config, e2e_temp_dir
    ):
        """Test that NS-AW sampler can be created from config."""
        config = InferenceConfig(**blackjax_ns_aw_prior_config)

        prior = setup_prior(config)
        transform = setup_transform(config, prior=prior)
        likelihood = setup_likelihood(config, transform)

        sampler = create_sampler(
            config=config.sampler,
            prior=prior,
            likelihood=likelihood,
            likelihood_transforms=[transform],
            seed=config.seed,
        )

        from jesterTOV.inference.samplers.blackjax_ns_aw import BlackJAXNSAWSampler

        assert isinstance(sampler, BlackJAXNSAWSampler)
