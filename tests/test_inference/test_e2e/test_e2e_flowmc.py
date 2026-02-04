"""End-to-end tests for FlowMC sampler.

FlowMC is a production-ready normalizing flow-enhanced MCMC sampler.
These tests verify the full pipeline: config -> sampler.sample() -> SamplerOutput.
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

from .conftest import validate_sampler_output, NEP_PARAMS


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.e2e
class TestFlowMCE2E:
    """End-to-end tests for FlowMC sampler."""

    def test_flowmc_prior_only_full_pipeline(self, flowmc_prior_config, e2e_temp_dir):
        """Test full FlowMC pipeline with prior-only likelihood.

        This exercises:
        - Config loading and validation
        - Prior setup
        - Transform setup (MetaModel)
        - Likelihood setup (zero + constraints_eos)
        - Sampler creation via factory
        - Full sampling run with training and production phases
        - SamplerOutput generation
        """
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

        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=10)

        # FlowMC-specific: metadata should be empty dict (no weights/ess)
        assert (
            output.metadata == {}
        ), f"FlowMC should have empty metadata, got {output.metadata}"

    def test_flowmc_chieft_full_pipeline(self, flowmc_chieft_config, e2e_temp_dir):
        """Test full FlowMC pipeline with chiEFT likelihood.

        This exercises the realistic use case with:
        - MetaModel+CSE transform
        - chiEFT likelihood
        - EOS constraint checking
        """
        config = InferenceConfig(**flowmc_chieft_config)

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

        # Should have samples
        assert "K_sat" in output.samples
        assert "nbreak" in output.samples

        # All samples should be finite
        for param, arr in output.samples.items():
            assert jnp.isfinite(arr).all(), f"Parameter {param} has non-finite values"

    def test_flowmc_training_vs_production_samples(
        self, flowmc_prior_config, e2e_temp_dir
    ):
        """Test FlowMC provides separate training and production samples."""
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

        # Test FlowMC-specific methods
        from jesterTOV.inference.samplers.flowmc import FlowMCSampler

        assert isinstance(sampler, FlowMCSampler)

        # Production samples (default)
        prod_output = sampler.get_sampler_output()

        # Training samples
        train_output = sampler.get_training_sampler_output()

        # Both should have samples
        assert len(prod_output.samples["K_sat"]) > 0
        assert len(train_output.samples["K_sat"]) > 0

        # Can be different lengths
        n_prod = sampler.get_n_samples()
        n_train = sampler.get_n_training_samples()
        assert n_prod > 0
        assert n_train > 0

    def test_flowmc_produces_valid_posterior(self, flowmc_prior_config, e2e_temp_dir):
        """Test that FlowMC produces samples in prior bounds."""
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

        # Check samples are within prior bounds
        # K_sat: [150, 300]
        assert jnp.all(output.samples["K_sat"] >= 150.0)
        assert jnp.all(output.samples["K_sat"] <= 300.0)

        # L_sym: [10, 200]
        assert jnp.all(output.samples["L_sym"] >= 10.0)
        assert jnp.all(output.samples["L_sym"] <= 200.0)
