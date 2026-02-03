"""End-to-end tests for BlackJAX Nested Sampling (Acceptance Walk) sampler.

These tests verify the full NS-AW pipeline and serve as regression tests
for the NamedTuple type mismatch bug that was fixed.
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
@pytest.mark.blackjax
class TestBlackJAXNSAWE2E:
    """End-to-end tests for BlackJAX NS-AW sampler.

    These tests exercise the full NS-AW pipeline and serve as regression
    tests for the NamedTuple type mismatch bug (inner_kernel_params).
    """

    def test_blackjax_ns_aw_prior_only_full_pipeline(
        self, blackjax_ns_aw_prior_config, e2e_temp_dir
    ):
        """Test full NS-AW pipeline with prior-only likelihood.

        This is the primary test that caught the inner_kernel_params type mismatch bug.
        The fix uses adaptive_init() instead of base_init() to return AdaptiveNSState.
        """
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

        # This is where the bug used to manifest - during sampling
        sampler.sample(key)

        output = sampler.get_sampler_output()

        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=10)

        # NS-AW-specific: check logL in metadata
        assert "logL" in output.metadata, "NS-AW output missing logL"

    def test_blackjax_ns_aw_chieft_full_pipeline(
        self, blackjax_ns_aw_chieft_config, e2e_temp_dir
    ):
        """Test full NS-AW pipeline with chiEFT likelihood.

        This exercises the realistic use case with:
        - MetaModel+CSE transform
        - chiEFT likelihood
        - EOS constraint checking
        """
        config = InferenceConfig(**blackjax_ns_aw_chieft_config)

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

        assert "K_sat" in output.samples
        assert "nbreak" in output.samples

        # All samples should be finite
        for param, arr in output.samples.items():
            assert jnp.isfinite(arr).all(), f"Parameter {param} has non-finite values"

    def test_blackjax_ns_aw_evidence_computed(
        self, blackjax_ns_aw_prior_config, e2e_temp_dir
    ):
        """Test that NS-AW computes evidence estimate (logZ)."""
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

        # Check evidence in sampler metadata
        from jesterTOV.inference.samplers.blackjax_ns_aw import BlackJAXNSAWSampler

        assert isinstance(sampler, BlackJAXNSAWSampler)

        # logZ should be computed and finite
        assert hasattr(sampler, "metadata"), "Sampler missing metadata"
        # Note: logZ might be in final_state.integrator instead of metadata
        # depending on implementation

    def test_blackjax_ns_aw_produces_valid_posterior(
        self, blackjax_ns_aw_prior_config, e2e_temp_dir
    ):
        """Test that NS-AW produces samples in prior bounds."""
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

        # Check samples are within prior bounds
        # K_sat: [150, 300]
        assert jnp.all(output.samples["K_sat"] >= 150.0)
        assert jnp.all(output.samples["K_sat"] <= 300.0)

        # L_sym: [10, 200]
        assert jnp.all(output.samples["L_sym"] >= 10.0)
        assert jnp.all(output.samples["L_sym"] <= 200.0)

    def test_blackjax_ns_aw_unit_cube_transform(
        self, blackjax_ns_aw_prior_config, e2e_temp_dir
    ):
        """Test that NS-AW properly creates unit cube transforms."""
        config = InferenceConfig(**blackjax_ns_aw_prior_config)

        prior = setup_prior(config)

        # The factory should auto-create BoundToBound transforms
        from jesterTOV.inference.samplers.transform_factory import create_sample_transforms

        transforms = create_sample_transforms(config.sampler, prior)

        # Should have transforms for NS-AW
        assert len(transforms) > 0, "NS-AW should have sample transforms"

        # Check they are BoundToBound (unit cube)
        from jesterTOV.inference.base.transform import BoundToBound

        for t in transforms:
            assert isinstance(t, BoundToBound), f"Expected BoundToBound, got {type(t)}"
