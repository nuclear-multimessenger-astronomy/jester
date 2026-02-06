"""Integration tests for end-to-end inference workflows."""

import pytest
import jax
import jax.numpy as jnp

from jesterTOV.inference.config import parser as config_parser
from jesterTOV.inference.config import schema
from jesterTOV.inference.priors import parser as prior_parser
from jesterTOV.inference.transforms import JesterTransform
from jesterTOV.inference.likelihoods import factory as likelihood_factory


class TestConfigToComponents:
    """Test creating all components from configuration."""

    def test_config_to_prior_integration(self, sample_config_file, sample_prior_file):
        """Test loading config and creating prior from it."""
        # Load config
        config = config_parser.load_config(sample_config_file)

        # Get prior specification file from config
        prior_spec_file = config.prior.specification_file

        # Create prior
        prior = prior_parser.parse_prior_file(
            prior_spec_file, nb_CSE=config.transform.nb_CSE
        )

        # Prior should have correct number of dimensions
        assert prior.n_dim == 8  # 8 NEP parameters for nb_CSE=0

        # Can sample from prior
        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=5)

        # Should have all expected parameters
        expected_params = [
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]
        for param in expected_params:
            assert param in samples

    def test_config_to_transform_integration(self, sample_config_dict):
        """Test creating transform from configuration."""
        config = schema.InferenceConfig(**sample_config_dict)

        # Create transform using from_config (no need for manual name_mapping)
        transform = JesterTransform.from_config(config.transform)

        # Transform should have correct type
        assert (
            "MetaModel_EOS_model" in transform.get_eos_type()
        )  # MetaModel for nb_CSE=0

    def test_config_to_likelihood_integration(self, sample_config_dict):
        """Test creating likelihood from configuration."""
        config = schema.InferenceConfig(**sample_config_dict)

        # Create likelihood (ZeroLikelihood in sample config)
        likelihood = likelihood_factory.create_combined_likelihood(config.likelihoods)

        # Should create ZeroLikelihood
        from jesterTOV.inference.likelihoods.combined import ZeroLikelihood

        assert isinstance(likelihood, ZeroLikelihood)

        # Can evaluate likelihood
        result = likelihood.evaluate({})
        assert result == 0.0


class TestPriorTransformLikelihoodChain:
    """Test chaining prior → transform → likelihood."""

    @pytest.fixture
    def full_config(self, temp_dir):
        """Create a complete config for testing the full chain."""
        # Create prior file
        prior_file = temp_dir / "test.prior"
        prior_content = """K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])
"""
        prior_file.write_text(prior_content)

        # Create config
        config_dict = {
            "seed": 42,
            "dry_run": False,
            "validate_only": False,
            "transform": {
                "type": "metamodel",
                "ndat_metamodel": 50,  # Smaller for faster tests
                "nmax_nsat": 2.0,
                "nb_CSE": 0,
                "min_nsat_TOV": 0.75,
                "ndat_TOV": 50,
                "nb_masses": 50,
                "crust_name": "DH",
            },
            "prior": {"specification_file": str(prior_file)},
            "likelihoods": [
                {
                    "type": "constraints_eos",
                    "enabled": True,
                    "parameters": {
                        "penalty_causality": -1e10,
                        "penalty_stability": -1e5,
                    },
                }
            ],
            "sampler": {
                "type": "flowmc",
                "n_chains": 2,
                "n_loop_training": 1,
                "n_loop_production": 1,
                "n_local_steps": 5,
                "n_global_steps": 5,
                "n_epochs": 5,
                "learning_rate": 0.001,
                "output_dir": str(temp_dir / "output"),
            },
            "data_paths": {},
        }

        return schema.InferenceConfig(**config_dict)

    def test_prior_sample_to_likelihood(self, full_config):
        """Test sampling from prior and evaluating likelihood.

        This tests the full chain: prior → transform → likelihood
        BUT we skip the actual transform (TOV solve) and just test
        that the components are compatible.
        """
        # Create prior
        prior = prior_parser.parse_prior_file(
            full_config.prior.specification_file, nb_CSE=full_config.transform.nb_CSE
        )

        # Sample from prior
        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=1)

        # Convert array samples to single dict
        single_sample = {key: values[0] for key, values in samples.items()}

        # Create likelihood
        likelihood = likelihood_factory.create_combined_likelihood(
            full_config.likelihoods
        )

        # Mock constraint outputs (pretend transform added these)
        single_sample["n_causality_violations"] = 0.0
        single_sample["n_stability_violations"] = 0.0
        single_sample["n_pressure_violations"] = 0.0

        # Evaluate likelihood
        log_likelihood = likelihood.evaluate(single_sample)

        # Should get 0.0 for no violations
        assert log_likelihood == 0.0

    def test_prior_log_prob_and_likelihood(self, full_config):
        """Test evaluating prior log probability and likelihood together."""
        # Create prior
        prior = prior_parser.parse_prior_file(
            full_config.prior.specification_file, nb_CSE=full_config.transform.nb_CSE
        )

        # Sample from prior
        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=1)
        single_sample = {key: values[0] for key, values in samples.items()}

        # Evaluate prior log probability
        log_prior = prior.log_prob(single_sample)
        assert jnp.isfinite(log_prior)

        # Create likelihood
        likelihood = likelihood_factory.create_combined_likelihood(
            full_config.likelihoods
        )

        # Mock constraint outputs
        single_sample["n_causality_violations"] = 0.0
        single_sample["n_stability_violations"] = 0.0
        single_sample["n_pressure_violations"] = 0.0

        # Evaluate likelihood
        log_likelihood = likelihood.evaluate(single_sample)

        # Compute log posterior (prior + likelihood)
        log_posterior = log_prior + log_likelihood

        assert jnp.isfinite(log_posterior)


class TestConstraintEnforcement:
    """Test that constraints properly reject invalid EOS."""

    def test_causality_violation_is_rejected(self):
        """Test that causality violations lead to large negative log likelihood."""
        from jesterTOV.inference.likelihoods.constraints import ConstraintEOSLikelihood

        likelihood = ConstraintEOSLikelihood(penalty_causality=-1e10)

        # Mock params with causality violation
        params = {
            "n_causality_violations": 5.0,  # Multiple violations
            "n_stability_violations": 0.0,
            "n_pressure_violations": 0.0,
        }

        log_likelihood = likelihood.evaluate(params)

        # Should apply penalty
        assert log_likelihood == -1e10

    def test_tov_failure_is_rejected(self):
        """Test that TOV integration failures lead to large negative log likelihood."""
        from jesterTOV.inference.likelihoods.constraints import ConstraintTOVLikelihood

        likelihood = ConstraintTOVLikelihood(penalty_tov=-1e10)

        # Mock params with TOV failure
        params = {
            "n_tov_failures": 1.0,  # TOV integration failed
        }

        log_likelihood = likelihood.evaluate(params)

        # Should apply penalty
        assert log_likelihood == -1e10


class TestParameterNamePropagation:
    """Test that parameter names propagate correctly through the system."""

    def test_prior_parameter_names_match_transform(self, sample_prior_file):
        """Test that prior parameter names match what transform expects."""
        # Create prior
        prior = prior_parser.parse_prior_file(sample_prior_file, nb_CSE=0)

        # Get parameter names
        param_names = prior.parameter_names

        # Should have NEP parameters
        expected_params = [
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]

        for param in expected_params:
            assert param in param_names, f"Missing parameter: {param}"

    def test_cse_parameter_names_when_enabled(self, sample_prior_file_with_cse):
        """Test that CSE parameters are added when nb_CSE > 0."""
        # Create prior with CSE
        nb_CSE = 8
        prior = prior_parser.parse_prior_file(sample_prior_file_with_cse, nb_CSE=nb_CSE)

        # Get parameter names
        param_names = prior.parameter_names

        # Should have nbreak
        assert "nbreak" in param_names

        # Should have CSE grid parameters
        for i in range(nb_CSE):
            assert f"n_CSE_{i}_u" in param_names
            assert f"cs2_CSE_{i}" in param_names

        # Should have final cs2
        assert f"cs2_CSE_{nb_CSE}" in param_names


class TestSampleValidation:
    """Test sample validation throughout the pipeline."""

    def test_prior_samples_are_in_bounds(self, sample_prior_file):
        """Test that prior samples respect bounds."""
        prior = prior_parser.parse_prior_file(sample_prior_file, nb_CSE=0)

        # Sample many times
        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=100)

        # Check K_sat bounds: [150, 300]
        assert jnp.all(samples["K_sat"] >= 150.0)
        assert jnp.all(samples["K_sat"] <= 300.0)

        # Check L_sym bounds: [10, 200]
        assert jnp.all(samples["L_sym"] >= 10.0)
        assert jnp.all(samples["L_sym"] <= 200.0)

    def test_samples_are_deterministic_with_same_seed(self, sample_prior_file):
        """Test that sampling is deterministic given same RNG key."""
        prior = prior_parser.parse_prior_file(sample_prior_file, nb_CSE=0)

        # Sample twice with same key
        rng_key = jax.random.PRNGKey(42)
        samples1 = prior.sample(rng_key, n_samples=10)
        samples2 = prior.sample(rng_key, n_samples=10)

        # Should be identical
        for key in samples1.keys():
            assert jnp.allclose(samples1[key], samples2[key], atol=1e-10)


class TestRadioTimingIntegration:
    """Integration tests for radio timing likelihood with realistic EOS."""

    def test_radio_timing_with_mock_eos(self):
        """Test RadioTimingLikelihood with mock EOS curve."""
        from jesterTOV.inference.likelihoods.radio import RadioTimingLikelihood

        # Create likelihood for J0348+0432 (Mobs = 2.01 ± 0.04 Msun)
        likelihood = RadioTimingLikelihood(
            psr_name="J0348+0432",
            mean=2.01,
            std=0.04,
            nb_masses=100,
        )

        # Mock stiff EOS with Mmax > 2.01
        masses_eos = jnp.linspace(0.5, 2.3, 100)  # Stiff EOS

        params = {"masses_EOS": masses_eos}

        # Evaluate
        log_likelihood = likelihood.evaluate(params)

        # Should be finite and reasonable (not large penalty)
        assert jnp.isfinite(log_likelihood)
        assert log_likelihood > -100.0

    def test_radio_timing_penalizes_soft_eos(self):
        """Test that RadioTimingLikelihood penalizes soft EOS with smooth Gaussian.

        NOTE: RadioTimingLikelihood uses a marginalized Gaussian likelihood,
        not a hard rejection. An EOS with Mmax < Mobs gets a negative log
        likelihood based on how many sigma away it is, not a -inf penalty.

        Hard penalties (< -1e10) only apply if Mmax < m_min (default 0.1 Msun).
        """
        from jesterTOV.inference.likelihoods.radio import RadioTimingLikelihood

        # Create likelihood for J0348+0432 (Mobs = 2.01 ± 0.04 Msun)
        likelihood = RadioTimingLikelihood(
            psr_name="J0348+0432",
            mean=2.01,
            std=0.04,
            nb_masses=100,
        )

        # Mock soft EOS with Mmax < 2.01 but > m_min (0.1)
        masses_eos = jnp.linspace(0.5, 1.8, 100)  # Mmax = 1.8 Msun

        params = {"masses_EOS": masses_eos}

        # Evaluate
        log_likelihood = likelihood.evaluate(params)

        # Should be finite (Gaussian penalty, not hard rejection)
        assert jnp.isfinite(log_likelihood)

        # Should be negative (Mmax is 0.21 Msun below mean, ~5 sigma away)
        assert log_likelihood < 0.0

        # For reference: Mmax=1.8 is (2.01-1.8)/0.04 = 5.25 sigma below mean
        # Gaussian log likelihood ~ -0.5 * (5.25)^2 ~ -13.8
        # Plus marginalization effects, expect ~-16
        assert -20.0 < log_likelihood < -10.0


class TestConfigValidationIntegration:
    """Test that invalid configurations are caught early."""

    def test_invalid_config_type_raises_validation_error(self):
        """Test that invalid transform type is caught by Pydantic."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            schema.TransformConfig(
                type="invalid_transform",  # Should fail
                nb_CSE=0,
            )

    def test_metamodel_with_nonzero_cse_raises_validation_error(self):
        """Test that MetaModel with nb_CSE != 0 is caught."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="nb_CSE must be 0"):
            schema.TransformConfig(
                type="metamodel",
                nb_CSE=8,  # Should fail for type=metamodel
            )

    def test_likelihood_without_required_params_raises_validation_error(self):
        """Test that likelihoods without required params are caught."""
        from pydantic import ValidationError

        # GW likelihood requires 'events' parameter
        with pytest.raises(ValidationError):
            schema.LikelihoodConfig(
                type="gw",
                enabled=True,
                parameters={},  # Missing 'events' - should fail
            )


class TestEOSSampleGeneration:
    """Test EOS sample generation from posterior samples."""

    def test_log_prob_filtered_when_fewer_eos_samples(self, temp_dir):
        """Test that log_prob is correctly filtered when generating fewer EOS samples.

        This is a regression test for the bug where log_prob was not filtered
        to match the randomly selected EOS samples, causing index out of bounds
        errors in postprocessing.
        """
        import numpy as np
        from jesterTOV.inference.result import InferenceResult
        from jesterTOV.inference.config.schema import InferenceConfig
        from jesterTOV.inference.transforms import JesterTransform

        # Create a mock result with 100 posterior samples
        n_full_samples = 100
        posterior = {
            "K_sat": np.random.uniform(150, 300, n_full_samples),
            "L_sym": np.random.uniform(10, 200, n_full_samples),
            "Q_sat": np.random.uniform(100, 300, n_full_samples),
            "Q_sym": np.random.uniform(-200, 200, n_full_samples),
            "Z_sat": np.random.uniform(-100, 100, n_full_samples),
            "Z_sym": np.random.uniform(-200, 200, n_full_samples),
            "E_sym": np.ones(n_full_samples) * 31.6,
            "K_sym": np.ones(n_full_samples) * -100.0,
            "log_prob": np.random.uniform(-100, -10, n_full_samples),
        }
        metadata = {
            "sampler": "flowmc",
            "n_samples": n_full_samples,
            "seed": 42,
        }

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        # Create minimal config for generate_eos_samples
        config_dict = {
            "seed": 42,
            "transform": {"type": "metamodel", "nb_CSE": 0},
            "prior": {"specification_file": "dummy.prior"},
            "likelihoods": [{"type": "zero", "enabled": True, "parameters": {}}],
            "sampler": {
                "type": "flowmc",
                "n_chains": 10,
                "n_loop_training": 2,
                "n_loop_production": 2,
                "n_local_steps": 50,
                "n_global_steps": 50,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "batch_size": 10000,
                "use_global": True,
                "output_dir": str(temp_dir),
                "n_eos_samples": 50,  # Request only 50 EOS samples
                "log_prob_batch_size": 10,
            },
        }
        config = InferenceConfig(**config_dict)

        # Create transform using from_config
        transform = JesterTransform.from_config(config.transform)

        # Store original log_prob length
        original_log_prob_len = len(result.posterior["log_prob"])
        assert original_log_prob_len == 100

        # Generate 50 EOS samples from 100 posterior samples using new method
        n_eos_samples = 50
        result.add_eos_from_transform(
            transform=transform,
            n_eos_samples=n_eos_samples,
            batch_size=10,
        )

        # CRITICAL: log_prob should now be filtered to match EOS sample count
        assert (
            len(result.posterior["log_prob"]) == n_eos_samples
        ), f"log_prob should have {n_eos_samples} entries, got {len(result.posterior['log_prob'])}"

        # Verify EOS quantities were added and have correct length
        assert "masses_EOS" in result.posterior
        assert len(result.posterior["masses_EOS"]) == n_eos_samples

        assert "radii_EOS" in result.posterior
        assert len(result.posterior["radii_EOS"]) == n_eos_samples

        assert "Lambdas_EOS" in result.posterior
        assert len(result.posterior["Lambdas_EOS"]) == n_eos_samples

        # Verify full log_prob was backed up
        assert "log_prob_full" in result.posterior
        assert len(result.posterior["log_prob_full"]) == original_log_prob_len

        # Verify filtered log_prob is a subset of original
        # (values should be from the original array, though possibly reordered due to random selection)
        for val in result.posterior["log_prob"]:
            # Each filtered value should be close to at least one original value
            assert np.any(
                np.isclose(val, result.posterior["log_prob_full"])
            ), f"Filtered log_prob value {val} not found in original log_prob"

    def test_sampler_specific_fields_also_filtered(self, temp_dir):
        """Test that sampler-specific fields (weights, ess) are also filtered."""
        import numpy as np
        from jesterTOV.inference.result import InferenceResult
        from jesterTOV.inference.config.schema import InferenceConfig
        from jesterTOV.inference.transforms import JesterTransform

        # Create a mock SMC result with weights and ess
        n_full_samples = 100
        posterior = {
            "K_sat": np.random.uniform(150, 300, n_full_samples),
            "L_sym": np.random.uniform(10, 200, n_full_samples),
            "Q_sat": np.random.uniform(100, 300, n_full_samples),
            "Q_sym": np.random.uniform(-200, 200, n_full_samples),
            "Z_sat": np.random.uniform(-100, 100, n_full_samples),
            "Z_sym": np.random.uniform(-200, 200, n_full_samples),
            "E_sym": np.ones(n_full_samples) * 31.6,
            "K_sym": np.ones(n_full_samples) * -100.0,
            "log_prob": np.random.uniform(-100, -10, n_full_samples),
            "weights": np.ones(n_full_samples) / n_full_samples,  # SMC weights
            "ess": np.random.uniform(0.7, 0.95, n_full_samples),  # SMC ESS
        }
        metadata = {
            "sampler": "blackjax_smc_rw",
            "n_samples": n_full_samples,
            "seed": 123,
        }

        result = InferenceResult(
            sampler_type="blackjax_smc_rw",
            posterior=posterior,
            metadata=metadata,
        )

        # Create minimal config
        config_dict = {
            "seed": 123,
            "transform": {"type": "metamodel", "nb_CSE": 0},
            "prior": {"specification_file": "dummy.prior"},
            "likelihoods": [{"type": "zero", "enabled": True, "parameters": {}}],
            "sampler": {
                "type": "smc-rw",
                "n_particles": 100,
                "n_mcmc_steps": 10,
                "target_ess": 0.9,
                "output_dir": str(temp_dir),
                "n_eos_samples": 40,
                "log_prob_batch_size": 10,
            },
        }
        config = InferenceConfig(**config_dict)

        # Create transform using from_config
        transform = JesterTransform.from_config(config.transform)

        # Generate 40 EOS samples from 100 posterior samples using new method
        n_eos_samples = 40
        result.add_eos_from_transform(
            transform=transform,
            n_eos_samples=n_eos_samples,
            batch_size=10,
        )

        # Verify all sampler-specific fields are filtered
        assert len(result.posterior["log_prob"]) == n_eos_samples
        assert len(result.posterior["weights"]) == n_eos_samples
        assert len(result.posterior["ess"]) == n_eos_samples

        # Verify full versions were backed up
        assert len(result.posterior["log_prob_full"]) == n_full_samples
        assert len(result.posterior["weights_full"]) == n_full_samples
        assert len(result.posterior["ess_full"]) == n_full_samples
