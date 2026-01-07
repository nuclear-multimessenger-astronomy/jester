"""Tests for InferenceResult class (HDF5 storage)."""

import pytest
import numpy as np
import json
from datetime import datetime

from jesterTOV.inference.result import InferenceResult


class TestInferenceResultBasic:
    """Test basic InferenceResult functionality."""

    def test_initialization(self):
        """Test InferenceResult initializes correctly."""
        posterior = {
            "K_sat": np.array([220.0, 230.0, 240.0]),
            "L_sym": np.array([90.0, 95.0, 100.0]),
            "log_prob": np.array([-10.0, -11.0, -12.0]),
        }
        metadata = {
            "sampler": "flowmc",
            "sampling_time": 3600.0,
            "n_samples": 3,
            "seed": 42,
        }
        histories = {
            "local_accs": np.array([0.3, 0.4, 0.5]),
        }

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
            histories=histories,
        )

        assert result.sampler_type == "flowmc"
        assert "K_sat" in result.posterior
        assert result.metadata["n_samples"] == 3
        assert result.histories is not None
        assert "local_accs" in result.histories

    def test_initialization_without_histories(self):
        """Test InferenceResult works without histories."""
        posterior = {"K_sat": np.array([220.0])}
        metadata = {"sampler": "blackjax_smc"}

        result = InferenceResult(
            sampler_type="blackjax_smc",
            posterior=posterior,
            metadata=metadata,
            histories=None,
        )

        assert result.histories is None


class TestInferenceResultSaveLoad:
    """Test save/load functionality for all sampler types."""

    def test_save_load_flowmc_basic(self, temp_dir):
        """Test save/load roundtrip for FlowMC results."""
        # Create FlowMC-like result
        posterior = {
            "K_sat": np.array([220.0, 230.0, 240.0]),
            "L_sym": np.array([90.0, 95.0, 100.0]),
            "Q_sat": np.array([100.0, 150.0, 200.0]),
            "log_prob": np.array([-10.0, -11.0, -12.0]),
        }
        metadata = {
            "sampler": "flowmc",
            "sampling_time": 3600.5,
            "n_samples": 3,
            "seed": 42,
            "creation_timestamp": datetime.now().isoformat(),
            "config_json": '{"seed": 42, "transform": {"type": "metamodel"}}',
            "n_chains": 10,
            "n_loop_training": 5,
            "n_loop_production": 5,
        }
        histories = {
            "local_accs": np.array([0.3, 0.4, 0.5, 0.6]),
            "global_accs": np.array([0.7, 0.75, 0.8, 0.85]),
            "loss_vals": np.array([1.5, 1.2, 1.0, 0.9]),
        }

        # Create and save
        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
            histories=histories,
        )

        filepath = temp_dir / "test_flowmc.h5"
        result.save(filepath)

        # Load and verify
        loaded = InferenceResult.load(filepath)

        assert loaded.sampler_type == "flowmc"
        assert loaded.metadata["n_samples"] == 3
        assert loaded.metadata["n_chains"] == 10
        np.testing.assert_array_equal(loaded.posterior["K_sat"], posterior["K_sat"])
        np.testing.assert_array_equal(loaded.posterior["L_sym"], posterior["L_sym"])
        np.testing.assert_array_equal(
            loaded.posterior["log_prob"], posterior["log_prob"]
        )
        assert loaded.histories is not None
        np.testing.assert_array_equal(
            loaded.histories["local_accs"], histories["local_accs"]
        )

    def test_save_load_smc_basic(self, temp_dir):
        """Test save/load roundtrip for BlackJAX SMC results."""
        # Create SMC-like result with sampler-specific data
        posterior = {
            "K_sat": np.array([220.0, 230.0, 240.0, 250.0]),
            "L_sym": np.array([90.0, 95.0, 100.0, 105.0]),
            "log_prob": np.array([-10.0, -11.0, -12.0, -13.0]),
            "_sampler_specific": {
                "weights": np.array([0.25, 0.25, 0.25, 0.25]),
                "ess": np.array([0.9, 0.85, 0.8, 0.75]),
            },
        }
        metadata = {
            "sampler": "blackjax_smc",
            "sampling_time": 1800.0,
            "n_samples": 4,
            "seed": 123,
            "creation_timestamp": datetime.now().isoformat(),
            "config_json": '{"seed": 123}',
            "kernel_type": "nuts",
            "n_particles": 100,
            "n_mcmc_steps": 10,
            "target_ess": 0.9,
            "annealing_steps": 50,
            "final_ess": 90.5,
            "final_ess_percent": 90.5,
            "mean_ess": 85.0,
            "min_ess": 75.0,
            "mean_acceptance": 0.65,
            "logZ": -1234.5,
            "logZ_err": 0.5,
        }
        histories = {
            "lmbda_history": np.linspace(0, 1, 50),
            "ess_history": np.ones(50) * 85.0,
            "acceptance_history": np.ones(50) * 0.65,
        }

        # Create and save
        result = InferenceResult(
            sampler_type="blackjax_smc",
            posterior=posterior,
            metadata=metadata,
            histories=histories,
        )

        filepath = temp_dir / "test_smc.h5"
        result.save(filepath)

        # Load and verify
        loaded = InferenceResult.load(filepath)

        assert loaded.sampler_type == "blackjax_smc"
        assert loaded.metadata["kernel_type"] == "nuts"
        assert loaded.metadata["n_particles"] == 100
        assert loaded.metadata["logZ"] == pytest.approx(-1234.5)
        np.testing.assert_array_equal(loaded.posterior["K_sat"], posterior["K_sat"])

        # Check sampler-specific data
        assert "_sampler_specific" in loaded.posterior
        sampler_data = loaded.posterior["_sampler_specific"]
        np.testing.assert_array_equal(
            sampler_data["weights"], posterior["_sampler_specific"]["weights"]
        )
        np.testing.assert_array_equal(
            sampler_data["ess"], posterior["_sampler_specific"]["ess"]
        )

        # Check histories
        assert loaded.histories is not None
        np.testing.assert_array_equal(
            loaded.histories["lmbda_history"], histories["lmbda_history"]
        )

    def test_save_load_ns_aw_basic(self, temp_dir):
        """Test save/load roundtrip for BlackJAX NS-AW results."""
        # Create NS-AW-like result
        posterior = {
            "K_sat": np.array([220.0, 225.0, 230.0]),
            "L_sym": np.array([90.0, 92.0, 94.0]),
            "log_prob": np.array([-10.0, -10.5, -11.0]),
            "_sampler_specific": {
                "logL": np.array([-5.0, -5.5, -6.0]),
                "logL_birth": np.array([-100.0, -95.0, -90.0]),
            },
        }
        metadata = {
            "sampler": "blackjax_ns_aw",
            "sampling_time": 7200.0,
            "n_samples": 3,
            "seed": 999,
            "creation_timestamp": datetime.now().isoformat(),
            "config_json": '{"seed": 999}',
            "n_live": 1000,
            "n_delete": 10,
            "n_delete_frac": 0.01,
            "n_target": 20,
            "max_mcmc": 5000,
            "max_proposals": 10000,
            "termination_dlogz": 0.01,
            "n_iterations": 15000,
            "n_likelihood_evaluations": 500000,
            "logZ": -2000.5,
            "logZ_err": 0.2,
            "logZ_anesthetic": -2000.6,
            "logZ_err_anesthetic": 0.25,
        }

        # Create and save (no histories for NS-AW in this test)
        result = InferenceResult(
            sampler_type="blackjax_ns_aw",
            posterior=posterior,
            metadata=metadata,
            histories=None,
        )

        filepath = temp_dir / "test_ns_aw.h5"
        result.save(filepath)

        # Load and verify
        loaded = InferenceResult.load(filepath)

        assert loaded.sampler_type == "blackjax_ns_aw"
        assert loaded.metadata["n_live"] == 1000
        assert loaded.metadata["logZ"] == pytest.approx(-2000.5)
        assert loaded.metadata["logZ_anesthetic"] == pytest.approx(-2000.6)
        np.testing.assert_array_equal(loaded.posterior["K_sat"], posterior["K_sat"])

        # Check sampler-specific data
        sampler_data = loaded.posterior["_sampler_specific"]
        np.testing.assert_array_equal(
            sampler_data["logL"], posterior["_sampler_specific"]["logL"]
        )
        np.testing.assert_array_equal(
            sampler_data["logL_birth"], posterior["_sampler_specific"]["logL_birth"]
        )

        assert loaded.histories is None

    def test_save_load_with_derived_eos(self, temp_dir):
        """Test save/load with derived EOS quantities."""
        posterior = {
            "K_sat": np.array([220.0, 230.0]),
            "log_prob": np.array([-10.0, -11.0]),
            # Derived EOS quantities
            "masses_EOS": np.random.rand(2, 100),
            "radii_EOS": np.random.rand(2, 100),
            "Lambdas_EOS": np.random.rand(2, 100),
            "n": np.random.rand(2, 50),
            "p": np.random.rand(2, 50),
            "e": np.random.rand(2, 50),
            "cs2": np.random.rand(2, 50),
        }
        metadata = {
            "sampler": "flowmc",
            "sampling_time": 100.0,
            "n_samples": 2,
            "seed": 1,
            "config_json": "{}",
        }

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        filepath = temp_dir / "test_with_eos.h5"
        result.save(filepath)

        # Load and verify
        loaded = InferenceResult.load(filepath)

        # Check derived quantities are preserved
        assert "masses_EOS" in loaded.posterior
        assert "radii_EOS" in loaded.posterior
        assert "Lambdas_EOS" in loaded.posterior
        np.testing.assert_array_equal(
            loaded.posterior["masses_EOS"], posterior["masses_EOS"]
        )
        np.testing.assert_array_equal(loaded.posterior["n"], posterior["n"])


class TestInferenceResultAddDerivedEOS:
    """Test add_derived_eos method."""

    def test_add_derived_eos(self):
        """Test adding derived EOS quantities."""
        import jax.numpy as jnp

        posterior = {
            "K_sat": np.array([220.0, 230.0]),
            "log_prob": np.array([-10.0, -11.0]),
        }
        metadata = {"sampler": "flowmc"}

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        # Add derived EOS (using JAX arrays to test conversion)
        eos_dict = {
            "masses_EOS": jnp.array([[1.4, 1.6, 1.8], [1.5, 1.7, 1.9]]),
            "radii_EOS": jnp.array([[11.0, 12.0, 13.0], [11.5, 12.5, 13.5]]),
            "Lambdas_EOS": jnp.array([[500.0, 400.0, 300.0], [450.0, 350.0, 250.0]]),
        }

        result.add_derived_eos(eos_dict)

        # Check that JAX arrays were converted to NumPy
        assert "masses_EOS" in result.posterior
        assert isinstance(result.posterior["masses_EOS"], np.ndarray)
        assert result.posterior["masses_EOS"].shape == (2, 3)
        np.testing.assert_array_equal(
            result.posterior["masses_EOS"], np.array([[1.4, 1.6, 1.8], [1.5, 1.7, 1.9]])
        )


class TestInferenceResultConfigProperty:
    """Test config_dict property."""

    def test_config_dict_deserialization(self):
        """Test that config_dict correctly deserializes JSON."""
        config_data = {
            "seed": 42,
            "transform": {"type": "metamodel", "nb_CSE": 0},
            "sampler": {"type": "flowmc", "n_chains": 10},
        }
        config_json = json.dumps(config_data)

        metadata = {
            "sampler": "flowmc",
            "config_json": config_json,
        }
        posterior = {"K_sat": np.array([220.0])}

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        # Test deserialization
        config_dict = result.config_dict
        assert config_dict["seed"] == 42
        assert config_dict["transform"]["type"] == "metamodel"
        assert config_dict["sampler"]["n_chains"] == 10

    def test_config_dict_empty_if_missing(self):
        """Test config_dict returns empty dict if config_json missing."""
        metadata = {"sampler": "flowmc"}
        posterior = {"K_sat": np.array([220.0])}

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        assert result.config_dict == {}


class TestInferenceResultSummary:
    """Test summary method."""

    def test_summary_flowmc(self):
        """Test summary for FlowMC results."""
        posterior = {"K_sat": np.array([220.0])}
        metadata = {
            "sampler": "flowmc",
            "sampling_time": 3600.5,
            "n_samples": 1000,
            "seed": 42,
            "creation_timestamp": "2024-12-28T10:00:00",
            "n_chains": 10,
            "n_loop_training": 5,
            "n_loop_production": 5,
        }

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        summary = result.summary()

        assert "FlowMC" in summary
        assert "Chains: 10" in summary
        assert "Training loops: 5" in summary
        assert "3600.5 seconds" in summary
        assert "seed: 42" in summary.lower()

    def test_summary_smc(self):
        """Test summary for SMC results."""
        posterior = {"K_sat": np.array([220.0])}
        metadata = {
            "sampler": "blackjax_smc",
            "sampling_time": 1800.0,
            "n_samples": 500,
            "seed": 123,
            "kernel_type": "nuts",
            "n_particles": 100,
            "annealing_steps": 50,
            "final_ess_percent": 85.5,
            "mean_acceptance": 0.65,
            "logZ": -1234.5,
        }

        result = InferenceResult(
            sampler_type="blackjax_smc",
            posterior=posterior,
            metadata=metadata,
        )

        summary = result.summary()

        assert "BlackJAX SMC" in summary
        assert "Kernel type: nuts" in summary
        assert "Particles: 100" in summary
        assert "Final ESS: 85.5%" in summary
        assert "Mean acceptance: 0.650" in summary
        assert "Evidence: log(Z) = -1234.5" in summary

    def test_summary_ns_aw(self):
        """Test summary for NS-AW results."""
        posterior = {"K_sat": np.array([220.0])}
        metadata = {
            "sampler": "blackjax_ns_aw",
            "sampling_time": 7200.0,
            "n_samples": 300,
            "seed": 999,
            "n_live": 1000,
            "n_iterations": 15000,
            "logZ": -2000.5,
            "logZ_err": 0.2,
        }

        result = InferenceResult(
            sampler_type="blackjax_ns_aw",
            posterior=posterior,
            metadata=metadata,
        )

        summary = result.summary()

        assert "Nested Sampling" in summary
        assert "Live points: 1000" in summary
        assert "Iterations: 15000" in summary
        assert "Evidence: log(Z) = -2000.50 ± 0.20" in summary

    def test_summary_with_derived_eos(self):
        """Test summary indicates when derived EOS quantities are present."""
        posterior = {
            "K_sat": np.array([220.0]),
            "masses_EOS": np.array([[1.4, 1.6]]),
        }
        metadata = {"sampler": "flowmc", "n_samples": 1}

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        summary = result.summary()
        assert "Derived EOS quantities: Yes" in summary


class TestInferenceResultEdgeCases:
    """Test edge cases and error handling."""

    def test_save_creates_directory(self, temp_dir):
        """Test that save creates parent directories if needed."""
        posterior = {"K_sat": np.array([220.0])}
        metadata = {"sampler": "flowmc"}

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        # Save to nested path that doesn't exist
        filepath = temp_dir / "subdir1" / "subdir2" / "results.h5"
        result.save(filepath)

        assert filepath.exists()

    def test_load_nonexistent_file_raises_error(self, temp_dir):
        """Test that loading nonexistent file raises FileNotFoundError."""
        filepath = temp_dir / "nonexistent.h5"

        with pytest.raises(FileNotFoundError, match="Results file not found"):
            InferenceResult.load(filepath)

    def test_save_load_empty_histories(self, temp_dir):
        """Test save/load with empty histories dict."""
        posterior = {"K_sat": np.array([220.0])}
        metadata = {"sampler": "flowmc"}
        histories = {}  # Empty dict

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
            histories=histories,
        )

        filepath = temp_dir / "test_empty_histories.h5"
        result.save(filepath)

        loaded = InferenceResult.load(filepath)

        # Empty histories should not create histories group
        assert loaded.histories is None

    def test_save_load_with_pathlib_path(self, temp_dir):
        """Test that save/load work with pathlib.Path objects."""
        posterior = {"K_sat": np.array([220.0])}
        metadata = {"sampler": "flowmc"}

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        filepath = temp_dir / "test_pathlib.h5"  # Path object, not string
        result.save(filepath)

        loaded = InferenceResult.load(filepath)
        assert loaded.sampler_type == "flowmc"

    def test_save_load_with_string_path(self, temp_dir):
        """Test that save/load work with string paths."""
        posterior = {"K_sat": np.array([220.0])}
        metadata = {"sampler": "flowmc"}

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        filepath = str(temp_dir / "test_string.h5")  # String, not Path
        result.save(filepath)

        loaded = InferenceResult.load(filepath)
        assert loaded.sampler_type == "flowmc"

    def test_scalar_vs_array_sampler_specific(self, temp_dir):
        """Test handling of scalar vs array datasets in sampler_specific."""
        posterior = {
            "K_sat": np.array([220.0, 230.0]),
            "_sampler_specific": {
                "weights": np.array([0.5, 0.5]),  # Array
                "ess": np.array([0.9, 0.85]),  # Array
                # Note: Current implementation doesn't have scalar sampler_specific,
                # but we test array handling explicitly
            },
        }
        metadata = {"sampler": "blackjax_smc"}

        result = InferenceResult(
            sampler_type="blackjax_smc",
            posterior=posterior,
            metadata=metadata,
        )

        filepath = temp_dir / "test_scalar_array.h5"
        result.save(filepath)

        loaded = InferenceResult.load(filepath)
        sampler_data = loaded.posterior["_sampler_specific"]

        # Verify arrays are loaded correctly
        np.testing.assert_array_equal(sampler_data["weights"], np.array([0.5, 0.5]))
        np.testing.assert_array_equal(sampler_data["ess"], np.array([0.9, 0.85]))

    def test_metadata_without_timestamps(self):
        """Test that result works without timestamps in metadata."""
        posterior = {"K_sat": np.array([220.0])}
        metadata = {
            "sampler": "flowmc",
            "n_samples": 1,
        }

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        summary = result.summary()
        assert "unknown" in summary  # Should handle missing timestamp gracefully
