"""Tests for normalizing flows module - configuration and data operations."""

import pytest
import numpy as np
import yaml
import jax
import jax.numpy as jnp
from pydantic import ValidationError

from jesterTOV.inference.flows.config import FlowTrainingConfig
from jesterTOV.inference.flows.train_flow import (
    load_posterior,
    standardize_data_zscore,
    inverse_standardize_data_zscore,
    standardize_data_minmax,
    inverse_standardize_data_minmax,
)
from jesterTOV.inference.flows.flow import create_flow, Flow


# ======================
# Fixtures
# ======================


@pytest.fixture
def synthetic_gw_posterior(tmp_path):
    """Create small synthetic GW posterior for testing."""
    n_samples = 1000
    np.random.seed(42)

    data = {
        "mass_1_source": np.random.uniform(1.2, 2.0, n_samples),
        "mass_2_source": np.random.uniform(1.0, 1.8, n_samples),
        "lambda_1": np.random.uniform(0, 1000, n_samples),
        "lambda_2": np.random.uniform(0, 1000, n_samples),
    }

    # Ensure m1 >= m2
    for i in range(n_samples):
        if data["mass_2_source"][i] > data["mass_1_source"][i]:
            data["mass_1_source"][i], data["mass_2_source"][i] = (
                data["mass_2_source"][i],
                data["mass_1_source"][i],
            )
            data["lambda_1"][i], data["lambda_2"][i] = (
                data["lambda_2"][i],
                data["lambda_1"][i],
            )

    file_path = tmp_path / "test_posterior.npz"
    np.savez(file_path, **data)
    return file_path


@pytest.fixture
def sample_flow_config_dict(tmp_path):
    """Sample flow training configuration dictionary."""
    posterior_file = tmp_path / "test.npz"
    output_dir = tmp_path / "output"

    return {
        "posterior_file": str(posterior_file),
        "output_dir": str(output_dir),
        "parameter_names": ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "max_patience": 10,
        "nn_depth": 3,
        "nn_block_dim": 4,
        "flow_layers": 1,
        "invert": True,
        "max_samples": 1000,
        "seed": 42,
        "plot_corner": False,
        "plot_losses": False,
        "flow_type": "masked_autoregressive_flow",
        "standardize": False,
    }


@pytest.fixture
def sample_flow_config_yaml(tmp_path, sample_flow_config_dict):
    """Create a sample YAML config file for testing."""
    config_file = tmp_path / "flow_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_flow_config_dict, f)
    return config_file


# ======================
# FlowTrainingConfig Tests
# ======================


class TestFlowTrainingConfig:
    """Test FlowTrainingConfig validation."""

    def test_valid_config_from_dict(self, sample_flow_config_dict):
        """Test creating config from dict."""
        config = FlowTrainingConfig(**sample_flow_config_dict)
        assert config.num_epochs == 100
        assert config.learning_rate == 1e-3
        assert config.flow_type == "masked_autoregressive_flow"

    def test_from_yaml_loading(self, sample_flow_config_yaml):
        """Test loading config from YAML file."""
        config = FlowTrainingConfig.from_yaml(sample_flow_config_yaml)
        assert isinstance(config, FlowTrainingConfig)
        assert config.num_epochs == 100
        assert config.seed == 42

    def test_invalid_num_epochs_fails(self, sample_flow_config_dict):
        """Test that negative epochs fail validation."""
        sample_flow_config_dict["num_epochs"] = -10
        with pytest.raises(ValidationError, match="Value must be positive"):
            FlowTrainingConfig(**sample_flow_config_dict)

    def test_zero_num_epochs_fails(self, sample_flow_config_dict):
        """Test that zero epochs fail validation."""
        sample_flow_config_dict["num_epochs"] = 0
        with pytest.raises(ValidationError, match="Value must be positive"):
            FlowTrainingConfig(**sample_flow_config_dict)

    def test_invalid_learning_rate_fails(self, sample_flow_config_dict):
        """Test that negative learning rate fails."""
        sample_flow_config_dict["learning_rate"] = -0.001
        with pytest.raises(ValidationError, match="Value must be positive"):
            FlowTrainingConfig(**sample_flow_config_dict)

    def test_zero_learning_rate_fails(self, sample_flow_config_dict):
        """Test that zero learning rate fails."""
        sample_flow_config_dict["learning_rate"] = 0.0
        with pytest.raises(ValidationError, match="Value must be positive"):
            FlowTrainingConfig(**sample_flow_config_dict)

    def test_invalid_val_prop_too_low_fails(self, sample_flow_config_dict):
        """Test that val_prop <= 0 fails validation."""
        sample_flow_config_dict["val_prop"] = 0.0
        with pytest.raises(ValidationError, match="Value must be positive"):
            FlowTrainingConfig(**sample_flow_config_dict)

    def test_invalid_val_prop_too_high_fails(self, sample_flow_config_dict):
        """Test that val_prop >= 1 fails validation."""
        sample_flow_config_dict["val_prop"] = 1.0
        with pytest.raises(ValidationError, match="val_prop must be in"):
            FlowTrainingConfig(**sample_flow_config_dict)

    def test_invalid_val_prop_negative_fails(self, sample_flow_config_dict):
        """Test that negative val_prop fails."""
        sample_flow_config_dict["val_prop"] = -0.1
        with pytest.raises(ValidationError):
            FlowTrainingConfig(**sample_flow_config_dict)

    def test_invalid_flow_type_fails(self, sample_flow_config_dict):
        """Test that invalid flow types fail."""
        sample_flow_config_dict["flow_type"] = "invalid_flow_type"
        with pytest.raises(ValidationError):
            FlowTrainingConfig(**sample_flow_config_dict)

    def test_valid_flow_types(self, sample_flow_config_dict):
        """Test that all valid flow types are accepted."""
        valid_types = [
            "block_neural_autoregressive_flow",
            "masked_autoregressive_flow",
            "coupling_flow",
        ]
        for flow_type in valid_types:
            sample_flow_config_dict["flow_type"] = flow_type
            config = FlowTrainingConfig(**sample_flow_config_dict)
            assert config.flow_type == flow_type

    def test_default_values(self, tmp_path):
        """Test that defaults are set correctly (NEW DEFAULTS as of PR #55)."""
        config = FlowTrainingConfig(
            posterior_file=str(tmp_path / "test.npz"),
            output_dir=str(tmp_path / "output"),
            parameter_names=["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
        )
        assert config.num_epochs == 600
        assert config.learning_rate == 1e-3
        assert config.max_patience == 50
        assert config.nn_depth == 5
        assert config.nn_block_dim == 8
        assert config.flow_layers == 1
        assert config.invert is True
        assert config.max_samples == 50_000
        assert config.seed == 0
        assert config.plot_corner is True
        assert config.plot_losses is True
        assert config.flow_type == "masked_autoregressive_flow"
        # NEW DEFAULTS
        assert config.standardize is True  # Changed from False
        assert config.standardization_method == "zscore"  # New field
        assert config.transformer == "rational_quadratic_spline"  # Changed from affine
        assert config.transformer_knots == 10  # Changed from 8
        assert config.transformer_interval == 5.0  # Changed from 4.0
        assert config.parameter_names == [
            "mass_1_source",
            "mass_2_source",
            "lambda_1",
            "lambda_2",
        ]
        assert config.val_prop == 0.2
        assert config.batch_size == 128

    def test_transformer_types(self, sample_flow_config_dict):
        """Test valid transformer types."""
        valid_transformers = ["affine", "rational_quadratic_spline"]
        for transformer in valid_transformers:
            sample_flow_config_dict["transformer"] = transformer
            config = FlowTrainingConfig(**sample_flow_config_dict)
            assert config.transformer == transformer

    def test_invalid_transformer_type_fails(self, sample_flow_config_dict):
        """Test that invalid transformer type fails."""
        sample_flow_config_dict["transformer"] = "invalid_transformer"
        with pytest.raises(ValidationError):
            FlowTrainingConfig(**sample_flow_config_dict)


# ======================
# Data Loading Tests
# ======================


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_posterior_basic(self, synthetic_gw_posterior):
        """Test loading posterior from npz file with GW parameters."""
        data, metadata = load_posterior(
            str(synthetic_gw_posterior),
            parameter_names=["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
            max_samples=50000,
        )

        assert data.shape[0] == 1000  # All samples loaded
        assert data.shape[1] == 4  # 4 features
        assert metadata["n_samples_total"] == 1000
        assert metadata["n_samples_used"] == 1000
        assert metadata["parameter_names"] == [
            "mass_1_source",
            "mass_2_source",
            "lambda_1",
            "lambda_2",
        ]
        assert "filepath" in metadata

    def test_load_posterior_with_downsampling(self, synthetic_gw_posterior):
        """Test downsampling when n_samples > max_samples."""
        data, metadata = load_posterior(
            str(synthetic_gw_posterior),
            parameter_names=["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
            max_samples=500,
        )

        # Should downsample to ~500 samples
        assert data.shape[0] <= 500
        assert data.shape[0] >= 450  # Allow some margin
        assert data.shape[1] == 4
        assert metadata["n_samples_total"] == 1000
        assert metadata["n_samples_used"] < 1000

    def test_load_posterior_custom_parameters(self, tmp_path):
        """Test loading with custom parameter names (e.g., NICER M-R)."""
        # Create file with NICER-style parameters
        n_samples = 100
        nicer_file = tmp_path / "nicer_posterior.npz"
        np.savez(
            nicer_file,
            mass=np.random.uniform(1.0, 2.0, n_samples),
            radius=np.random.uniform(10.0, 14.0, n_samples),
        )

        data, metadata = load_posterior(
            str(nicer_file), parameter_names=["mass", "radius"]
        )

        assert data.shape == (n_samples, 2)  # 2D flow
        assert metadata["parameter_names"] == ["mass", "radius"]

    def test_load_missing_parameter_fails(self, tmp_path):
        """Test that missing requested parameters raise KeyError."""
        incomplete_file = tmp_path / "incomplete.npz"
        np.savez(
            incomplete_file,
            mass=np.random.randn(100),
            # Missing radius
        )

        with pytest.raises(KeyError, match="Missing required parameter names"):
            load_posterior(str(incomplete_file), parameter_names=["mass", "radius"])

    def test_load_nonexistent_file_fails(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        nonexistent_file = tmp_path / "does_not_exist.npz"
        with pytest.raises(FileNotFoundError, match="Posterior file not found"):
            load_posterior(
                str(nonexistent_file),
                parameter_names=[
                    "mass_1_source",
                    "mass_2_source",
                    "lambda_1",
                    "lambda_2",
                ],
            )

    def test_load_handles_flattening(self, tmp_path):
        """Test that multi-dimensional arrays are flattened correctly."""
        posterior_file = tmp_path / "2d_posterior.npz"
        n_samples = 100
        np.savez(
            posterior_file,
            mass_1_source=np.random.randn(n_samples, 1),  # 2D array
            mass_2_source=np.random.randn(n_samples, 1),
            lambda_1=np.random.randn(n_samples, 1),
            lambda_2=np.random.randn(n_samples, 1),
        )

        data, metadata = load_posterior(
            str(posterior_file),
            parameter_names=["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
        )
        assert data.shape == (n_samples, 4)


# ======================
# Data Preprocessing Tests
# ======================


class TestDataPreprocessing:
    """Test data preprocessing functions."""

    # Z-score standardization tests
    def test_standardize_zscore_roundtrip(self):
        """Test z-score standardize -> inverse identity."""
        np.random.seed(42)
        original_data = np.random.randn(100, 4) * 100 + 500

        # Standardize
        standardized, statistics = standardize_data_zscore(original_data)

        # Check standardized data has mean~0, std~1
        np.testing.assert_allclose(standardized.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(standardized.std(axis=0), 1, atol=1e-10)

        # Inverse transform
        recovered_data = inverse_standardize_data_zscore(standardized, statistics)

        # Check roundtrip identity
        np.testing.assert_allclose(original_data, recovered_data, rtol=1e-10)

    def test_standardize_zscore_statistics(self):
        """Test that z-score stores correct mean and std."""
        np.random.seed(42)
        data = np.random.uniform(10, 100, (100, 4))

        standardized, statistics = standardize_data_zscore(data)

        assert "mean" in statistics
        assert "std" in statistics
        assert statistics["mean"].shape == (4,)
        assert statistics["std"].shape == (4,)
        # Verify mean and std match data
        np.testing.assert_allclose(statistics["mean"], data.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(statistics["std"], data.std(axis=0), rtol=1e-10)

    def test_standardize_zscore_constant_feature(self):
        """Test z-score standardization handles constant features."""
        data = np.ones((100, 4)) * 42.0  # All features constant

        standardized, statistics = standardize_data_zscore(data)

        # Should handle constant features gracefully (avoid division by zero)
        assert not np.any(np.isnan(standardized))
        assert not np.any(np.isinf(standardized))

    # Min-max standardization tests (legacy)
    def test_standardize_minmax_roundtrip(self):
        """Test min-max standardize -> inverse identity."""
        np.random.seed(42)
        original_data = np.random.randn(100, 4) * 100 + 500

        # Standardize
        standardized, bounds = standardize_data_minmax(original_data)

        # Check standardized data is in [0, 1]
        assert np.all(standardized >= 0)
        assert np.all(standardized <= 1)

        # Inverse transform
        recovered_data = inverse_standardize_data_minmax(standardized, bounds)

        # Check roundtrip identity
        np.testing.assert_allclose(original_data, recovered_data, rtol=1e-10)

    def test_standardize_minmax_bounds(self):
        """Test that min-max standardized data is in [0, 1]."""
        np.random.seed(42)
        data = np.random.uniform(10, 100, (100, 4))

        standardized, bounds = standardize_data_minmax(data)

        assert np.all(standardized >= 0)
        assert np.all(standardized <= 1)
        assert "min" in bounds
        assert "max" in bounds
        assert bounds["min"].shape == (4,)
        assert bounds["max"].shape == (4,)

    def test_standardize_minmax_constant_feature(self):
        """Test min-max standardization handles constant features."""
        data = np.ones((100, 4)) * 42.0  # All features constant

        standardized, bounds = standardize_data_minmax(data)

        # Should handle constant features gracefully (avoid division by zero)
        assert not np.any(np.isnan(standardized))
        assert not np.any(np.isinf(standardized))


# ======================
# New Features Tests (PR #55)
# ======================


class TestNewFlowFeatures:
    """Test new flow features: parameter_names, standardization_method, flexible dimensionality."""

    def test_parameter_names_required(self, tmp_path):
        """Test that parameter_names is required."""
        with pytest.raises(ValidationError, match="Field required"):
            FlowTrainingConfig(
                posterior_file=str(tmp_path / "test.npz"),
                output_dir=str(tmp_path / "output"),
                # Missing parameter_names - should fail
            )

    def test_parameter_names_non_empty_list_is_valid(self, tmp_path):
        """Test that non-empty parameter_names list is valid."""
        config = FlowTrainingConfig(
            posterior_file=str(tmp_path / "test.npz"),
            output_dir=str(tmp_path / "output"),
            parameter_names=["mass", "radius"],
        )
        assert config.parameter_names == ["mass", "radius"]

    def test_parameter_names_empty_list_fails(self, tmp_path):
        """Test that empty parameter_names list raises validation error."""
        with pytest.raises(ValidationError, match="cannot be an empty list"):
            FlowTrainingConfig(
                posterior_file=str(tmp_path / "test.npz"),
                output_dir=str(tmp_path / "output"),
                parameter_names=[],  # Empty list should fail
            )

    def test_standardization_method_zscore(self, tmp_path):
        """Test that standardization_method='zscore' is valid."""
        config = FlowTrainingConfig(
            posterior_file=str(tmp_path / "test.npz"),
            output_dir=str(tmp_path / "output"),
            parameter_names=["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
            standardization_method="zscore",
        )
        assert config.standardization_method == "zscore"

    def test_standardization_method_minmax(self, tmp_path):
        """Test that standardization_method='minmax' is valid."""
        config = FlowTrainingConfig(
            posterior_file=str(tmp_path / "test.npz"),
            output_dir=str(tmp_path / "output"),
            parameter_names=["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
            standardization_method="minmax",
        )
        assert config.standardization_method == "minmax"

    def test_standardization_method_invalid_fails(self, tmp_path):
        """Test that invalid standardization_method fails."""
        with pytest.raises(ValidationError):
            FlowTrainingConfig(
                posterior_file=str(tmp_path / "test.npz"),
                output_dir=str(tmp_path / "output"),
                parameter_names=[
                    "mass_1_source",
                    "mass_2_source",
                    "lambda_1",
                    "lambda_2",
                ],
                standardization_method="invalid",  # Invalid method
            )

    def test_create_flow_flexible_dimensionality_2d(self):
        """Test creating 2D flow (e.g., for NICER M-R)."""
        flow = create_flow(
            key=jax.random.key(0),
            dim=2,  # 2D flow
            flow_type="masked_autoregressive_flow",
        )
        assert flow.shape == (2,)  # Base distribution dimension

    def test_create_flow_flexible_dimensionality_4d(self):
        """Test creating 4D flow (default for GW)."""
        flow = create_flow(
            key=jax.random.key(0),
            dim=4,  # 4D flow
            flow_type="masked_autoregressive_flow",
        )
        assert flow.shape == (4,)  # Base distribution dimension

    def test_create_flow_default_dim_is_4(self):
        """Test that default dim=4 for backward compatibility."""
        flow = create_flow(
            key=jax.random.key(0),
            flow_type="masked_autoregressive_flow",
        )
        assert flow.shape == (4,)  # Default 4D


class TestFlowStandardizationMethods:
    """Test Flow class with both standardization methods."""

    def test_flow_zscore_standardization(self):
        """Test Flow class with z-score standardization."""
        # Create mock flow
        base_flow = create_flow(
            key=jax.random.key(0),
            dim=2,
            flow_type="masked_autoregressive_flow",
        )

        # Mock metadata with z-score
        metadata = {
            "standardize": True,
            "data_mean": [1.5, 12.0],
            "data_std": [0.2, 1.0],
        }

        flow_kwargs = {
            "flow_type": "masked_autoregressive_flow",
            "seed": 0,
            "standardize": True,
            "standardization_method": "zscore",
        }

        # Create Flow wrapper
        flow = Flow(base_flow, metadata, flow_kwargs)

        # Test standardization method detection
        assert flow.standardization_method == "zscore"

        # Test standardize/destandardize roundtrip
        original = jnp.array([[1.5, 12.0], [1.7, 13.0]])
        standardized = flow.standardize_input(original)
        recovered = flow.destandardize_output(standardized)

        np.testing.assert_allclose(original, recovered, rtol=1e-6)

    def test_flow_minmax_standardization(self):
        """Test Flow class with min-max standardization."""
        # Create mock flow
        base_flow = create_flow(
            key=jax.random.key(0),
            dim=2,
            flow_type="masked_autoregressive_flow",
        )

        # Mock metadata with min-max bounds
        metadata = {
            "standardize": True,
            "data_bounds_min": [1.0, 10.0],
            "data_bounds_max": [2.0, 14.0],
        }

        flow_kwargs = {
            "flow_type": "masked_autoregressive_flow",
            "seed": 0,
            "standardize": True,
            "standardization_method": "minmax",
        }

        # Create Flow wrapper
        flow = Flow(base_flow, metadata, flow_kwargs)

        # Test standardization method detection
        assert flow.standardization_method == "minmax"

        # Test standardize/destandardize roundtrip
        original = jnp.array([[1.5, 12.0], [1.7, 13.0]])
        standardized = flow.standardize_input(original)
        recovered = flow.destandardize_output(standardized)

        np.testing.assert_allclose(original, recovered, rtol=1e-6)

    def test_flow_no_standardization(self):
        """Test Flow class with standardization disabled."""
        # Create mock flow
        base_flow = create_flow(
            key=jax.random.key(0),
            dim=2,
            flow_type="masked_autoregressive_flow",
        )

        # Mock metadata with no standardization
        metadata = {"standardize": False}

        flow_kwargs = {
            "flow_type": "masked_autoregressive_flow",
            "seed": 0,
            "standardize": False,
        }

        # Create Flow wrapper
        flow = Flow(base_flow, metadata, flow_kwargs)

        # Test standardization method detection
        assert flow.standardization_method == "none"

        # Test that operations are identity
        original = jnp.array([[1.5, 12.0], [1.7, 13.0]])
        standardized = flow.standardize_input(original)
        recovered = flow.destandardize_output(standardized)

        np.testing.assert_allclose(original, standardized, rtol=1e-6)
        np.testing.assert_allclose(original, recovered, rtol=1e-6)
