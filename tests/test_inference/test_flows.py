"""Tests for normalizing flows module - configuration and data operations."""

import pytest
import numpy as np
import yaml
from pydantic import ValidationError

from jesterTOV.inference.flows.config import FlowTrainingConfig
from jesterTOV.inference.flows.train_flow import (
    load_gw_posterior,
    standardize_data,
    inverse_standardize_data,
)


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
        "flow_type": "triangular_spline_flow",
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
        assert config.flow_type == "triangular_spline_flow"

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
            "triangular_spline_flow",
        ]
        for flow_type in valid_types:
            sample_flow_config_dict["flow_type"] = flow_type
            config = FlowTrainingConfig(**sample_flow_config_dict)
            assert config.flow_type == flow_type

    def test_default_values(self, tmp_path):
        """Test that defaults are set correctly."""
        config = FlowTrainingConfig(
            posterior_file=str(tmp_path / "test.npz"),
            output_dir=str(tmp_path / "output"),
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
        assert config.flow_type == "triangular_spline_flow"
        assert config.standardize is False
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

    def test_load_gw_posterior_basic(self, synthetic_gw_posterior):
        """Test loading posterior from npz file with required keys."""
        data, metadata = load_gw_posterior(
            str(synthetic_gw_posterior), max_samples=50000
        )

        assert data.shape[0] == 1000  # All samples loaded
        assert data.shape[1] == 4  # 4 features
        assert metadata["n_samples_total"] == 1000
        assert metadata["n_samples_used"] == 1000
        assert "filepath" in metadata

    def test_load_gw_posterior_with_downsampling(self, synthetic_gw_posterior):
        """Test downsampling when n_samples > max_samples."""
        data, metadata = load_gw_posterior(str(synthetic_gw_posterior), max_samples=500)

        # Should downsample to ~500 samples
        assert data.shape[0] <= 500
        assert data.shape[0] >= 450  # Allow some margin
        assert data.shape[1] == 4
        assert metadata["n_samples_total"] == 1000
        assert metadata["n_samples_used"] < 1000

    def test_load_missing_keys_fails(self, tmp_path):
        """Test that missing required keys raise KeyError."""
        # Create file with missing keys
        incomplete_file = tmp_path / "incomplete.npz"
        np.savez(
            incomplete_file,
            mass_1_source=np.random.randn(100),
            mass_2_source=np.random.randn(100),
            # Missing lambda_1 and lambda_2
        )

        with pytest.raises(KeyError, match="Missing required keys"):
            load_gw_posterior(str(incomplete_file))

    def test_load_nonexistent_file_fails(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        nonexistent_file = tmp_path / "does_not_exist.npz"
        with pytest.raises(FileNotFoundError, match="Posterior file not found"):
            load_gw_posterior(str(nonexistent_file))

    def test_load_handles_flattening(self, tmp_path):
        """Test that multi-dimensional arrays are flattened correctly."""
        # Create file with 2D arrays
        posterior_file = tmp_path / "2d_posterior.npz"
        n_samples = 100
        np.savez(
            posterior_file,
            mass_1_source=np.random.randn(n_samples, 1),  # 2D array
            mass_2_source=np.random.randn(n_samples, 1),
            lambda_1=np.random.randn(n_samples, 1),
            lambda_2=np.random.randn(n_samples, 1),
        )

        data, metadata = load_gw_posterior(str(posterior_file))
        assert data.shape == (n_samples, 4)


# ======================
# Data Preprocessing Tests
# ======================


class TestDataPreprocessing:
    """Test data preprocessing functions."""

    def test_standardize_data_roundtrip(self):
        """Test standardize -> inverse_standardize identity."""
        np.random.seed(42)
        original_data = np.random.randn(100, 4) * 100 + 500

        # Standardize
        standardized, bounds = standardize_data(original_data)

        # Check standardized data is in [0, 1]
        assert np.all(standardized >= 0)
        assert np.all(standardized <= 1)

        # Inverse transform
        recovered_data = inverse_standardize_data(standardized, bounds)

        # Check roundtrip identity
        np.testing.assert_allclose(original_data, recovered_data, rtol=1e-10)

    def test_standardize_data_bounds(self):
        """Test that standardized data is in [0, 1]."""
        np.random.seed(42)
        data = np.random.uniform(10, 100, (100, 4))

        standardized, bounds = standardize_data(data)

        assert np.all(standardized >= 0)
        assert np.all(standardized <= 1)
        assert "min" in bounds
        assert "max" in bounds
        assert bounds["min"].shape == (4,)
        assert bounds["max"].shape == (4,)

    def test_standardize_constant_feature(self):
        """Test standardization handles constant features."""
        data = np.ones((100, 4)) * 42.0  # All features constant

        standardized, bounds = standardize_data(data)

        # Should handle constant features gracefully (avoid division by zero)
        assert not np.any(np.isnan(standardized))
        assert not np.any(np.isinf(standardized))
