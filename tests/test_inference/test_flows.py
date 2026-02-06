"""Tests for normalizing flows module - configuration and data operations."""

import pytest
import numpy as np
import yaml
from pydantic import ValidationError

from jesterTOV.inference.flows.config import FlowTrainingConfig
from jesterTOV.inference.flows.train_flow import (
    load_posterior,
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
def synthetic_nicer_posterior(tmp_path):
    """Create small synthetic NICER posterior with mass and radius."""
    n_samples = 1000
    np.random.seed(42)

    data = {
        "mass": np.random.uniform(1.0, 2.5, n_samples),  # Solar masses
        "radius": np.random.uniform(10.0, 15.0, n_samples),  # km
    }

    file_path = tmp_path / "test_nicer_posterior.npz"
    np.savez(file_path, **data)
    return file_path


@pytest.fixture
def synthetic_eos_posterior(tmp_path):
    """Create synthetic EOS posterior with multiple parameters."""
    n_samples = 1000
    np.random.seed(42)

    data = {
        "log_p1": np.random.uniform(32.0, 34.0, n_samples),
        "gamma1": np.random.uniform(1.5, 4.0, n_samples),
        "gamma2": np.random.uniform(1.0, 5.0, n_samples),
        "gamma3": np.random.uniform(1.0, 5.0, n_samples),
    }

    file_path = tmp_path / "test_eos_posterior.npz"
    np.savez(file_path, **data)
    return file_path


@pytest.fixture
def synthetic_single_param_posterior(tmp_path):
    """Create synthetic posterior with single parameter."""
    n_samples = 1000
    np.random.seed(42)

    data = {
        "hubble_constant": np.random.uniform(60.0, 80.0, n_samples),
    }

    file_path = tmp_path / "test_single_param_posterior.npz"
    np.savez(file_path, **data)
    return file_path


@pytest.fixture
def synthetic_mixed_posterior(tmp_path):
    """Create synthetic posterior with mixed parameter types."""
    n_samples = 1000
    np.random.seed(42)

    data = {
        "mass": np.random.uniform(1.0, 2.5, n_samples),
        "radius": np.random.uniform(10.0, 15.0, n_samples),
        "tidal_deformability": np.random.uniform(0, 1000, n_samples),
        "compactness": np.random.uniform(0.1, 0.3, n_samples),
        "frequency": np.random.uniform(1000, 3000, n_samples),
    }

    file_path = tmp_path / "test_mixed_posterior.npz"
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
        assert config.flow_type == "masked_autoregressive_flow"
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
        """Test loading GW posterior from npz file with required keys."""
        parameter_names = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]
        data, metadata = load_posterior(
            str(synthetic_gw_posterior), parameter_names, max_samples=50000
        )

        assert data.shape[0] == 1000  # All samples loaded
        assert data.shape[1] == 4  # 4 features
        assert metadata["n_samples_total"] == 1000
        assert metadata["n_samples_used"] == 1000
        assert metadata["parameter_names"] == parameter_names
        assert metadata["n_parameters"] == 4
        assert "filepath" in metadata

    def test_load_nicer_posterior_basic(self, synthetic_nicer_posterior):
        """Test loading NICER-style posterior with mass and radius."""
        parameter_names = ["mass", "radius"]
        data, metadata = load_posterior(
            str(synthetic_nicer_posterior), parameter_names, max_samples=50000
        )

        assert data.shape[0] == 1000  # All samples loaded
        assert data.shape[1] == 2  # 2 features (mass, radius)
        assert metadata["n_samples_total"] == 1000
        assert metadata["n_samples_used"] == 1000
        assert metadata["parameter_names"] == parameter_names
        assert metadata["n_parameters"] == 2

        # Check data ranges are reasonable for mass and radius
        assert np.all(data[:, 0] >= 1.0) and np.all(data[:, 0] <= 2.5)  # mass
        assert np.all(data[:, 1] >= 10.0) and np.all(data[:, 1] <= 15.0)  # radius

    def test_load_eos_posterior_basic(self, synthetic_eos_posterior):
        """Test loading EOS posterior with multiple parameters."""
        parameter_names = ["log_p1", "gamma1", "gamma2", "gamma3"]
        data, metadata = load_posterior(
            str(synthetic_eos_posterior), parameter_names, max_samples=50000
        )

        assert data.shape[0] == 1000
        assert data.shape[1] == 4
        assert metadata["parameter_names"] == parameter_names
        assert metadata["n_parameters"] == 4

    def test_load_single_parameter_posterior(self, synthetic_single_param_posterior):
        """Test loading posterior with single parameter."""
        parameter_names = ["hubble_constant"]
        data, metadata = load_posterior(
            str(synthetic_single_param_posterior), parameter_names, max_samples=50000
        )

        assert data.shape[0] == 1000
        assert data.shape[1] == 1  # Single parameter
        assert metadata["parameter_names"] == parameter_names
        assert metadata["n_parameters"] == 1

    def test_load_mixed_posterior_subset(self, synthetic_mixed_posterior):
        """Test loading subset of parameters from mixed posterior."""
        # Only load mass and radius, ignoring other parameters
        parameter_names = ["mass", "radius"]
        data, metadata = load_posterior(
            str(synthetic_mixed_posterior), parameter_names, max_samples=50000
        )

        assert data.shape[0] == 1000
        assert data.shape[1] == 2  # Only 2 parameters loaded
        assert metadata["parameter_names"] == parameter_names
        assert metadata["n_parameters"] == 2

    def test_load_mixed_posterior_all_params(self, synthetic_mixed_posterior):
        """Test loading all parameters from mixed posterior."""
        parameter_names = [
            "mass",
            "radius",
            "tidal_deformability",
            "compactness",
            "frequency",
        ]
        data, metadata = load_posterior(
            str(synthetic_mixed_posterior), parameter_names, max_samples=50000
        )

        assert data.shape[0] == 1000
        assert data.shape[1] == 5  # All 5 parameters
        assert metadata["parameter_names"] == parameter_names
        assert metadata["n_parameters"] == 5

    def test_load_parameters_different_order(self, synthetic_mixed_posterior):
        """Test that parameter order is preserved when loading."""
        # Load parameters in different order than stored
        parameter_names = ["radius", "mass", "compactness"]
        data, metadata = load_posterior(
            str(synthetic_mixed_posterior), parameter_names, max_samples=50000
        )

        assert data.shape[1] == 3
        assert metadata["parameter_names"] == parameter_names

        # Verify order is correct by checking ranges
        # Column 0 should be radius (10-15), column 1 should be mass (1-2.5)
        assert np.all(data[:, 0] >= 10.0) and np.all(data[:, 0] <= 15.0)  # radius
        assert np.all(data[:, 1] >= 1.0) and np.all(data[:, 1] <= 2.5)  # mass

    def test_load_gw_posterior_with_downsampling(self, synthetic_gw_posterior):
        """Test downsampling when n_samples > max_samples."""
        parameter_names = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]
        data, metadata = load_posterior(
            str(synthetic_gw_posterior), parameter_names, max_samples=500
        )

        # Should downsample to ~500 samples
        assert data.shape[0] <= 500
        assert data.shape[0] >= 450  # Allow some margin
        assert data.shape[1] == 4
        assert metadata["n_samples_total"] == 1000
        assert metadata["n_samples_used"] < 1000

    def test_load_nicer_posterior_with_downsampling(self, synthetic_nicer_posterior):
        """Test downsampling NICER posterior."""
        parameter_names = ["mass", "radius"]
        data, metadata = load_posterior(
            str(synthetic_nicer_posterior), parameter_names, max_samples=500
        )

        assert data.shape[0] <= 500
        assert data.shape[0] >= 450
        assert data.shape[1] == 2

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

        parameter_names = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]
        with pytest.raises(KeyError, match="Missing required parameters"):
            load_posterior(str(incomplete_file), parameter_names)

    def test_load_missing_nicer_parameter_fails(self, synthetic_nicer_posterior):
        """Test that requesting non-existent parameter fails."""
        # Try to load a parameter that doesn't exist
        parameter_names = ["mass", "radius", "nonexistent_param"]
        with pytest.raises(KeyError, match="Missing required parameters"):
            load_posterior(str(synthetic_nicer_posterior), parameter_names)

    def test_load_nonexistent_file_fails(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        nonexistent_file = tmp_path / "does_not_exist.npz"
        parameter_names = ["mass", "radius"]
        with pytest.raises(FileNotFoundError, match="Posterior file not found"):
            load_posterior(str(nonexistent_file), parameter_names)

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

        parameter_names = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]
        data, metadata = load_posterior(str(posterior_file), parameter_names)
        assert data.shape == (n_samples, 4)

    def test_load_nicer_handles_flattening(self, tmp_path):
        """Test that NICER posterior handles 2D arrays correctly."""
        posterior_file = tmp_path / "2d_nicer_posterior.npz"
        n_samples = 100
        np.savez(
            posterior_file,
            mass=np.random.uniform(1.0, 2.5, (n_samples, 1)),  # 2D array
            radius=np.random.uniform(10.0, 15.0, (n_samples, 1)),  # 2D array
        )

        parameter_names = ["mass", "radius"]
        data, metadata = load_posterior(str(posterior_file), parameter_names)
        assert data.shape == (n_samples, 2)

    def test_empty_parameter_list_fails(self, synthetic_gw_posterior):
        """Test that empty parameter list raises ValueError."""
        parameter_names = []
        with pytest.raises(ValueError, match="parameter_names cannot be empty"):
            load_posterior(
                str(synthetic_gw_posterior), parameter_names, max_samples=50000
            )


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

    def test_standardize_nicer_data_roundtrip(self):
        """Test standardization roundtrip for NICER-style data."""
        np.random.seed(42)
        # Simulate NICER mass-radius data
        mass = np.random.uniform(1.0, 2.5, 100)
        radius = np.random.uniform(10.0, 15.0, 100)
        original_data = np.column_stack([mass, radius])

        standardized, bounds = standardize_data(original_data)

        assert np.all(standardized >= 0)
        assert np.all(standardized <= 1)

        recovered_data = inverse_standardize_data(standardized, bounds)
        np.testing.assert_allclose(original_data, recovered_data, rtol=1e-10)

    def test_standardize_single_parameter(self):
        """Test standardization with single parameter."""
        np.random.seed(42)
        original_data = np.random.uniform(60.0, 80.0, (100, 1))

        standardized, bounds = standardize_data(original_data)

        assert np.all(standardized >= 0)
        assert np.all(standardized <= 1)
        assert bounds["min"].shape == (1,)
        assert bounds["max"].shape == (1,)

        recovered_data = inverse_standardize_data(standardized, bounds)
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

    def test_standardize_different_scales(self):
        """Test standardization with parameters on vastly different scales."""
        np.random.seed(42)
        # Simulate parameters with different scales
        mass = np.random.uniform(1.0, 2.5, 100)  # O(1)
        radius = np.random.uniform(10.0, 15.0, 100)  # O(10)
        lambda_param = np.random.uniform(0, 1000, 100)  # O(1000)
        compactness = np.random.uniform(0.1, 0.3, 100)  # O(0.1)

        original_data = np.column_stack([mass, radius, lambda_param, compactness])

        standardized, bounds = standardize_data(original_data)

        # All should be scaled to [0, 1]
        assert np.all(standardized >= 0)
        assert np.all(standardized <= 1)

        # Check each column independently
        for i in range(4):
            assert standardized[:, i].min() >= 0
            assert standardized[:, i].max() <= 1

        # Roundtrip
        recovered_data = inverse_standardize_data(standardized, bounds)
        np.testing.assert_allclose(original_data, recovered_data, rtol=1e-10)

    def test_standardize_constant_feature(self):
        """Test standardization handles constant features."""
        data = np.ones((100, 4)) * 42.0  # All features constant

        standardized, bounds = standardize_data(data)

        # Should handle constant features gracefully (avoid division by zero)
        assert not np.any(np.isnan(standardized))
        assert not np.any(np.isinf(standardized))

    def test_standardize_preserves_shape(self):
        """Test that standardization preserves data shape."""
        for n_params in [1, 2, 4, 10]:
            np.random.seed(42)
            data = np.random.randn(100, n_params)
            standardized, bounds = standardize_data(data)

            assert standardized.shape == data.shape
            assert bounds["min"].shape == (n_params,)
            assert bounds["max"].shape == (n_params,)
