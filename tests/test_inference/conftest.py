"""Test fixtures for inference module tests."""

import pytest
import tempfile
from pathlib import Path
import yaml
import jax.numpy as jnp


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_prior_file(temp_dir):
    """Create a sample .prior file for testing."""
    prior_content = """K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])
"""
    prior_file = temp_dir / "test.prior"
    prior_file.write_text(prior_content)
    return prior_file


@pytest.fixture
def sample_prior_file_with_cse(temp_dir):
    """Create a sample .prior file with CSE parameters for testing."""
    prior_content = """K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])
nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])
"""
    prior_file = temp_dir / "test_cse.prior"
    prior_file.write_text(prior_content)
    return prior_file


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        "seed": 42,
        "dry_run": False,
        "validate_only": False,
        "transform": {
            "type": "metamodel",
            "ndat_metamodel": 100,
            "nmax_nsat": 2.0,
            "nb_CSE": 0,
            "min_nsat_TOV": 0.75,
            "ndat_TOV": 100,
            "nb_masses": 100,
            "crust_name": "DH",
        },
        "prior": {"specification_file": "test.prior"},
        "likelihoods": [{"type": "zero", "enabled": True, "parameters": {}}],
        "sampler": {
            "type": "flowmc",
            "n_chains": 4,
            "n_loop_training": 2,
            "n_loop_production": 2,
            "n_local_steps": 10,
            "n_global_steps": 10,
            "n_epochs": 20,
            "learning_rate": 0.001,
            "output_dir": "./test_output/",
        },
        "data_paths": {},
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_dict, sample_prior_file):
    """Create a sample config.yaml file for testing."""
    # Update prior path to point to temp prior file
    config = sample_config_dict.copy()
    config["prior"]["specification_file"] = str(sample_prior_file)

    config_file = temp_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def sample_nep_params():
    """Sample NEP (Nuclear Empirical Parameters) for testing."""
    return {
        "E_sat": -16.0,
        "K_sat": 220.0,
        "Q_sat": 0.0,
        "Z_sat": 0.0,
        "E_sym": 32.0,
        "L_sym": 90.0,
        "K_sym": 0.0,
        "Q_sym": 0.0,
        "Z_sym": 0.0,
    }


@pytest.fixture
def sample_nep_params_array(sample_nep_params):
    """Sample NEP parameters as JAX array for testing."""
    return jnp.array(list(sample_nep_params.values()))


@pytest.fixture
def sample_cse_params():
    """Sample CSE grid parameters for testing."""
    return {
        "nbreak": 0.24,
        "e0": 150.0,
        "e1": 200.0,
        "e2": 250.0,
        "e3": 300.0,
        "e4": 350.0,
        "e5": 400.0,
        "e6": 450.0,
        "e7": 500.0,
    }


@pytest.fixture
def realistic_nep_stiff():
    """Realistic stiff EOS NEP parameters for testing (should produce ~2 Msun NS)."""
    return {
        "E_sat": -16.0,
        "K_sat": 240.0,  # Stiff
        "Q_sat": 0.0,
        "Z_sat": 0.0,
        "E_sym": 32.0,
        "L_sym": 90.0,  # High L_sym for stiff EOS
        "K_sym": 0.0,
        "Q_sym": 0.0,
        "Z_sym": 0.0,
    }


@pytest.fixture
def realistic_nep_soft():
    """Realistic soft EOS NEP parameters for testing (should produce ~1.5 Msun NS)."""
    return {
        "E_sat": -16.0,
        "K_sat": 200.0,  # Soft
        "Q_sat": 0.0,
        "Z_sat": 0.0,
        "E_sym": 32.0,
        "L_sym": 40.0,  # Low L_sym for soft EOS
        "K_sym": 0.0,
        "Q_sym": 0.0,
        "Z_sym": 0.0,
    }


@pytest.fixture
def mock_gw_data():
    """Mock GW170817 data for testing likelihoods."""
    return {
        "m1_source": 1.46,  # Solar masses
        "m2_source": 1.27,
        "m1_source_err": 0.1,
        "m2_source_err": 0.1,
        "lambda_tilde": 300.0,
        "lambda_tilde_err": 100.0,
    }


@pytest.fixture
def mock_nicer_data():
    """Mock NICER data for testing likelihoods."""
    # Simple mock: just return some sample points
    return {
        "mass_samples": jnp.linspace(1.3, 1.5, 100),
        "radius_samples": jnp.linspace(11.0, 13.0, 100),
        "weights": jnp.ones(100) / 100.0,
    }
