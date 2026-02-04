"""End-to-end test fixtures for inference pipeline.

These fixtures provide lightweight configurations for testing the complete
sampling pipeline. Target runtime is less than 2 minutes per test.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp

from jesterTOV.inference.samplers import SamplerOutput

# Enable 64-bit precision for all E2E tests
jax.config.update("jax_enable_x64", True)


# ============================================================================
# LIGHTWEIGHT HYPERPARAMETER CONSTANTS
# Target: <2 min per test
# ============================================================================

# Transform parameters (reduced for speed)
LIGHTWEIGHT_TRANSFORM = {
    "ndat_metamodel": 30,  # 100 -> 30
    "ndat_TOV": 30,  # 100 -> 30
    "nb_masses": 20,  # 100 -> 20
}

# FlowMC lightweight params
FLOWMC_LIGHTWEIGHT = {
    "n_chains": 50,  # 1000 -> 50
    "n_loop_training": 3,  # 30 -> 3
    "n_loop_production": 3,  # 20 -> 3
    "n_local_steps": 10,  # 100 -> 10
    "n_global_steps": 10,  # 100 -> 10
    "n_epochs": 5,  # 30 -> 5
    "learning_rate": 0.001,
}

# SMC-RW lightweight params
SMC_RW_LIGHTWEIGHT = {
    "n_particles": 100,  # 2000 -> 100
    "n_mcmc_steps": 3,  # 10 -> 3
    "target_ess": 0.9,
    "random_walk_sigma": 0.1,
}

# BlackJAX NS-AW lightweight params
BLACKJAX_NS_AW_LIGHTWEIGHT = {
    "n_live": 100,  # 1400 -> 100
    "n_delete_frac": 0.5,
    "n_target": 20,  # 60 -> 20
    "max_mcmc": 500,  # 5000 -> 500
    "max_proposals": 100,  # 1000 -> 100
    "termination_dlogz": 0.5,  # 0.1 -> 0.5 (less strict for speed)
}

# NEP parameter names (without CSE)
NEP_PARAMS = ["K_sat", "Q_sat", "Z_sat", "E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym"]


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def e2e_temp_dir() -> Iterator[Path]:
    """Create a temporary directory for E2E test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_prior_file(e2e_temp_dir: Path) -> Path:
    """Create a minimal prior file for testing (NEP params only, no CSE)."""
    prior_content = """K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])
"""
    prior_file = e2e_temp_dir / "minimal.prior"
    prior_file.write_text(prior_content)
    return prior_file


@pytest.fixture
def chieft_prior_file(e2e_temp_dir: Path) -> Path:
    """Create a prior file with nbreak for chiEFT tests (requires CSE)."""
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
    prior_file = e2e_temp_dir / "chieft.prior"
    prior_file.write_text(prior_content)
    return prior_file


# ============================================================================
# CONFIG BUILDER FUNCTIONS
# ============================================================================


def build_prior_only_config(
    sampler_config: dict[str, Any], prior_file: Path, output_dir: Path
) -> dict[str, Any]:
    """Build a prior-only config (cheapest - no likelihood computation)."""
    return {
        "seed": 42,
        "dry_run": False,
        "validate_only": False,
        "transform": {
            "type": "metamodel",
            "nb_CSE": 0,
            "nmax_nsat": 2.0,
            "min_nsat_TOV": 0.75,
            "crust_name": "DH",
            **LIGHTWEIGHT_TRANSFORM,
        },
        "prior": {"specification_file": str(prior_file)},
        "likelihoods": [
            {"type": "constraints_eos", "enabled": True, "parameters": {}},
            {"type": "zero", "enabled": True, "parameters": {}},
        ],
        "sampler": {
            **sampler_config,
            "output_dir": str(output_dir),
            "n_eos_samples": 50,  # Very small for speed
        },
        "postprocessing": {"enabled": False},
    }


def build_chieft_config(
    sampler_config: dict[str, Any], prior_file: Path, output_dir: Path
) -> dict[str, Any]:
    """Build a chiEFT config (cheap but exercises real likelihood)."""
    return {
        "seed": 42,
        "dry_run": False,
        "validate_only": False,
        "transform": {
            "type": "metamodel_cse",
            "nb_CSE": 8,
            "nmax_nsat": 25.0,
            "min_nsat_TOV": 0.75,
            "crust_name": "DH",
            **LIGHTWEIGHT_TRANSFORM,
        },
        "prior": {"specification_file": str(prior_file)},
        "likelihoods": [
            {"type": "constraints_eos", "enabled": True, "parameters": {}},
            {
                "type": "chieft",
                "enabled": True,
                "parameters": {
                    "nb_n": 30,  # 100 -> 30 for speed
                },
            },
        ],
        "sampler": {
            **sampler_config,
            "output_dir": str(output_dir),
            "n_eos_samples": 50,
        },
        "postprocessing": {"enabled": False},
    }


# ============================================================================
# SAMPLER CONFIG FIXTURES
# ============================================================================


@pytest.fixture
def flowmc_prior_config(minimal_prior_file: Path, e2e_temp_dir: Path) -> dict[str, Any]:
    """FlowMC config with prior-only likelihood."""
    sampler_config = {"type": "flowmc", **FLOWMC_LIGHTWEIGHT}
    return build_prior_only_config(sampler_config, minimal_prior_file, e2e_temp_dir)


@pytest.fixture
def flowmc_chieft_config(chieft_prior_file: Path, e2e_temp_dir: Path) -> dict[str, Any]:
    """FlowMC config with chiEFT likelihood."""
    sampler_config = {"type": "flowmc", **FLOWMC_LIGHTWEIGHT}
    return build_chieft_config(sampler_config, chieft_prior_file, e2e_temp_dir)


@pytest.fixture
def smc_rw_prior_config(minimal_prior_file: Path, e2e_temp_dir: Path) -> dict[str, Any]:
    """SMC-RW config with prior-only likelihood."""
    sampler_config = {"type": "smc-rw", **SMC_RW_LIGHTWEIGHT}
    return build_prior_only_config(sampler_config, minimal_prior_file, e2e_temp_dir)


@pytest.fixture
def smc_rw_chieft_config(chieft_prior_file: Path, e2e_temp_dir: Path) -> dict[str, Any]:
    """SMC-RW config with chiEFT likelihood."""
    sampler_config = {"type": "smc-rw", **SMC_RW_LIGHTWEIGHT}
    return build_chieft_config(sampler_config, chieft_prior_file, e2e_temp_dir)


@pytest.fixture
def blackjax_ns_aw_prior_config(minimal_prior_file: Path, e2e_temp_dir: Path) -> dict[str, Any]:
    """BlackJAX NS-AW config with prior-only likelihood."""
    sampler_config = {"type": "blackjax-ns-aw", **BLACKJAX_NS_AW_LIGHTWEIGHT}
    return build_prior_only_config(sampler_config, minimal_prior_file, e2e_temp_dir)


@pytest.fixture
def blackjax_ns_aw_chieft_config(chieft_prior_file: Path, e2e_temp_dir: Path) -> dict[str, Any]:
    """BlackJAX NS-AW config with chiEFT likelihood."""
    sampler_config = {"type": "blackjax-ns-aw", **BLACKJAX_NS_AW_LIGHTWEIGHT}
    return build_chieft_config(sampler_config, chieft_prior_file, e2e_temp_dir)


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def validate_sampler_output(
    output: SamplerOutput, expected_params: list[str], min_samples: int = 10
) -> None:
    """Validate SamplerOutput structure and content.

    Parameters
    ----------
    output : SamplerOutput
        Output from sampler.get_sampler_output()
    expected_params : list[str]
        Parameter names expected in samples dict
    min_samples : int
        Minimum number of samples expected

    Raises
    ------
    AssertionError
        If validation fails
    """

    # Check type
    assert isinstance(
        output, SamplerOutput
    ), f"Expected SamplerOutput, got {type(output)}"

    # Check samples dict has expected parameters
    for param in expected_params:
        assert param in output.samples, f"Missing parameter: {param}"
        arr = output.samples[param]
        assert jnp.isfinite(arr).all(), f"Parameter {param} has non-finite values"
        assert len(arr) >= min_samples, f"Parameter {param} has only {len(arr)} samples"

    # Check log_prob
    assert output.log_prob is not None, "log_prob is None"
    assert (
        len(output.log_prob) >= min_samples
    ), f"log_prob has only {len(output.log_prob)} samples"
    # Note: log_prob can be -inf for rejected samples, so we check for NaN only
    assert not jnp.isnan(output.log_prob).any(), "log_prob contains NaN"

    # Check all sample arrays have same length
    first_param = list(output.samples.keys())[0]
    n_samples = len(output.samples[first_param])
    for param, arr in output.samples.items():
        assert (
            len(arr) == n_samples
        ), f"Sample length mismatch: {param} has {len(arr)}, expected {n_samples}"
    assert (
        len(output.log_prob) == n_samples
    ), f"log_prob length {len(output.log_prob)} != sample length {n_samples}"
