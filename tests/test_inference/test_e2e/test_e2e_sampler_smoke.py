"""Nightly smoke tests: every sampler x every EOS parametrization.

Each test samples from the prior only (zero likelihood) with ultra-light
hyperparameters. The goal is to catch fundamental breakage — a broken sampler
or EOS that errors on sampling — before it reaches users.

Matrix covered:
  Samplers : smc-rw, flowmc, blackjax-ns-aw
  EOS      : spectral, metamodel_cse, metamodel_peak_cse
"""

from pathlib import Path
from typing import Any

import jax
import pytest

from jesterTOV.inference.config.schema import InferenceConfig
from jesterTOV.inference.run_inference import (
    determine_keep_names,
    setup_likelihood,
    setup_prior,
    setup_transform,
)
from jesterTOV.inference.samplers import create_sampler

from .conftest import (
    BLACKJAX_NS_AW_LIGHTWEIGHT,
    FLOWMC_LIGHTWEIGHT,
    LIGHTWEIGHT_TOV,
    PEAK_CSE_PARAMS,
    SMC_RW_LIGHTWEIGHT,
    SPECTRAL_PARAMS,
    validate_sampler_output,
)

# ---------------------------------------------------------------------------
# Per-sampler lightweight overrides (applied on top of the base config)
# ---------------------------------------------------------------------------

_SAMPLER_PARAMS: dict[str, dict[str, Any]] = {
    "smc-rw": {
        "type": "smc-rw",
        **SMC_RW_LIGHTWEIGHT,
        "n_particles": 50,
        "n_mcmc_steps": 2,
    },
    "flowmc": {
        "type": "flowmc",
        **FLOWMC_LIGHTWEIGHT,
        "n_chains": 20,
        "n_loop_training": 2,
        "n_loop_production": 2,
        "n_local_steps": 5,
        "n_global_steps": 5,
    },
    "blackjax-ns-aw": {
        "type": "blackjax-ns-aw",
        **BLACKJAX_NS_AW_LIGHTWEIGHT,
        "n_live": 50,
        "n_target": 10,
        "max_mcmc": 200,
        "termination_dlogz": 1.0,
    },
}

_SAMPLER_IDS = list(_SAMPLER_PARAMS.keys())

# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

_ZERO_LIKELIHOODS = [
    {"type": "constraints_eos", "enabled": True},
    {"type": "zero", "enabled": True},
]


def _base(sampler_type: str, prior_file: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "seed": 42,
        "dry_run": False,
        "validate_only": False,
        "prior": {"specification_file": str(prior_file)},
        "likelihoods": _ZERO_LIKELIHOODS,
        "sampler": {
            **_SAMPLER_PARAMS[sampler_type],
            "output_dir": str(output_dir),
            "n_eos_samples": 50,
        },
        "postprocessing": {"enabled": False},
    }


def _spectral_config(
    sampler_type: str, prior_file: Path, output_dir: Path
) -> dict[str, Any]:
    cfg = _base(sampler_type, prior_file, output_dir)
    cfg["eos"] = {
        "type": "spectral",
        "crust_name": "SLy",
        "n_points_high": 30,
    }
    cfg["tov"] = {"type": "gr", **LIGHTWEIGHT_TOV}
    return cfg


def _metamodel_cse_config(
    sampler_type: str, prior_file: Path, output_dir: Path
) -> dict[str, Any]:
    cfg = _base(sampler_type, prior_file, output_dir)
    cfg["eos"] = {
        "type": "metamodel_cse",
        "nb_CSE": 8,
        "nmax_nsat": 25.0,
        "crust_name": "DH",
        "ndat_metamodel": 30,
    }
    cfg["tov"] = {"type": "gr", **LIGHTWEIGHT_TOV}
    return cfg


def _peak_cse_config(
    sampler_type: str, prior_file: Path, output_dir: Path
) -> dict[str, Any]:
    cfg = _base(sampler_type, prior_file, output_dir)
    cfg["eos"] = {
        "type": "metamodel_peak_cse",
        "ndat_CSE": 30,
        "ndat_metamodel": 30,
        "nmax_nsat": 25.0,
        "crust_name": "DH",
    }
    cfg["tov"] = {"type": "gr", **LIGHTWEIGHT_TOV}
    return cfg


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run(config_dict: dict[str, Any]):
    config = InferenceConfig(**config_dict)
    prior, _ = setup_prior(config)
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
    sampler.sample(jax.random.PRNGKey(config.seed))
    return sampler.get_sampler_output()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.blackjax
class TestSamplerEOSSmoke:
    """Smoke tests: all sampler x EOS combinations sampling from their prior.

    A failure here means a sampler or EOS is fundamentally broken, not a
    statistical issue.
    """

    @pytest.mark.parametrize("sampler_type", _SAMPLER_IDS)
    def test_spectral_prior_only(self, sampler_type, spectral_prior_file, e2e_temp_dir):
        cfg = _spectral_config(sampler_type, spectral_prior_file, e2e_temp_dir)
        output = _run(cfg)
        validate_sampler_output(output, expected_params=SPECTRAL_PARAMS, min_samples=5)

    @pytest.mark.parametrize("sampler_type", _SAMPLER_IDS)
    def test_metamodel_cse_prior_only(
        self, sampler_type, chieft_prior_file, e2e_temp_dir
    ):
        cfg = _metamodel_cse_config(sampler_type, chieft_prior_file, e2e_temp_dir)
        output = _run(cfg)
        validate_sampler_output(
            output, expected_params=["K_sat", "L_sym", "nbreak"], min_samples=5
        )

    @pytest.mark.parametrize("sampler_type", _SAMPLER_IDS)
    def test_metamodel_peak_cse_prior_only(
        self, sampler_type, peak_cse_prior_file, e2e_temp_dir
    ):
        cfg = _peak_cse_config(sampler_type, peak_cse_prior_file, e2e_temp_dir)
        output = _run(cfg)
        validate_sampler_output(output, expected_params=PEAK_CSE_PARAMS, min_samples=5)
