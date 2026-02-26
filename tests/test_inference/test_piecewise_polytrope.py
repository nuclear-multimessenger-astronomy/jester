r"""Tests for the piecewise polytrope EOS implementation.

Covers:
- Unit tests for PiecewisePolytrope_EOS_model.construct_eos()
- Config parsing for PiecewisePolytropeEOSConfig
- JesterTransform creation via from_config()
- E2E SMC-RW prior-only inference run
"""

import pytest
import jax
import jax.numpy as jnp

from jesterTOV.eos.piecewise_polytrope import PiecewisePolytrope_EOS_model
from jesterTOV.inference.config.schema import (
    PiecewisePolytropeEOSConfig,
    InferenceConfig,
    TOVConfig,
)
from jesterTOV.inference.transforms import JesterTransform
from jesterTOV.inference.run_inference import (
    setup_prior,
    setup_transform,
    setup_likelihood,
    determine_keep_names,
)
from jesterTOV.inference.samplers import create_sampler
from jesterTOV.tov.gr import GRTOVSolver

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Fiducial parameters from the APR3 EOS (Read et al. 2009, Table III)
FIDUCIAL_PARAMS = {
    "logp1_si": 34.384,
    "gamma1": 3.005,
    "gamma2": 2.988,
    "gamma3": 2.851,
}

# Prior bounds used in the example config
PRIOR_BOUNDS = {
    "logp1_si": (33.5, 34.5),
    "gamma1": (1.2, 4.5),
    "gamma2": (1.2, 4.5),
    "gamma3": (1.2, 4.5),
}

PP_PARAMS = ["logp1_si", "gamma1", "gamma2", "gamma3"]


# ============================================================================
# Unit tests: EOS model
# ============================================================================


class TestPiecewisePolytropeEOSModel:
    """Unit tests for PiecewisePolytrope_EOS_model."""

    def test_construct_eos_returns_correct_shapes(self):
        """EOS output arrays should have shape (n_points,)."""
        n_points = 200
        model = PiecewisePolytrope_EOS_model(n_points=n_points)
        eos_data = model.construct_eos(FIDUCIAL_PARAMS)

        assert eos_data.ns.shape == (n_points,)
        assert eos_data.ps.shape == (n_points,)
        assert eos_data.hs.shape == (n_points,)
        assert eos_data.es.shape == (n_points,)
        assert eos_data.dloge_dlogps.shape == (n_points,)
        assert eos_data.cs2.shape == (n_points,)

    def test_required_parameters(self):
        """Model should declare the four PP parameters."""
        model = PiecewisePolytrope_EOS_model()
        assert model.get_required_parameters() == PP_PARAMS

    def test_pressure_is_monotone(self):
        """Pressure grid must be strictly increasing (log-spaced)."""
        model = PiecewisePolytrope_EOS_model(n_points=200)
        eos_data = model.construct_eos(FIDUCIAL_PARAMS)
        diffs = jnp.diff(eos_data.ps)
        assert jnp.all(diffs > 0), "Pressure must be monotonically increasing"

    def test_number_density_is_positive(self):
        """Number densities must be strictly positive."""
        model = PiecewisePolytrope_EOS_model(n_points=200)
        eos_data = model.construct_eos(FIDUCIAL_PARAMS)
        assert jnp.all(eos_data.ns > 0), "Number densities must be positive"

    def test_energy_density_is_positive(self):
        """Energy densities must be strictly positive."""
        model = PiecewisePolytrope_EOS_model(n_points=200)
        eos_data = model.construct_eos(FIDUCIAL_PARAMS)
        assert jnp.all(eos_data.es > 0), "Energy densities must be positive"

    def test_cs2_positive(self):
        """Speed of sound squared must be positive for fiducial APR3-like params."""
        model = PiecewisePolytrope_EOS_model(n_points=200)
        eos_data = model.construct_eos(FIDUCIAL_PARAMS)
        assert jnp.all(eos_data.cs2 > 0), "cs2 must be positive"

    def test_no_nans_in_eos(self):
        """EOS output must not contain NaN for fiducial parameters."""
        model = PiecewisePolytrope_EOS_model(n_points=200)
        eos_data = model.construct_eos(FIDUCIAL_PARAMS)
        for field_name, arr in [
            ("ns", eos_data.ns),
            ("ps", eos_data.ps),
            ("hs", eos_data.hs),
            ("es", eos_data.es),
            ("dloge_dlogps", eos_data.dloge_dlogps),
            ("cs2", eos_data.cs2),
        ]:
            assert not jnp.any(jnp.isnan(arr)), f"NaN found in {field_name}"

    def test_construct_eos_with_soft_params(self):
        """Soft EOS (low logp1, low gammas) should still produce valid output."""
        soft_params = {
            "logp1_si": 33.5,
            "gamma1": 1.5,
            "gamma2": 1.5,
            "gamma3": 1.5,
        }
        model = PiecewisePolytrope_EOS_model(n_points=100)
        eos_data = model.construct_eos(soft_params)
        assert eos_data.ns.shape == (100,)
        assert not jnp.any(jnp.isnan(eos_data.ps))

    def test_construct_eos_with_stiff_params(self):
        """Stiff EOS (high logp1, high gammas) should still produce valid output."""
        stiff_params = {
            "logp1_si": 34.5,
            "gamma1": 4.0,
            "gamma2": 4.0,
            "gamma3": 4.0,
        }
        model = PiecewisePolytrope_EOS_model(n_points=100)
        eos_data = model.construct_eos(stiff_params)
        assert eos_data.ns.shape == (100,)
        assert not jnp.any(jnp.isnan(eos_data.ps))

    def test_n_points_respected(self):
        """n_points parameter should control the EOS grid size."""
        for n_pts in [50, 100, 500]:
            model = PiecewisePolytrope_EOS_model(n_points=n_pts)
            eos_data = model.construct_eos(FIDUCIAL_PARAMS)
            assert eos_data.ns.shape == (n_pts,)

    def test_tov_solve_gives_finite_masses(self):
        """TOV solve on fiducial EOS should give finite masses."""
        model = PiecewisePolytrope_EOS_model(n_points=300)
        eos_data = model.construct_eos(FIDUCIAL_PARAMS)
        solver = GRTOVSolver()
        family = solver.construct_family(eos_data, ndat=50, min_nsat=0.75)
        assert jnp.any(jnp.isfinite(family.masses)), "No finite masses from TOV solve"
        # APR3 has max mass > 2 Msun
        valid_masses = family.masses[jnp.isfinite(family.masses)]
        assert jnp.max(valid_masses) > 2.0, "Max mass should exceed 2 Msun for APR3"


# ============================================================================
# Config tests
# ============================================================================


class TestPiecewisePolytropeConfig:
    """Tests for PiecewisePolytropeEOSConfig validation."""

    def test_default_config_creation(self):
        """Default config should be created with correct defaults."""
        config = PiecewisePolytropeEOSConfig()
        assert config.type == "piecewise_polytrope"
        assert config.n_points == 500
        assert config.nb_CSE == 0

    def test_custom_n_points(self):
        """n_points can be customized."""
        config = PiecewisePolytropeEOSConfig(n_points=200)
        assert config.n_points == 200

    def test_nb_cse_must_be_zero(self):
        """nb_CSE must be 0; non-zero value should raise ValueError."""
        with pytest.raises(Exception):
            PiecewisePolytropeEOSConfig(nb_CSE=4)

    def test_type_discriminator(self):
        """Type field should be 'piecewise_polytrope'."""
        config = PiecewisePolytropeEOSConfig()
        assert config.type == "piecewise_polytrope"

    def test_inference_config_parses_pp_eos(self, tmp_path):
        """InferenceConfig should accept piecewise_polytrope EOS type."""
        prior_file = tmp_path / "test.prior"
        prior_file.write_text(
            'logp1_si = UniformPrior(33.5, 34.5, parameter_names=["logp1_si"])\n'
            'gamma1 = UniformPrior(1.2, 4.5, parameter_names=["gamma1"])\n'
            'gamma2 = UniformPrior(1.2, 4.5, parameter_names=["gamma2"])\n'
            'gamma3 = UniformPrior(1.2, 4.5, parameter_names=["gamma3"])\n'
        )
        config_dict = {
            "seed": 42,
            "eos": {"type": "piecewise_polytrope", "n_points": 100},
            "tov": {"type": "gr", "ndat_TOV": 50},
            "prior": {"specification_file": str(prior_file)},
            "likelihoods": [
                {"type": "constraints_eos", "enabled": True},
                {"type": "zero", "enabled": True},
            ],
            "sampler": {
                "type": "smc-rw",
                "n_particles": 10,
                "n_mcmc_steps": 2,
                "output_dir": str(tmp_path),
            },
        }
        config = InferenceConfig(**config_dict)
        assert config.eos.type == "piecewise_polytrope"  # type: ignore[union-attr]


# ============================================================================
# Transform factory tests
# ============================================================================


class TestPiecewisePolytropeTransform:
    """Tests for JesterTransform with piecewise polytrope EOS."""

    def test_from_config_creates_pp_transform(self):
        """JesterTransform.from_config should create a PP transform."""
        eos_config = PiecewisePolytropeEOSConfig(n_points=100)
        tov_config = TOVConfig(type="gr", ndat_TOV=50)

        transform = JesterTransform.from_config(eos_config, tov_config)

        assert transform is not None
        assert "PiecewisePolytrope_EOS_model" in transform.get_eos_type()

    def test_transform_parameter_names(self):
        """Transform should require the four PP parameter names."""
        eos_config = PiecewisePolytropeEOSConfig(n_points=100)
        tov_config = TOVConfig(type="gr", ndat_TOV=50)

        transform = JesterTransform.from_config(eos_config, tov_config)

        assert transform.get_parameter_names() == PP_PARAMS

    def test_transform_forward_produces_masses(self):
        """Forward pass should return masses, radii, and lambdas."""
        eos_config = PiecewisePolytropeEOSConfig(n_points=100)
        tov_config = TOVConfig(type="gr", ndat_TOV=30, min_nsat_TOV=0.75)

        transform = JesterTransform.from_config(eos_config, tov_config)
        result = transform.forward(FIDUCIAL_PARAMS)

        assert "masses_EOS" in result
        assert "radii_EOS" in result
        assert "Lambdas_EOS" in result
        assert result["masses_EOS"].shape == (30,)

    def test_transform_forward_has_no_nans_for_fiducial(self):
        """Forward pass should be NaN-free for fiducial APR3 parameters."""
        eos_config = PiecewisePolytropeEOSConfig(n_points=100)
        tov_config = TOVConfig(type="gr", ndat_TOV=30, min_nsat_TOV=0.75)

        transform = JesterTransform.from_config(eos_config, tov_config)
        result = transform.forward(FIDUCIAL_PARAMS)

        # After nan_to_num cleaning, all finite
        for key in ["masses_EOS", "radii_EOS", "logpc_EOS"]:
            assert jnp.all(jnp.isfinite(result[key])), f"Non-finite values in {key}"


# ============================================================================
# E2E test
# ============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.e2e
class TestPiecewisePolytropeSMCRWE2E:
    """End-to-end SMC-RW inference test with piecewise polytrope EOS."""

    def test_smc_rw_pp_prior_only_pipeline(self, tmp_path):
        """Full SMC-RW pipeline with PP EOS and prior-only likelihood.

        Uses lightweight hyperparameters for speed (<2 min).
        """
        prior_file = tmp_path / "pp.prior"
        prior_file.write_text(
            'logp1_si = UniformPrior(33.5, 34.5, parameter_names=["logp1_si"])\n'
            'gamma1 = UniformPrior(1.2, 4.5, parameter_names=["gamma1"])\n'
            'gamma2 = UniformPrior(1.2, 4.5, parameter_names=["gamma2"])\n'
            'gamma3 = UniformPrior(1.2, 4.5, parameter_names=["gamma3"])\n'
        )

        config_dict = {
            "seed": 42,
            "eos": {"type": "piecewise_polytrope", "n_points": 50},
            "tov": {
                "type": "gr",
                "min_nsat_TOV": 0.75,
                "ndat_TOV": 30,
                "nb_masses": 20,
            },
            "prior": {"specification_file": str(prior_file)},
            "likelihoods": [
                {"type": "constraints_eos", "enabled": True},
                {"type": "zero", "enabled": True},
            ],
            "sampler": {
                "type": "smc-rw",
                "n_particles": 100,
                "n_mcmc_steps": 3,
                "target_ess": 0.9,
                "random_walk_sigma": 0.1,
                "output_dir": str(tmp_path),
            },
            "postprocessing": {"enabled": False},
        }

        config = InferenceConfig(**config_dict)

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

        # Check all four PP parameters are in samples
        for param in PP_PARAMS:
            assert param in output.samples, f"Missing parameter: {param}"
            assert jnp.isfinite(
                output.samples[param]
            ).all(), f"Non-finite values in {param}"

        # Samples must be within prior bounds
        for param, (lo, hi) in PRIOR_BOUNDS.items():
            assert jnp.all(
                output.samples[param] >= lo
            ), f"{param} samples below lower bound {lo}"
            assert jnp.all(
                output.samples[param] <= hi
            ), f"{param} samples above upper bound {hi}"

        # SMC metadata
        assert "weights" in output.metadata, "SMC output missing weights"
        weights = output.metadata["weights"]
        assert jnp.isclose(
            jnp.sum(weights), 1.0, atol=0.01
        ), f"SMC weights don't sum to 1: {jnp.sum(weights)}"
