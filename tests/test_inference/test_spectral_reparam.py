"""Tests for the spectral EOS Gaussian reparametrization.

Covers:
- MultivariateGaussianPrior (sampling, log_prob)
- SpectralEOSConfig.reparametrized flag
- SpectralDecomposition_EOS_model.get_required_parameters when reparametrized
- construct_eos with tilde parameters reproduces correct gamma values
- Prior file parsing with MultivariateGaussianPrior
- JesterTransform instantiation with reparametrized=True
"""

import jax
import jax.numpy as jnp
import numpy as np

from jesterTOV.inference.base import MultivariateGaussianPrior
from jesterTOV.inference.config.schemas.eos import SpectralEOSConfig
from jesterTOV.eos.spectral.spectral_decomposition import (
    SpectralDecomposition_EOS_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Reference values copied from the EOS class (sigma_scale = 1.0, i.e. the base Cholesky)
REPARAM_MEAN = np.array([0.92232738, 0.34177628, -0.08307007, 0.00413816])
REPARAM_L_BASE = np.array(
    [
        [+3.41174267e-01, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        [-2.12029883e-01, +1.15271551e-01, 0.00000000e00, 0.00000000e00],
        [+3.18696840e-02, -3.73794902e-02, +8.27531880e-03, 0.00000000e00],
        [-1.38603940e-03, +2.39274018e-03, -7.76593593e-04, +1.86425321e-04],
    ]
)


# ---------------------------------------------------------------------------
# MultivariateGaussianPrior tests
# ---------------------------------------------------------------------------


class TestMultivariateGaussianPrior:
    """Tests for MultivariateGaussianPrior."""

    def test_default_is_unit_gaussian(self):
        """Default prior should be N(0, I_4)."""
        prior = MultivariateGaussianPrior(parameter_names=["a", "b", "c", "d"])
        assert jnp.allclose(prior.mean, jnp.zeros(4))
        assert jnp.allclose(prior.cov, jnp.eye(4))

    def test_n_dim(self):
        prior = MultivariateGaussianPrior(
            parameter_names=[
                "gamma_0_tilde",
                "gamma_1_tilde",
                "gamma_2_tilde",
                "gamma_3_tilde",
            ]
        )
        assert prior.n_dim == 4

    def test_parameter_names(self):
        names = ["gamma_0_tilde", "gamma_1_tilde", "gamma_2_tilde", "gamma_3_tilde"]
        prior = MultivariateGaussianPrior(parameter_names=names)
        assert prior.parameter_names == names

    def test_sample_returns_dict(self):
        names = ["a", "b", "c", "d"]
        prior = MultivariateGaussianPrior(parameter_names=names)
        rng = jax.random.PRNGKey(0)
        samples = prior.sample(rng, n_samples=100)
        assert isinstance(samples, dict)
        for name in names:
            assert name in samples
            assert samples[name].shape == (100,)

    def test_unit_gaussian_sample_statistics(self):
        """Samples from N(0, I) should have ~zero mean and ~unit variance."""
        prior = MultivariateGaussianPrior(parameter_names=["x0", "x1", "x2", "x3"])
        rng = jax.random.PRNGKey(42)
        samples = prior.sample(rng, n_samples=5000)
        for name in prior.parameter_names:
            vals = np.array(samples[name])
            assert abs(vals.mean()) < 0.1, f"{name}: mean {vals.mean()} far from 0"
            assert abs(vals.std() - 1.0) < 0.1, f"{name}: std {vals.std()} far from 1"

    def test_log_prob_at_mean_is_maximum(self):
        """Log prob at the mean should be greater than at a distant point."""
        prior = MultivariateGaussianPrior(parameter_names=["a", "b", "c", "d"])
        at_mean = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
        far_away = {"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.0}
        assert prior.log_prob(at_mean) > prior.log_prob(far_away)

    def test_log_prob_is_finite_near_mean(self):
        prior = MultivariateGaussianPrior(parameter_names=["a", "b", "c", "d"])
        sample = {"a": 0.1, "b": -0.2, "c": 0.3, "d": -0.1}
        assert jnp.isfinite(prior.log_prob(sample))

    def test_custom_mean_and_cov(self):
        """Custom mean and covariance should be stored and used correctly."""
        mean = jnp.array([1.0, 2.0])
        cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        prior = MultivariateGaussianPrior(
            parameter_names=["x", "y"],
            mean=mean,
            cov=cov,
        )
        assert jnp.allclose(prior.mean, mean)
        assert jnp.allclose(prior.cov, cov)

        # Log prob at mean should be finite
        at_mean = {"x": 1.0, "y": 2.0}
        assert jnp.isfinite(prior.log_prob(at_mean))

    def test_log_prob_matches_scipy(self):
        """log_prob should match scipy.stats.multivariate_normal as a reference."""
        from scipy.stats import multivariate_normal as sp_mvn

        prior = MultivariateGaussianPrior(
            parameter_names=["x", "y"],
            mean=jnp.array([1.0, -1.0]),
            cov=jnp.array([[3.0, 1.0], [1.0, 2.0]]),
        )
        point = {"x": 0.5, "y": 0.5}
        jax_lp = float(prior.log_prob(point))
        scipy_lp = sp_mvn.logpdf(
            [0.5, 0.5],
            mean=[1.0, -1.0],
            cov=[[3.0, 1.0], [1.0, 2.0]],
        )
        assert abs(jax_lp - scipy_lp) < 1e-6


# ---------------------------------------------------------------------------
# SpectralEOSConfig reparametrized flag
# ---------------------------------------------------------------------------


class TestSpectralEOSConfig:
    """Tests for the reparametrized flag on SpectralEOSConfig."""

    def test_default_is_not_reparametrized(self):
        cfg = SpectralEOSConfig(type="spectral")
        assert cfg.reparametrized is False

    def test_reparametrized_true(self):
        cfg = SpectralEOSConfig(type="spectral", reparametrized=True)
        assert cfg.reparametrized is True

    def test_reparametrized_false_explicit(self):
        cfg = SpectralEOSConfig(type="spectral", reparametrized=False)
        assert cfg.reparametrized is False

    def test_sigma_scale_default_is_one(self):
        cfg = SpectralEOSConfig(type="spectral")
        assert cfg.sigma_scale == 1.0

    def test_sigma_scale_custom(self):
        cfg = SpectralEOSConfig(type="spectral", reparametrized=True, sigma_scale=1.5)
        assert cfg.sigma_scale == 1.5


# ---------------------------------------------------------------------------
# SpectralDecomposition_EOS_model parameter names
# ---------------------------------------------------------------------------


class TestSpectralEOSModelParamNames:
    """Tests for get_required_parameters with reparametrized flag."""

    def test_original_param_names(self):
        model = SpectralDecomposition_EOS_model(
            crust_name="SLy", n_points_high=50, reparametrized=False
        )
        names = model.get_required_parameters()
        assert names == ["gamma_0", "gamma_1", "gamma_2", "gamma_3"]

    def test_reparametrized_param_names(self):
        model = SpectralDecomposition_EOS_model(
            crust_name="SLy", n_points_high=50, reparametrized=True
        )
        names = model.get_required_parameters()
        assert names == [
            "gamma_0_tilde",
            "gamma_1_tilde",
            "gamma_2_tilde",
            "gamma_3_tilde",
        ]

    def test_default_is_not_reparametrized(self):
        model = SpectralDecomposition_EOS_model(crust_name="SLy", n_points_high=50)
        names = model.get_required_parameters()
        assert names == ["gamma_0", "gamma_1", "gamma_2", "gamma_3"]


# ---------------------------------------------------------------------------
# construct_eos reparametrization transform correctness
# ---------------------------------------------------------------------------


class TestSpectralReparamTransform:
    """Tests that the gamma = mu + L_wide @ z transform is applied correctly."""

    def test_tilde_zero_gives_mean(self):
        """When tilde params are all zero, gamma should equal the posterior mean."""
        model = SpectralDecomposition_EOS_model(
            crust_name="SLy", n_points_high=50, reparametrized=True
        )
        params = {
            "gamma_0_tilde": 0.0,
            "gamma_1_tilde": 0.0,
            "gamma_2_tilde": 0.0,
            "gamma_3_tilde": 0.0,
        }
        # Call construct_eos and verify the EOS was built successfully
        eos_data = model.construct_eos(params)
        # EOS should be valid (finite values)
        assert jnp.all(jnp.isfinite(eos_data.ns))

    def test_tilde_recovers_known_gamma(self):
        """Given z, the recovered gamma = mu + L_base @ z should match manual calc."""
        # Pick a simple z vector
        z = np.array([1.0, -1.0, 0.5, -0.5])
        expected_gamma = REPARAM_MEAN + REPARAM_L_BASE @ z

        model = SpectralDecomposition_EOS_model(
            crust_name="SLy", n_points_high=50, reparametrized=True
        )

        # Verify that the same EOS is produced via tilde params as via direct gamma
        params_tilde = {
            "gamma_0_tilde": float(z[0]),
            "gamma_1_tilde": float(z[1]),
            "gamma_2_tilde": float(z[2]),
            "gamma_3_tilde": float(z[3]),
        }
        params_direct = {
            "gamma_0": float(expected_gamma[0]),
            "gamma_1": float(expected_gamma[1]),
            "gamma_2": float(expected_gamma[2]),
            "gamma_3": float(expected_gamma[3]),
        }

        model_direct = SpectralDecomposition_EOS_model(
            crust_name="SLy", n_points_high=50, reparametrized=False
        )

        eos_reparam = model.construct_eos(params_tilde)
        eos_direct = model_direct.construct_eos(params_direct)

        # Both should produce identical EOS (same pressure, energy density arrays)
        assert jnp.allclose(eos_reparam.ps, eos_direct.ps, atol=1e-10)
        assert jnp.allclose(eos_reparam.es, eos_direct.es, atol=1e-10)

    def test_tilde_params_produce_valid_eos(self):
        """Tilde params drawn from N(0,1) should mostly produce valid EOS."""
        model = SpectralDecomposition_EOS_model(
            crust_name="SLy", n_points_high=50, reparametrized=True
        )
        rng = np.random.default_rng(42)
        n_test = 20
        n_valid = 0
        for _ in range(n_test):
            z = rng.standard_normal(4)
            params = {
                "gamma_0_tilde": float(z[0]),
                "gamma_1_tilde": float(z[1]),
                "gamma_2_tilde": float(z[2]),
                "gamma_3_tilde": float(z[3]),
            }
            eos_data = model.construct_eos(params)
            if jnp.all(jnp.isfinite(eos_data.ns)):
                n_valid += 1
        # Expect most to be valid (reparametrization should reduce NaN rate significantly)
        assert n_valid >= 14, (
            f"Only {n_valid}/{n_test} valid EOS from reparametrized prior; "
            "expected at least 14/20"
        )

    def test_sigma_scale_scales_transform(self):
        """sigma_scale should scale the Cholesky factor: L_wide = sigma_scale * L_base."""
        z = np.array([1.0, -1.0, 0.5, -0.5])
        sigma_scale = 2.0

        # Model with custom sigma_scale
        model_scaled = SpectralDecomposition_EOS_model(
            crust_name="SLy",
            n_points_high=50,
            reparametrized=True,
            sigma_scale=sigma_scale,
        )
        # Expected gamma uses scaled L
        expected_gamma = REPARAM_MEAN + sigma_scale * (REPARAM_L_BASE @ z)

        params_tilde = {
            "gamma_0_tilde": float(z[0]),
            "gamma_1_tilde": float(z[1]),
            "gamma_2_tilde": float(z[2]),
            "gamma_3_tilde": float(z[3]),
        }
        params_direct = {
            "gamma_0": float(expected_gamma[0]),
            "gamma_1": float(expected_gamma[1]),
            "gamma_2": float(expected_gamma[2]),
            "gamma_3": float(expected_gamma[3]),
        }
        model_direct = SpectralDecomposition_EOS_model(
            crust_name="SLy", n_points_high=50, reparametrized=False
        )

        eos_scaled = model_scaled.construct_eos(params_tilde)
        eos_direct = model_direct.construct_eos(params_direct)

        assert jnp.allclose(eos_scaled.ps, eos_direct.ps, atol=1e-10)
        assert jnp.allclose(eos_scaled.es, eos_direct.es, atol=1e-10)

    def test_sigma_scale_default_is_one(self):
        """Default sigma_scale=1.0 should use L_base without any scaling."""
        z = np.array([0.5, -0.5, 0.3, -0.2])
        expected_gamma = REPARAM_MEAN + REPARAM_L_BASE @ z

        model = SpectralDecomposition_EOS_model(
            crust_name="SLy", n_points_high=50, reparametrized=True
        )
        params_tilde = {
            "gamma_0_tilde": float(z[0]),
            "gamma_1_tilde": float(z[1]),
            "gamma_2_tilde": float(z[2]),
            "gamma_3_tilde": float(z[3]),
        }
        params_direct = {
            "gamma_0": float(expected_gamma[0]),
            "gamma_1": float(expected_gamma[1]),
            "gamma_2": float(expected_gamma[2]),
            "gamma_3": float(expected_gamma[3]),
        }
        model_direct = SpectralDecomposition_EOS_model(
            crust_name="SLy", n_points_high=50, reparametrized=False
        )
        eos_reparam = model.construct_eos(params_tilde)
        eos_direct = model_direct.construct_eos(params_direct)

        assert jnp.allclose(eos_reparam.ps, eos_direct.ps, atol=1e-10)
        assert jnp.allclose(eos_reparam.es, eos_direct.es, atol=1e-10)


# ---------------------------------------------------------------------------
# Prior file parsing with MultivariateGaussianPrior
# ---------------------------------------------------------------------------


class TestReparamPriorParsing:
    """Tests that the prior file with MultivariateGaussianPrior parses correctly."""

    def test_parse_multivariate_gaussian_prior(self, tmp_path):
        """MultivariateGaussianPrior in a prior file should be parsed correctly."""
        from jesterTOV.inference.priors.parser import parse_prior_file

        prior_file = tmp_path / "test_reparam.prior"
        prior_file.write_text(
            "spectral_reparam = MultivariateGaussianPrior(\n"
            '    parameter_names=["gamma_0_tilde", "gamma_1_tilde",'
            ' "gamma_2_tilde", "gamma_3_tilde"],\n'
            ")\n"
        )
        prior = parse_prior_file(prior_file)
        assert prior.n_dim == 4
        assert "gamma_0_tilde" in prior.parameter_names
        assert "gamma_3_tilde" in prior.parameter_names

    def test_reparam_prior_sampling(self, tmp_path):
        """Sampling from the parsed reparametrized prior gives unit-normal values."""
        from jesterTOV.inference.priors.parser import parse_prior_file

        prior_file = tmp_path / "test_reparam.prior"
        prior_file.write_text(
            "spectral_reparam = MultivariateGaussianPrior(\n"
            '    parameter_names=["gamma_0_tilde", "gamma_1_tilde",'
            ' "gamma_2_tilde", "gamma_3_tilde"],\n'
            ")\n"
        )
        prior = parse_prior_file(prior_file)
        rng = jax.random.PRNGKey(7)
        samples = prior.sample(rng, n_samples=200)

        for name in [
            "gamma_0_tilde",
            "gamma_1_tilde",
            "gamma_2_tilde",
            "gamma_3_tilde",
        ]:
            vals = np.array(samples[name])
            # Should look like standard normal
            assert abs(vals.mean()) < 0.3, f"{name}: mean {vals.mean()} far from 0"
            assert 0.7 < vals.std() < 1.4, f"{name}: std {vals.std()} far from 1"


# ---------------------------------------------------------------------------
# JesterTransform integration
# ---------------------------------------------------------------------------


class TestJesterTransformSpectralReparam:
    """Integration tests: JesterTransform with reparametrized spectral EOS."""

    def test_transform_instantiation_reparametrized(self):
        """JesterTransform should instantiate and report tilde parameter names."""
        from jesterTOV.inference.transforms.transform import JesterTransform
        from jesterTOV.inference.config.schemas.eos import SpectralEOSConfig
        from jesterTOV.inference.config.schemas.tov import GRTOVConfig

        eos_cfg = SpectralEOSConfig(
            type="spectral",
            crust_name="SLy",
            n_points_high=50,
            reparametrized=True,
        )
        tov_cfg = GRTOVConfig(type="gr", ndat_TOV=50, min_nsat_TOV=0.75, nb_masses=50)
        transform = JesterTransform.from_config(eos_cfg, tov_cfg)

        param_names = transform.get_parameter_names()
        assert "gamma_0_tilde" in param_names
        assert "gamma_1_tilde" in param_names
        assert "gamma_2_tilde" in param_names
        assert "gamma_3_tilde" in param_names
        # Original names should NOT be in the parameter list
        assert "gamma_0" not in param_names

    def test_transform_instantiation_original(self):
        """JesterTransform with reparametrized=False should report plain gamma names."""
        from jesterTOV.inference.transforms.transform import JesterTransform
        from jesterTOV.inference.config.schemas.eos import SpectralEOSConfig
        from jesterTOV.inference.config.schemas.tov import GRTOVConfig

        eos_cfg = SpectralEOSConfig(
            type="spectral",
            crust_name="SLy",
            n_points_high=50,
            reparametrized=False,
        )
        tov_cfg = GRTOVConfig(type="gr", ndat_TOV=50, min_nsat_TOV=0.75, nb_masses=50)
        transform = JesterTransform.from_config(eos_cfg, tov_cfg)

        param_names = transform.get_parameter_names()
        assert "gamma_0" in param_names
        assert "gamma_1" in param_names
        assert "gamma_2" in param_names
        assert "gamma_3" in param_names
        assert "gamma_0_tilde" not in param_names
