"""Tests for unified JesterTransform system."""

import pytest
import jax
import jax.numpy as jnp

from jesterTOV.inference.config.schema import (
    MetamodelEOSConfig,
    MetamodelCSEEOSConfig,
    MetamodelPeakCSEEOSConfig,
    SpectralEOSConfig,
    GRTOVConfig,
)
from jesterTOV.inference.transforms import JesterTransform


class TestJesterTransform:
    """Test unified JesterTransform class."""

    def test_from_config_metamodel(self):
        """Test creating MetaModel transform via from_config."""
        eos_config = MetamodelEOSConfig(
            type="metamodel",
            ndat_metamodel=100,
            nmax_nsat=2.0,
            nb_CSE=0,
            crust_name="DH",
        )
        tov_config = GRTOVConfig(
            type="gr",
            min_nsat_TOV=0.75,
            ndat_TOV=100,
        )

        transform = JesterTransform.from_config(eos_config, tov_config)

        assert transform is not None
        assert "MetaModel_EOS_model" in transform.get_eos_type()
        assert transform.ndat_TOV == 100
        assert transform.min_nsat_TOV == 0.75

    def test_from_config_metamodel_cse(self):
        """Test creating MetaModel+CSE transform via from_config."""
        eos_config = MetamodelCSEEOSConfig(
            type="metamodel_cse",
            ndat_metamodel=100,
            nmax_nsat=25.0,
            nb_CSE=8,
            crust_name="DH",
        )
        tov_config = GRTOVConfig(
            type="gr",
            min_nsat_TOV=0.75,
            ndat_TOV=100,
        )

        transform = JesterTransform.from_config(eos_config, tov_config)

        assert transform is not None
        assert "MetaModel_with_CSE_EOS_model" in transform.get_eos_type()
        assert transform.ndat_TOV == 100

    def test_from_config_spectral(self):
        """Test creating Spectral transform via from_config."""
        eos_config = SpectralEOSConfig(
            type="spectral",
            crust_name="SLy",  # Spectral requires SLy for LALSuite compatibility
        )
        tov_config = GRTOVConfig(
            type="gr",
            min_nsat_TOV=0.75,
            ndat_TOV=100,
        )

        transform = JesterTransform.from_config(eos_config, tov_config)

        assert transform is not None
        assert "SpectralDecomposition_EOS_model" in transform.get_eos_type()

    def test_from_config_metamodel_peak_cse(self):
        """Test creating MetaModel+peakCSE transform via from_config."""
        eos_config = MetamodelPeakCSEEOSConfig(
            type="metamodel_peak_cse",
            ndat_metamodel=100,
            nmax_nsat=25.0,
            ndat_CSE=100,
            crust_name="DH",
        )
        tov_config = GRTOVConfig(
            type="gr",
            min_nsat_TOV=0.75,
            ndat_TOV=100,
        )

        transform = JesterTransform.from_config(eos_config, tov_config)

        assert transform is not None
        assert "MetaModel_with_peakCSE_EOS_model" in transform.get_eos_type()
        assert transform.ndat_TOV == 100

    def test_get_parameter_names_metamodel_peak_cse(self):
        """Test that MetaModel+peakCSE transform reports correct parameter names."""
        eos_config = MetamodelPeakCSEEOSConfig(type="metamodel_peak_cse")
        tov_config = GRTOVConfig()

        transform = JesterTransform.from_config(eos_config, tov_config)
        param_names = transform.get_parameter_names()

        expected_params = [
            "E_sat",
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
            "nbreak",
            "gaussian_peak",
            "gaussian_mu",
            "gaussian_sigma",
            "logit_growth_rate",
            "logit_midpoint",
        ]
        for param in expected_params:
            assert param in param_names, f"Missing parameter: {param}"

    def test_invalid_eos_type_fails(self):
        """Test that unknown EOS config type raises ValueError at runtime."""
        from unittest.mock import MagicMock

        # Create a mock config that passes isinstance checks for none of the known types
        mock_config = MagicMock(spec=[])  # Empty spec so isinstance returns False

        with pytest.raises((ValueError, AttributeError)):
            JesterTransform._create_eos(mock_config)

    def test_invalid_crust_name_fails(self):
        """Test that invalid crust name raises error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MetamodelEOSConfig(
                type="metamodel",
                crust_name="InvalidCrust",  # type: ignore
                nb_CSE=0,
            )

    def test_get_parameter_names_metamodel(self):
        """Test that MetaModel transform reports correct parameter names."""
        eos_config = MetamodelEOSConfig(type="metamodel", nb_CSE=0)
        tov_config = GRTOVConfig()

        transform = JesterTransform.from_config(eos_config, tov_config)
        param_names = transform.get_parameter_names()

        # Should have 9 NEP parameters
        expected_params = [
            "E_sat",
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

    def test_get_parameter_names_metamodel_cse(self):
        """Test that MetaModel+CSE transform reports correct parameter names."""
        eos_config = MetamodelCSEEOSConfig(type="metamodel_cse", nb_CSE=8)
        tov_config = GRTOVConfig()

        transform = JesterTransform.from_config(eos_config, tov_config)
        param_names = transform.get_parameter_names()

        # Should have 9 NEP parameters
        assert "E_sat" in param_names
        assert "K_sat" in param_names
        # CSE parameters are added dynamically by prior parser, not by transform
        # Transform only knows about NEP parameters

    def test_forward_preserves_keep_names(self):
        """Test that transform preserves specified parameters in output."""
        eos_config = MetamodelEOSConfig(
            type="metamodel",
            ndat_metamodel=50,  # Smaller for faster test
            nmax_nsat=2.0,
            nb_CSE=0,
        )
        tov_config = GRTOVConfig(ndat_TOV=50)

        keep_names = ["K_sat", "L_sym"]
        transform = JesterTransform.from_config(
            eos_config, tov_config, keep_names=keep_names
        )

        # Create minimal realistic params
        params = {
            "E_sat": -16.0,
            "K_sat": 220.0,
            "Q_sat": 0.0,
            "Z_sat": 0.0,
            "E_sym": 31.7,
            "L_sym": 90.0,
            "K_sym": 0.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }

        result = transform.forward(params)

        # Check that keep_names are preserved
        assert "K_sat" in result
        assert result["K_sat"] == 220.0
        assert "L_sym" in result
        assert result["L_sym"] == 90.0

        # Check that output quantities are present
        assert "masses_EOS" in result
        assert "radii_EOS" in result
        assert "Lambdas_EOS" in result


class TestJesterTransformIntegration:
    """Integration tests for JesterTransform."""

    @pytest.mark.slow
    def test_metamodel_forward_realistic_params(self, realistic_nep_stiff):
        """Test forward transform with realistic stiff EOS parameters.

        NOTE: This is a slow integration test as it solves TOV equations.
        """
        eos_config = MetamodelEOSConfig(
            type="metamodel",
            ndat_metamodel=100,
            nmax_nsat=2.0,
            nb_CSE=0,
        )
        tov_config = GRTOVConfig(ndat_TOV=100)

        transform = JesterTransform.from_config(eos_config, tov_config)
        result = transform.forward(realistic_nep_stiff)

        # Check that output contains expected keys
        assert "masses_EOS" in result
        assert "radii_EOS" in result
        assert "Lambdas_EOS" in result

        # Check that outputs are arrays
        assert isinstance(result["masses_EOS"], jnp.ndarray)
        assert isinstance(result["radii_EOS"], jnp.ndarray)
        assert isinstance(result["Lambdas_EOS"], jnp.ndarray)

        # Check that all arrays have same length
        n_points = len(result["masses_EOS"])
        assert len(result["radii_EOS"]) == n_points
        assert len(result["Lambdas_EOS"]) == n_points

        # Check that we got some valid neutron stars
        max_mass = jnp.max(result["masses_EOS"])
        assert (
            max_mass > 1.0
        ), f"Maximum mass {max_mass} too low - EOS may be unphysical"

        # Check that radii are in reasonable range
        max_radius = jnp.max(result["radii_EOS"])
        assert 8.0 < max_radius < 30.0, f"Maximum radius {max_radius} km unreasonable"

        # Original NEP parameters should be preserved in output
        for param in realistic_nep_stiff.keys():
            assert param in result
            assert result[param] == realistic_nep_stiff[param]

    @pytest.mark.slow
    def test_metamodel_cse_forward_realistic_params(self, realistic_nep_stiff):
        """Test MetaModel+CSE forward transform.

        NOTE: This is a slow integration test. CSE allows higher densities.
        """
        eos_config = MetamodelCSEEOSConfig(
            type="metamodel_cse",
            ndat_metamodel=100,
            nmax_nsat=25.0,
            nb_CSE=8,
        )
        tov_config = GRTOVConfig(ndat_TOV=100)

        transform = JesterTransform.from_config(eos_config, tov_config)

        # Add CSE parameters to NEP params
        params = realistic_nep_stiff.copy()
        params["nbreak"] = 0.24  # Breaking density

        # Add CSE grid parameters (uniform [0,1])
        for i in range(8):
            params[f"n_CSE_{i}_u"] = 0.5  # Uniform grid spacing
            params[f"cs2_CSE_{i}"] = 0.3  # cs^2 values
        params["cs2_CSE_8"] = 0.3  # Final cs2

        result = transform.forward(params)

        # Check that output contains expected keys
        assert "masses_EOS" in result
        assert "radii_EOS" in result
        assert "Lambdas_EOS" in result

        # Check that we got valid neutron stars
        max_mass = jnp.max(result["masses_EOS"])
        assert max_mass > 1.0, f"Maximum mass {max_mass} too low"

        # CSE should allow higher maximum masses than MetaModel alone
        assert max_mass < 3.5, f"Maximum mass {max_mass} too high - likely unphysical"

    @pytest.mark.slow
    def test_metamodel_peak_cse_forward_realistic_params(self, realistic_nep_stiff):
        """Test MetaModel+peakCSE forward transform with realistic parameters."""
        eos_config = MetamodelPeakCSEEOSConfig(
            type="metamodel_peak_cse",
            ndat_metamodel=100,
            nmax_nsat=25.0,
            ndat_CSE=100,
        )
        tov_config = GRTOVConfig(ndat_TOV=100)

        transform = JesterTransform.from_config(eos_config, tov_config)

        params = realistic_nep_stiff.copy()
        params["nbreak"] = 0.24
        params["gaussian_peak"] = 0.3
        params["gaussian_mu"] = 0.5
        params["gaussian_sigma"] = 0.15
        params["logit_growth_rate"] = 5.0
        params["logit_midpoint"] = 1.5

        result = transform.forward(params)

        assert "masses_EOS" in result
        assert "radii_EOS" in result
        assert "Lambdas_EOS" in result

        max_mass = jnp.max(result["masses_EOS"])
        assert max_mass > 1.0, f"Maximum mass {max_mass} too low"
        assert max_mass < 3.5, f"Maximum mass {max_mass} too high - likely unphysical"

    def test_transform_preserves_input_parameters(self, realistic_nep_stiff):
        """Test that transforms preserve input parameters in output."""
        eos_config = MetamodelEOSConfig(
            type="metamodel",
            ndat_metamodel=50,  # Use fewer points for speed
            nmax_nsat=2.0,
            nb_CSE=0,
        )
        tov_config = GRTOVConfig(ndat_TOV=50)

        keep_names = list(realistic_nep_stiff.keys())
        transform = JesterTransform.from_config(
            eos_config, tov_config, keep_names=keep_names
        )

        result = transform.forward(realistic_nep_stiff)

        # All input parameters should be in output
        for param, value in realistic_nep_stiff.items():
            assert param in result
            assert result[param] == value


# Realistic spectral params (LALSuite-compatible, within standard prior bounds)
SPECTRAL_PARAMS = {
    "gamma_0": 1.35,
    "gamma_1": 0.5,
    "gamma_2": 0.1,
    "gamma_3": 0.005,
}


class TestSpectralTransform:
    """Tests specific to the spectral decomposition transform."""

    @pytest.mark.slow
    def test_spectral_forward_single(self):
        """Test single forward pass of spectral transform."""
        eos_config = SpectralEOSConfig(type="spectral", crust_name="SLy")
        tov_config = GRTOVConfig(ndat_TOV=30)
        transform = JesterTransform.from_config(eos_config, tov_config)

        result = transform.forward(SPECTRAL_PARAMS)

        assert "masses_EOS" in result
        assert "radii_EOS" in result
        assert "Lambdas_EOS" in result
        assert jnp.isfinite(result["masses_EOS"]).any()

    @pytest.mark.slow
    def test_spectral_forward_vmap(self):
        """Regression test: spectral forward must not crash under jax.vmap.

        The bug was that construct_eos called float(gamma_violation) on a JAX
        traced array, which raises ConcretizationTypeError inside vmap.
        """
        eos_config = SpectralEOSConfig(type="spectral", crust_name="SLy")
        tov_config = GRTOVConfig(ndat_TOV=30)
        transform = JesterTransform.from_config(eos_config, tov_config)

        # Batch of 3 spectral parameter sets
        params_batch = {
            "gamma_0": jnp.array([1.35, 1.5, 0.8]),
            "gamma_1": jnp.array([0.5, 0.2, 0.3]),
            "gamma_2": jnp.array([0.1, 0.05, 0.0]),
            "gamma_3": jnp.array([0.005, 0.001, -0.001]),
        }

        # This must not raise jax.errors.ConcretizationTypeError
        results = jax.vmap(transform.forward)(params_batch)

        assert "masses_EOS" in results
        assert results["masses_EOS"].shape[0] == 3

    @pytest.mark.slow
    def test_spectral_gamma_constraint_key_present(self):
        """Test that spectral EOS populates n_gamma_violations in transform output.

        This key is consumed by ConstraintGammaLikelihood.
        """
        eos_config = SpectralEOSConfig(type="spectral", crust_name="SLy")
        tov_config = GRTOVConfig(ndat_TOV=30)
        transform = JesterTransform.from_config(eos_config, tov_config)

        result = transform.forward(SPECTRAL_PARAMS)

        assert "n_gamma_violations" in result, (
            "Spectral transform must include 'n_gamma_violations' for "
            "ConstraintGammaLikelihood to work correctly"
        )


class TestNTOV:
    """Tests for n_TOV computation in JesterTransform."""

    @pytest.mark.slow
    def test_n_TOV_present_in_output(self, realistic_nep_stiff):
        """Test that n_TOV is present in transform output."""
        eos_config = MetamodelEOSConfig(type="metamodel", ndat_metamodel=50, nb_CSE=0)
        tov_config = GRTOVConfig(ndat_TOV=50)
        transform = JesterTransform.from_config(eos_config, tov_config)

        result = transform.forward(realistic_nep_stiff)

        assert "n_TOV" in result, "n_TOV must be present in transform output"

    @pytest.mark.slow
    def test_n_TOV_is_scalar(self, realistic_nep_stiff):
        """Test that n_TOV is a scalar (not an array)."""
        eos_config = MetamodelEOSConfig(type="metamodel", ndat_metamodel=50, nb_CSE=0)
        tov_config = GRTOVConfig(ndat_TOV=50)
        transform = JesterTransform.from_config(eos_config, tov_config)

        result = transform.forward(realistic_nep_stiff)

        n_tov = result["n_TOV"]
        assert (
            jnp.ndim(n_tov) == 0
        ), f"n_TOV should be scalar, got shape {jnp.shape(n_tov)}"

    @pytest.mark.slow
    def test_n_TOV_is_positive(self, realistic_nep_stiff):
        """Test that n_TOV is positive for a valid EOS."""
        eos_config = MetamodelCSEEOSConfig(ndat_metamodel=50, nb_CSE=8)
        tov_config = GRTOVConfig(ndat_TOV=50)
        transform = JesterTransform.from_config(eos_config, tov_config)

        # Add CSE parameters to NEP params
        params = realistic_nep_stiff.copy()
        params["nbreak"] = 0.24  # Breaking density

        # Add CSE grid parameters (uniform [0,1])
        for i in range(8):
            params[f"n_CSE_{i}_u"] = 0.5  # Uniform grid spacing
            params[f"cs2_CSE_{i}"] = 0.3  # cs^2 values
        params["cs2_CSE_8"] = 0.3  # Final cs2

        result = transform.forward(params)

        assert float(result["n_TOV"]) > 0.0, "n_TOV must be positive for valid EOS"

    @pytest.mark.slow
    def test_n_TOV_exceeds_MTOV_central_density(self, realistic_nep_stiff):
        """Test that n_TOV is physically reasonable.

        n_TOV should be the density at the central pressure of MTOV.
        For realistic neutron stars, this should be 2-8 n_sat in geometric units.
        """
        from jesterTOV import utils

        eos_config = MetamodelCSEEOSConfig(
            type="metamodel_cse", ndat_metamodel=50, nb_CSE=8
        )
        tov_config = GRTOVConfig(ndat_TOV=50)
        transform = JesterTransform.from_config(eos_config, tov_config)

        # Add CSE parameters to NEP params
        params = realistic_nep_stiff.copy()
        params["nbreak"] = 0.24  # Breaking density

        # Add CSE grid parameters (uniform [0,1])
        for i in range(8):
            params[f"n_CSE_{i}_u"] = 0.5  # Uniform grid spacing
            params[f"cs2_CSE_{i}"] = 0.3  # cs^2 values
        params["cs2_CSE_8"] = 0.3  # Final cs2

        result = transform.forward(params)

        n_TOV_nsat = float(result["n_TOV"]) / utils.fm_inv3_to_geometric / 0.16
        # Typical MTOV central density is between 2 and 8 n_sat
        assert (
            1.0 < n_TOV_nsat < 10.0
        ), f"n_TOV = {n_TOV_nsat:.2f} n_sat is outside expected range [1, 10] n_sat"

    @pytest.mark.slow
    def test_n_TOV_stored_in_hdf5(self, realistic_nep_stiff, tmp_path):
        """Test that n_TOV is stored and retrieved from HDF5 results."""
        import numpy as np
        from jesterTOV.inference.result import InferenceResult

        eos_config = MetamodelEOSConfig(type="metamodel", ndat_metamodel=50, nb_CSE=0)
        tov_config = GRTOVConfig(ndat_TOV=50)
        transform = JesterTransform.from_config(eos_config, tov_config)

        result = transform.forward(realistic_nep_stiff)
        n_TOV_val = float(result["n_TOV"])

        # Build a minimal InferenceResult with n_TOV
        posterior = {
            "log_prob": np.array([-10.0]),
            "masses_EOS": np.array([[1.4, 2.0]]),
            "radii_EOS": np.array([[12.0, 11.0]]),
            "Lambdas_EOS": np.array([[500.0, 100.0]]),
            "n": np.array([[0.1, 0.5]]),
            "p": np.array([[1.0, 10.0]]),
            "e": np.array([[100.0, 500.0]]),
            "cs2": np.array([[0.1, 0.4]]),
            "n_TOV": np.array([n_TOV_val]),
        }
        metadata = {"sampler": "flowmc", "n_samples": 1}
        inference_result = InferenceResult(
            sampler_type="flowmc", posterior=posterior, metadata=metadata
        )

        filepath = tmp_path / "results.h5"
        inference_result.save(filepath)

        loaded = InferenceResult.load(filepath)
        assert "n_TOV" in loaded.posterior
        assert np.isclose(loaded.posterior["n_TOV"][0], n_TOV_val)
