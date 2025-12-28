"""Tests for inference transform system (base, metamodel, factory)."""

import pytest
import jax.numpy as jnp

from jesterTOV.inference.config.schema import TransformConfig
from jesterTOV.inference.transforms import factory
from jesterTOV.inference.transforms.metamodel import MetaModelTransform
from jesterTOV.inference.transforms.metamodel_cse import MetaModelCSETransform


class TestTransformFactory:
    """Test transform factory functionality."""

    def test_create_metamodel_transform(self):
        """Test creating MetaModel transform via factory."""
        config = TransformConfig(
            type="metamodel",
            ndat_metamodel=100,
            nmax_nsat=2.0,
            nb_CSE=0,
            min_nsat_TOV=0.75,
            ndat_TOV=100,
            nb_masses=100,
            crust_name="DH",
        )

        transform = factory.create_transform(config)

        assert isinstance(transform, MetaModelTransform)
        assert transform.ndat_metamodel == 100
        assert transform.nmax_nsat == 2.0
        assert transform.nb_CSE == 0

    def test_create_metamodel_cse_transform(self):
        """Test creating MetaModel+CSE transform via factory."""
        config = TransformConfig(
            type="metamodel_cse",
            ndat_metamodel=100,
            nmax_nsat=25.0,
            nb_CSE=8,
            min_nsat_TOV=0.75,
            ndat_TOV=100,
            nb_masses=100,
            crust_name="DH",
        )

        transform = factory.create_transform(config)

        assert isinstance(transform, MetaModelCSETransform)
        assert transform.nb_CSE == 8
        assert transform.nmax_nsat == 25.0

    def test_invalid_transform_type_fails(self):
        """Test that invalid transform type raises error."""
        # Pydantic will catch the invalid type during config creation
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TransformConfig(
                type="invalid_transform",  # type: ignore
                nb_CSE=0,
            )

    def test_invalid_crust_name_fails(self):
        """Test that invalid crust name raises error."""
        # Pydantic will catch the invalid crust name during config creation
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TransformConfig(
                type="metamodel",
                crust_name="InvalidCrust",  # type: ignore
                nb_CSE=0,
            )


class TestMetaModelTransform:
    """Test MetaModel transform."""

    @pytest.fixture
    def metamodel_transform(self):
        """Create a MetaModel transform for testing."""
        name_mapping = (
            ["K_sat", "Q_sat", "Z_sat", "E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym"],
            ["masses_EOS", "radii_EOS", "Lambdas_EOS"],
        )
        return MetaModelTransform(
            name_mapping=name_mapping,
            ndat_metamodel=100,
            nmax_nsat=2.0,  # Use 2.0 for MetaModel (causality limit)
            min_nsat_TOV=0.75,
            ndat_TOV=100,
            nb_masses=100,
            crust_name="DH",
        )

    def test_metamodel_initialization(self, metamodel_transform):
        """Test MetaModel transform initialization."""
        assert metamodel_transform.nb_CSE == 0
        assert metamodel_transform.ndat_metamodel == 100
        assert metamodel_transform.nmax_nsat == 2.0
        assert metamodel_transform.get_eos_type() == "MM"

    def test_metamodel_parameter_names(self, metamodel_transform):
        """Test that MetaModel expects correct parameter names."""
        expected_params = [
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]
        assert metamodel_transform.get_parameter_names() == expected_params

    def test_metamodel_forward_realistic_params(self, metamodel_transform, realistic_nep_stiff):
        """Test forward transform with realistic stiff EOS parameters.

        NOTE: This is a slow integration test as it solves TOV equations.
        If it fails, check that:
        1. NEP parameters are physically reasonable
        2. EOS is not too soft (may not produce stable NSs)
        3. nmax_nsat is appropriate (2.0 for MetaModel)
        """
        result = metamodel_transform.forward(realistic_nep_stiff)

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
        # (at least one mass should be > 1.0 Msun for realistic EOS)
        max_mass = jnp.max(result["masses_EOS"])
        assert max_mass > 1.0, f"Maximum mass {max_mass} too low - EOS may be unphysical"

        # Check that radii are in reasonable range (8-25 km for various EOS)
        max_radius = jnp.max(result["radii_EOS"])
        assert 8.0 < max_radius < 25.0, f"Maximum radius {max_radius} km unreasonable"

        # Original NEP parameters should be preserved in output
        for param in realistic_nep_stiff.keys():
            assert param in result
            assert result[param] == realistic_nep_stiff[param]


class TestMetaModelCSETransform:
    """Test MetaModel+CSE transform."""

    @pytest.fixture
    def metamodel_cse_transform(self):
        """Create a MetaModel+CSE transform for testing."""
        # With nb_CSE=8, we expect 8 NEP + 1 nbreak + 8*2 CSE grid = 25 params
        nep_params = [
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]
        cse_params = ["nbreak"]
        for i in range(8):
            cse_params.extend([f"n_CSE_{i}_u", f"cs2_CSE_{i}"])
        cse_params.append("cs2_CSE_8")  # Final cs2

        name_mapping = (
            nep_params + cse_params,
            ["masses_EOS", "radii_EOS", "Lambdas_EOS"],
        )

        return MetaModelCSETransform(
            name_mapping=name_mapping,
            ndat_metamodel=100,
            nmax_nsat=25.0,  # Can use higher density with CSE
            nb_CSE=8,
            min_nsat_TOV=0.75,
            ndat_TOV=100,
            nb_masses=100,
            crust_name="DH",
        )

    def test_metamodel_cse_initialization(self, metamodel_cse_transform):
        """Test MetaModel+CSE transform initialization."""
        assert metamodel_cse_transform.nb_CSE == 8
        assert metamodel_cse_transform.nmax_nsat == 25.0
        assert metamodel_cse_transform.get_eos_type() == "MM_CSE"

    def test_metamodel_cse_parameter_count(self, metamodel_cse_transform):
        """Test that MetaModel+CSE expects correct number of parameters."""
        # 8 NEP + 1 nbreak + 8 n_CSE_*_u + 8 cs2_CSE_* = 25
        expected_count = 8 + 1 + 8 + 8
        assert len(metamodel_cse_transform.get_parameter_names()) == expected_count

    def test_metamodel_cse_forward_realistic_params(
        self, metamodel_cse_transform, realistic_nep_stiff
    ):
        """Test forward transform with realistic parameters + CSE.

        NOTE: This is a slow integration test. CSE allows higher densities
        so we can test with nmax_nsat=25.0.
        """
        # Add CSE parameters to NEP params
        params = realistic_nep_stiff.copy()
        params["nbreak"] = 0.24  # Breaking density

        # Add CSE grid parameters (uniform [0,1])
        for i in range(8):
            params[f"n_CSE_{i}_u"] = 0.5  # Uniform grid spacing
            params[f"cs2_CSE_{i}"] = 0.3  # cs^2 values
        params["cs2_CSE_8"] = 0.3  # Final cs2

        result = metamodel_cse_transform.forward(params)

        # Check that output contains expected keys
        assert "masses_EOS" in result
        assert "radii_EOS" in result
        assert "Lambdas_EOS" in result

        # Check that we got valid neutron stars
        max_mass = jnp.max(result["masses_EOS"])
        assert max_mass > 1.0, f"Maximum mass {max_mass} too low"

        # CSE should allow higher maximum masses than MetaModel alone
        # (though not guaranteed for all parameter choices)
        assert max_mass < 3.5, f"Maximum mass {max_mass} too high - likely unphysical"


class TestTransformIntegration:
    """Integration tests for transform system."""

    def test_factory_creates_correct_type_for_config(self):
        """Test that factory creates correct transform type for each config."""
        configs = [
            (TransformConfig(type="metamodel", nb_CSE=0), MetaModelTransform),
            (TransformConfig(type="metamodel_cse", nb_CSE=8), MetaModelCSETransform),
        ]

        for config, expected_type in configs:
            transform = factory.create_transform(config)
            assert isinstance(
                transform, expected_type
            ), f"Expected {expected_type}, got {type(transform)}"

    def test_transform_preserves_input_parameters(self, realistic_nep_stiff):
        """Test that transforms preserve input parameters in output."""
        name_mapping = (
            list(realistic_nep_stiff.keys()),
            ["masses_EOS", "radii_EOS", "Lambdas_EOS"],
        )

        transform = MetaModelTransform(
            name_mapping=name_mapping,
            ndat_metamodel=50,  # Use fewer points for speed
            nmax_nsat=2.0,
            nb_masses=50,
        )

        result = transform.forward(realistic_nep_stiff)

        # All input parameters should be in output
        for param, value in realistic_nep_stiff.items():
            assert param in result
            assert result[param] == value
