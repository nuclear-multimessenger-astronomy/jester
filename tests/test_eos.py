"""Unit tests for eos module."""

import pytest
import jax.numpy as jnp
import os
from jesterTOV import eos, utils
from jesterTOV.tov import GRTOVSolver
from jesterTOV.tov.data_classes import EOSData


# Minimal concrete implementation for testing base interpolation
class _TestInterpolateEOS(eos.Interpolate_EOS_model):
    """Minimal concrete EOS for testing interpolate_eos method."""

    def construct_eos(self, params: dict[str, float]) -> EOSData:
        """Dummy implementation."""
        raise NotImplementedError("Test class only")

    def get_required_parameters(self) -> list[str]:
        """Dummy implementation."""
        return []


class TestCrust:
    """Test Crust class functionality."""

    def test_crust_list_available(self) -> None:
        """Test listing available crusts."""
        available = eos.Crust.list_available()
        assert isinstance(available, list)
        assert "DH" in available
        assert "BPS" in available
        # SLy should be available but may be updated by user
        assert len(available) >= 2

    def test_crust_validate(self) -> None:
        """Test crust validation."""
        assert eos.Crust.validate("DH") is True
        assert eos.Crust.validate("BPS") is True
        assert eos.Crust.validate("invalid_crust") is False

    def test_crust_get_crust_dir(self) -> None:
        """Test getting crust directory path."""
        crust_dir = eos.Crust.get_crust_dir()
        assert os.path.exists(crust_dir)
        assert os.path.isdir(crust_dir)
        assert crust_dir == eos.CRUST_DIR

    def test_crust_initialization(self) -> None:
        """Test basic crust initialization."""
        crust = eos.Crust("DH")
        assert len(crust) > 0
        assert jnp.all(crust.n > 0)
        assert jnp.all(crust.p > 0)
        assert jnp.all(crust.e > 0)

    def test_crust_properties(self) -> None:
        """Test crust property access."""
        crust = eos.Crust("DH")

        # Test properties return arrays
        assert isinstance(crust.n, jnp.ndarray) or hasattr(crust.n, "shape")
        assert isinstance(crust.p, jnp.ndarray) or hasattr(crust.p, "shape")
        assert isinstance(crust.e, jnp.ndarray) or hasattr(crust.e, "shape")

        # Test all have same length
        assert len(crust.n) == len(crust.p) == len(crust.e)

        # Test physical constraints
        assert jnp.all(crust.n > 0)  # Densities should be positive
        assert jnp.all(crust.p > 0)  # Pressures should be positive
        assert jnp.all(crust.e > 0)  # Energy densities should be positive

        # Test monotonicity
        assert jnp.all(jnp.diff(crust.n) > 0)  # Density should increase

    def test_crust_density_masking(self) -> None:
        """Test density range masking."""
        crust_full = eos.Crust("DH")
        crust_masked = eos.Crust("DH", min_density=0.001, max_density=0.1)

        # Masked crust should have fewer points
        assert len(crust_masked) < len(crust_full)

        # All densities should be within range
        assert jnp.all(crust_masked.n >= 0.001)
        assert jnp.all(crust_masked.n <= 0.1)

        # Test that min/max density properties match
        assert crust_masked.min_density >= 0.001
        assert crust_masked.max_density <= 0.1

    def test_crust_zero_pressure_filtering(self) -> None:
        """Test zero pressure filtering."""
        crust = eos.Crust("DH", filter_zero_pressure=True)
        assert jnp.all(crust.p > 0)

        # Test that disabling filter might include zero pressure points
        # (depends on crust data - some crusts may not have zero pressure points)
        crust_unfiltered = eos.Crust("DH", filter_zero_pressure=False)
        assert len(crust_unfiltered) >= len(crust)

    def test_crust_mu_lowest(self) -> None:
        """Test chemical potential calculation."""
        crust = eos.Crust("DH")
        expected_mu = (crust.e[0] + crust.p[0]) / crust.n[0]
        assert jnp.isclose(crust.mu_lowest, expected_mu)

        # Should be positive
        assert crust.mu_lowest > 0

    def test_crust_cs2(self) -> None:
        """Test speed of sound squared."""
        crust = eos.Crust("DH")

        # Should have same length as other arrays
        assert len(crust.cs2) == len(crust.n)

        # Should be positive (assuming physical crust)
        assert jnp.all(crust.cs2 > 0)

        # Should not exceed speed of light
        assert jnp.all(crust.cs2 <= 1.0)

    def test_crust_get_data(self) -> None:
        """Test get_data() convenience method."""
        crust = eos.Crust("DH")
        n, p, e = crust.get_data()

        assert jnp.array_equal(n, crust.n)
        assert jnp.array_equal(p, crust.p)
        assert jnp.array_equal(e, crust.e)

    def test_crust_invalid_name(self) -> None:
        """Test invalid crust name raises error."""
        with pytest.raises(ValueError, match="not found"):
            eos.Crust("invalid_crust")

    def test_crust_with_npz_path(self) -> None:
        """Test loading with full .npz path."""
        crust_dir = eos.Crust.get_crust_dir()
        full_path = os.path.join(crust_dir, "DH.npz")
        crust = eos.Crust(full_path)

        assert len(crust) > 0
        assert jnp.all(crust.n > 0)

    def test_crust_repr(self) -> None:
        """Test string representation."""
        crust = eos.Crust("DH")
        repr_str = repr(crust)

        assert "Crust" in repr_str
        assert "DH" in repr_str
        assert "n_points" in repr_str
        assert "density_range" in repr_str

    def test_crust_len(self) -> None:
        """Test __len__ method."""
        crust = eos.Crust("DH")
        assert len(crust) == len(crust.n)
        assert isinstance(len(crust), int)

    def test_crust_bps(self) -> None:
        """Test loading BPS crust specifically."""
        crust = eos.Crust("BPS")

        assert len(crust) > 10
        assert jnp.all(crust.n > 0)
        assert jnp.all(crust.p > 0)
        assert jnp.all(crust.e > 0)

    def test_crust_empty_after_filtering(self) -> None:
        """Test that appropriate error is raised if filtering removes all points."""
        # Try to create a crust with impossible density range
        with pytest.raises(ValueError, match="No crust points remain"):
            eos.Crust("DH", min_density=10.0, max_density=20.0)


class TestInterpolateEOSModel:
    """Test base interpolation EOS model."""

    def test_interpolate_eos_basic(self, sample_density_arrays):
        """Test basic EOS interpolation functionality."""
        n, p, e = sample_density_arrays

        model = _TestInterpolateEOS()
        ns, ps, hs, es, dloge_dlogps = model.interpolate_eos(n, p, e)

        # Check output shapes
        assert ns.shape == n.shape
        assert ps.shape == p.shape
        assert hs.shape == n.shape
        assert es.shape == e.shape
        assert dloge_dlogps.shape == n.shape

        # Check unit conversions
        assert jnp.allclose(ns, n * utils.fm_inv3_to_geometric)
        assert jnp.allclose(ps, p * utils.MeV_fm_inv3_to_geometric)
        assert jnp.allclose(es, e * utils.MeV_fm_inv3_to_geometric)

    def test_interpolate_eos_enthalpy_calculation(self, sample_density_arrays):
        """Test that enthalpy calculation is reasonable."""
        n, p, e = sample_density_arrays

        model = _TestInterpolateEOS()
        ns, ps, hs, es, dloge_dlogps = model.interpolate_eos(n, p, e)

        # Enthalpy should be positive and finite
        assert jnp.all(jnp.isfinite(hs))
        assert jnp.all(hs > 0)  # Should be positive for realistic EOS


class TestMetaModelEOSModel:
    """Test MetaModel EOS implementation."""

    def test_metamodel_initialization(self, metamodel_params):
        """Test MetaModel initialization with default parameters."""
        model = eos.MetaModel_EOS_model(**metamodel_params)

        # Check that attributes are set correctly
        assert model.nsat == metamodel_params["nsat"]
        assert len(model.v_nq) == 5
        assert model.b_sat == metamodel_params["b_sat"]
        assert model.b_sym == metamodel_params["b_sym"]

        # Check that crust data is loaded
        assert hasattr(model, "ns_crust")
        assert hasattr(model, "ps_crust")
        assert hasattr(model, "es_crust")
        assert len(model.ns_crust) > 0

    def test_metamodel_kappa_assignment(self):
        """Test that kappa parameters are assigned correctly."""
        kappas = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        model = eos.MetaModel_EOS_model(kappas=kappas)

        assert model.kappa_sat == 0.1
        assert model.kappa_sat2 == 0.2
        assert model.kappa_sat3 == 0.3
        assert model.kappa_NM == 0.4
        assert model.kappa_NM2 == 0.5
        assert model.kappa_NM3 == 0.6

        # Check derived quantities
        assert model.kappa_sym == 0.4 - 0.1  # kappa_NM - kappa_sat
        assert model.kappa_sym2 == 0.5 - 0.2
        assert model.kappa_sym3 == 0.6 - 0.3

    def test_metamodel_construct_eos(self, metamodel_params, nep_dict):
        """Test EOS construction with MetaModel."""
        model = eos.MetaModel_EOS_model(**metamodel_params)

        eos_data = model.construct_eos(nep_dict)

        # Check that all outputs have reasonable shapes and values
        assert len(eos_data.ns) > 0
        assert len(eos_data.ps) == len(eos_data.ns)
        assert len(eos_data.hs) == len(eos_data.ns)
        assert len(eos_data.es) == len(eos_data.ns)
        assert len(eos_data.dloge_dlogps) == len(eos_data.ns)
        assert eos_data.mu is not None
        assert len(eos_data.mu) == len(eos_data.ns)
        assert len(eos_data.cs2) == len(eos_data.ns)

        # Check physical constraints
        assert jnp.all(eos_data.ns > 0)  # Density should be positive
        assert jnp.all(eos_data.ps > 0)  # Pressure should be positive
        assert jnp.all(eos_data.es > 0)  # Energy density should be positive
        assert jnp.all(eos_data.cs2 > 0)  # Speed of sound squared should be positive
        assert jnp.all(eos_data.cs2 <= 1.0)  # Should not exceed speed of light

    def test_metamodel_auxiliary_functions(self, metamodel_params):
        """Test auxiliary functions in MetaModel."""
        model = eos.MetaModel_EOS_model(**metamodel_params)

        # Test compute_x function
        n_test = jnp.array([0.16, 0.32, 0.48])  # Test around nsat
        x = model.compute_x(n_test)
        expected_x = (n_test - model.nsat) / (3 * model.nsat)
        assert jnp.allclose(x, expected_x)

        # Test compute_f_1 function
        delta = jnp.array([0.0, 0.1, -0.1])
        f_1 = model.compute_f_1(delta)
        expected = (1 + delta) ** (5 / 3) + (1 - delta) ** (5 / 3)
        assert jnp.allclose(f_1, expected)

        # Test compute_b function
        b = model.compute_b(delta)
        expected_b = model.b_sat + model.b_sym * delta**2
        assert jnp.allclose(b, expected_b)

    def test_metamodel_proton_fraction_bounds(self, metamodel_params, nep_dict):
        """Test that proton fraction stays within physical bounds."""
        model = eos.MetaModel_EOS_model(**metamodel_params)

        # Create coefficient array for symmetry energy
        coefficient_sym = jnp.array(
            [
                nep_dict["E_sym"],
                nep_dict["L_sym"],
                nep_dict["K_sym"],
                nep_dict["Q_sym"],
                nep_dict["Z_sym"],
            ]
        )

        n_test = jnp.linspace(0.2, 1.0, 10)  # Test range of densities
        yp = model.compute_proton_fraction(coefficient_sym, n_test)

        # Proton fraction should be between 0 and 0.5 for neutron star matter
        assert jnp.all(yp >= 0.0)
        assert jnp.all(yp <= 0.5)


class TestMetaModelWithCSEEOSModel:
    """Test MetaModel with CSE extension."""

    def test_metamodel_cse_initialization(self):
        """Test MetaModel with CSE initialization."""
        model = eos.MetaModel_with_CSE_EOS_model(
            nsat=0.16, nmin_MM_nsat=0.75, nmax_nsat=6.0, ndat_metamodel=50, ndat_CSE=50
        )

        assert model.nsat == 0.16
        assert model.nmax == 6.0 * 0.16
        assert model.ndat_CSE == 50
        assert model.ndat_metamodel == 50

    def test_metamodel_cse_construct_eos(self, nep_dict):
        """Test EOS construction with CSE extension."""
        # CSE model requires nb_CSE parameter
        nb_CSE = 3
        model = eos.MetaModel_with_CSE_EOS_model(
            nsat=0.16, ndat_metamodel=30, ndat_CSE=30, nb_CSE=nb_CSE
        )

        # Add break density and CSE grid parameters to NEP dict
        nep_dict_extended = nep_dict.copy()
        nep_dict_extended["nbreak"] = 0.5  # fm^-3

        # Add individual CSE grid point parameters (normalized positions and cs2 values)
        nep_dict_extended["n_CSE_0_u"] = 0.1
        nep_dict_extended["cs2_CSE_0"] = 0.3
        nep_dict_extended["n_CSE_1_u"] = 0.4
        nep_dict_extended["cs2_CSE_1"] = 0.4
        nep_dict_extended["n_CSE_2_u"] = 0.7
        nep_dict_extended["cs2_CSE_2"] = 0.5
        nep_dict_extended["cs2_CSE_3"] = 0.6  # Final cs2 value

        eos_data = model.construct_eos(nep_dict_extended)

        # Check basic properties
        assert len(eos_data.ns) > 0
        assert jnp.all(eos_data.ns > 0)
        assert jnp.all(eos_data.ps > 0)
        assert jnp.all(eos_data.es > 0)
        assert jnp.all(eos_data.cs2 > 0)
        assert jnp.all(eos_data.cs2 <= 1.0)


class TestConstructFamily:
    """Test neutron star family construction."""

    def test_construct_family_basic(self, sample_eos_dict):
        """Test basic family construction functionality."""
        # Create simple EOS data for testing
        ns = jnp.linspace(0.1, 1.0, 50) * utils.fm_inv3_to_geometric
        ps = jnp.linspace(10, 100, 50) * utils.MeV_fm_inv3_to_geometric
        es = jnp.linspace(20, 200, 50) * utils.MeV_fm_inv3_to_geometric
        hs = utils.cumtrapz(ps / (es + ps), jnp.log(ps))
        dloge_dlogps = jnp.diff(jnp.log(es)) / jnp.diff(jnp.log(ps))
        dloge_dlogps = jnp.concatenate([jnp.array([dloge_dlogps[0]]), dloge_dlogps])
        # Compute cs2 from p and e
        cs2 = ps / es / dloge_dlogps

        eos_data = EOSData(
            ns=ns, ps=ps, hs=hs, es=es, dloge_dlogps=dloge_dlogps, cs2=cs2
        )

        # Test family construction using GRTOVSolver
        solver = GRTOVSolver()
        family_data = solver.construct_family(eos_data, ndat=10, min_nsat=1.0)

        # Check output shapes
        assert len(family_data.log10pcs) == 10
        assert len(family_data.masses) == 10
        assert len(family_data.radii) == 10
        assert len(family_data.lambdas) == 10

        # Check physical properties
        assert jnp.all(family_data.masses > 0)  # Masses should be positive
        assert jnp.all(family_data.radii > 0)  # Radii should be positive
        assert jnp.all(
            family_data.lambdas > 0
        )  # Tidal deformabilities should be positive

        # Check that mass increases initially (before MTOV)
        max_idx = jnp.argmax(family_data.masses)
        if max_idx > 0:
            assert jnp.all(jnp.diff(family_data.masses[:max_idx]) >= 0)

    def test_locate_lowest_non_causal_point(self):
        """Test the function that locates non-causal points."""
        # Create speed of sound array with causal violation
        cs2 = jnp.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 0.9])  # Violation at index 4

        idx = utils.locate_lowest_non_causal_point(cs2)
        assert idx == 4

        # Test case with no violations
        cs2_causal = jnp.array([0.1, 0.3, 0.5, 0.8, 0.9])
        idx_causal = utils.locate_lowest_non_causal_point(cs2_causal)
        assert idx_causal == -1


@pytest.mark.slow
class TestMetaModelIntegration:
    """Integration tests for MetaModel (marked as slow)."""

    def test_full_metamodel_pipeline(self, metamodel_params, nep_dict):
        """Test complete MetaModel pipeline from initialization to family construction."""
        # Create model
        model = eos.MetaModel_EOS_model(**metamodel_params)

        # Construct EOS
        eos_data = model.construct_eos(nep_dict)

        # Construct neutron star family using GRTOVSolver
        solver = GRTOVSolver()
        family_data = solver.construct_family(eos_data, ndat=20, min_nsat=0.75)

        # Check that we get reasonable neutron star properties for limited EOS (2 nsat)
        assert jnp.max(family_data.masses) > 0.5  # Maximum mass for soft/limited EOS
        assert jnp.max(family_data.masses) < 1.5  # Expected for EOS limited to 2 nsat
        assert jnp.min(family_data.radii) > 8.0  # Minimum radius should be > 8 km
        assert (
            jnp.max(family_data.radii) < 30.0
        )  # Maximum radius (soft EOS = larger radii)


# Test fixtures and parameterized tests
@pytest.mark.parametrize("crust_name", ["DH", "BPS"])
def test_all_available_crusts(crust_name):
    """Test that all available crust files can be loaded."""
    crust = eos.Crust(crust_name)

    assert len(crust) > 10
    assert jnp.all(crust.n > 0)
    assert jnp.all(crust.p > 0)
    assert jnp.all(crust.e > 0)

    # Check that arrays are sorted by density
    assert jnp.all(jnp.diff(crust.n) > 0)


@pytest.mark.parametrize("nsat", [0.15, 0.16, 0.17])
@pytest.mark.parametrize("nmax_nsat", [1.5, 2.0, 2.5])
def test_metamodel_parameter_variations(nsat, nmax_nsat, nep_dict):
    """Test MetaModel with different parameter choices."""
    model = eos.MetaModel_EOS_model(
        nsat=nsat, nmax_nsat=nmax_nsat, ndat=50  # Reduced for faster testing
    )

    eos_data = model.construct_eos(nep_dict)

    # Basic sanity checks
    assert len(eos_data.ns) > 0
    assert jnp.all(eos_data.ns > 0)
    assert jnp.all(eos_data.ps > 0)
    assert jnp.all(eos_data.es > 0)
