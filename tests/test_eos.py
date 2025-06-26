"""Unit tests for eos module."""

import pytest
import jax.numpy as jnp
import os
from jesterTOV import eos, utils


class TestLoadCrust:
    """Test crust loading functionality."""

    def test_load_crust_available_names(self):
        """Test loading crust with available names."""
        # Test loading DH crust (should be available)
        n, p, e = eos.load_crust("DH")

        assert isinstance(n, jnp.ndarray) or hasattr(n, "shape")  # JAX arrays
        assert isinstance(p, jnp.ndarray) or hasattr(p, "shape")  # JAX arrays
        assert isinstance(e, jnp.ndarray) or hasattr(e, "shape")  # JAX arrays
        assert len(n) == len(p) == len(e)
        assert jnp.all(n > 0)  # Densities should be positive
        assert jnp.all(p > 0)  # Pressures should be positive
        assert jnp.all(e > 0)  # Energy densities should be positive

    def test_load_crust_bps(self):
        """Test loading BPS crust."""
        n, p, e = eos.load_crust("BPS")

        assert isinstance(n, jnp.ndarray) or hasattr(n, "shape")  # JAX arrays
        assert isinstance(p, jnp.ndarray) or hasattr(p, "shape")  # JAX arrays
        assert isinstance(e, jnp.ndarray) or hasattr(e, "shape")  # JAX arrays
        assert len(n) > 10  # Should have reasonable number of points

    def test_load_crust_invalid_name(self):
        """Test that invalid crust names raise appropriate errors."""
        with pytest.raises(ValueError, match="Crust invalid_crust not found"):
            eos.load_crust("invalid_crust")

    def test_load_crust_with_npz_extension(self):
        """Test loading crust with .npz extension."""
        # This should work the same as without extension
        crust_dir = eos.CRUST_DIR
        available_files = [f for f in os.listdir(crust_dir) if f.endswith(".npz")]

        if "DH.npz" in available_files:
            full_path = os.path.join(crust_dir, "DH.npz")
            n, p, e = eos.load_crust(full_path)

            assert isinstance(n, jnp.ndarray) or hasattr(n, "shape")  # JAX arrays
            assert len(n) > 0


class TestInterpolateEOSModel:
    """Test base interpolation EOS model."""

    def test_interpolate_eos_basic(self, sample_density_arrays):
        """Test basic EOS interpolation functionality."""
        n, p, e = sample_density_arrays

        model = eos.Interpolate_EOS_model()
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

        model = eos.Interpolate_EOS_model()
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

        result = model.construct_eos(nep_dict)
        ns, ps, hs, es, dloge_dlogps, mu, cs2 = result

        # Check that all outputs have reasonable shapes and values
        assert len(ns) > 0
        assert len(ps) == len(ns)
        assert len(hs) == len(ns)
        assert len(es) == len(ns)
        assert len(dloge_dlogps) == len(ns)
        assert len(mu) == len(ns)
        assert len(cs2) == len(ns)

        # Check physical constraints
        assert jnp.all(ns > 0)  # Density should be positive
        assert jnp.all(ps > 0)  # Pressure should be positive
        assert jnp.all(es > 0)  # Energy density should be positive
        assert jnp.all(cs2 > 0)  # Speed of sound squared should be positive
        assert jnp.all(cs2 <= 1.0)  # Should not exceed speed of light

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
            nsat=0.16, nmin_MM_nsat=0.75, nmax_nsat=8.0, ndat_metamodel=50, ndat_CSE=50
        )

        assert model.nsat == 0.16
        assert model.nmax == 8.0 * 0.16
        assert model.ndat_CSE == 50
        assert model.ndat_metamodel == 50

    def test_metamodel_cse_construct_eos(self, nep_dict):
        """Test EOS construction with CSE extension."""
        model = eos.MetaModel_with_CSE_EOS_model(
            nsat=0.16, ndat_metamodel=30, ndat_CSE=30
        )

        # Add break density to NEP dict
        nep_dict_extended = nep_dict.copy()
        nep_dict_extended["nbreak"] = 0.5  # fm^-3

        # Create simple CSE grids
        ngrids = jnp.array([0.6, 0.8, 1.0, 1.2])
        cs2grids = jnp.array([0.3, 0.4, 0.5, 0.6])

        result = model.construct_eos(nep_dict_extended, ngrids, cs2grids)
        ns, ps, hs, es, dloge_dlogps, mu, cs2 = result

        # Check basic properties
        assert len(ns) > 0
        assert jnp.all(ns > 0)
        assert jnp.all(ps > 0)
        assert jnp.all(es > 0)
        assert jnp.all(cs2 > 0)
        assert jnp.all(cs2 <= 1.0)


class TestConstructFamily:
    """Test neutron star family construction."""

    def test_construct_family_basic(self, sample_eos_dict):
        """Test basic family construction functionality."""
        # Create simple EOS tuple for testing
        ns = jnp.linspace(0.1, 1.0, 50) * utils.fm_inv3_to_geometric
        ps = jnp.linspace(10, 100, 50) * utils.MeV_fm_inv3_to_geometric
        es = jnp.linspace(20, 200, 50) * utils.MeV_fm_inv3_to_geometric
        hs = utils.cumtrapz(ps / (es + ps), jnp.log(ps))
        dloge_dlogps = jnp.diff(jnp.log(es)) / jnp.diff(jnp.log(ps))
        dloge_dlogps = jnp.concatenate([jnp.array([dloge_dlogps[0]]), dloge_dlogps])

        eos_tuple = (ns, ps, hs, es, dloge_dlogps)

        # Test family construction
        log_pcs, ms, rs, lambdas = eos.construct_family(
            eos_tuple, ndat=10, min_nsat=1.0
        )

        # Check output shapes
        assert len(log_pcs) == 10
        assert len(ms) == 10
        assert len(rs) == 10
        assert len(lambdas) == 10

        # Check physical properties
        assert jnp.all(ms > 0)  # Masses should be positive
        assert jnp.all(rs > 0)  # Radii should be positive
        assert jnp.all(lambdas > 0)  # Tidal deformabilities should be positive

        # Check that mass increases initially (before MTOV)
        max_idx = jnp.argmax(ms)
        if max_idx > 0:
            assert jnp.all(jnp.diff(ms[:max_idx]) >= 0)

    def test_locate_lowest_non_causal_point(self):
        """Test the function that locates non-causal points."""
        # Create speed of sound array with causal violation
        cs2 = jnp.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 0.9])  # Violation at index 4

        idx = eos.locate_lowest_non_causal_point(cs2)
        assert idx == 4

        # Test case with no violations
        cs2_causal = jnp.array([0.1, 0.3, 0.5, 0.8, 0.9])
        idx_causal = eos.locate_lowest_non_causal_point(cs2_causal)
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
        ns, ps, hs, es, dloge_dlogps, mu, cs2 = eos_data

        # Construct neutron star family
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        log_pcs, ms, rs, lambdas = eos.construct_family(eos_tuple, ndat=20)

        # Check that we get reasonable neutron star properties
        assert jnp.max(ms) > 1.0  # Maximum mass should be > 1 solar mass
        assert jnp.max(ms) < 3.0  # But not unreasonably high
        assert jnp.min(rs) > 8.0  # Minimum radius should be > 8 km
        assert jnp.max(rs) < 20.0  # Maximum radius should be < 20 km


# Test fixtures and parameterized tests
@pytest.mark.parametrize("crust_name", ["DH", "BPS"])
def test_all_available_crusts(crust_name):
    """Test that all available crust files can be loaded."""
    n, p, e = eos.load_crust(crust_name)

    assert len(n) > 10
    assert jnp.all(n > 0)
    assert jnp.all(p > 0)
    assert jnp.all(e > 0)

    # Check that arrays are sorted by density
    assert jnp.all(jnp.diff(n) > 0)


@pytest.mark.parametrize("nsat", [0.15, 0.16, 0.17])
@pytest.mark.parametrize("nmax_nsat", [8.0, 10.0, 12.0])
def test_metamodel_parameter_variations(nsat, nmax_nsat, nep_dict):
    """Test MetaModel with different parameter choices."""
    model = eos.MetaModel_EOS_model(
        nsat=nsat, nmax_nsat=nmax_nsat, ndat=50  # Reduced for faster testing
    )

    result = model.construct_eos(nep_dict)
    ns, ps, hs, es, dloge_dlogps, mu, cs2 = result

    # Basic sanity checks
    assert len(ns) > 0
    assert jnp.all(ns > 0)
    assert jnp.all(ps > 0)
    assert jnp.all(es > 0)
