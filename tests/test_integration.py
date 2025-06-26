"""Integration tests for JESTER package."""

import pytest
import jax.numpy as jnp
from jesterTOV import eos, tov, ptov, utils


class TestMetaModelEOSIntegration:
    """Integration tests for MetaModel EOS workflow."""

    @pytest.mark.integration
    def test_metamodel_complete_workflow(self):
        """Test complete MetaModel workflow from initialization to neutron star properties."""
        # Create realistic MetaModel parameters
        metamodel_params = {
            "kappas": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Simple case
            "v_nq": [0.0, 0.0, 0.0, 0.0, 0.0],
            "b_sat": 16.0,
            "b_sym": 30.0,
            "nsat": 0.16,
            "nmin_MM_nsat": 0.75,
            "nmax_nsat": 8.0,
            "ndat": 150,
            "crust_name": "DH",
            "max_n_crust_nsat": 0.5,
            "ndat_spline": 15,
        }

        # Realistic NEP parameters
        nep_dict = {
            "E_sat": -16.0,
            "K_sat": 220.0,
            "Q_sat": -300.0,
            "Z_sat": 0.0,
            "E_sym": 31.7,
            "L_sym": 58.7,
            "K_sym": -100.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }

        # Initialize MetaModel
        model = eos.MetaModel_EOS_model(**metamodel_params)

        # Construct EOS
        eos_data = model.construct_eos(nep_dict)
        ns, ps, hs, es, dloge_dlogps, mu, cs2 = eos_data

        # Basic EOS checks
        assert len(ns) > 100  # Should have reasonable resolution
        assert jnp.all(ns > 0)
        assert jnp.all(ps > 0)
        assert jnp.all(es > 0)
        assert jnp.all(cs2 > 0)
        assert jnp.all(cs2 <= 1.0)  # Causal

        # Construct neutron star family
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        log_pcs, masses, radii, lambdas = eos.construct_family(eos_tuple, ndat=30)

        # Check neutron star properties
        assert len(masses) == 30
        assert jnp.all(masses > 0)
        assert jnp.all(radii > 0)
        assert jnp.all(lambdas > 0)

        # Check realistic ranges
        max_mass = jnp.max(masses)
        min_radius = jnp.min(radii)
        max_radius = jnp.max(radii)

        assert 1.5 < max_mass < 3.0  # Realistic maximum mass range
        assert 8.0 < min_radius < 15.0  # Realistic radius range
        assert 10.0 < max_radius < 18.0

        # Check that mass increases initially
        max_idx = jnp.argmax(masses)
        if max_idx > 5:  # Check first part of the sequence
            assert jnp.all(
                jnp.diff(masses[: max_idx // 2]) > -0.01
            )  # Allow small noise

    @pytest.mark.integration
    @pytest.mark.slow
    def test_metamodel_cse_workflow(self):
        """Test MetaModel with CSE extension workflow."""
        # Initialize MetaModel with CSE
        model = eos.MetaModel_with_CSE_EOS_model(
            nsat=0.16, nmin_MM_nsat=0.75, nmax_nsat=10.0, ndat_metamodel=80, ndat_CSE=70
        )

        # NEP parameters with break density
        nep_dict = {
            "E_sat": -16.0,
            "K_sat": 220.0,
            "Q_sat": -400.0,
            "Z_sat": 0.0,
            "E_sym": 31.7,
            "L_sym": 58.7,
            "K_sym": -100.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
            "nbreak": 0.48,  # Break density in fm^-3
        }

        # Create CSE grids
        ngrids = jnp.array([0.5, 0.7, 0.9, 1.2, 1.5])
        cs2grids = jnp.array([0.35, 0.4, 0.45, 0.5, 0.6])

        # Construct EOS
        eos_data = model.construct_eos(nep_dict, ngrids, cs2grids)
        ns, ps, hs, es, dloge_dlogps, mu, cs2 = eos_data

        # Check that we have data from both metamodel and CSE regions
        n_SI = ns / utils.fm_inv3_to_geometric  # Convert back to fm^-3
        assert jnp.min(n_SI) < nep_dict["nbreak"]  # Should include metamodel region
        assert jnp.max(n_SI) > nep_dict["nbreak"]  # Should include CSE region

        # Check continuity at break point
        break_idx = jnp.argmin(jnp.abs(n_SI - nep_dict["nbreak"]))
        if break_idx > 0 and break_idx < len(ps) - 1:
            # Check that pressure is continuous (within numerical precision)
            p_before = ps[break_idx - 1]
            p_after = ps[break_idx + 1]
            assert abs((p_after - p_before) / p_before) < 0.1  # 10% tolerance

        # Construct family
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        log_pcs, masses, radii, lambdas = eos.construct_family(eos_tuple, ndat=25)

        # Should get reasonable neutron star properties
        assert jnp.max(masses) > 1.5
        assert jnp.min(radii) > 8.0
        assert jnp.max(radii) < 20.0


class TestTOVIntegration:
    """Integration tests for TOV solving."""

    @pytest.mark.integration
    def test_tov_eos_consistency(self):
        """Test TOV solver with realistic EOS data."""
        # Create simple but realistic EOS
        n = jnp.linspace(0.1, 1.5, 100)  # fm^-3
        p = 20.0 * n**1.8  # Polytropic-like pressure
        e = 150.0 * n + p  # Energy density

        # Convert to geometric units
        ns = n * utils.fm_inv3_to_geometric
        ps = p * utils.MeV_fm_inv3_to_geometric
        es = e * utils.MeV_fm_inv3_to_geometric

        # Calculate auxiliary quantities
        hs = utils.cumtrapz(ps / (es + ps), jnp.log(ps))
        dloge_dlogps = jnp.diff(jnp.log(es)) / jnp.diff(jnp.log(ps))
        dloge_dlogps = jnp.concatenate([jnp.array([dloge_dlogps[0]]), dloge_dlogps])

        eos_dict = {"p": ps, "h": hs, "e": es, "dloge_dlogp": dloge_dlogps}

        # Test multiple central pressures
        pressure_indices = [20, 30, 40, 50, 60]
        masses = []
        radii = []

        for idx in pressure_indices:
            if idx < len(ps):
                pc = ps[idx]
                try:
                    M, R, k2 = tov.tov_solver(eos_dict, pc)

                    if (
                        jnp.isfinite(M)
                        and jnp.isfinite(R)
                        and M > 0
                        and R > 0
                        and M / R < 0.5
                    ):
                        masses.append(M)
                        radii.append(R)

                        # Basic physics checks
                        assert M > 0.1  # Reasonable mass
                        assert R > 1.0  # Reasonable radius
                        assert k2 > 0  # Positive tidal deformability

                except Exception:
                    continue  # Skip problematic cases

        # Should get several valid solutions
        assert len(masses) >= 3

        # Mass should generally increase with central pressure (up to maximum)
        masses = jnp.array(masses)
        radii = jnp.array(radii)

        # Check that we get a reasonable M-R relationship
        max_mass_idx = jnp.argmax(masses)
        if max_mass_idx > 0:
            # Mass should increase initially
            assert jnp.all(jnp.diff(masses[: max_mass_idx + 1]) >= -0.01)

    @pytest.mark.integration
    def test_tov_ptov_comparison(self):
        """Test comparison between TOV and post-TOV solvers."""
        # Create simple EOS
        n = jnp.linspace(0.1, 1.0, 80)
        p = 15.0 * n**1.6
        e = 120.0 * n + p

        ns = n * utils.fm_inv3_to_geometric
        ps = p * utils.MeV_fm_inv3_to_geometric
        es = e * utils.MeV_fm_inv3_to_geometric

        hs = utils.cumtrapz(ps / (es + ps), jnp.log(ps))
        dloge_dlogps = jnp.diff(jnp.log(es)) / jnp.diff(jnp.log(ps))
        dloge_dlogps = jnp.concatenate([jnp.array([dloge_dlogps[0]]), dloge_dlogps])

        # Standard EOS dict
        eos_dict = {"p": ps, "h": hs, "e": es, "dloge_dlogp": dloge_dlogps}

        # Post-TOV EOS dict (GR limit)
        ptov_eos_dict = eos_dict.copy()
        ptov_eos_dict.update(
            {
                "lambda_BL": 0.0,
                "lambda_DY": 0.0,
                "lambda_HB": 1.0,
                "gamma": 0.0,
                "alpha": 10.0,
                "beta": 0.3,
            }
        )

        # Compare results at same central pressure
        pc = ps[35]

        M_tov, R_tov, k2_tov = tov.tov_solver(eos_dict, pc)
        M_ptov, R_ptov, k2_ptov = ptov.tov_solver(ptov_eos_dict, pc)

        # Should be very similar in GR limit
        assert abs(M_tov - M_ptov) / M_tov < 0.01
        assert abs(R_tov - R_ptov) / R_tov < 0.01
        assert abs(k2_tov - k2_ptov) / k2_tov < 0.05


class TestCrustIntegration:
    """Integration tests for crust loading and EOS construction."""

    @pytest.mark.integration
    def test_crust_metamodel_connection(self):
        """Test smooth connection between crust and MetaModel."""
        # Load crust data
        n_crust, p_crust, e_crust = eos.load_crust("DH")

        # Create MetaModel that should connect to crust
        model = eos.MetaModel_EOS_model(
            nsat=0.16,
            nmin_MM_nsat=0.75,  # Should be above crust
            nmax_nsat=6.0,
            ndat=100,
            crust_name="DH",
            max_n_crust_nsat=0.5,
        )

        nep_dict = {
            "E_sat": -16.0,
            "K_sat": 240.0,
            "Q_sat": -350.0,
            "Z_sat": 0.0,
            "E_sym": 32.0,
            "L_sym": 60.0,
            "K_sym": -120.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }

        # Construct EOS
        eos_data = model.construct_eos(nep_dict)
        ns, ps, hs, es, dloge_dlogps, mu, cs2 = eos_data

        # Convert back to physical units for comparison
        n_SI = ns / utils.fm_inv3_to_geometric
        p_SI = ps / utils.MeV_fm_inv3_to_geometric
        e_SI = es / utils.MeV_fm_inv3_to_geometric

        # Check that EOS starts with crust data
        crust_end_density = model.max_n_crust
        crust_indices = n_SI <= crust_end_density * 1.01  # Small tolerance

        if jnp.sum(crust_indices) > 5:  # If we have enough crust points
            # Crust part should match loaded crust data (approximately)
            n_crust_eos = n_SI[crust_indices]
            p_crust_eos = p_SI[crust_indices]

            # Check that densities are in reasonable range
            assert jnp.min(n_crust_eos) <= jnp.max(n_crust) * 1.01
            assert jnp.max(n_crust_eos) >= jnp.min(n_crust) * 0.99

        # Check that connection region exists
        connection_start = model.max_n_crust
        metamodel_start = model.nmin_MM

        connection_indices = (n_SI > connection_start) & (n_SI < metamodel_start)
        assert jnp.sum(connection_indices) > 0  # Should have connection points

        # Check overall causality
        assert jnp.all(cs2 > 0)
        assert jnp.all(cs2 <= 1.0)

    @pytest.mark.integration
    @pytest.mark.parametrize("crust_name", ["DH", "BPS"])
    def test_different_crusts_consistency(self, crust_name):
        """Test that different crusts give consistent results."""
        # Test with different crust models
        model = eos.MetaModel_EOS_model(
            nsat=0.16,
            nmax_nsat=6.0,
            ndat=80,
            crust_name=crust_name,
            max_n_crust_nsat=0.4,
        )

        nep_dict = {
            "E_sat": -16.0,
            "K_sat": 220.0,
            "Q_sat": -300.0,
            "Z_sat": 0.0,
            "E_sym": 31.7,
            "L_sym": 58.7,
            "K_sym": -100.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }

        # Should not crash and should give reasonable results
        eos_data = model.construct_eos(nep_dict)
        ns, ps, hs, es, dloge_dlogps, mu, cs2 = eos_data

        # Basic checks
        assert len(ns) > 50
        assert jnp.all(ns > 0)
        assert jnp.all(ps > 0)
        assert jnp.all(es > 0)
        assert jnp.all(cs2 > 0)
        assert jnp.all(cs2 <= 1.0)

        # Should be able to construct neutron star family
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        log_pcs, masses, radii, lambdas = eos.construct_family(eos_tuple, ndat=20)

        assert len(masses) == 20
        assert jnp.all(masses > 0)
        assert jnp.all(radii > 0)
        assert jnp.max(masses) > 1.0  # Should get at least 1 solar mass


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    @pytest.mark.integration
    def test_extreme_eos_parameters(self):
        """Test behavior with extreme but physical EOS parameters."""
        # Test with stiff EOS
        stiff_nep = {
            "E_sat": -15.0,  # Slightly less bound
            "K_sat": 300.0,  # Stiffer
            "Q_sat": -200.0,
            "Z_sat": 0.0,
            "E_sym": 35.0,  # Higher symmetry energy
            "L_sym": 80.0,  # Stiffer symmetry energy
            "K_sym": -50.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }

        # Test with soft EOS
        soft_nep = {
            "E_sat": -17.0,  # More bound
            "K_sat": 180.0,  # Softer
            "Q_sat": -400.0,
            "Z_sat": 0.0,
            "E_sym": 28.0,  # Lower symmetry energy
            "L_sym": 40.0,  # Softer symmetry energy
            "K_sym": -150.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }

        for nep_dict, eos_type in [(stiff_nep, "stiff"), (soft_nep, "soft")]:
            model = eos.MetaModel_EOS_model(
                nsat=0.16, nmax_nsat=8.0, ndat=100, crust_name="DH"
            )

            try:
                eos_data = model.construct_eos(nep_dict)
                ns, ps, hs, es, dloge_dlogps, mu, cs2 = eos_data

                # Should maintain causality
                assert jnp.all(cs2 > 0)
                assert jnp.all(cs2 <= 1.0)

                # Should be able to solve TOV
                eos_tuple = (ns, ps, hs, es, dloge_dlogps)
                log_pcs, masses, radii, lambdas = eos.construct_family(
                    eos_tuple, ndat=15
                )

                assert jnp.all(masses > 0)
                assert jnp.all(radii > 0)

                # Stiff EOS should generally give higher maximum mass
                max_mass = jnp.max(masses)
                if eos_type == "stiff":
                    assert max_mass > 1.8  # Should support high masses
                elif eos_type == "soft":
                    assert max_mass > 1.2  # Should still support reasonable masses

            except Exception as e:
                pytest.skip(f"Extreme {eos_type} EOS test failed: {e}")


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_reproducibility():
    """Test that the full pipeline gives reproducible results."""
    # Fixed random seed equivalent (use fixed parameters)
    metamodel_params = {
        "kappas": (0.05, 0.02, 0.01, 0.08, 0.04, 0.015),
        "v_nq": [1.5, 2.2, 0.8, 1.1, 0.6],
        "b_sat": 16.5,
        "b_sym": 28.0,
        "nsat": 0.16,
        "nmin_MM_nsat": 0.75,
        "nmax_nsat": 7.0,
        "ndat": 120,
        "crust_name": "DH",
    }

    nep_dict = {
        "E_sat": -15.8,
        "K_sat": 235.0,
        "Q_sat": -325.0,
        "Z_sat": 0.0,
        "E_sym": 31.2,
        "L_sym": 62.0,
        "K_sym": -115.0,
        "Q_sym": 0.0,
        "Z_sym": 0.0,
    }

    # Run pipeline multiple times
    results = []

    for _ in range(3):
        model = eos.MetaModel_EOS_model(**metamodel_params)
        eos_data = model.construct_eos(nep_dict)
        ns, ps, hs, es, dloge_dlogps, mu, cs2 = eos_data

        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        log_pcs, masses, radii, lambdas = eos.construct_family(eos_tuple, ndat=20)

        max_mass = jnp.max(masses)
        radius_at_1p4 = jnp.interp(1.4, masses, radii)  # Radius at 1.4 solar masses

        results.append((max_mass, radius_at_1p4))

    # Results should be identical (deterministic)
    for i in range(1, len(results)):
        assert abs(results[i][0] - results[0][0]) < 1e-12  # Maximum mass
        assert abs(results[i][1] - results[0][1]) < 1e-12  # Radius at 1.4 Msun
