"""Integration tests for JESTER package."""

import pytest
import jax.numpy as jnp
from jesterTOV import eos, utils
from jesterTOV.tov import GRTOVSolver
from jesterTOV.tov.data_classes import EOSData


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
            "nmax_nsat": 2.0,
            "ndat": 150,
            "crust_name": "DH",
            "max_n_crust_nsat": 0.5,
            "ndat_spline": 15,
        }

        # Realistic NEP parameters
        nep_dict = {
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

        # Initialize MetaModel
        model = eos.MetaModel_EOS_model(**metamodel_params)

        # Construct EOS
        eos_data = model.construct_eos(nep_dict)

        # Basic EOS checks
        assert len(eos_data.ns) > 100  # Should have reasonable resolution
        assert jnp.all(eos_data.ns > 0)
        assert jnp.all(eos_data.ps > 0)
        assert jnp.all(eos_data.es > 0)
        assert jnp.all(eos_data.cs2 > 0)
        assert jnp.all(eos_data.cs2 <= 1.0)  # Causal

        # Construct neutron star family using GRTOVSolver
        solver = GRTOVSolver()
        family_data = solver.construct_family(eos_data, ndat=30, min_nsat=0.75)

        # Check neutron star properties
        assert len(family_data.masses) == 30
        assert jnp.all(family_data.masses > 0)
        assert jnp.all(family_data.radii > 0)
        assert jnp.all(family_data.lambdas > 0)

        # Check realistic ranges for limited EOS (2 nsat)
        max_mass = jnp.max(family_data.masses)
        min_radius = jnp.min(family_data.radii)
        max_radius = jnp.max(family_data.radii)

        assert 0.5 < max_mass < 1.5  # Expected for EOS limited to 2 nsat
        assert 8.0 < min_radius < 20.0  # Radius range for soft EOS
        assert 10.0 < max_radius < 30.0  # Soft EOS produces larger radii

        # Check that mass increases initially
        max_idx = jnp.argmax(family_data.masses)
        if max_idx > 5:  # Check first part of the sequence
            assert jnp.all(
                jnp.diff(family_data.masses[: max_idx // 2]) > -0.01
            )  # Allow small noise

    @pytest.mark.integration
    @pytest.mark.slow
    def test_metamodel_cse_workflow(self):
        """Test MetaModel with CSE extension workflow."""
        # Initialize MetaModel with CSE
        nb_CSE = 4
        model = eos.MetaModel_with_CSE_EOS_model(
            nsat=0.16,
            nmin_MM_nsat=0.75,
            nmax_nsat=6.0,
            ndat_metamodel=80,
            ndat_CSE=70,
            nb_CSE=nb_CSE,
        )

        # NEP parameters with break density
        nep_dict = {
            "E_sat": -16.0,
            "K_sat": 220.0,
            "Q_sat": 0.0,
            "Z_sat": 0.0,
            "E_sym": 31.7,
            "L_sym": 90.0,
            "K_sym": 0.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
            "nbreak": 0.80,  # Break density in fm^-3 (below nmax=6*nsat=0.96)
        }

        # Add CSE grid parameters (normalized positions and cs2 values)
        nep_dict["n_CSE_0_u"] = 0.1
        nep_dict["cs2_CSE_0"] = 0.35
        nep_dict["n_CSE_1_u"] = 0.3
        nep_dict["cs2_CSE_1"] = 0.4
        nep_dict["n_CSE_2_u"] = 0.6
        nep_dict["cs2_CSE_2"] = 0.45
        nep_dict["n_CSE_3_u"] = 0.9
        nep_dict["cs2_CSE_3"] = 0.5
        nep_dict["cs2_CSE_4"] = 0.6  # Final cs2 value

        # Construct EOS
        eos_data = model.construct_eos(nep_dict)

        # Check that we have data from both metamodel and CSE regions
        n_SI = eos_data.ns / utils.fm_inv3_to_geometric  # Convert back to fm^-3
        assert jnp.min(n_SI) < nep_dict["nbreak"]  # Should include metamodel region
        assert jnp.max(n_SI) > nep_dict["nbreak"]  # Should include CSE region

        # Check continuity at break point
        break_idx = jnp.argmin(jnp.abs(n_SI - nep_dict["nbreak"]))
        if break_idx > 0 and break_idx < len(eos_data.ps) - 1:
            # Check that pressure is continuous (within numerical precision)
            p_before = eos_data.ps[break_idx - 1]
            p_after = eos_data.ps[break_idx + 1]
            assert abs((p_after - p_before) / p_before) < 0.1  # 10% tolerance

        # Construct family using GRTOVSolver
        solver = GRTOVSolver()
        family_data = solver.construct_family(eos_data, ndat=25, min_nsat=0.75)

        # Should get reasonable neutron star properties (CSE with 6 nsat base)
        assert (
            jnp.max(family_data.masses) > 1.5
        )  # Expected for CSE extension from 6 nsat base
        assert jnp.min(family_data.radii) > 8.0
        assert jnp.max(family_data.radii) < 25.0


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
        dedps = es / ps * dloge_dlogps
        cs2s = 1.0 / dedps

        eos_data = EOSData(
            ns=ns, ps=ps, hs=hs, es=es, dloge_dlogps=dloge_dlogps, cs2=cs2s
        )
        solver = GRTOVSolver()

        # Test multiple central pressures
        pressure_indices = [20, 30, 40, 50, 60]
        masses = []
        radii = []

        for idx in pressure_indices:
            if idx < len(ps):
                pc = float(ps[idx])
                try:
                    solution = solver.solve(eos_data, pc)

                    if (
                        jnp.isfinite(solution.M)
                        and jnp.isfinite(solution.R)
                        and solution.M > 0
                        and solution.R > 0
                        and solution.M / solution.R < 0.5
                    ):
                        masses.append(solution.M)
                        radii.append(solution.R)

                        # Basic physics checks
                        assert solution.M > 0.1  # Reasonable mass
                        assert solution.R > 1.0  # Reasonable radius
                        assert solution.k2 > 0  # Positive tidal deformability

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
    def test_tov_post_comparison(self):
        """Test comparison between GR TOV and post-TOV solvers."""
        from jesterTOV.tov import GRTOVSolver, PostTOVSolver
        from jesterTOV.tov.data_classes import EOSData

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
        dedps = es / ps * dloge_dlogps
        cs2s = 1.0 / dedps

        # Create EOSData
        eos_data = EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=cs2s,
            mu=jnp.zeros_like(ns),  # Chemical potential (not used)
            extra_constraints=None,
        )

        # Compare results at same central pressure
        pc = ps[35]

        # GR TOV solver
        gr_solver = GRTOVSolver()
        tov_solution = gr_solver.solve(eos_data, pc)
        M_tov, R_tov, k2_tov = tov_solution.M, tov_solution.R, tov_solution.k2

        # Post-TOV solver in GR limit (all MG parameters = 0)
        post_solver = PostTOVSolver()
        post_params = {
            "lambda_BL": 0.0,
            "lambda_DY": 0.0,
            "lambda_HB": 1.0,
            "gamma": 0.0,
            "alpha": 10.0,
            "beta": 0.3,
        }
        post_solution = post_solver.solve(eos_data, pc, **post_params)
        M_post, R_post, k2_post = post_solution.M, post_solution.R, post_solution.k2

        # Should be very similar in GR limit
        assert abs(M_tov - M_post) / M_tov < 0.01
        assert abs(R_tov - R_post) / R_tov < 0.01
        assert abs(k2_tov - k2_post) / k2_tov < 0.05


class TestCrustIntegration:
    """Integration tests for crust loading and EOS construction."""

    @pytest.mark.integration
    def test_crust_metamodel_connection(self):
        """Test smooth connection between crust and MetaModel."""
        # Load crust data
        crust = eos.Crust("DH")
        n_crust, p_crust, e_crust = crust.n, crust.p, crust.e

        # Create MetaModel that should connect to crust
        model = eos.MetaModel_EOS_model(
            nsat=0.16,
            nmin_MM_nsat=0.75,  # Should be above crust
            nmax_nsat=2.0,
            ndat=100,
            crust_name="DH",
            max_n_crust_nsat=0.5,
        )

        nep_dict = {
            "E_sat": -16.0,
            "K_sat": 240.0,
            "Q_sat": 0.0,
            "Z_sat": 0.0,
            "E_sym": 32.0,
            "L_sym": 90.0,
            "K_sym": 0.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }

        # Construct EOS
        eos_data = model.construct_eos(nep_dict)

        # Convert back to physical units for comparison
        n_SI = eos_data.ns / utils.fm_inv3_to_geometric
        p_SI = eos_data.ps / utils.MeV_fm_inv3_to_geometric
        e_SI = eos_data.es / utils.MeV_fm_inv3_to_geometric

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
        assert jnp.all(eos_data.cs2 > 0)
        assert jnp.all(eos_data.cs2 <= 1.0)

    @pytest.mark.integration
    @pytest.mark.parametrize("crust_name", ["DH", "BPS"])
    def test_different_crusts_consistency(self, crust_name):
        """Test that different crusts give consistent results."""
        # Test with different crust models
        model = eos.MetaModel_EOS_model(
            nsat=0.16,
            nmax_nsat=2.0,
            ndat=80,
            crust_name=crust_name,
            max_n_crust_nsat=0.4,
        )

        nep_dict = {
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

        # Should not crash and should give reasonable results
        eos_data = model.construct_eos(nep_dict)

        # Basic checks
        assert len(eos_data.ns) > 50
        assert jnp.all(eos_data.ns > 0)
        assert jnp.all(eos_data.ps > 0)
        assert jnp.all(eos_data.es > 0)
        assert jnp.all(eos_data.cs2 > 0)
        assert jnp.all(eos_data.cs2 <= 1.0)

        # Should be able to construct neutron star family using GRTOVSolver
        solver = GRTOVSolver()
        family_data = solver.construct_family(eos_data, ndat=20, min_nsat=0.75)

        assert len(family_data.masses) == 20
        assert jnp.all(family_data.masses > 0)
        assert jnp.all(family_data.radii > 0)
        assert jnp.max(family_data.masses) > 0.5  # Expected for EOS limited to 2 nsat


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
                nsat=0.16, nmax_nsat=2.0, ndat=100, crust_name="DH"
            )

            try:
                eos_data = model.construct_eos(nep_dict)

                # Should maintain causality
                assert jnp.all(eos_data.cs2 > 0)
                assert jnp.all(eos_data.cs2 <= 1.0)

                # Should be able to solve TOV using GRTOVSolver
                solver = GRTOVSolver()
                family_data = solver.construct_family(eos_data, ndat=15, min_nsat=0.75)

                assert jnp.all(family_data.masses > 0)
                assert jnp.all(family_data.radii > 0)

                # Stiff EOS should generally give higher maximum mass
                max_mass = jnp.max(family_data.masses)
                if eos_type == "stiff":
                    assert max_mass > 1.0  # Realistic for EOS limited to 2 nsat
                elif eos_type == "soft":
                    assert max_mass > 0.2  # Soft EOS naturally produces lower masses

            except Exception as e:
                # Let the test fail with the actual error instead of skipping
                raise


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
        "nmax_nsat": 2.0,
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

        # Use GRTOVSolver to construct family
        solver = GRTOVSolver()
        family_data = solver.construct_family(eos_data, ndat=20, min_nsat=0.75)

        max_mass = jnp.max(family_data.masses)
        radius_at_1p4 = jnp.interp(
            1.4, family_data.masses, family_data.radii
        )  # Radius at 1.4 solar masses

        results.append((max_mass, radius_at_1p4))

    # Results should be identical (deterministic)
    for i in range(1, len(results)):
        assert abs(results[i][0] - results[0][0]) < 1e-12  # Maximum mass
        assert abs(results[i][1] - results[0][1]) < 1e-12  # Radius at 1.4 Msun
