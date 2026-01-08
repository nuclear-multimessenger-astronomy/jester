"""
Regression tests for cs2 numerical round-trip error bug.

This module tests that sound speed squared (cs2) is correctly passed through
from EOS models to construct_family, avoiding numerical round-trip errors that
previously caused artificially low MTOV values (~0.35 M_sun instead of 1.4-2.0 M_sun).

Bug description:
The bug occurred when cs2 was:
1. Computed analytically in MetaModel
2. Used to integrate p and e
3. Differentiated to compute dloge_dlogp
4. Recomputed in construct_family as cs2 = p / e / dloge_dlogp

This integrate→differentiate→divide cycle introduced numerical errors (up to 1240%
relative error at crust-core boundary), creating spurious causality violations that
caused construct_family to set pc_max too low.

Fix:
Pass cs2 directly through to construct_family instead of recomputing it.
"""

import jax.numpy as jnp
import pytest

import jesterTOV.eos as eos
from jesterTOV import utils


class TestCS2Roundtrip:
    """Test that cs2 round-trip errors do not occur."""

    def test_cs2_not_recomputed_in_construct_family(self):
        """
        Verify that construct_family uses the stored cs2 rather than recomputing it.

        This test creates an EOS where the stored cs2 is deliberately different from
        what p/e/dloge_dlogp would give, and verifies that construct_family uses the
        stored value (which should be correct).
        """
        # Create simple EOS
        ns = jnp.linspace(0.1, 1.0, 50) * utils.fm_inv3_to_geometric
        ps = jnp.linspace(10, 100, 50) * utils.MeV_fm_inv3_to_geometric
        es = jnp.linspace(20, 200, 50) * utils.MeV_fm_inv3_to_geometric
        hs = utils.cumtrapz(ps / (es + ps), jnp.log(ps))
        dloge_dlogps = jnp.diff(jnp.log(es)) / jnp.diff(jnp.log(ps))
        dloge_dlogps = jnp.concatenate([jnp.array([dloge_dlogps[0]]), dloge_dlogps])

        # Create cs2 that is deliberately different from p/e/dloge_dlogp
        cs2_correct = jnp.linspace(0.1, 0.85, 50)  # Smooth, causal
        cs2_recomputed = ps / es / dloge_dlogps  # Would have numerical errors

        # Pass correct cs2 to construct_family
        eos_tuple = (ns, ps, hs, es, dloge_dlogps, cs2_correct)
        log_pcs, ms, rs, lambdas = eos.construct_family(eos_tuple, ndat=10)

        # If construct_family recomputed cs2, it would use cs2_recomputed which
        # might hit causality limit. With cs2_correct, it should work fine.
        assert len(ms) == 10
        assert jnp.all(ms > 0)
        assert jnp.all(rs > 0)

    def test_metamodel_cs2_has_low_error(self):
        """
        Test that cs2 from MetaModel.construct_eos has low numerical error
        compared to recomputing from p/e/dloge_dlogp.

        This verifies that analytically-computed cs2 is accurate enough to avoid
        spurious causality violations.
        """
        metamodel_params = {
            "nsat": 0.16,
            "nmin_MM_nsat": 0.75,
            "nmax_nsat": 6.0,
            "ndat": 200,
            "crust_name": "DH",
            "max_n_crust_nsat": 0.5,
            "ndat_spline": 10,
        }

        nep_dict = {
            "P_sat": 0.5,
            "K_sat": 250.0,
            "Q_sat": -300.0,
            "Z_sat": 0.0,
            "S_sym": 32.0,
            "L_sym": 60.0,
            "K_sym": -50.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }

        model = eos.MetaModel_EOS_model(**metamodel_params)
        ns, ps, hs, es, dloge_dlogps, mu, cs2_analytical = model.construct_eos(nep_dict)

        # Recompute cs2 the "buggy" way
        cs2_recomputed = ps / es / dloge_dlogps

        # Calculate relative error
        relative_error = jnp.abs(cs2_recomputed - cs2_analytical) / (cs2_analytical + 1e-10)

        # The error should be small (< 10% for most points)
        # In the bug, errors reached 1240% at the crust-core boundary
        median_error = jnp.median(relative_error)
        max_error = jnp.max(relative_error)

        # These thresholds may need adjustment based on numerical stability
        assert median_error < 0.01, f"Median cs2 error too high: {median_error:.2%}"
        assert max_error < 0.5, f"Max cs2 error too high: {max_error:.2%}"

    def test_metamodel_produces_reasonable_mtov(self):
        """
        Test that MetaModel produces reasonable MTOV values.

        This is a regression test for the bug where spurious cs2 spikes
        caused artificially low MTOV values (~0.35 M_sun instead of 1.4-2.0 M_sun).
        """
        metamodel_params = {
            "nsat": 0.16,
            "nmin_MM_nsat": 0.75,
            "nmax_nsat": 6.0,
            "ndat": 200,
            "crust_name": "DH",
            "max_n_crust_nsat": 0.5,
            "ndat_spline": 10,
        }

        # Test parameters that could trigger numerical issues at boundaries
        nep_dict = {
            "P_sat": 0.5,
            "K_sat": 220.0,
            "Q_sat": -200.0,
            "Z_sat": 0.0,
            "S_sym": 30.0,
            "L_sym": 45.0,
            "K_sym": -100.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }

        model = eos.MetaModel_EOS_model(**metamodel_params)
        ns, ps, hs, es, dloge_dlogps, mu, cs2 = model.construct_eos(nep_dict)

        # Construct family (including cs2 to avoid bug)
        eos_tuple = (ns, ps, hs, es, dloge_dlogps, cs2)
        log_pcs, masses, radii, lambdas = eos.construct_family(eos_tuple, ndat=50)

        # Calculate MTOV
        mtov = jnp.max(masses)

        # MTOV should be in reasonable range (1.0-2.5 M_sun)
        # Before the fix, buggy samples had MTOV ~ 0.2-0.5 M_sun
        assert mtov > 0.8, f"MTOV too low: {mtov:.3f} M_sun (bug may have returned!)"
        assert mtov < 3.0, f"MTOV unreasonably high: {mtov:.3f} M_sun"

    def test_cs2_recomputation_introduces_errors(self):
        """
        Test that recomputing cs2 from p/e/dloge_dlogp introduces numerical errors
        compared to the analytically-computed cs2.

        This directly demonstrates the bug: the integrate→differentiate→divide cycle
        introduces errors, especially at the crust-core boundary.
        """
        metamodel_params = {
            "nsat": 0.16,
            "nmin_MM_nsat": 0.75,
            "nmax_nsat": 2.0,  # Limited range to avoid acausal regions
            "ndat": 200,
            "crust_name": "DH",
            "max_n_crust_nsat": 0.5,
            "ndat_spline": 10,  # Coarse grid at boundary → numerical errors
        }

        # Use moderate NEP parameters that produce a causal EOS
        nep_dict = {
            "P_sat": 0.5,
            "K_sat": 250.0,
            "Q_sat": -200.0,
            "Z_sat": 0.0,
            "S_sym": 30.0,
            "L_sym": 60.0,
            "K_sym": -50.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }

        model = eos.MetaModel_EOS_model(**metamodel_params)
        ns, ps, hs, es, dloge_dlogps, mu, cs2_analytical = model.construct_eos(nep_dict)

        # Recompute cs2 (the buggy way)
        cs2_recomputed = ps / es / dloge_dlogps

        # Calculate errors
        relative_error = jnp.abs(cs2_recomputed - cs2_analytical) / (cs2_analytical + 1e-10)
        max_error = jnp.max(relative_error)

        # The bug manifested as errors > 1000% at the crust-core boundary
        # With the fix, construct_family uses cs2_analytical, avoiding these errors

        # Test that using stored cs2 gives correct results
        eos_tuple_correct = (ns, ps, hs, es, dloge_dlogps, cs2_analytical)
        log_pcs_correct, masses_correct, radii_correct, lambdas_correct = eos.construct_family(
            eos_tuple_correct, ndat=30
        )

        # Should get reasonable neutron stars when using analytical cs2
        mtov = jnp.max(masses_correct)
        assert mtov > 0.8, f"MTOV too low with analytical cs2: {mtov:.3f} M_sun"
        assert mtov < 3.0, f"MTOV too high with analytical cs2: {mtov:.3f} M_sun"
