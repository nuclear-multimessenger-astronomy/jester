"""Tests for inference likelihood system (base, factory, specific likelihoods)."""

import json

import pytest
import jax
import jax.numpy as jnp
from unittest.mock import MagicMock, patch

from jesterTOV.inference.config import schema
from jesterTOV.inference.likelihoods import factory
from jesterTOV.inference.likelihoods.combined import ZeroLikelihood, CombinedLikelihood
from jesterTOV.inference.likelihoods.constraints import (
    ConstraintEOSLikelihood,
    ConstraintTOVLikelihood,
    ConstraintEsymLikelihood,
    ConstraintGammaLikelihood,
    check_tov_validity,
    check_causality_violation,
    check_stability,
    check_pressure_monotonicity,
    check_all_constraints,
    check_gamma_bounds,
)
from jesterTOV.inference.likelihoods.chieft import ChiEFTLikelihood
from jesterTOV.inference.likelihoods.radio import RadioTimingLikelihood
from jesterTOV.inference.likelihoods.mock_mr import MockMassRadiusLikelihood
from jesterTOV.inference.likelihoods.gw import GWLikelihood, StackedGWLikelihood
from jesterTOV.inference.base import LikelihoodBase


class TestZeroLikelihood:
    """Test ZeroLikelihood functionality."""

    def test_zero_likelihood_returns_zero(self):
        """Test that ZeroLikelihood always returns 0.0."""
        likelihood = ZeroLikelihood()

        # Any params should give 0.0
        params = {"K_sat": 220.0, "L_sym": 90.0}
        result = likelihood.evaluate(params)

        assert result == 0.0

    def test_zero_likelihood_with_empty_params(self):
        """Test ZeroLikelihood with empty parameter dict."""
        likelihood = ZeroLikelihood()
        result = likelihood.evaluate({})
        assert result == 0.0


class TestCombinedLikelihood:
    """Test CombinedLikelihood functionality."""

    def test_combined_likelihood_sums_correctly(self):
        """Test that CombinedLikelihood sums log likelihoods."""
        # Create two zero likelihoods - should sum to 0.0
        likelihood1 = ZeroLikelihood()
        likelihood2 = ZeroLikelihood()

        combined = CombinedLikelihood([likelihood1, likelihood2])
        result = combined.evaluate({})

        assert result == 0.0

    def test_combined_likelihood_with_single_likelihood(self):
        """Test CombinedLikelihood with single likelihood."""
        likelihood = ZeroLikelihood()
        combined = CombinedLikelihood([likelihood])

        result = combined.evaluate({})
        assert result == 0.0

    def test_combined_likelihood_initialization(self):
        """Test CombinedLikelihood initialization."""
        likelihood1 = ZeroLikelihood()
        likelihood2 = ZeroLikelihood()

        combined = CombinedLikelihood([likelihood1, likelihood2])

        assert len(combined.likelihoods_list) == 2
        assert combined.counter == 0


class TestConstraintHelperFunctions:
    """Test constraint checking helper functions."""

    def test_check_tov_validity_valid_arrays(self):
        """Test TOV validity check with valid arrays (no NaN)."""
        masses = jnp.array([1.4, 1.8, 2.0])
        radii = jnp.array([12.0, 11.5, 11.0])
        lambdas = jnp.array([400.0, 300.0, 200.0])

        n_violations = check_tov_validity(masses, radii, lambdas)
        assert n_violations == 0.0

    def test_check_tov_validity_with_nan(self):
        """Test TOV validity check with NaN values."""
        masses = jnp.array([1.4, jnp.nan, 2.0])
        radii = jnp.array([12.0, 11.5, jnp.nan])
        lambdas = jnp.array([400.0, 300.0, 200.0])

        n_violations = check_tov_validity(masses, radii, lambdas)
        assert n_violations == 2.0  # Two NaN values

    def test_check_causality_violation_valid(self):
        """Test causality check with valid cs^2 values."""
        cs2 = jnp.array([0.1, 0.3, 0.5, 0.9])

        n_violations = check_causality_violation(cs2)
        assert n_violations == 0.0

    def test_check_causality_violation_invalid(self):
        """Test causality check with cs^2 > 1 (violates causality)."""
        cs2 = jnp.array([0.5, 1.2, 0.8, 1.5])  # Two violations

        n_violations = check_causality_violation(cs2)
        assert n_violations == 2.0

    def test_check_stability_valid(self):
        """Test stability check with positive cs^2."""
        cs2 = jnp.array([0.1, 0.3, 0.5])

        n_violations = check_stability(cs2)
        assert n_violations == 0.0

    def test_check_stability_invalid(self):
        """Test stability check with negative cs^2 (unstable)."""
        cs2 = jnp.array([0.5, -0.1, 0.3, -0.2])  # Two violations

        n_violations = check_stability(cs2)
        assert n_violations == 2.0

    def test_check_pressure_monotonicity_valid(self):
        """Test pressure monotonicity with increasing pressure."""
        p = jnp.array([1.0, 2.0, 3.0, 4.0])

        n_violations = check_pressure_monotonicity(p)
        assert n_violations == 0.0

    def test_check_pressure_monotonicity_invalid(self):
        """Test pressure monotonicity with decreasing pressure."""
        p = jnp.array([1.0, 3.0, 2.0, 4.0])  # One decrease

        n_violations = check_pressure_monotonicity(p)
        assert n_violations == 1.0

    def test_check_all_constraints_valid(self):
        """Test check_all_constraints with all valid inputs."""
        masses = jnp.array([1.4, 1.8, 2.0])
        radii = jnp.array([12.0, 11.5, 11.0])
        lambdas = jnp.array([400.0, 300.0, 200.0])
        cs2 = jnp.array([0.3, 0.5, 0.7])
        p = jnp.array([1.0, 2.0, 3.0])

        constraints = check_all_constraints(masses, radii, lambdas, cs2, p)

        assert constraints["n_tov_failures"] == 0.0
        assert constraints["n_causality_violations"] == 0.0
        assert constraints["n_stability_violations"] == 0.0
        assert constraints["n_pressure_violations"] == 0.0

    def test_check_all_constraints_with_violations(self):
        """Test check_all_constraints with multiple violations."""
        masses = jnp.array([1.4, jnp.nan, 2.0])  # 1 NaN
        radii = jnp.array([12.0, 11.5, 11.0])
        lambdas = jnp.array([400.0, 300.0, 200.0])
        cs2 = jnp.array([0.3, 1.5, -0.1])  # 1 causality, 1 stability violation
        p = jnp.array([1.0, 3.0, 2.0])  # 1 pressure decrease

        constraints = check_all_constraints(masses, radii, lambdas, cs2, p)

        assert constraints["n_tov_failures"] == 1.0
        assert constraints["n_causality_violations"] == 1.0
        assert constraints["n_stability_violations"] == 1.0
        assert constraints["n_pressure_violations"] == 1.0


class TestConstraintEOSLikelihood:
    """Test ConstraintEOSLikelihood (EOS-level constraints only)."""

    def test_constraint_eos_likelihood_all_valid(self):
        """Test ConstraintEOSLikelihood with all valid constraints."""
        likelihood = ConstraintEOSLikelihood()

        # Valid params (no violations)
        params = {
            "n_causality_violations": 0.0,
            "n_stability_violations": 0.0,
            "n_pressure_violations": 0.0,
        }

        result = likelihood.evaluate(params)
        assert result == 0.0

    def test_constraint_eos_likelihood_causality_violation(self):
        """Test ConstraintEOSLikelihood with causality violation."""
        likelihood = ConstraintEOSLikelihood(penalty_causality=-1e10)

        # Causality violation
        params = {
            "n_causality_violations": 1.0,  # One violation
            "n_stability_violations": 0.0,
            "n_pressure_violations": 0.0,
        }

        result = likelihood.evaluate(params)
        assert result == -1e10

    def test_constraint_eos_likelihood_multiple_violations(self):
        """Test ConstraintEOSLikelihood with multiple violations."""
        likelihood = ConstraintEOSLikelihood(
            penalty_causality=-1e10,
            penalty_stability=-1e5,
            penalty_pressure=-1e5,
        )

        # Multiple violations
        params = {
            "n_causality_violations": 1.0,
            "n_stability_violations": 2.0,
            "n_pressure_violations": 1.0,
        }

        result = likelihood.evaluate(params)
        # Should sum all penalties
        assert result == -1e10 + -1e5 + -1e5

    def test_constraint_eos_likelihood_missing_keys(self):
        """Test ConstraintEOSLikelihood with missing violation keys (defaults to 0)."""
        likelihood = ConstraintEOSLikelihood()

        # Empty params - should use defaults
        params = {}

        result = likelihood.evaluate(params)
        assert result == 0.0

    def test_check_gamma_bounds_valid(self):
        """Test gamma bounds check with valid Gamma values."""
        gamma_vals = jnp.array([0.8, 1.2, 2.0, 3.5, 4.0])  # All in [0.6, 4.5]

        n_violations = check_gamma_bounds(gamma_vals)
        assert n_violations == 0.0

    def test_check_gamma_bounds_below_lower_bound(self):
        """Test gamma bounds check with Gamma < 0.6 (violates lower bound)."""
        gamma_vals = jnp.array([0.5, 0.8, 1.2, 2.0])  # One violation (0.5 < 0.6)

        n_violations = check_gamma_bounds(gamma_vals)
        assert n_violations == 1.0

    def test_check_gamma_bounds_above_upper_bound(self):
        """Test gamma bounds check with Gamma > 4.5 (violates upper bound)."""
        gamma_vals = jnp.array([1.0, 2.0, 4.6, 5.0])  # Two violations (4.6, 5.0 > 4.5)

        n_violations = check_gamma_bounds(gamma_vals)
        assert n_violations == 2.0

    def test_check_gamma_bounds_both_bounds_violated(self):
        """Test gamma bounds check with violations at both bounds."""
        gamma_vals = jnp.array(
            [0.5, 1.0, 2.0, 4.6]
        )  # Two violations (0.5 < 0.6, 4.6 > 4.5)

        n_violations = check_gamma_bounds(gamma_vals)
        assert n_violations == 2.0

    def test_check_gamma_bounds_at_boundaries(self):
        """Test gamma bounds check with Gamma exactly at boundaries."""
        gamma_vals = jnp.array([0.6, 1.0, 4.5])  # All valid, including boundaries

        n_violations = check_gamma_bounds(gamma_vals)
        assert n_violations == 0.0


class TestConstraintGammaLikelihood:
    """Test ConstraintGammaLikelihood (Gamma bound constraints for spectral EOS)."""

    def test_constraint_gamma_likelihood_valid(self):
        """Test ConstraintGammaLikelihood with valid Gamma bounds."""
        likelihood = ConstraintGammaLikelihood()

        # Valid Gamma (no violations)
        params = {"n_gamma_violations": 0.0}

        result = likelihood.evaluate(params)
        assert result == 0.0

    def test_constraint_gamma_likelihood_with_violations(self):
        """Test ConstraintGammaLikelihood with Gamma bound violations."""
        likelihood = ConstraintGammaLikelihood(penalty_gamma=-1e10)

        # Gamma violations
        params = {"n_gamma_violations": 3.0}  # Multiple violations

        result = likelihood.evaluate(params)
        # Penalty is multiplied by number of violations: 3.0 * (-1e10) = -3e10
        assert result == -3e10

    def test_constraint_gamma_likelihood_missing_key(self):
        """Test ConstraintGammaLikelihood with missing n_gamma_violations key."""
        likelihood = ConstraintGammaLikelihood()

        # Empty params - should use default (0.0)
        params = {}

        result = likelihood.evaluate(params)
        assert result == 0.0

    def test_constraint_gamma_likelihood_with_non_spectral_transform(self):
        """Test ConstraintGammaLikelihood gracefully handles non-spectral transforms.

        Non-spectral transforms (metamodel, metamodel_cse) won't have n_gamma_violations
        in their output. The likelihood should return 0.0 in this case.
        """
        likelihood = ConstraintGammaLikelihood()

        # Params from non-spectral transform (no n_gamma_violations key)
        params = {
            "masses_EOS": jnp.array([1.4, 1.8, 2.0]),
            "radii_EOS": jnp.array([12.0, 11.5, 11.0]),
        }

        result = likelihood.evaluate(params)
        assert result == 0.0


class TestConstraintEsymLikelihood:
    """Test ConstraintEsymLikelihood (symmetry energy constraint for metamodel EOS)."""

    def test_constraint_esym_likelihood_valid(self):
        """Returns 0.0 when no esym violations."""
        likelihood = ConstraintEsymLikelihood()
        params = {"n_esym_violations": 0.0}
        result = likelihood.evaluate(params)
        assert float(result) == 0.0

    def test_constraint_esym_likelihood_with_violations(self):
        """Penalty is proportional to number of violation points."""
        likelihood = ConstraintEsymLikelihood(penalty_esym=-1e10)
        params = {"n_esym_violations": 5.0}
        result = likelihood.evaluate(params)
        assert float(result) == -5e10

    def test_constraint_esym_likelihood_missing_key(self):
        """Missing key defaults to 0 → 0.0 log-likelihood."""
        likelihood = ConstraintEsymLikelihood()
        result = likelihood.evaluate({})
        assert float(result) == 0.0

    def test_constraint_esym_likelihood_non_metamodel_params(self):
        """No n_esym_violations key (non-metamodel transform) → returns 0.0."""
        likelihood = ConstraintEsymLikelihood()
        params = {
            "masses_EOS": jnp.array([1.4, 2.0]),
            "radii_EOS": jnp.array([12.0, 11.0]),
        }
        result = likelihood.evaluate(params)
        assert float(result) == 0.0

    def test_metamodel_good_params_zero_violations(self):
        """MetaModel with typical NEPs stores zero esym violations in extra_constraints."""
        from jesterTOV.eos.metamodel.base import MetaModel_EOS_model

        eos = MetaModel_EOS_model()
        params = {
            "E_sat": -16.0,
            "K_sat": 230.0,
            "Q_sat": 300.0,
            "Z_sat": -500.0,
            "E_sym": 32.0,
            "L_sym": 60.0,
            "K_sym": -100.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }
        eos_data = eos.construct_eos(params)
        assert eos_data.extra_constraints is not None
        assert "n_esym_violations" in eos_data.extra_constraints
        assert float(eos_data.extra_constraints["n_esym_violations"]) == 0.0

    def test_metamodel_bad_params_nonzero_violations(self):
        """MetaModel with pathological NEPs (very negative L_sym) gives esym < 0."""
        from jesterTOV.eos.metamodel.base import MetaModel_EOS_model

        eos = MetaModel_EOS_model()
        params = {
            "E_sat": -16.0,
            "K_sat": 230.0,
            "Q_sat": 300.0,
            "Z_sat": -500.0,
            "E_sym": 5.0,
            "L_sym": -200.0,
            "K_sym": -400.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }
        eos_data = eos.construct_eos(params)
        assert eos_data.extra_constraints is not None
        assert float(eos_data.extra_constraints["n_esym_violations"]) > 0.0

    def test_metamodel_cse_propagates_violations(self):
        """MetaModel+CSE propagates esym violations from the inner metamodel."""
        from jesterTOV.eos.metamodel.metamodel_CSE import MetaModel_with_CSE_EOS_model

        eos = MetaModel_with_CSE_EOS_model(nb_CSE=4)
        good_params = {
            "E_sat": -16.0,
            "K_sat": 230.0,
            "Q_sat": 300.0,
            "Z_sat": -500.0,
            "E_sym": 32.0,
            "L_sym": 60.0,
            "K_sym": -100.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
            "nbreak": 0.24,
        }
        for i in range(4):
            good_params[f"n_CSE_{i}_u"] = (i + 1) / 5.0
            good_params[f"cs2_CSE_{i}"] = 0.3
        good_params["cs2_CSE_4"] = 0.3

        eos_data = eos.construct_eos(good_params)
        assert eos_data.extra_constraints is not None
        assert "n_esym_violations" in eos_data.extra_constraints
        assert float(eos_data.extra_constraints["n_esym_violations"]) == 0.0

    def test_full_pipeline_likelihood_evaluates(self):
        """ConstraintEsymLikelihood evaluates correctly on metamodel extra_constraints."""
        from jesterTOV.eos.metamodel.base import MetaModel_EOS_model

        eos = MetaModel_EOS_model()
        bad_params = {
            "E_sat": -16.0,
            "K_sat": 230.0,
            "Q_sat": 300.0,
            "Z_sat": -500.0,
            "E_sym": 5.0,
            "L_sym": -200.0,
            "K_sym": -400.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }
        eos_data = eos.construct_eos(bad_params)
        likelihood = ConstraintEsymLikelihood(penalty_esym=-1e10)
        logL = likelihood.evaluate(eos_data.extra_constraints)  # type: ignore[arg-type]
        assert float(logL) < -1e8


class TestConstraintTOVLikelihood:
    """Test ConstraintTOVLikelihood (TOV-level constraints only)."""

    def test_constraint_tov_likelihood_valid(self):
        """Test ConstraintTOVLikelihood with valid TOV integration."""
        likelihood = ConstraintTOVLikelihood()

        # Valid TOV (no NaN)
        params = {"n_tov_failures": 0.0}

        result = likelihood.evaluate(params)
        assert result == 0.0

    def test_constraint_tov_likelihood_with_failures(self):
        """Test ConstraintTOVLikelihood with TOV integration failure."""
        likelihood = ConstraintTOVLikelihood(penalty_tov=-1e10)

        # TOV failure (NaN in output)
        params = {"n_tov_failures": 3.0}  # Multiple NaN

        result = likelihood.evaluate(params)
        assert result == -1e10

    def test_constraint_tov_likelihood_missing_key(self):
        """Test ConstraintTOVLikelihood with missing n_tov_failures key."""
        likelihood = ConstraintTOVLikelihood()

        # Empty params - should use default
        params = {}

        result = likelihood.evaluate(params)
        assert result == 0.0


class TestChiEFTLikelihood:
    """Test ChiEFTLikelihood initialization and basic properties."""

    def test_chieft_likelihood_initialization(self):
        """Test ChiEFTLikelihood initializes correctly."""
        # Note: This may fail if data files are missing - if so, document in CLAUDE.md
        try:
            likelihood = ChiEFTLikelihood(
                low_filename=None,  # Will use default
                high_filename=None,  # Will use default
                nb_n=100,
            )

            # Check basic properties
            assert likelihood.nb_n == 100

        except FileNotFoundError as e:
            pytest.skip(f"ChiEFT data files not found: {e}")

    def test_chieft_likelihood_with_custom_files(self):
        """Test ChiEFTLikelihood with custom file paths."""
        from pathlib import Path

        # Use the actual ChiEFT data files
        data_dir = (
            Path(__file__).parent.parent.parent
            / "jesterTOV"
            / "inference"
            / "data"
            / "chiEFT"
            / "2402.04172"
        )
        low_file = data_dir / "low.dat"
        high_file = data_dir / "high.dat"

        # Verify files exist
        assert low_file.exists(), f"ChiEFT low data file not found: {low_file}"
        assert high_file.exists(), f"ChiEFT high data file not found: {high_file}"

        # Create likelihood with custom file paths
        likelihood = ChiEFTLikelihood(
            low_filename=str(low_file),
            high_filename=str(high_file),
            nb_n=100,
        )

        # Check basic properties
        assert likelihood.nb_n == 100
        assert len(likelihood.n_low) > 0
        assert len(likelihood.p_low) > 0
        assert len(likelihood.n_high) > 0
        assert len(likelihood.p_high) > 0

        # Check interpolation functions exist and work
        test_density = 1.0  # 1.0 * n_sat
        low_pressure = likelihood.EFT_low(test_density)
        high_pressure = likelihood.EFT_high(test_density)
        assert low_pressure < high_pressure  # Low bound should be less than high bound


class TestRadioTimingLikelihood:
    """Test RadioTimingLikelihood functionality."""

    def test_radio_timing_likelihood_initialization(self):
        """Test RadioTimingLikelihood initializes correctly."""
        likelihood = RadioTimingLikelihood(
            psr_name="J0348+0432",
            mean=2.01,  # Solar masses
            std=0.04,  # Uncertainty
        )

        assert likelihood.psr_name == "J0348+0432"
        assert likelihood.mean == 2.01
        assert likelihood.std == 0.04

    def test_radio_timing_likelihood_evaluate(self):
        """Test RadioTimingLikelihood evaluation.

        NOTE: This is an integration test that requires a valid transform
        output with masses_EOS array. If it fails, check that:
        1. Transform provides 'masses_EOS' in params
        2. masses_EOS is a JAX array with sufficient points
        """
        likelihood = RadioTimingLikelihood(
            psr_name="J0348+0432",
            mean=2.01,
            std=0.04,
        )

        # Mock transform output with masses_EOS
        # Realistic NS mass range: 1.0 - 2.5 solar masses
        masses_eos = jnp.linspace(1.0, 2.5, 100)
        params = {
            "masses_EOS": masses_eos,
        }

        result = likelihood.evaluate(params)

        # Should return a finite log likelihood
        assert jnp.isfinite(result)

        # For a stiff EOS with max mass > 2.01, likelihood should be reasonable
        # (not a large negative penalty)
        assert result > -1000.0, f"Likelihood too negative: {result}"


class TestMockMassRadiusLikelihood:
    """Test MockMassRadiusLikelihood functionality."""

    def test_initialization(self):
        """Test MockMassRadiusLikelihood initializes correctly and pre-samples masses."""
        likelihood = MockMassRadiusLikelihood(
            psr_name="PSR0",
            mean_mass=1.4,
            mean_radius=12.0,
            std_mass=0.1,
            std_radius=0.5,
            correlation=0.1,
            N_masses_evaluation=50,
            seed=0,
        )

        assert likelihood.psr_name == "PSR0"
        assert likelihood.fixed_mass_samples.shape == (50,)
        # Pre-sampled masses should be centred near mean_mass
        assert 1.0 < float(jnp.mean(likelihood.fixed_mass_samples)) < 1.8

    def test_invalid_correlation_raises(self):
        """Test that a correlation outside (-1, 1) raises ValueError."""
        with pytest.raises(ValueError, match="must be strictly between -1 and 1"):
            MockMassRadiusLikelihood(
                psr_name="PSR0",
                mean_mass=1.4,
                mean_radius=12.0,
                std_mass=0.1,
                std_radius=0.5,
                correlation=1.0,
            )

    def test_evaluate_is_finite(self):
        """Test that evaluate() returns a finite log likelihood for a realistic M-R curve."""
        likelihood = MockMassRadiusLikelihood(
            psr_name="PSR0",
            mean_mass=1.4,
            mean_radius=12.0,
            std_mass=0.1,
            std_radius=0.5,
            correlation=0.1,
            N_masses_evaluation=50,
            seed=0,
        )

        masses_eos = jnp.linspace(1.0, 2.2, 100)
        radii_eos = jnp.linspace(13.0, 11.0, 100)
        params = {"masses_EOS": masses_eos, "radii_EOS": radii_eos}

        result = likelihood.evaluate(params)

        assert jnp.isfinite(result)

    def test_evaluate_sensitivity(self):
        """Test that a worse-fitting M-R curve gives a lower log likelihood."""
        likelihood = MockMassRadiusLikelihood(
            psr_name="PSR0",
            mean_mass=1.4,
            mean_radius=12.0,
            std_mass=0.1,
            std_radius=0.5,
            correlation=0.0,
            N_masses_evaluation=50,
            seed=0,
        )

        masses_eos = jnp.linspace(1.0, 2.2, 100)
        good_radii = jnp.linspace(13.0, 11.0, 100)  # passes through ~12 km at 1.4 Msun
        bad_radii = good_radii + 5.0  # shifted far away from the mock observation

        log_prob_good = likelihood.evaluate(
            {"masses_EOS": masses_eos, "radii_EOS": good_radii}
        )
        log_prob_bad = likelihood.evaluate(
            {"masses_EOS": masses_eos, "radii_EOS": bad_radii}
        )

        assert log_prob_bad < log_prob_good

    def test_evaluate_applies_penalty_beyond_mtov(self):
        """Test that pre-sampled masses above M_TOV incur the configured penalty."""
        # Mean mass placed well above the EOS maximum mass so (almost) all
        # pre-sampled masses exceed M_TOV.
        masses_eos = jnp.linspace(1.0, 2.0, 100)
        radii_eos = jnp.linspace(13.0, 11.0, 100)
        params = {"masses_EOS": masses_eos, "radii_EOS": radii_eos}

        likelihood_no_penalty = MockMassRadiusLikelihood(
            psr_name="PSR0",
            mean_mass=3.0,
            mean_radius=11.0,
            std_mass=0.05,
            std_radius=0.5,
            correlation=0.0,
            penalty_value=0.0,
            N_masses_evaluation=50,
            seed=0,
        )
        likelihood_with_penalty = MockMassRadiusLikelihood(
            psr_name="PSR0",
            mean_mass=3.0,
            mean_radius=11.0,
            std_mass=0.05,
            std_radius=0.5,
            correlation=0.0,
            penalty_value=-1e4,
            N_masses_evaluation=50,
            seed=0,
        )

        log_prob_no_penalty = likelihood_no_penalty.evaluate(params)
        log_prob_with_penalty = likelihood_with_penalty.evaluate(params)

        assert log_prob_with_penalty < log_prob_no_penalty


def _save_toy_flow(
    output_dir,
    seed: int,
    nn_width: int = 8,
    nn_depth: int = 2,
    standardize: bool = False,
):
    """Build and save a tiny masked_autoregressive_flow (dim=4) to `output_dir`,
    loadable via ``Flow.from_directory`` -- a stand-in for a trained GW-event flow.
    """
    from jesterTOV.inference.flows.flow import create_flow
    from jesterTOV.inference.flows.train_flow import save_model

    flow_kwargs = {
        "seed": seed,
        "flow_type": "masked_autoregressive_flow",
        "nn_depth": nn_depth,
        "nn_block_dim": 4,
        "nn_width": nn_width,
        "flow_layers": 1,
        "invert": True,
        "cond_dim": None,
        "transformer_type": "affine",
        "transformer_knots": 4,
        "transformer_interval": 4.0,
    }
    flow = create_flow(
        key=jax.random.key(seed),
        dim=4,
        **{k: v for k, v in flow_kwargs.items() if k != "seed"},
    )
    metadata: dict = {"standardize": standardize}
    if standardize:
        # Arbitrary but distinct-per-event stats, so stacking bugs (e.g. mixing up
        # which row belongs to which event) would show up as numerical mismatches.
        metadata["data_mean"] = [1.4 + 0.1 * seed, 1.3, 300.0, 300.0]
        metadata["data_std"] = [0.2, 0.2, 100.0, 100.0]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(flow, str(output_dir), flow_kwargs, metadata)
    return output_dir


class TestStackedGWLikelihood:
    """Test StackedGWLikelihood: batched/stacked evaluation of many GW events,
    replacing one GWLikelihood per event with a single lax.map-based computation.
    See likelihoods/gw.py::StackedGWLikelihood docstring and dev/FINDINGS.md
    (Part 4) for the motivating memory/compile-time issue.
    """

    @pytest.mark.parametrize("standardize", [False, True])
    def test_matches_sum_of_individual_gw_likelihoods(self, tmp_path, standardize):
        """StackedGWLikelihood(events) must equal sum(GWLikelihood(event) for event
        in events) -- it's a batching change, not a different likelihood."""
        n_events = 3
        model_dirs = [
            _save_toy_flow(tmp_path / f"event_{i}", seed=i, standardize=standardize)
            for i in range(n_events)
        ]
        event_names = [f"event_{i}" for i in range(n_events)]

        stacked = StackedGWLikelihood(
            event_names=event_names,
            model_dirs=[str(d) for d in model_dirs],
            N_masses_evaluation=20,
            N_masses_batch_size=5,
            event_batch_size=2,
            seed=42,
        )

        individual = [
            GWLikelihood(
                event_name=name,
                model_dir=str(d),
                N_masses_evaluation=20,
                N_masses_batch_size=5,
                seed=42,
            )
            for name, d in zip(event_names, model_dirs)
        ]

        masses_eos = jnp.linspace(1.0, 2.2, 100)
        lambdas_eos = jnp.linspace(2000.0, 10.0, 100)
        params = {"masses_EOS": masses_eos, "Lambdas_EOS": lambdas_eos}

        stacked_result = stacked.evaluate(params)
        expected = sum(lik.evaluate(params) for lik in individual)

        assert jnp.isfinite(stacked_result)
        assert jnp.allclose(stacked_result, expected, rtol=1e-6, atol=1e-8)

    def test_event_batch_size_does_not_change_result(self, tmp_path):
        """Chunking the event axis differently must not change the answer, only
        how much is materialized concurrently (see dev/FINDINGS.md Part 4)."""
        n_events = 4
        model_dirs = [
            _save_toy_flow(tmp_path / f"event_{i}", seed=i) for i in range(n_events)
        ]
        event_names = [f"event_{i}" for i in range(n_events)]
        masses_eos = jnp.linspace(1.0, 2.2, 100)
        lambdas_eos = jnp.linspace(2000.0, 10.0, 100)
        params = {"masses_EOS": masses_eos, "Lambdas_EOS": lambdas_eos}

        results = {}
        for event_batch_size in [1, 2, 4]:
            likelihood = StackedGWLikelihood(
                event_names=event_names,
                model_dirs=[str(d) for d in model_dirs],
                N_masses_evaluation=20,
                N_masses_batch_size=5,
                event_batch_size=event_batch_size,
                seed=42,
            )
            results[event_batch_size] = likelihood.evaluate(params)

        assert jnp.allclose(results[1], results[2], rtol=1e-6)
        assert jnp.allclose(results[1], results[4], rtol=1e-6)

    def test_default_event_batch_size_is_one(self, tmp_path):
        """The default must be the safe (scan-over-events) choice, matching
        GWLikelihood.N_masses_batch_size's default - see dev/FINDINGS.md Part 4."""
        model_dirs = [_save_toy_flow(tmp_path / "event_0", seed=0)]
        likelihood = StackedGWLikelihood(
            event_names=["event_0"],
            model_dirs=[str(model_dirs[0])],
            N_masses_evaluation=10,
        )
        assert likelihood.event_batch_size == 1

    def test_mismatched_architecture_raises_clear_error(self, tmp_path):
        """Events with different flow architectures cannot be stacked -- must fail
        fast at construction time with a message naming the offending event."""
        model_dir_a = _save_toy_flow(tmp_path / "event_a", seed=0, nn_width=8)
        model_dir_b = _save_toy_flow(tmp_path / "event_b", seed=1, nn_width=16)

        with pytest.raises(ValueError, match="event_b"):
            StackedGWLikelihood(
                event_names=["event_a", "event_b"],
                model_dirs=[str(model_dir_a), str(model_dir_b)],
                N_masses_evaluation=10,
                N_masses_batch_size=5,
            )

    def test_penalty_applied_beyond_mtov(self, tmp_path):
        """Masses above M_TOV should incur the configured penalty, matching
        GWLikelihood's behaviour."""
        # standardize=True centres pre-sampled masses near 1.4 +/- 0.2 (see
        # _save_toy_flow) -- an untrained/unstandardized flow instead samples
        # near its base distribution N(0, I), which is nowhere near a
        # physically realistic M_TOV cutoff and would never trigger the penalty.
        model_dirs = [_save_toy_flow(tmp_path / "event_0", seed=0, standardize=True)]
        masses_eos = jnp.linspace(
            1.0, 1.4, 50
        )  # M_TOV in the middle of the flow's mass range
        lambdas_eos = jnp.linspace(2000.0, 10.0, 50)
        params = {"masses_EOS": masses_eos, "Lambdas_EOS": lambdas_eos}

        no_penalty = StackedGWLikelihood(
            event_names=["event_0"],
            model_dirs=[str(model_dirs[0])],
            penalty_value=0.0,
            N_masses_evaluation=50,
            N_masses_batch_size=10,
            seed=0,
        )
        with_penalty = StackedGWLikelihood(
            event_names=["event_0"],
            model_dirs=[str(model_dirs[0])],
            penalty_value=-1e4,
            N_masses_evaluation=50,
            N_masses_batch_size=10,
            seed=0,
        )

        assert with_penalty.evaluate(params) < no_penalty.evaluate(params)

    def test_mismatched_event_names_and_model_dirs_length_raises(self):
        with pytest.raises(ValueError, match="must have the same length"):
            StackedGWLikelihood(
                event_names=["event_0", "event_1"],
                model_dirs=["/some/dir"],
            )


class TestLikelihoodFactory:
    """Test likelihood factory functionality."""

    def test_create_zero_likelihood(self):
        """Test creating ZeroLikelihood via factory."""
        config = schema.ZeroLikelihoodConfig(enabled=True)

        likelihood = factory.create_likelihood(config)

        assert isinstance(likelihood, ZeroLikelihood)

    def test_create_constraint_eos_likelihood(self):
        """Test creating ConstraintEOSLikelihood via factory."""
        config = schema.EOSConstraintsLikelihoodConfig(
            enabled=True,
            penalty_causality=-1e10,
            penalty_stability=-1e5,
        )

        likelihood = factory.create_likelihood(config)

        assert isinstance(likelihood, ConstraintEOSLikelihood)
        assert likelihood.penalty_causality == -1e10
        assert likelihood.penalty_stability == -1e5

    def test_create_constraint_tov_likelihood(self):
        """Test creating ConstraintTOVLikelihood via factory."""
        config = schema.TOVConstraintsLikelihoodConfig(
            enabled=True,
            penalty_tov=-1e10,
        )

        likelihood = factory.create_likelihood(config)

        assert isinstance(likelihood, ConstraintTOVLikelihood)
        assert likelihood.penalty_tov == -1e10

    def test_create_constraint_esym_likelihood(self):
        """Test creating ConstraintEsymLikelihood via factory."""
        config = schema.EsymConstraintsLikelihoodConfig(
            enabled=True,
            penalty_esym=-1e10,
        )
        likelihood = factory.create_likelihood(config)
        assert isinstance(likelihood, ConstraintEsymLikelihood)
        assert likelihood.penalty_esym == -1e10

    def test_create_constraint_gamma_likelihood(self):
        """Test creating ConstraintGammaLikelihood via factory."""
        config = schema.GammaConstraintsLikelihoodConfig(
            enabled=True,
            penalty_gamma=-1e10,
        )

        likelihood = factory.create_likelihood(config)

        assert isinstance(likelihood, ConstraintGammaLikelihood)
        assert likelihood.penalty_gamma == -1e10

    def test_create_chieft_likelihood(self):
        """Test creating ChiEFTLikelihood via factory."""
        config = schema.ChiEFTLikelihoodConfig(
            enabled=True,
            nb_n=100,
        )

        try:
            likelihood = factory.create_likelihood(config)
            assert isinstance(likelihood, ChiEFTLikelihood)
            assert likelihood.nb_n == 100
        except FileNotFoundError:
            pytest.skip("ChiEFT data files not found")

    def test_create_disabled_likelihood_returns_none(self):
        """Test that factory returns None for disabled likelihoods."""
        config = schema.ZeroLikelihoodConfig(enabled=False)

        likelihood = factory.create_likelihood(config)
        assert likelihood is None

    def test_create_gw_likelihood_via_factory_raises_error(self):
        """Test that GW likelihoods must be created via create_combined_likelihood."""
        config = schema.GWLikelihoodConfig(
            enabled=True,
            events=[{"name": "GW170817", "nf_model_dir": "/path/to/data"}],  # type: ignore[arg-type]
        )

        with pytest.raises(
            RuntimeError, match="should be created via create_combined_likelihood"
        ):
            factory.create_likelihood(config)

    def test_create_nicer_likelihood_via_factory_raises_error(self):
        """Test that NICER likelihoods must be created via create_combined_likelihood."""
        config = schema.NICERLikelihoodConfig(
            enabled=True,
            pulsars=[
                {
                    "name": "J0030",
                    "amsterdam_samples_file": "/path/to/amsterdam.txt",
                    "maryland_samples_file": "/path/to/maryland.txt",
                }
            ],
        )

        with pytest.raises(
            RuntimeError, match="should be created via create_combined_likelihood"
        ):
            factory.create_likelihood(config)

    def test_create_mock_mr_likelihood_via_factory_raises_error(self):
        """Test that mock M-R likelihoods must be created via create_combined_likelihood."""
        config = schema.MockMassRadiusLikelihoodConfig(
            enabled=True,
            json_file="/path/to/mock_observations.json",
        )

        with pytest.raises(
            RuntimeError, match="should be created via create_combined_likelihood"
        ):
            factory.create_likelihood(config)

    def test_invalid_likelihood_type_raises_error(self):
        """Test that invalid likelihood type raises ValidationError.

        NOTE: Pydantic catches this during config creation, not in factory.
        This is the correct behavior - validation happens at config time.
        """
        from pydantic import ValidationError, TypeAdapter

        # Test using TypeAdapter since LikelihoodConfig is now a Union
        adapter = TypeAdapter(schema.LikelihoodConfig)

        with pytest.raises(ValidationError, match="Input tag"):
            adapter.validate_python(
                {
                    "type": "invalid_type",
                    "enabled": True,
                }
            )


class TestCombinedLikelihoodFactory:
    """Test create_combined_likelihood factory function."""

    def test_create_gw_likelihood_builds_stacked_gw_likelihood(self):
        """GWLikelihoodConfig must go through StackedGWLikelihood (not one
        GWLikelihood per event) - see likelihoods/gw.py::StackedGWLikelihood and
        dev/FINDINGS.md Part 4. Uses the real shipped GW170817/GW190425 presets,
        which share architecture (verified in dev/FINDINGS.md Part 4), so they
        must stack without error and match summing individual GWLikelihoods.
        """
        config = schema.GWLikelihoodConfig(
            events=[
                schema.GWEventConfig(name="GW170817"),
                schema.GWEventConfig(name="GW190425"),
            ],
            N_masses_evaluation=50,
        )

        likelihood = factory.create_combined_likelihood([config])
        assert isinstance(likelihood, StackedGWLikelihood)
        assert likelihood.event_batch_size == 1  # shipped default

        masses_eos = jnp.linspace(0.5, 2.3, 100)
        lambdas_eos = 3000.0 * (masses_eos / 0.5) ** (-6.0) + 1.0
        params = {"masses_EOS": masses_eos, "Lambdas_EOS": lambdas_eos}

        from jesterTOV.inference.likelihoods.factory import get_gw_model_dir

        individual = [
            GWLikelihood(
                event_name=event.name,
                model_dir=get_gw_model_dir(event),
                N_masses_evaluation=50,
            )
            for event in config.events
        ]
        expected = sum(lik.evaluate(params) for lik in individual)

        assert jnp.allclose(likelihood.evaluate(params), expected, rtol=1e-6)

    def test_create_combined_likelihood_single(self):
        """Test that single enabled likelihood is returned directly (not wrapped)."""
        configs = [schema.ZeroLikelihoodConfig(enabled=True)]

        likelihood = factory.create_combined_likelihood(configs)

        # Single likelihood should be returned directly
        assert isinstance(likelihood, ZeroLikelihood)

    def test_create_combined_likelihood_multiple(self):
        """Test that multiple likelihoods are combined."""
        configs = [
            schema.ZeroLikelihoodConfig(enabled=True),
            schema.EOSConstraintsLikelihoodConfig(enabled=True),
        ]

        likelihood = factory.create_combined_likelihood(configs)

        # Multiple likelihoods should be wrapped in CombinedLikelihood
        assert isinstance(likelihood, CombinedLikelihood)
        assert len(likelihood.likelihoods_list) == 2

    def test_create_combined_likelihood_with_disabled(self):
        """Test that disabled likelihoods are skipped."""
        configs = [
            schema.ZeroLikelihoodConfig(enabled=True),
            schema.ZeroLikelihoodConfig(enabled=False),
        ]

        likelihood = factory.create_combined_likelihood(configs)

        # Only one enabled - should return directly
        assert isinstance(likelihood, ZeroLikelihood)

    def test_create_combined_likelihood_all_disabled_raises_error(self):
        """Test that all disabled likelihoods raises ValueError."""
        configs = [
            schema.ZeroLikelihoodConfig(enabled=False),
            schema.ZeroLikelihoodConfig(enabled=False),
        ]

        with pytest.raises(ValueError, match="No likelihoods enabled"):
            factory.create_combined_likelihood(configs)

    def test_create_combined_likelihood_with_radio_timing(self):
        """Test creating combined likelihood with radio timing constraint."""
        configs = [
            schema.RadioLikelihoodConfig(
                enabled=True,
                pulsars=[
                    {
                        "name": "J0348+0432",
                        "mass_mean": 2.01,
                        "mass_std": 0.04,
                    },
                ],
            ),
        ]

        likelihood = factory.create_combined_likelihood(configs)

        # Single radio likelihood should be returned directly
        assert isinstance(likelihood, RadioTimingLikelihood)

    def test_create_combined_likelihood_with_single_mock_mr_observation(self, tmp_path):
        """Test creating combined likelihood with a single mock M-R observation."""
        json_file = tmp_path / "mock_observations.json"
        json_file.write_text(
            json.dumps(
                [
                    {
                        "name": "PSR0",
                        "mean_mass": 1.4,
                        "mean_radius": 12.0,
                        "std_mass": 0.1,
                        "std_radius": 0.1,
                        "correlation": 0.1,
                    }
                ]
            )
        )
        configs = [
            schema.MockMassRadiusLikelihoodConfig(
                enabled=True,
                json_file=str(json_file),
            ),
        ]

        likelihood = factory.create_combined_likelihood(configs)

        # Single mock M-R likelihood should be returned directly
        assert isinstance(likelihood, MockMassRadiusLikelihood)
        assert likelihood.psr_name == "PSR0"

    def test_create_combined_likelihood_with_multiple_mock_mr_observations(
        self, tmp_path
    ):
        """Test that one likelihood is created per entry in the mock observations JSON file."""
        json_file = tmp_path / "mock_observations.json"
        json_file.write_text(
            json.dumps(
                [
                    {
                        "name": "PSR0",
                        "mean_mass": 1.4,
                        "mean_radius": 12.0,
                        "std_mass": 0.1,
                        "std_radius": 0.1,
                        "correlation": 0.1,
                    },
                    {
                        "name": "PSR1",
                        "mean_mass": 2.0,
                        "mean_radius": 11.5,
                        "std_mass": 0.1,
                        "std_radius": 0.1,
                        "correlation": -0.2,
                    },
                ]
            )
        )
        configs = [
            schema.MockMassRadiusLikelihoodConfig(
                enabled=True,
                json_file=str(json_file),
            ),
        ]

        likelihood = factory.create_combined_likelihood(configs)

        assert isinstance(likelihood, CombinedLikelihood)
        assert len(likelihood.likelihoods_list) == 2
        assert {lik.psr_name for lik in likelihood.likelihoods_list} == {
            "PSR0",
            "PSR1",
        }


class TestGWEventPresets:
    """Test GW event preset path functionality."""

    def test_get_gw_model_dir_gw170817_preset(self):
        """Test that GW170817 uses preset path when nf_model_dir not provided."""
        result = factory.get_gw_model_dir(schema.GWEventConfig(name="GW170817"))

        # Should contain the expected path components
        assert "gw170817_gwtc1_lowspin" in result
        assert result.endswith("gw170817_gwtc1_lowspin")

    def test_get_gw_model_dir_gw190425_preset(self):
        """Test that GW190425 uses preset path when nf_model_dir not provided."""
        result = factory.get_gw_model_dir(schema.GWEventConfig(name="GW190425"))

        # Should contain the expected path components
        assert "gw190425_phenompnrt_ls" in result
        assert result.endswith("gw190425_phenompnrt_ls")

    def test_get_gw_model_dir_case_insensitive(self):
        """Test that event name matching is case-insensitive."""
        # Lowercase should work
        result_lower = factory.get_gw_model_dir(schema.GWEventConfig(name="gw170817"))
        assert "gw170817_gwtc1_lowspin" in result_lower

        # Mixed case should work
        result_mixed = factory.get_gw_model_dir(schema.GWEventConfig(name="Gw170817"))
        assert "gw170817_gwtc1_lowspin" in result_mixed

        # Uppercase should work
        result_upper = factory.get_gw_model_dir(schema.GWEventConfig(name="GW170817"))
        assert "gw170817_gwtc1_lowspin" in result_upper

    def test_get_gw_model_dir_custom_path(self):
        """Test that custom nf_model_dir takes precedence over preset."""
        custom_path = "/custom/path/to/model"
        result = factory.get_gw_model_dir(
            schema.GWEventConfig(name="GW170817", nf_model_dir=custom_path)
        )

        # Should use the custom path, not preset
        assert "custom/path/to/model" in result
        assert "gw170817_gwtc1_lowspin" not in result

    def test_get_gw_model_dir_no_nf_model_dir_uses_preset(self):
        """Test that omitting nf_model_dir triggers preset lookup."""
        result = factory.get_gw_model_dir(schema.GWEventConfig(name="GW170817"))

        # No nf_model_dir should trigger preset
        assert "gw170817_gwtc1_lowspin" in result

    def test_get_gw_model_dir_unknown_event_raises_error(self):
        """Test that unknown event without nf_model_dir raises ValueError."""
        with pytest.raises(ValueError, match="not in presets"):
            factory.get_gw_model_dir(schema.GWEventConfig(name="GW999999"))

        # Error message should list available presets
        with pytest.raises(ValueError, match="GW170817.*GW190425"):
            factory.get_gw_model_dir(schema.GWEventConfig(name="GW999999"))

    def test_get_gw_model_dir_returns_absolute_path(self):
        """Test that preset paths are resolved to absolute paths."""
        result = factory.get_gw_model_dir(schema.GWEventConfig(name="GW170817"))

        # Should be an absolute path
        from pathlib import Path

        assert Path(result).is_absolute()

    def test_gw_preset_paths_exist(self):
        """Test that preset flow model directories actually exist."""
        from pathlib import Path

        for event_name in ["GW170817", "GW190425"]:
            model_dir = factory.get_gw_model_dir(schema.GWEventConfig(name=event_name))
            model_path = Path(model_dir)

            # Directory should exist
            assert model_path.exists(), f"Preset path does not exist: {model_dir}"
            assert model_path.is_dir(), f"Preset path is not a directory: {model_dir}"

            # Required flow files should exist
            required_files = ["flow_weights.eqx", "flow_kwargs.json", "metadata.json"]
            for fname in required_files:
                file_path = model_path / fname
                assert file_path.exists(), f"Missing required file: {file_path}"

    def test_load_flow_from_preset_gw170817(self):
        """Test that Flow can actually be loaded from GW170817 preset path."""
        from jesterTOV.inference.flows.flow import Flow

        model_dir = factory.get_gw_model_dir(schema.GWEventConfig(name="GW170817"))

        # Should load without errors
        flow = Flow.from_directory(model_dir)

        # Verify flow is properly initialized
        assert flow is not None
        assert hasattr(flow, "flow")
        assert hasattr(flow, "metadata")
        assert hasattr(flow, "standardize")

    def test_load_flow_from_preset_gw190425(self):
        """Test that Flow can actually be loaded from GW190425 preset path."""
        from jesterTOV.inference.flows.flow import Flow

        model_dir = factory.get_gw_model_dir(schema.GWEventConfig(name="GW190425"))

        # Should load without errors
        flow = Flow.from_directory(model_dir)

        # Verify flow is properly initialized
        assert flow is not None
        assert hasattr(flow, "flow")
        assert hasattr(flow, "metadata")
        assert hasattr(flow, "standardize")


class TestLikelihoodIntegration:
    """Integration tests for likelihood system."""

    def test_likelihood_base_interface(self):
        """Test that all likelihoods implement LikelihoodBase interface."""
        # Create a few likelihoods and verify they have evaluate method
        likelihoods = [
            ZeroLikelihood(),
            ConstraintEOSLikelihood(),
            ConstraintTOVLikelihood(),
            RadioTimingLikelihood("J0348+0432", 2.01, 0.04, 100),
        ]

        for likelihood in likelihoods:
            assert isinstance(likelihood, LikelihoodBase)
            assert hasattr(likelihood, "evaluate")
            assert callable(likelihood.evaluate)

    def test_likelihood_chaining(self):
        """Test that likelihoods can be chained via CombinedLikelihood."""
        l1 = ZeroLikelihood()
        l2 = ConstraintEOSLikelihood()
        l3 = ConstraintTOVLikelihood()

        combined = CombinedLikelihood([l1, l2, l3])

        # All valid params should give 0.0
        params = {
            "n_tov_failures": 0.0,
            "n_causality_violations": 0.0,
            "n_stability_violations": 0.0,
            "n_pressure_violations": 0.0,
        }

        result = combined.evaluate(params)
        assert result == 0.0

    def test_likelihood_with_violations_propagates(self):
        """Test that constraint violations propagate through CombinedLikelihood."""
        eos_constraint = ConstraintEOSLikelihood(penalty_causality=-1e10)
        tov_constraint = ConstraintTOVLikelihood(penalty_tov=-1e10)

        combined = CombinedLikelihood([eos_constraint, tov_constraint])

        # Both violated
        params = {
            "n_causality_violations": 1.0,
            "n_tov_failures": 1.0,
            "n_stability_violations": 0.0,
            "n_pressure_violations": 0.0,
        }

        result = combined.evaluate(params)
        # Should sum both penalties
        assert result == -2e10


def _make_mock_flow() -> MagicMock:
    """Create a mock Flow with realistic sample/log_prob behaviour."""
    flow = MagicMock()
    # sample returns (N, 2) array of (mass, radius) pairs
    flow.sample.side_effect = lambda k, shape: jax.random.uniform(
        k,
        shape=(*shape, 2),
        minval=jnp.array([1.0, 10.0]),
        maxval=jnp.array([2.5, 14.0]),
    )
    # log_prob returns a scalar
    flow.log_prob.return_value = jnp.array(-2.5)
    return flow


class TestNICERLikelihoodGroups:
    """Tests that NICERLikelihood handles one or both analysis groups correctly."""

    def _make_params(self) -> dict:
        masses = jnp.linspace(1.0, 2.5, 50)
        radii = jnp.linspace(13.0, 11.0, 50)
        return {"masses_EOS": masses, "radii_EOS": radii}

    def test_raises_when_neither_group_provided(self):
        """NICERLikelihood must raise if both model dirs are None."""
        from jesterTOV.inference.likelihoods.nicer import NICERLikelihood

        with pytest.raises(ValueError, match="At least one"):
            NICERLikelihood("J0437", amsterdam_model_dir=None, maryland_model_dir=None)

    def test_amsterdam_only_initialization(self):
        """Amsterdam-only: flow is loaded and masses are pre-sampled."""
        from jesterTOV.inference.likelihoods.nicer import NICERLikelihood

        mock_flow = _make_mock_flow()

        with patch(
            "jesterTOV.inference.flows.flow.Flow.from_directory", return_value=mock_flow
        ):
            likelihood = NICERLikelihood(
                "J0437",
                amsterdam_model_dir="/fake/amsterdam",
                maryland_model_dir=None,
                N_masses_evaluation=10,
                seed=42,
            )

        assert likelihood.amsterdam_flow is mock_flow
        assert likelihood.maryland_flow is None
        assert likelihood.amsterdam_fixed_mass_samples is not None
        assert likelihood.maryland_fixed_mass_samples is None
        assert likelihood.amsterdam_fixed_mass_samples.shape == (10,)

    def test_maryland_only_initialization(self):
        """Maryland-only: flow is loaded and masses are pre-sampled."""
        from jesterTOV.inference.likelihoods.nicer import NICERLikelihood

        mock_flow = _make_mock_flow()

        with patch(
            "jesterTOV.inference.flows.flow.Flow.from_directory", return_value=mock_flow
        ):
            likelihood = NICERLikelihood(
                "J0614",
                amsterdam_model_dir=None,
                maryland_model_dir="/fake/maryland",
                N_masses_evaluation=10,
                seed=42,
            )

        assert likelihood.amsterdam_flow is None
        assert likelihood.maryland_flow is mock_flow
        assert likelihood.amsterdam_fixed_mass_samples is None
        assert likelihood.maryland_fixed_mass_samples is not None
        assert likelihood.maryland_fixed_mass_samples.shape == (10,)

    def test_both_groups_initialization(self):
        """Both groups: both flows are loaded and masses pre-sampled."""
        from jesterTOV.inference.likelihoods.nicer import NICERLikelihood

        mock_amsterdam = _make_mock_flow()
        mock_maryland = _make_mock_flow()

        with patch(
            "jesterTOV.inference.flows.flow.Flow.from_directory",
            side_effect=[mock_amsterdam, mock_maryland],
        ):
            likelihood = NICERLikelihood(
                "J0030",
                amsterdam_model_dir="/fake/amsterdam",
                maryland_model_dir="/fake/maryland",
                N_masses_evaluation=10,
                seed=42,
            )

        assert likelihood.amsterdam_flow is mock_amsterdam
        assert likelihood.maryland_flow is mock_maryland
        assert likelihood.amsterdam_fixed_mass_samples is not None
        assert likelihood.maryland_fixed_mass_samples is not None

    def test_amsterdam_only_evaluate_returns_finite(self):
        """Amsterdam-only evaluate returns a finite log-likelihood."""
        from jesterTOV.inference.likelihoods.nicer import NICERLikelihood

        mock_flow = _make_mock_flow()

        with patch(
            "jesterTOV.inference.flows.flow.Flow.from_directory", return_value=mock_flow
        ):
            likelihood = NICERLikelihood(
                "J0437",
                amsterdam_model_dir="/fake/amsterdam",
                maryland_model_dir=None,
                N_masses_evaluation=10,
                N_masses_batch_size=5,
                seed=42,
            )

        result = likelihood.evaluate(self._make_params())

        assert jnp.isfinite(result), f"Expected finite log-likelihood, got {result}"

    def test_maryland_only_evaluate_returns_finite(self):
        """Maryland-only evaluate returns a finite log-likelihood."""
        from jesterTOV.inference.likelihoods.nicer import NICERLikelihood

        mock_flow = _make_mock_flow()

        with patch(
            "jesterTOV.inference.flows.flow.Flow.from_directory", return_value=mock_flow
        ):
            likelihood = NICERLikelihood(
                "J0614",
                amsterdam_model_dir=None,
                maryland_model_dir="/fake/maryland",
                N_masses_evaluation=10,
                N_masses_batch_size=5,
                seed=42,
            )

        result = likelihood.evaluate(self._make_params())

        assert jnp.isfinite(result), f"Expected finite log-likelihood, got {result}"

    def test_single_group_equals_that_group_logL(self):
        """Single-group result equals the per-group log-mean (no averaging dilution)."""
        from jesterTOV.inference.likelihoods.nicer import NICERLikelihood

        mock_amsterdam = _make_mock_flow()
        mock_maryland = _make_mock_flow()

        with patch(
            "jesterTOV.inference.flows.flow.Flow.from_directory",
            side_effect=[mock_amsterdam],
        ):
            lhood_amsterdam_only = NICERLikelihood(
                "J0030",
                amsterdam_model_dir="/fake/amsterdam",
                maryland_model_dir=None,
                N_masses_evaluation=10,
                N_masses_batch_size=5,
                seed=42,
            )

        with patch(
            "jesterTOV.inference.flows.flow.Flow.from_directory",
            side_effect=[mock_amsterdam, mock_maryland],
        ):
            lhood_both = NICERLikelihood(
                "J0030",
                amsterdam_model_dir="/fake/amsterdam",
                maryland_model_dir="/fake/maryland",
                N_masses_evaluation=10,
                N_masses_batch_size=5,
                seed=42,
            )

        params = self._make_params()
        result_single = lhood_amsterdam_only.evaluate(params)
        result_both = lhood_both.evaluate(params)

        # With a constant mock log_prob=-2.5, single-group result == both-groups result
        # because logsumexp([x, x]) - log(2) == x
        assert jnp.isfinite(result_single)
        assert jnp.isfinite(result_both)
        assert jnp.allclose(result_single, result_both, atol=1e-5), (
            f"Single-group ({result_single:.4f}) != both-groups ({result_both:.4f}) "
            "for constant mock log_prob"
        )
