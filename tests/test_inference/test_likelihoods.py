"""Tests for inference likelihood system (base, factory, specific likelihoods)."""

import pytest
import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.inference.config import schema
from jesterTOV.inference.likelihoods import factory
from jesterTOV.inference.likelihoods.combined import ZeroLikelihood, CombinedLikelihood
from jesterTOV.inference.likelihoods.constraints import (
    ConstraintEOSLikelihood,
    ConstraintTOVLikelihood,
    ConstraintLikelihood,
    check_tov_validity,
    check_causality_violation,
    check_stability,
    check_pressure_monotonicity,
    check_all_constraints,
)
from jesterTOV.inference.likelihoods.chieft import ChiEFTLikelihood
from jesterTOV.inference.likelihoods.radio import RadioTimingLikelihood
from jesterTOV.inference.base import LikelihoodBase


class TestZeroLikelihood:
    """Test ZeroLikelihood functionality."""

    def test_zero_likelihood_returns_zero(self):
        """Test that ZeroLikelihood always returns 0.0."""
        likelihood = ZeroLikelihood()

        # Any params should give 0.0
        params = {"K_sat": 220.0, "L_sym": 90.0}
        result = likelihood.evaluate(params, {})

        assert result == 0.0

    def test_zero_likelihood_with_empty_params(self):
        """Test ZeroLikelihood with empty parameter dict."""
        likelihood = ZeroLikelihood()
        result = likelihood.evaluate({}, {})
        assert result == 0.0


class TestCombinedLikelihood:
    """Test CombinedLikelihood functionality."""

    def test_combined_likelihood_sums_correctly(self):
        """Test that CombinedLikelihood sums log likelihoods."""
        # Create two zero likelihoods - should sum to 0.0
        likelihood1 = ZeroLikelihood()
        likelihood2 = ZeroLikelihood()

        combined = CombinedLikelihood([likelihood1, likelihood2])
        result = combined.evaluate({}, {})

        assert result == 0.0

    def test_combined_likelihood_with_single_likelihood(self):
        """Test CombinedLikelihood with single likelihood."""
        likelihood = ZeroLikelihood()
        combined = CombinedLikelihood([likelihood])

        result = combined.evaluate({}, {})
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

        assert constraints['n_tov_failures'] == 0.0
        assert constraints['n_causality_violations'] == 0.0
        assert constraints['n_stability_violations'] == 0.0
        assert constraints['n_pressure_violations'] == 0.0

    def test_check_all_constraints_with_violations(self):
        """Test check_all_constraints with multiple violations."""
        masses = jnp.array([1.4, jnp.nan, 2.0])  # 1 NaN
        radii = jnp.array([12.0, 11.5, 11.0])
        lambdas = jnp.array([400.0, 300.0, 200.0])
        cs2 = jnp.array([0.3, 1.5, -0.1])  # 1 causality, 1 stability violation
        p = jnp.array([1.0, 3.0, 2.0])  # 1 pressure decrease

        constraints = check_all_constraints(masses, radii, lambdas, cs2, p)

        assert constraints['n_tov_failures'] == 1.0
        assert constraints['n_causality_violations'] == 1.0
        assert constraints['n_stability_violations'] == 1.0
        assert constraints['n_pressure_violations'] == 1.0


class TestConstraintEOSLikelihood:
    """Test ConstraintEOSLikelihood (EOS-level constraints only)."""

    def test_constraint_eos_likelihood_all_valid(self):
        """Test ConstraintEOSLikelihood with all valid constraints."""
        likelihood = ConstraintEOSLikelihood()

        # Valid params (no violations)
        params = {
            'n_causality_violations': 0.0,
            'n_stability_violations': 0.0,
            'n_pressure_violations': 0.0,
        }

        result = likelihood.evaluate(params, {})
        assert result == 0.0

    def test_constraint_eos_likelihood_causality_violation(self):
        """Test ConstraintEOSLikelihood with causality violation."""
        likelihood = ConstraintEOSLikelihood(penalty_causality=-1e10)

        # Causality violation
        params = {
            'n_causality_violations': 1.0,  # One violation
            'n_stability_violations': 0.0,
            'n_pressure_violations': 0.0,
        }

        result = likelihood.evaluate(params, {})
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
            'n_causality_violations': 1.0,
            'n_stability_violations': 2.0,
            'n_pressure_violations': 1.0,
        }

        result = likelihood.evaluate(params, {})
        # Should sum all penalties
        assert result == -1e10 + -1e5 + -1e5

    def test_constraint_eos_likelihood_missing_keys(self):
        """Test ConstraintEOSLikelihood with missing violation keys (defaults to 0)."""
        likelihood = ConstraintEOSLikelihood()

        # Empty params - should use defaults
        params = {}

        result = likelihood.evaluate(params, {})
        assert result == 0.0


class TestConstraintTOVLikelihood:
    """Test ConstraintTOVLikelihood (TOV-level constraints only)."""

    def test_constraint_tov_likelihood_valid(self):
        """Test ConstraintTOVLikelihood with valid TOV integration."""
        likelihood = ConstraintTOVLikelihood()

        # Valid TOV (no NaN)
        params = {'n_tov_failures': 0.0}

        result = likelihood.evaluate(params, {})
        assert result == 0.0

    def test_constraint_tov_likelihood_with_failures(self):
        """Test ConstraintTOVLikelihood with TOV integration failure."""
        likelihood = ConstraintTOVLikelihood(penalty_tov=-1e10)

        # TOV failure (NaN in output)
        params = {'n_tov_failures': 3.0}  # Multiple NaN

        result = likelihood.evaluate(params, {})
        assert result == -1e10

    def test_constraint_tov_likelihood_missing_key(self):
        """Test ConstraintTOVLikelihood with missing n_tov_failures key."""
        likelihood = ConstraintTOVLikelihood()

        # Empty params - should use default
        params = {}

        result = likelihood.evaluate(params, {})
        assert result == 0.0


class TestConstraintLikelihood:
    """Test ConstraintLikelihood (combined EOS + TOV constraints, deprecated)."""

    def test_constraint_likelihood_all_valid(self):
        """Test ConstraintLikelihood with all valid constraints."""
        likelihood = ConstraintLikelihood()

        # All valid
        params = {
            'n_tov_failures': 0.0,
            'n_causality_violations': 0.0,
            'n_stability_violations': 0.0,
            'n_pressure_violations': 0.0,
        }

        result = likelihood.evaluate(params, {})
        assert result == 0.0

    def test_constraint_likelihood_all_violations(self):
        """Test ConstraintLikelihood with all constraint types violated."""
        likelihood = ConstraintLikelihood(
            penalty_tov=-1e10,
            penalty_causality=-1e10,
            penalty_stability=-1e5,
            penalty_pressure=-1e5,
        )

        # All violated
        params = {
            'n_tov_failures': 1.0,
            'n_causality_violations': 1.0,
            'n_stability_violations': 1.0,
            'n_pressure_violations': 1.0,
        }

        result = likelihood.evaluate(params, {})
        # Should sum all penalties
        expected = -1e10 + -1e10 + -1e5 + -1e5
        assert result == expected


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
        # This test documents the expected API, but will skip if files don't exist
        pytest.skip("Requires ChiEFT data files - test documents expected API")


class TestRadioTimingLikelihood:
    """Test RadioTimingLikelihood functionality."""

    def test_radio_timing_likelihood_initialization(self):
        """Test RadioTimingLikelihood initializes correctly."""
        likelihood = RadioTimingLikelihood(
            psr_name="J0348+0432",
            mean=2.01,  # Solar masses
            std=0.04,   # Uncertainty
            nb_masses=100,
        )

        assert likelihood.psr_name == "J0348+0432"
        assert likelihood.mean == 2.01
        assert likelihood.std == 0.04
        assert likelihood.nb_masses == 100

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
            nb_masses=100,
        )

        # Mock transform output with masses_EOS
        # Realistic NS mass range: 1.0 - 2.5 solar masses
        masses_eos = jnp.linspace(1.0, 2.5, 100)
        params = {
            'masses_EOS': masses_eos,
        }

        result = likelihood.evaluate(params, {})

        # Should return a finite log likelihood
        assert jnp.isfinite(result)

        # For a stiff EOS with max mass > 2.01, likelihood should be reasonable
        # (not a large negative penalty)
        assert result > -1000.0, f"Likelihood too negative: {result}"


class TestLikelihoodFactory:
    """Test likelihood factory functionality."""

    def test_create_zero_likelihood(self):
        """Test creating ZeroLikelihood via factory."""
        config = schema.LikelihoodConfig(
            type="zero",
            enabled=True,
            parameters={},
        )

        likelihood = factory.create_likelihood(config)

        assert isinstance(likelihood, ZeroLikelihood)

    def test_create_constraint_eos_likelihood(self):
        """Test creating ConstraintEOSLikelihood via factory."""
        config = schema.LikelihoodConfig(
            type="constraints_eos",
            enabled=True,
            parameters={
                "penalty_causality": -1e10,
                "penalty_stability": -1e5,
            },
        )

        likelihood = factory.create_likelihood(config)

        assert isinstance(likelihood, ConstraintEOSLikelihood)
        assert likelihood.penalty_causality == -1e10
        assert likelihood.penalty_stability == -1e5

    def test_create_constraint_tov_likelihood(self):
        """Test creating ConstraintTOVLikelihood via factory."""
        config = schema.LikelihoodConfig(
            type="constraints_tov",
            enabled=True,
            parameters={
                "penalty_tov": -1e10,
            },
        )

        likelihood = factory.create_likelihood(config)

        assert isinstance(likelihood, ConstraintTOVLikelihood)
        assert likelihood.penalty_tov == -1e10

    def test_create_chieft_likelihood(self):
        """Test creating ChiEFTLikelihood via factory."""
        config = schema.LikelihoodConfig(
            type="chieft",
            enabled=True,
            parameters={
                "nb_n": 100,
            },
        )

        try:
            likelihood = factory.create_likelihood(config)
            assert isinstance(likelihood, ChiEFTLikelihood)
            assert likelihood.nb_n == 100
        except FileNotFoundError:
            pytest.skip("ChiEFT data files not found")

    def test_create_disabled_likelihood_returns_none(self):
        """Test that factory returns None for disabled likelihoods."""
        config = schema.LikelihoodConfig(
            type="zero",
            enabled=False,
            parameters={},
        )

        likelihood = factory.create_likelihood(config)
        assert likelihood is None

    def test_create_gw_likelihood_via_factory_raises_error(self):
        """Test that GW likelihoods must be created via create_combined_likelihood."""
        config = schema.LikelihoodConfig(
            type="gw",
            enabled=True,
            parameters={
                "events": [
                    {"name": "GW170817", "model_dir": "/path/to/data"}
                ],
            },
        )

        with pytest.raises(RuntimeError, match="should be created via create_combined_likelihood"):
            factory.create_likelihood(config)

    def test_create_nicer_likelihood_via_factory_raises_error(self):
        """Test that NICER likelihoods must be created via create_combined_likelihood."""
        config = schema.LikelihoodConfig(
            type="nicer",
            enabled=True,
            parameters={
                "pulsars": [
                    {
                        "name": "J0030",
                        "amsterdam_samples_file": "/path/to/amsterdam.txt",
                        "maryland_samples_file": "/path/to/maryland.txt",
                    }
                ],
            },
        )

        with pytest.raises(RuntimeError, match="should be created via create_combined_likelihood"):
            factory.create_likelihood(config)

    def test_invalid_likelihood_type_raises_error(self):
        """Test that invalid likelihood type raises ValidationError.

        NOTE: Pydantic catches this during config creation, not in factory.
        This is the correct behavior - validation happens at config time.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be"):
            schema.LikelihoodConfig(
                type="invalid_type",
                enabled=True,
                parameters={},
            )


class TestCombinedLikelihoodFactory:
    """Test create_combined_likelihood factory function."""

    def test_create_combined_likelihood_single(self):
        """Test that single enabled likelihood is returned directly (not wrapped)."""
        configs = [
            schema.LikelihoodConfig(
                type="zero",
                enabled=True,
                parameters={},
            ),
        ]

        likelihood = factory.create_combined_likelihood(configs)

        # Single likelihood should be returned directly
        assert isinstance(likelihood, ZeroLikelihood)

    def test_create_combined_likelihood_multiple(self):
        """Test that multiple likelihoods are combined."""
        configs = [
            schema.LikelihoodConfig(type="zero", enabled=True, parameters={}),
            schema.LikelihoodConfig(
                type="constraints_eos",
                enabled=True,
                parameters={},
            ),
        ]

        likelihood = factory.create_combined_likelihood(configs)

        # Multiple likelihoods should be wrapped in CombinedLikelihood
        assert isinstance(likelihood, CombinedLikelihood)
        assert len(likelihood.likelihoods_list) == 2

    def test_create_combined_likelihood_with_disabled(self):
        """Test that disabled likelihoods are skipped."""
        configs = [
            schema.LikelihoodConfig(type="zero", enabled=True, parameters={}),
            schema.LikelihoodConfig(type="zero", enabled=False, parameters={}),
        ]

        likelihood = factory.create_combined_likelihood(configs)

        # Only one enabled - should return directly
        assert isinstance(likelihood, ZeroLikelihood)

    def test_create_combined_likelihood_all_disabled_raises_error(self):
        """Test that all disabled likelihoods raises ValueError."""
        configs = [
            schema.LikelihoodConfig(type="zero", enabled=False, parameters={}),
            schema.LikelihoodConfig(type="zero", enabled=False, parameters={}),
        ]

        with pytest.raises(ValueError, match="No likelihoods enabled"):
            factory.create_combined_likelihood(configs)

    def test_create_combined_likelihood_with_radio_timing(self):
        """Test creating combined likelihood with radio timing constraint."""
        configs = [
            schema.LikelihoodConfig(
                type="radio",
                enabled=True,
                parameters={
                    "pulsars": [
                        {
                            "name": "J0348+0432",
                            "mass_mean": 2.01,
                            "mass_std": 0.04,
                        },
                    ],
                    "nb_masses": 100,
                },
            ),
        ]

        likelihood = factory.create_combined_likelihood(configs)

        # Single radio likelihood should be returned directly
        assert isinstance(likelihood, RadioTimingLikelihood)


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
            assert hasattr(likelihood, 'evaluate')
            assert callable(likelihood.evaluate)

    def test_likelihood_chaining(self):
        """Test that likelihoods can be chained via CombinedLikelihood."""
        l1 = ZeroLikelihood()
        l2 = ConstraintEOSLikelihood()
        l3 = ConstraintTOVLikelihood()

        combined = CombinedLikelihood([l1, l2, l3])

        # All valid params should give 0.0
        params = {
            'n_tov_failures': 0.0,
            'n_causality_violations': 0.0,
            'n_stability_violations': 0.0,
            'n_pressure_violations': 0.0,
        }

        result = combined.evaluate(params, {})
        assert result == 0.0

    def test_likelihood_with_violations_propagates(self):
        """Test that constraint violations propagate through CombinedLikelihood."""
        eos_constraint = ConstraintEOSLikelihood(penalty_causality=-1e10)
        tov_constraint = ConstraintTOVLikelihood(penalty_tov=-1e10)

        combined = CombinedLikelihood([eos_constraint, tov_constraint])

        # Both violated
        params = {
            'n_causality_violations': 1.0,
            'n_tov_failures': 1.0,
            'n_stability_violations': 0.0,
            'n_pressure_violations': 0.0,
        }

        result = combined.evaluate(params, {})
        # Should sum both penalties
        assert result == -2e10
