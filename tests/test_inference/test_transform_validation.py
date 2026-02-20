"""Test parameter validation in transform setup."""

import pytest
from unittest.mock import MagicMock
from jesterTOV.inference.config.schema import MetamodelEOSConfig, TOVConfig
from jesterTOV.inference.run_inference import setup_transform
from jesterTOV.inference.base.prior import CombinePrior, UniformPrior


def test_missing_parameters_raises_error():
    """Test that setup_transform raises ValueError when parameters are missing from prior."""
    # Create a metamodel EOS config (requires 9 NEP parameters)
    eos_config = MetamodelEOSConfig(
        type="metamodel",
        ndat_metamodel=100,
        nmax_nsat=25.0,
    )
    tov_config = TOVConfig(
        ndat_TOV=100,
        min_nsat_TOV=0.75,
    )

    # Create minimal config mock with eos and tov
    config = MagicMock()
    config.eos = eos_config
    config.tov = tov_config

    # Create a prior with only SOME of the required parameters (missing E_sat, K_sat, Q_sat)
    # MetaModel requires: E_sat, K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym
    priors = [
        UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"]),
        UniformPrior(28.0, 45.0, parameter_names=["E_sym"]),
        UniformPrior(10.0, 200.0, parameter_names=["L_sym"]),
        UniformPrior(-400.0, 200.0, parameter_names=["K_sym"]),
        UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"]),
        UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"]),
    ]
    prior = CombinePrior(priors)

    # Test that ValueError is raised
    with pytest.raises(ValueError) as exc_info:
        setup_transform(config, prior=prior)

    # Check error message content
    error_msg = str(exc_info.value)
    assert "Transform with EOS" in error_msg, "Error should mention EOS"
    assert "TOV" in error_msg, "Error should mention TOV"
    assert "missing params" in error_msg, "Error should say 'missing params'"
    assert "E_sat" in error_msg, "Error should list E_sat as missing"
    assert "K_sat" in error_msg, "Error should list K_sat as missing"
    assert "Q_sat" in error_msg, "Error should list Q_sat as missing"
    assert "from the prior file" in error_msg, "Error should mention prior file"

    print(f"✓ Error message correctly raised:\n  {error_msg}")


def test_all_parameters_present_succeeds():
    """Test that setup_transform succeeds when all parameters are present."""
    # Create a metamodel EOS config
    eos_config = MetamodelEOSConfig(
        type="metamodel",
        ndat_metamodel=100,
        nmax_nsat=25.0,
    )
    tov_config = TOVConfig(
        ndat_TOV=100,
        min_nsat_TOV=0.75,
    )

    config = MagicMock()
    config.eos = eos_config
    config.tov = tov_config

    # Create a prior with ALL required parameters
    priors = [
        UniformPrior(-16.1, -15.9, parameter_names=["E_sat"]),
        UniformPrior(150.0, 300.0, parameter_names=["K_sat"]),
        UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"]),
        UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"]),
        UniformPrior(28.0, 45.0, parameter_names=["E_sym"]),
        UniformPrior(10.0, 200.0, parameter_names=["L_sym"]),
        UniformPrior(-400.0, 200.0, parameter_names=["K_sym"]),
        UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"]),
        UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"]),
    ]
    prior = CombinePrior(priors)

    # Should succeed without raising
    transform = setup_transform(config, prior=prior)
    assert transform is not None
    assert transform.get_eos_type() == "MetaModel_EOS_model"
    print("✓ Transform created successfully with all required parameters")


def test_unused_parameters_succeeds():
    """Test that unused parameters in prior don't cause errors (only warnings)."""
    # Create a metamodel EOS config
    eos_config = MetamodelEOSConfig(
        type="metamodel",
        ndat_metamodel=100,
        nmax_nsat=25.0,
    )
    tov_config = TOVConfig(
        ndat_TOV=100,
        min_nsat_TOV=0.75,
    )

    config = MagicMock()
    config.eos = eos_config
    config.tov = tov_config

    # Create prior with ALL required parameters PLUS extra unused ones
    priors = [
        UniformPrior(-16.1, -15.9, parameter_names=["E_sat"]),
        UniformPrior(150.0, 300.0, parameter_names=["K_sat"]),
        UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"]),
        UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"]),
        UniformPrior(28.0, 45.0, parameter_names=["E_sym"]),
        UniformPrior(10.0, 200.0, parameter_names=["L_sym"]),
        UniformPrior(-400.0, 200.0, parameter_names=["K_sym"]),
        UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"]),
        UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"]),
        # Extra unused parameters
        UniformPrior(0.0, 1.0, parameter_names=["unused_param1"]),
        UniformPrior(0.0, 1.0, parameter_names=["unused_param2"]),
    ]
    prior = CombinePrior(priors)

    # Should succeed without raising (just logs warning)
    transform = setup_transform(config, prior=prior)

    # Verify transform was created successfully
    assert transform is not None
    assert transform.get_eos_type() == "MetaModel_EOS_model"
    print("✓ Transform created successfully with unused parameters (warning logged)")


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("Testing parameter validation...\n")

    print("1. Testing missing parameters...")
    test_missing_parameters_raises_error()

    print("\n2. Testing all parameters present...")
    test_all_parameters_present_succeeds()

    print("\n3. Testing unused parameters warning...")
    test_unused_parameters_succeeds()

    print("\n✅ All validation tests passed!")
