"""Test parameter validation in transform setup."""

import pytest
from unittest.mock import MagicMock
from jesterTOV.inference.config.schema import TransformConfig
from jesterTOV.inference.run_inference import setup_transform
from jesterTOV.inference.base.prior import CombinePrior, UniformPrior


def test_missing_parameters_raises_error():
    """Test that setup_transform raises ValueError when parameters are missing from prior."""
    # Create a metamodel transform config (requires 9 NEP parameters)
    transform_config = TransformConfig(
        type="metamodel",
        ndat_metamodel=100,
        ndat_TOV=100,
        min_nsat_TOV=0.75,
        nmax_nsat=25.0,
    )

    # Create minimal config mock with just transform
    config = MagicMock()
    config.transform = transform_config

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
    # Create a metamodel transform config
    transform_config = TransformConfig(
        type="metamodel",
        ndat_metamodel=100,
        ndat_TOV=100,
        min_nsat_TOV=0.75,
        nmax_nsat=25.0,
    )

    config = MagicMock()
    config.transform = transform_config

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
    # Create a metamodel transform config
    transform_config = TransformConfig(
        type="metamodel",
        ndat_metamodel=100,
        ndat_TOV=100,
        min_nsat_TOV=0.75,
        nmax_nsat=25.0,
    )

    config = MagicMock()
    config.transform = transform_config

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
    from unittest.mock import MagicMock

    caplog = MagicMock()
    caplog.at_level = lambda level: caplog
    caplog.__enter__ = lambda self: self
    caplog.__exit__ = lambda self, *args: None
    caplog.records = []

    class LogRecord:
        def __init__(self, message):
            self.message = message

    # Capture actual warnings by temporarily modifying logger
    import jesterTOV.logging_config

    original_logger = jesterTOV.logging_config.get_logger("jester")
    warnings_logged = []

    def mock_warning(msg):
        warnings_logged.append(msg)
        original_logger.warning(msg)

    original_logger.warning = mock_warning
    caplog.records = [LogRecord(msg) for msg in warnings_logged]

    test_unused_parameters_warns(caplog)

    print("\n✅ All validation tests passed!")
