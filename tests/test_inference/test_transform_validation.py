"""Test parameter validation in transform setup."""

import pytest
from unittest.mock import MagicMock
from jesterTOV.inference.config.schema import MetamodelEOSConfig, GRTOVConfig
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
    tov_config = GRTOVConfig(
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
    tov_config = GRTOVConfig(
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
    tov_config = GRTOVConfig(
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


def _make_nep_prior() -> CombinePrior:
    """Return a CombinePrior with all 9 MetaModel NEP parameters."""
    return CombinePrior(
        [
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
    )


def test_fixed_param_satisfies_required_param():
    """A required param supplied via fixed_params is not flagged as missing."""
    from jesterTOV.tov.anisotropy import AnisotropyTOVSolver
    from jesterTOV.inference.config.schema import AnisotropyTOVConfig

    eos_config = MetamodelEOSConfig(
        type="metamodel", ndat_metamodel=100, nmax_nsat=25.0
    )
    tov_config = AnisotropyTOVConfig(type="anisotropy", ndat_TOV=100, min_nsat_TOV=0.75)
    config = MagicMock()
    config.eos = eos_config
    config.tov = tov_config

    # AnisotropyTOVSolver requires e.g. lambda_BL — supply it as fixed
    anisotropy_solver = AnisotropyTOVSolver()
    tov_required = anisotropy_solver.get_required_parameters()
    assert len(tov_required) > 0, "AnisotropyTOVSolver must require at least one param"

    # Fix all TOV params; prior has only NEP params
    fixed_params = {p: 0.0 for p in tov_required}
    prior = _make_nep_prior()

    # Should succeed — fixed params cover the TOV requirements
    transform = setup_transform(config, prior=prior, fixed_params=fixed_params)
    assert transform is not None
    # Fixed params should not appear in get_parameter_names()
    for p in tov_required:
        assert p not in transform.get_parameter_names()


def test_fixed_param_injected_in_forward():
    """JesterTransform.forward() injects fixed_params into the EOS/TOV call."""
    from jesterTOV.eos.metamodel import MetaModel_EOS_model
    from jesterTOV.tov.gr import GRTOVSolver
    from jesterTOV.inference.transforms import JesterTransform

    eos = MetaModel_EOS_model(crust_name="DH")
    tov = GRTOVSolver()

    # Pick one NEP param to fix; rest will be sampled
    all_nep = eos.get_required_parameters()  # 9 params
    fixed_name = "Z_sat"
    fixed_value = 0.0
    sampled_names = [p for p in all_nep if p != fixed_name]

    transform = JesterTransform(
        eos=eos,
        tov_solver=tov,
        fixed_params={fixed_name: fixed_value},
        ndat_TOV=10,  # small for speed
    )

    # Verify fixed param is excluded from the sampled parameter list
    assert fixed_name not in transform.get_parameter_names()
    assert set(transform.get_parameter_names()) == set(sampled_names)

    # Build a minimal param dict (without the fixed param)
    sample_params = {
        "E_sat": -16.0,
        "K_sat": 230.0,
        "Q_sat": 400.0,
        # Z_sat omitted — must be injected by transform
        "E_sym": 32.0,
        "L_sym": 60.0,
        "K_sym": -100.0,
        "Q_sym": 0.0,
        "Z_sym": 0.0,
    }

    result = transform.forward(sample_params)

    # Transform should succeed and produce M-R-Λ arrays
    assert "masses_EOS" in result
    assert "radii_EOS" in result
    assert "Lambdas_EOS" in result

    # Fixed param should appear in the output for traceability
    assert fixed_name in result
    assert float(result[fixed_name]) == pytest.approx(fixed_value)


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("Testing parameter validation...\n")

    print("1. Testing missing parameters...")
    test_missing_parameters_raises_error()

    print("\n2. Testing all parameters present...")
    test_all_parameters_present_succeeds()

    print("\n3. Testing unused parameters warning...")
    test_unused_parameters_succeeds()

    print("\n4. Testing fixed param satisfies required param...")
    test_fixed_param_satisfies_required_param()

    print("\n5. Testing fixed param injected in forward...")
    test_fixed_param_injected_in_forward()

    print("\n✅ All validation tests passed!")
