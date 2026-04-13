"""Tests for AnisotropyTOVSolver inference integration.

Covers config parsing, transform creation, gamma passthrough, and parameter
validation for the anisotropy TOV solver with beyond-GR corrections.
"""

import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError

import jax.numpy as jnp

from jesterTOV.inference.config.schema import (
    MetamodelEOSConfig,
    GRTOVConfig,
    AnisotropyTOVConfig,
    TOVConfig,
)
from jesterTOV.inference.transforms import JesterTransform
from jesterTOV.inference.run_inference import setup_transform
from jesterTOV.inference.base.prior import CombinePrior, UniformPrior
from jesterTOV.tov.anisotropy import AnisotropyTOVSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nep_priors() -> list[UniformPrior]:
    """Nine MetaModel nuclear empirical parameter priors."""
    return [
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


def _eos_config() -> MetamodelEOSConfig:
    return MetamodelEOSConfig(type="metamodel", ndat_metamodel=100, nmax_nsat=3.0)


_NEP_PARAMS: dict[str, float] = {
    "E_sat": -16.0,
    "K_sat": 240.0,
    "Q_sat": 400.0,
    "Z_sat": 0.0,
    "E_sym": 31.7,
    "L_sym": 58.7,
    "K_sym": -100.0,
    "Q_sym": 0.0,
    "Z_sym": 0.0,
}


# ---------------------------------------------------------------------------
# Config schema tests
# ---------------------------------------------------------------------------


class TestAnisotropyTOVConfig:
    """Test AnisotropyTOVConfig Pydantic model."""

    def test_anisotropy_config_defaults(self):
        """AnisotropyTOVConfig should parse with just type='anisotropy'."""
        config = AnisotropyTOVConfig(type="anisotropy")
        assert config.type == "anisotropy"
        assert config.min_nsat_TOV == 0.75
        assert config.ndat_TOV == 100
        assert config.nb_masses == 100

    def test_anisotropy_config_custom_base_fields(self):
        """AnisotropyTOVConfig should accept overridden base fields."""
        config = AnisotropyTOVConfig(type="anisotropy", ndat_TOV=50, min_nsat_TOV=0.5)
        assert config.ndat_TOV == 50
        assert config.min_nsat_TOV == 0.5

    def test_anisotropy_config_rejects_extra_fields(self):
        """AnisotropyTOVConfig must not accept unknown keys (extra='forbid')."""
        with pytest.raises(ValidationError):
            AnisotropyTOVConfig(type="anisotropy", unknown_field=1.0)  # type: ignore[call-arg]

    def test_tov_union_selects_gr_config(self):
        """TOVConfig discriminated union selects GRTOVConfig for type='gr'."""
        from pydantic import TypeAdapter

        ta: TypeAdapter[TOVConfig] = TypeAdapter(TOVConfig)  # type: ignore[type-arg]
        config = ta.validate_python({"type": "gr"})
        assert isinstance(config, GRTOVConfig)

    def test_tov_union_selects_anisotropy_config(self):
        """TOVConfig discriminated union selects AnisotropyTOVConfig for type='anisotropy'."""
        from pydantic import TypeAdapter

        ta: TypeAdapter[TOVConfig] = TypeAdapter(TOVConfig)  # type: ignore[type-arg]
        config = ta.validate_python({"type": "anisotropy"})
        assert isinstance(config, AnisotropyTOVConfig)

    def test_tov_union_rejects_unknown_type(self):
        """TOVConfig union should reject unrecognised type strings."""
        from pydantic import TypeAdapter

        ta: TypeAdapter[TOVConfig] = TypeAdapter(TOVConfig)  # type: ignore[type-arg]
        with pytest.raises(ValidationError):
            ta.validate_python({"type": "unknown_solver"})


# ---------------------------------------------------------------------------
# Transform factory tests
# ---------------------------------------------------------------------------


class TestAnisotropyTOVTransformFactory:
    """Test JesterTransform.from_config() with AnisotropyTOVConfig."""

    def test_creates_anisotropy_solver(self):
        """from_config() should instantiate an AnisotropyTOVSolver for type='anisotropy'."""
        transform = JesterTransform.from_config(
            eos_config=_eos_config(),
            tov_config=AnisotropyTOVConfig(type="anisotropy"),
        )
        assert isinstance(transform.tov_solver, AnisotropyTOVSolver)

    def test_anisotropy_required_parameters(self):
        """AnisotropyTOVSolver.get_required_parameters() must return all 3 coupling params."""
        expected = ["lambda_BL", "lambda_DY", "lambda_HB"]
        assert AnisotropyTOVSolver().get_required_parameters() == expected

    def test_transform_parameter_names_include_anisotropy_params(self):
        """With AnisotropyTOVSolver, get_parameter_names() includes both EOS and TOV params."""
        transform = JesterTransform.from_config(
            eos_config=_eos_config(),
            tov_config=AnisotropyTOVConfig(type="anisotropy"),
        )
        required = transform.get_parameter_names()
        for nep in [
            "E_sat",
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]:
            assert nep in required
        # All anisotropy params are now required
        for p in ["lambda_BL", "lambda_DY", "lambda_HB"]:
            assert p in required


# ---------------------------------------------------------------------------
# Parameter-validation tests (setup_transform)
# ---------------------------------------------------------------------------


class TestAnisotropyParameterValidation:
    """Test run_inference.setup_transform() with anisotropy TOV config."""

    def _make_config(self) -> MagicMock:
        config = MagicMock()
        config.eos = _eos_config()
        config.tov = AnisotropyTOVConfig(type="anisotropy")
        return config

    def test_neps_only_prior_fails_for_anisotropy(self):
        """Prior with only 9 NEPs raises ValueError since all 3 anisotropy params are required."""
        prior = CombinePrior(_nep_priors())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="missing params"):
            setup_transform(self._make_config(), prior=prior)

    def test_neps_plus_all_anisotropy_params_succeeds(self):
        """Prior with 9 NEPs + all 3 anisotropy params passes validation."""
        priors = _nep_priors() + [
            UniformPrior(0.0, 0.2, parameter_names=["lambda_BL"]),
            UniformPrior(0.0, 0.1, parameter_names=["lambda_DY"]),
            UniformPrior(0.9, 1.1, parameter_names=["lambda_HB"]),
        ]
        prior = CombinePrior(priors)  # type: ignore[arg-type]
        transform = setup_transform(self._make_config(), prior=prior)
        assert isinstance(transform.tov_solver, AnisotropyTOVSolver)

    def test_missing_nep_raises_error(self):
        """Missing a required NEP still raises ValueError for anisotropy TOV."""
        incomplete = _nep_priors()[1:]  # drop E_sat
        prior = CombinePrior(incomplete)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="missing params"):
            setup_transform(self._make_config(), prior=prior)


# ---------------------------------------------------------------------------
# tov_kwargs passthrough tests
# ---------------------------------------------------------------------------


class TestAnisotropyKwargsPassthrough:
    """Test that non-EOS prior params are forwarded to the TOV solver."""

    @pytest.fixture
    def eos_data(self):
        """Valid EOS data built from a fixed MetaModel parameter set."""
        from jesterTOV.eos.metamodel import MetaModel_EOS_model

        return MetaModel_EOS_model().construct_eos(_NEP_PARAMS)

    def test_gamma_zero_matches_gr(self, eos_data):
        """gamma=0 with AnisotropyTOVSolver should match GR solution closely."""
        from jesterTOV.tov.gr import GRTOVSolver

        pc = eos_data.ps[25]
        gr_sol = GRTOVSolver().solve(eos_data, pc, {})
        gr_params = {
            "lambda_BL": 0.0,
            "lambda_DY": 0.0,
            "lambda_HB": 1.0,
        }
        aniso_sol = AnisotropyTOVSolver().solve(eos_data, pc, gr_params)

        assert float(abs(gr_sol.M - aniso_sol.M) / gr_sol.M) < 0.01
        assert float(abs(gr_sol.R - aniso_sol.R) / gr_sol.R) < 0.01

    def test_nonzero_params_differ_from_gr(self, eos_data):
        """Non-GR parameters should produce a solution different from pure GR."""
        from jesterTOV.tov.gr import GRTOVSolver

        pc = eos_data.ps[25]
        gr_sol = GRTOVSolver().solve(eos_data, pc, {})
        mg_params = {
            "lambda_BL": 0.1,
            "lambda_DY": 0.05,
            "lambda_HB": 1.0,
        }
        aniso_sol = AnisotropyTOVSolver().solve(eos_data, pc, mg_params)

        diff = float(abs(gr_sol.M - aniso_sol.M) / gr_sol.M)
        assert diff > 1e-6, "Non-GR corrections should shift the mass"

    def test_transform_forward_passes_lambda_DY(self):
        """construct_eos_and_solve_tov must pass lambda_DY from the param dict to the solver."""
        transform = JesterTransform.from_config(
            eos_config=_eos_config(),
            tov_config=AnisotropyTOVConfig(type="anisotropy"),
        )

        # Run transform with lambda_DY=0 (GR limit) vs non-zero lambda_DY
        base_aniso = {
            "lambda_BL": 0.0,
            "lambda_HB": 1.0,
        }
        result_gr = transform.construct_eos_and_solve_tov(
            {**_NEP_PARAMS, **base_aniso, "lambda_DY": 0.0}
        )
        result_mg = transform.construct_eos_and_solve_tov(
            {**_NEP_PARAMS, **base_aniso, "lambda_DY": 0.1}
        )

        mass_diff = jnp.max(jnp.abs(result_gr["masses_EOS"] - result_mg["masses_EOS"]))
        assert float(mass_diff) > 1e-6, "lambda_DY should shift the M-R family"
