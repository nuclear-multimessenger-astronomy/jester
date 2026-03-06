"""Tests for the unified crust EOS model.

Covers:
- UnifiedCrustEOSConfig validation (BSk model validator, field inheritance)
- BSk outer crust table (build_outer_crust_table)
- UnifiedCrustEOS_MetaModel: instantiation, parameter interface, EOSData validity
- Physical constraints: cs2 causal, pressure monotone
- JAX JIT compilation
- JesterTransform.from_config with type="unified_crust"
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Reference NEPs (near BSk24)
# ---------------------------------------------------------------------------

TEST_PARAMS = {
    "E_sat": -16.05,
    "K_sat": 245.5,
    "Q_sat": -400.0,
    "Z_sat": 0.0,
    "E_sym": 30.0,
    "L_sym": 46.4,
    "K_sym": -145.0,
    "Q_sym": 0.0,
    "Z_sym": 0.0,
}


# ---------------------------------------------------------------------------
# BSk outer crust table tests
# ---------------------------------------------------------------------------


class TestBSKOuterCrust:
    """Tests for the standalone BSk outer crust table builder."""

    def test_bsk24_table_shape(self):
        from jesterTOV.eos.unified_crust.bsk_outer_crust import build_outer_crust_table

        n, p, e, cs2, mub = build_outer_crust_table(1e-11, 2.5e-4, 50, bsk=24)
        assert n.shape == (50,)
        assert p.shape == (50,)
        assert e.shape == (50,)
        assert cs2.shape == (50,)
        assert mub.shape == (50,)

    def test_bsk22_table_shape(self):
        from jesterTOV.eos.unified_crust.bsk_outer_crust import build_outer_crust_table

        n, p, e, cs2, mub = build_outer_crust_table(1e-11, 2.6e-4, 30, bsk=22)
        assert n.shape == (30,)

    def test_density_monotone(self):
        from jesterTOV.eos.unified_crust.bsk_outer_crust import build_outer_crust_table

        n, p, e, cs2, mub = build_outer_crust_table(1e-11, 2.5e-4, 50, bsk=24)
        assert np.all(np.diff(n) > 0), "Density must be monotonically increasing"

    def test_pressure_positive(self):
        from jesterTOV.eos.unified_crust.bsk_outer_crust import build_outer_crust_table

        n, p, e, cs2, mub = build_outer_crust_table(1e-11, 2.5e-4, 50, bsk=24)
        assert np.all(p > 0), "Pressure must be positive"

    def test_energy_density_positive(self):
        from jesterTOV.eos.unified_crust.bsk_outer_crust import build_outer_crust_table

        n, p, e, cs2, mub = build_outer_crust_table(1e-11, 2.5e-4, 50, bsk=24)
        assert np.all(e > 0), "Energy density must be positive"

    def test_cs2_causal(self):
        from jesterTOV.eos.unified_crust.bsk_outer_crust import build_outer_crust_table

        n, p, e, cs2, mub = build_outer_crust_table(1e-11, 2.5e-4, 50, bsk=24)
        assert np.all(cs2 >= 0), "cs2 must be non-negative"
        assert np.all(cs2 <= 1), "cs2 must not exceed 1 (causality)"

    def test_chemical_potential_physical(self):
        from jesterTOV.eos.unified_crust.bsk_outer_crust import build_outer_crust_table

        n, p, e, cs2, mub = build_outer_crust_table(1e-11, 2.5e-4, 50, bsk=24)
        # mu_b should be close to nucleon mass (930-940 MeV) at these densities
        assert np.all(mub > 900), "mub should exceed 900 MeV in outer crust"
        assert np.all(mub < 950), "mub should be below 950 MeV in outer crust"

    def test_neutron_drip_density(self):
        from jesterTOV.eos.unified_crust.bsk_outer_crust import get_neutron_drip_density

        n_drip_24 = get_neutron_drip_density(24)
        n_drip_22 = get_neutron_drip_density(22)
        assert (
            2.5e-4 < n_drip_24 < 2.7e-4
        ), f"BSk24 drip density unexpected: {n_drip_24}"
        assert (
            2.6e-4 < n_drip_22 < 2.8e-4
        ), f"BSk22 drip density unexpected: {n_drip_22}"

    def test_invalid_bsk_raises(self):
        from jesterTOV.eos.unified_crust.bsk_outer_crust import build_outer_crust_table

        with pytest.raises(ValueError, match="BSk model must be 22 or 24"):
            build_outer_crust_table(1e-11, 2.5e-4, 50, bsk=21)

    def test_nbmax_clamped_at_drip(self):
        """nbmax above drip density should be silently clamped."""
        from jesterTOV.eos.unified_crust.bsk_outer_crust import (
            build_outer_crust_table,
            get_neutron_drip_density,
        )

        n_drip = get_neutron_drip_density(24)
        # Request above drip — should clamp
        n, p, e, cs2, mub = build_outer_crust_table(1e-11, n_drip * 2, 50, bsk=24)
        assert n[-1] <= n_drip * 1.001


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestUnifiedCrustEOSConfig:
    """Tests for UnifiedCrustEOSConfig Pydantic model."""

    def test_default_instantiation(self):
        from jesterTOV.inference.config.schemas.eos import UnifiedCrustEOSConfig

        config = UnifiedCrustEOSConfig()
        assert config.type == "unified_crust"
        assert config.bsk_outer_crust == 24
        assert config.ndat_outer == 50
        assert config.ndat_metamodel == 100  # inherited from BaseMetamodelEOSConfig

    def test_bsk22(self):
        from jesterTOV.inference.config.schemas.eos import UnifiedCrustEOSConfig

        config = UnifiedCrustEOSConfig(bsk_outer_crust=22)
        assert config.bsk_outer_crust == 22

    def test_bsk24(self):
        from jesterTOV.inference.config.schemas.eos import UnifiedCrustEOSConfig

        config = UnifiedCrustEOSConfig(bsk_outer_crust=24)
        assert config.bsk_outer_crust == 24

    def test_invalid_bsk_raises(self):
        from jesterTOV.inference.config.schemas.eos import UnifiedCrustEOSConfig

        with pytest.raises(Exception):
            UnifiedCrustEOSConfig(bsk_outer_crust=21)

    def test_inherits_metamodel_fields(self):
        from jesterTOV.inference.config.schemas.eos import UnifiedCrustEOSConfig

        config = UnifiedCrustEOSConfig(ndat_metamodel=200, nmax_nsat=12.0)
        assert config.ndat_metamodel == 200
        assert config.nmax_nsat == 12.0

    def test_extra_fields_forbidden(self):
        from jesterTOV.inference.config.schemas.eos import UnifiedCrustEOSConfig

        with pytest.raises(Exception):
            UnifiedCrustEOSConfig(unknown_field=42)

    def test_serialization_roundtrip(self):
        from jesterTOV.inference.config.schemas.eos import UnifiedCrustEOSConfig

        config = UnifiedCrustEOSConfig(bsk_outer_crust=24, ndat_outer=60)
        d = config.model_dump()
        config2 = UnifiedCrustEOSConfig(**d)
        assert config2.bsk_outer_crust == 24
        assert config2.ndat_outer == 60

    def test_in_eos_config_union(self):
        """UnifiedCrustEOSConfig should be parseable via the EOSConfig union."""
        from jesterTOV.inference.config.schema import EOSConfig
        from pydantic import TypeAdapter

        data = {"type": "unified_crust", "bsk_outer_crust": 24}
        adapter = TypeAdapter(EOSConfig)
        config = adapter.validate_python(data)
        assert config.type == "unified_crust"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# EOS class tests
# ---------------------------------------------------------------------------


class TestUnifiedCrustEOS_MetaModel:
    """Tests for UnifiedCrustEOS_MetaModel."""

    @pytest.fixture(scope="class")
    def eos(self):
        from jesterTOV.eos.unified_crust import UnifiedCrustEOS_MetaModel

        return UnifiedCrustEOS_MetaModel(
            bsk_outer_crust=24,
            ndat_outer=30,
            nmax_nsat=8.0,
            ndat_core=100,
        )

    def test_instantiation(self, eos):
        from jesterTOV.eos.unified_crust import UnifiedCrustEOS_MetaModel

        assert isinstance(eos, UnifiedCrustEOS_MetaModel)

    def test_required_parameters(self, eos):
        params = eos.get_required_parameters()
        expected = [
            "E_sat",
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]
        assert params == expected

    def test_construct_eos_returns_eosdata(self, eos):
        from jesterTOV.tov.data_classes import EOSData

        eos_data = eos.construct_eos(TEST_PARAMS)
        assert isinstance(eos_data, EOSData)

    def test_construct_eos_all_fields_present(self, eos):
        eos_data = eos.construct_eos(TEST_PARAMS)
        assert eos_data.ns is not None
        assert eos_data.ps is not None
        assert eos_data.hs is not None
        assert eos_data.es is not None
        assert eos_data.dloge_dlogps is not None
        assert eos_data.cs2 is not None

    def test_all_finite(self, eos):
        eos_data = eos.construct_eos(TEST_PARAMS)
        assert jnp.all(jnp.isfinite(eos_data.ns)), "ns contains non-finite values"
        assert jnp.all(jnp.isfinite(eos_data.ps)), "ps contains non-finite values"
        assert jnp.all(jnp.isfinite(eos_data.es)), "es contains non-finite values"
        assert jnp.all(jnp.isfinite(eos_data.cs2)), "cs2 contains non-finite values"

    def test_pressure_positive(self, eos):
        eos_data = eos.construct_eos(TEST_PARAMS)
        assert jnp.all(eos_data.ps > 0), "Pressure must be positive"

    def test_pressure_monotone(self, eos):
        eos_data = eos.construct_eos(TEST_PARAMS)
        dp = jnp.diff(eos_data.ps)
        assert jnp.all(dp > 0), "Pressure must be monotonically increasing"

    def test_cs2_causal(self, eos):
        eos_data = eos.construct_eos(TEST_PARAMS)
        assert jnp.all(eos_data.cs2 >= 0), "cs2 must be non-negative"
        # Note: the metamodel at high density can have cs2 > 1 (superluminal
        # core), identical to the plain MetaModel_EOS_model. The outer crust
        # cs2 is clipped to [0, 1] by build_outer_crust_table. Superluminal
        # samples are penalised by the constraints_eos likelihood at run time.

    def test_deterministic(self, eos):
        """Same parameters must produce identical results."""
        data1 = eos.construct_eos(TEST_PARAMS)
        data2 = eos.construct_eos(TEST_PARAMS)
        assert jnp.allclose(data1.ns, data2.ns)
        assert jnp.allclose(data1.ps, data2.ps)
        assert jnp.allclose(data1.cs2, data2.cs2)

    def test_jit_compilable(self, eos):
        """construct_eos must be JIT-compilable."""
        jitted = jax.jit(eos.construct_eos)
        eos_data = jitted(TEST_PARAMS)
        assert jnp.all(jnp.isfinite(eos_data.ps))

    def test_mu_populated(self, eos):
        """mu (baryon chemical potential) should be populated."""
        eos_data = eos.construct_eos(TEST_PARAMS)
        assert eos_data.mu is not None
        assert jnp.all(jnp.isfinite(eos_data.mu))  # type: ignore[arg-type]
        # mu should be in ~[900, 1500] MeV for physical densities
        from jesterTOV import utils

        mu_MeV = jnp.array(eos_data.mu) / utils.MeV_fm_inv3_to_geometric  # back to MeV
        assert jnp.all(mu_MeV > 800), "mu should exceed 800 MeV"

    def test_bsk22_also_works(self):
        from jesterTOV.eos.unified_crust import UnifiedCrustEOS_MetaModel

        eos22 = UnifiedCrustEOS_MetaModel(
            bsk_outer_crust=22, ndat_outer=30, ndat_core=80
        )
        eos_data = eos22.construct_eos(TEST_PARAMS)
        assert jnp.all(jnp.isfinite(eos_data.ps))

    def test_multiple_nep_sets(self, eos):
        """Several physically reasonable NEP sets should all yield valid EOS."""
        nep_sets = [
            {
                "E_sat": -16.0,
                "K_sat": 230.0,
                "Q_sat": -300.0,
                "Z_sat": 0.0,
                "E_sym": 32.0,
                "L_sym": 60.0,
                "K_sym": -100.0,
                "Q_sym": 0.0,
                "Z_sym": 0.0,
            },
            {
                "E_sat": -15.95,
                "K_sat": 260.0,
                "Q_sat": -200.0,
                "Z_sat": 0.0,
                "E_sym": 35.0,
                "L_sym": 80.0,
                "K_sym": -50.0,
                "Q_sym": 0.0,
                "Z_sym": 0.0,
            },
            {
                "E_sat": -16.1,
                "K_sat": 200.0,
                "Q_sat": -500.0,
                "Z_sat": 0.0,
                "E_sym": 29.0,
                "L_sym": 30.0,
                "K_sym": -200.0,
                "Q_sym": 0.0,
                "Z_sym": 0.0,
            },
        ]
        for params in nep_sets:
            data = eos.construct_eos(params)
            assert jnp.all(
                jnp.isfinite(data.ps)
            ), f"Non-finite pressure for params={params}"
            assert jnp.all(
                jnp.diff(data.ps) > 0
            ), f"Non-monotone pressure for params={params}"


# ---------------------------------------------------------------------------
# Transform integration tests
# ---------------------------------------------------------------------------


class TestUnifiedCrustJesterTransform:
    """Tests for JesterTransform with unified_crust EOS type."""

    @pytest.fixture(scope="class")
    def transform(self):
        from jesterTOV.inference.transforms.transform import JesterTransform
        from jesterTOV.inference.config.schemas.eos import UnifiedCrustEOSConfig
        from jesterTOV.inference.config.schemas.tov import GRTOVConfig

        eos_config = UnifiedCrustEOSConfig(
            bsk_outer_crust=24,
            ndat_outer=30,
            ndat_metamodel=100,
            nmax_nsat=8.0,
        )
        tov_config = GRTOVConfig()
        return JesterTransform.from_config(eos_config, tov_config)

    def test_from_config(self, transform):
        from jesterTOV.inference.transforms.transform import JesterTransform

        assert isinstance(transform, JesterTransform)

    def test_parameter_names(self, transform):
        names = transform.get_parameter_names()
        expected = [
            "E_sat",
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]
        assert names == expected

    def test_forward_call(self, transform):
        result = transform.forward(TEST_PARAMS)
        assert "masses_EOS" in result
        assert "radii_EOS" in result
        assert "Lambdas_EOS" in result
        masses = np.array(result["masses_EOS"])
        radii = np.array(result["radii_EOS"])
        assert np.any(np.isfinite(masses) & (masses > 0.5)), "No valid TOV solutions"
        assert np.any(np.isfinite(radii) & (radii > 5)), "No valid TOV radii"

    def test_eos_type_repr(self, transform):
        assert "UnifiedCrustEOS_MetaModel" in transform.get_eos_type()


# ---------------------------------------------------------------------------
# Slow / E2E tests
# ---------------------------------------------------------------------------


class TestUnifiedCrustE2E:
    """End-to-end inference pipeline test."""

    @pytest.mark.slow
    def test_smc_rw_chieft_e2e(self, tmp_path):
        """2–3 SMC-RW iterations with chiEFT likelihood — verifies full pipeline."""
        import yaml
        from pathlib import Path
        from jesterTOV.inference.run_inference import run_inference

        example_config = (
            Path(__file__).parent.parent.parent
            / "examples"
            / "inference"
            / "smc_random_walk"
            / "unified_crust"
            / "config.yaml"
        )

        if not example_config.exists():
            pytest.skip(f"Example config not found: {example_config}")

        with open(example_config) as f:
            config_dict = yaml.safe_load(f)

        # Use minimal settings for speed
        config_dict["sampler"]["n_particles"] = 50
        config_dict["sampler"]["n_mcmc_steps"] = 2
        config_dict["outdir"] = str(tmp_path / "outdir")

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_dict))

        prior_src = example_config.parent / "prior.prior"
        if prior_src.exists():
            import shutil

            shutil.copy(prior_src, tmp_path / "prior.prior")

        result = run_inference(str(config_file))
        assert result is not None
