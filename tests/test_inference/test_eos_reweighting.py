"""Unit tests for EOSReweightingSampler."""

from pathlib import Path

import jax
import numpy as np
import pytest

from jesterTOV.inference.likelihoods.combined import ZeroLikelihood
from jesterTOV.inference.samplers.eos_reweighting import (
    EOSReweightingSampler,
    resample_eos_posterior,
)
from jesterTOV.inference.config.schemas.samplers import EOSReweightingConfig
from jesterTOV.inference.config.schemas.eos_reweighting import (
    EOSReweightingInferenceConfig,
)
from jesterTOV.inference.config.schemas.likelihoods import (
    ZeroLikelihoodConfig,
    GWLikelihoodConfig,
    GWEventConfig,
    GWResampledLikelihoodConfig,
    NICERLikelihoodConfig,
    NICERKDELikelihoodConfig,
    RadioLikelihoodConfig,
    ChiEFTLikelihoodConfig,
    EOSConstraintsLikelihoodConfig,
    TOVConstraintsLikelihoodConfig,
    EsymConstraintsLikelihoodConfig,
    GammaConstraintsLikelihoodConfig,
)

jax.config.update("jax_enable_x64", True)

N_EOS = 5
N_GRID = 30


def _make_dummy_npz(tmp_dir: Path, n_eos: int = N_EOS, n_pts: int = 50) -> str:
    """Write a minimal NPZ file with toy EOS curves."""
    masses = np.tile(np.linspace(0.5, 2.0, n_pts), (n_eos, 1))
    lambdas = (
        np.random.default_rng(0).uniform(100, 1000, (n_eos, 1))
        * ((2.0 - masses) / 1.5) ** 3
    )
    radii = 12.0 * np.ones_like(masses)
    path = str(tmp_dir / "test_eos.npz")
    np.savez(path, masses=masses, lambdas=lambdas, radii=radii)
    return path


@pytest.fixture()
def tmp_npz(tmp_path: Path) -> str:
    return _make_dummy_npz(tmp_path)


@pytest.fixture()
def zero_likelihood() -> ZeroLikelihood:
    return ZeroLikelihood()


def _make_sampler(npz_path: str, likelihood) -> EOSReweightingSampler:
    cfg = EOSReweightingConfig(
        eos_file=npz_path,
        n_grid=N_GRID,
        m_min=0.5,
        m_max=None,
    )
    return EOSReweightingSampler(likelihood=likelihood, config=cfg)


class TestEOSReweightingSamplerBasic:
    def test_output_shapes(self, tmp_npz, zero_likelihood):
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        key = jax.random.key(0)
        out = sampler.sample(key)

        assert out.log_prob.shape == (N_EOS,), "log_prob must be [N_eos]"
        assert out.samples["eos_index"].shape == (N_EOS,)
        assert out.samples["log_likelihood"].shape == (N_EOS,)
        assert out.samples["posterior_weight"].shape == (N_EOS,)

    def test_evidence_finite(self, tmp_npz, zero_likelihood):
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        out = sampler.sample(jax.random.key(0))

        ev = out.metadata["evidence"]
        assert np.isfinite(ev["log_Z"]), "log_Z must be finite"
        assert ev["N_eff"] > 0, "N_eff must be positive"
        assert 0.0 < ev["N_eff_fraction"] <= 1.0

    def test_zero_likelihood_uniform_weights(self, tmp_npz, zero_likelihood):
        """ZeroLikelihood → all log-likelihoods equal → N_eff = N."""
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        out = sampler.sample(jax.random.key(0))

        lw = np.array(out.samples["log_likelihood"])
        assert np.allclose(
            lw, lw[0], atol=1e-6
        ), "ZeroLikelihood must give equal weights"
        assert abs(out.metadata["evidence"]["N_eff"] - N_EOS) < 0.5

    def test_posterior_weights_sum_to_one(self, tmp_npz, zero_likelihood):
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        out = sampler.sample(jax.random.key(0))
        w = np.array(out.samples["posterior_weight"])
        assert abs(w.sum() - 1.0) < 1e-5

    def test_n_eos_in_metadata(self, tmp_npz, zero_likelihood):
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        out = sampler.sample(jax.random.key(0))
        assert out.metadata["N_eos"] == N_EOS

    def test_resampled_posterior_present(self, tmp_npz, zero_likelihood):
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        out = sampler.sample(jax.random.key(0))

        n_resampled = out.metadata["N_resampled"]
        assert n_resampled > 0
        assert out.samples["masses_EOS"].shape == (n_resampled, N_GRID)
        assert out.samples["Lambdas_EOS"].shape == (n_resampled, N_GRID)
        assert out.samples["radii_EOS"].shape == (n_resampled, N_GRID)
        assert out.samples["resampled_eos_index"].shape == (n_resampled,)
        assert out.samples["resampled_log_likelihood"].shape == (n_resampled,)
        assert np.all(np.asarray(out.samples["resampled_eos_index"]) < N_EOS)

    def test_resampled_curves_match_source_curves(self, tmp_npz, zero_likelihood):
        """Each resampled curve must equal the source curve at its index."""
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        out = sampler.sample(jax.random.key(0))

        _, (all_masses, all_lambdas, all_radii) = sampler.load_and_grid(
            [tmp_npz], N_GRID, 0.5, None
        )
        idx = np.asarray(out.samples["resampled_eos_index"])
        np.testing.assert_allclose(
            np.asarray(out.samples["masses_EOS"]), np.asarray(all_masses)[idx]
        )
        np.testing.assert_allclose(
            np.asarray(out.samples["Lambdas_EOS"]), np.asarray(all_lambdas)[idx]
        )
        np.testing.assert_allclose(
            np.asarray(out.samples["radii_EOS"]), np.asarray(all_radii)[idx]
        )

    def test_zero_likelihood_resample_size_equals_n_eff(self, tmp_npz, zero_likelihood):
        """With ZeroLikelihood, N_eff == N_EOS, so the default resample size is N_EOS."""
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        out = sampler.sample(jax.random.key(0))
        assert out.metadata["N_resampled"] == N_EOS

    def test_n_resample_override(self, tmp_npz, zero_likelihood):
        cfg = EOSReweightingConfig(
            eos_file=tmp_npz, n_grid=N_GRID, m_min=0.5, n_resample=3
        )
        sampler = EOSReweightingSampler(likelihood=zero_likelihood, config=cfg)
        out = sampler.sample(jax.random.key(0))
        assert out.metadata["N_resampled"] == 3
        assert out.samples["masses_EOS"].shape == (3, N_GRID)


class TestResampleEosPosterior:
    """Unit tests for the standalone, publicly reusable resample_eos_posterior()."""

    @staticmethod
    def _toy_curves(n: int = 4, n_pts: int = 5):
        masses = np.tile(np.linspace(1.0, 2.0, n_pts), (n, 1))
        lambdas = np.arange(n)[:, None] * np.ones((n, n_pts))
        radii = 10.0 + np.arange(n)[:, None] * np.ones((n, n_pts))
        return masses, lambdas, radii

    def test_default_n_samples_is_n_eff(self):
        masses, lambdas, radii = self._toy_curves(n=4)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        out = resample_eos_posterior(masses, lambdas, radii, weights, seed=0)
        # N_eff = 1 / sum(w^2) = 4 for uniform weights over 4 points
        assert out["masses_EOS"].shape[0] == 4

    def test_explicit_n_samples(self):
        masses, lambdas, radii = self._toy_curves(n=4)
        weights = np.array([0.7, 0.1, 0.1, 0.1])
        out = resample_eos_posterior(
            masses, lambdas, radii, weights, n_samples=10, seed=0
        )
        assert out["eos_index"].shape == (10,)
        assert out["masses_EOS"].shape == (10, 5)

    def test_curves_match_indices(self):
        masses, lambdas, radii = self._toy_curves(n=4)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        out = resample_eos_posterior(
            masses, lambdas, radii, weights, n_samples=20, seed=1
        )
        idx = out["eos_index"]
        np.testing.assert_allclose(out["masses_EOS"], masses[idx])
        np.testing.assert_allclose(out["Lambdas_EOS"], lambdas[idx])
        np.testing.assert_allclose(out["radii_EOS"], radii[idx])

    def test_deterministic_given_seed(self):
        masses, lambdas, radii = self._toy_curves(n=4)
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        out1 = resample_eos_posterior(
            masses, lambdas, radii, weights, n_samples=50, seed=7
        )
        out2 = resample_eos_posterior(
            masses, lambdas, radii, weights, n_samples=50, seed=7
        )
        np.testing.assert_array_equal(out1["eos_index"], out2["eos_index"])

    def test_only_high_weight_index_sampled_when_others_are_zero(self):
        masses, lambdas, radii = self._toy_curves(n=4)
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        out = resample_eos_posterior(
            masses, lambdas, radii, weights, n_samples=20, seed=0
        )
        assert np.all(out["eos_index"] == 0)

    def test_unnormalised_weights_are_normalised(self):
        masses, lambdas, radii = self._toy_curves(n=4)
        weights = np.array([10.0, 0.0, 0.0, 0.0])  # not normalised to 1
        out = resample_eos_posterior(
            masses, lambdas, radii, weights, n_samples=20, seed=0
        )
        assert np.all(out["eos_index"] == 0)


class TestEOSReweightingConfig:
    def test_config_validation_bad_batch_size(self):
        with pytest.raises(Exception):
            EOSReweightingConfig(eos_file="x.npz", batch_size=-1, n_grid=50, m_min=0.5)

    def test_small_batch_size_still_correct(self, tmp_npz, zero_likelihood):
        """A batch_size smaller than N_EOS should still produce correct results."""
        cfg = EOSReweightingConfig(
            eos_file=tmp_npz, n_grid=N_GRID, m_min=0.5, batch_size=1
        )
        sampler = EOSReweightingSampler(likelihood=zero_likelihood, config=cfg)
        out = sampler.sample(jax.random.key(0))
        assert out.log_prob.shape == (N_EOS,)

    def test_missing_radii_raises(self, tmp_path, zero_likelihood):
        """NPZ file without 'radii' key must raise a clear ValueError."""
        p = str(tmp_path / "no_radii.npz")
        masses = np.tile(np.linspace(0.5, 2.0, 30), (3, 1))
        np.savez(p, masses=masses, lambdas=np.ones_like(masses) * 500)
        cfg = EOSReweightingConfig(eos_file=p, n_grid=20, m_min=0.5)
        sampler = EOSReweightingSampler(likelihood=zero_likelihood, config=cfg)
        with pytest.raises(ValueError, match="radii"):
            sampler.sample(jax.random.key(0))

    def test_mismatched_shapes_raises(self, tmp_path, zero_likelihood):
        """masses/lambdas/radii with mismatched shapes must raise a clear ValueError."""
        p = str(tmp_path / "ragged.npz")
        masses = np.tile(np.linspace(0.5, 2.0, 30), (3, 1))
        lambdas = np.tile(np.linspace(100, 1000, 20), (3, 1))  # wrong n_points
        radii = np.ones_like(masses) * 12.0
        np.savez(p, masses=masses, lambdas=lambdas, radii=radii)
        cfg = EOSReweightingConfig(eos_file=p, n_grid=20, m_min=0.5)
        sampler = EOSReweightingSampler(likelihood=zero_likelihood, config=cfg)
        with pytest.raises(ValueError, match="mismatched shapes"):
            sampler.sample(jax.random.key(0))

    def test_m_max_cap_only_limits_likelihood_grid_not_posterior(
        self, tmp_path, zero_likelihood
    ):
        """When M_TOV exceeds DEFAULT_M_MAX_CAP, the likelihood grid is capped
        but the resampled posterior curves (used for plotting) must still
        extend out to the true M_TOV of each EOS."""
        n_pts = 50
        m_tov = 4.0  # well above DEFAULT_M_MAX_CAP (3.0)
        masses = np.tile(np.linspace(0.5, m_tov, n_pts), (3, 1))
        lambdas = 500.0 * ((m_tov - masses) / (m_tov - 0.5)) ** 3
        radii = 12.0 * np.ones_like(masses)
        p = str(tmp_path / "stiff.npz")
        np.savez(p, masses=masses, lambdas=lambdas, radii=radii)

        cfg = EOSReweightingConfig(eos_file=p, n_grid=20, m_min=0.5, m_max=None)
        sampler = EOSReweightingSampler(likelihood=zero_likelihood, config=cfg)

        (lik_masses, _, _), (full_masses, _, _) = sampler.load_and_grid(
            [p], 20, 0.5, None
        )
        assert float(lik_masses[0, -1]) == pytest.approx(
            EOSReweightingSampler.DEFAULT_M_MAX_CAP
        )
        assert float(full_masses[0, -1]) == pytest.approx(m_tov)

        out = sampler.sample(jax.random.key(0))
        assert float(np.max(np.asarray(out.samples["masses_EOS"]))) == pytest.approx(
            m_tov
        )

    def test_m_min_only_limits_likelihood_grid_not_posterior(
        self, tmp_path, zero_likelihood
    ):
        """m_min restricts the likelihood grid's lower bound, but the
        resampled posterior curves (used for plotting/output) must still
        extend down to the EOS set's own natural minimum mass."""
        n_pts = 50
        m_natural_min = 0.2
        m_tov = 2.0
        masses = np.tile(np.linspace(m_natural_min, m_tov, n_pts), (3, 1))
        lambdas = 500.0 * ((m_tov - masses) / (m_tov - m_natural_min)) ** 3
        radii = 12.0 * np.ones_like(masses)
        p = str(tmp_path / "wide_range.npz")
        np.savez(p, masses=masses, lambdas=lambdas, radii=radii)

        cfg = EOSReweightingConfig(eos_file=p, n_grid=20, m_min=1.1, m_max=None)
        sampler = EOSReweightingSampler(likelihood=zero_likelihood, config=cfg)

        (lik_masses, _, _), (full_masses, _, _) = sampler.load_and_grid(
            [p], 20, 1.1, None
        )
        assert float(lik_masses[0, 0]) == pytest.approx(1.1)
        assert float(full_masses[0, 0]) == pytest.approx(m_natural_min)

        out = sampler.sample(jax.random.key(0))
        assert float(np.min(np.asarray(out.samples["masses_EOS"]))) == pytest.approx(
            m_natural_min
        )


class TestEOSReweightingLikelihoodValidation:
    """Only likelihoods that depend purely on tabulated M-Lambda-R curves
    (gw, nicer, radio, zero) are accepted in eos-reweighting mode; others
    require EOS-level structure that tabulated curves don't provide."""

    @staticmethod
    def _sampler_config() -> EOSReweightingConfig:
        return EOSReweightingConfig(eos_file="x.npz", n_grid=20, m_min=0.5)

    @pytest.mark.parametrize(
        "likelihood",
        [
            ZeroLikelihoodConfig(),
            GWLikelihoodConfig(events=[GWEventConfig(name="GW170817")]),
            NICERLikelihoodConfig(pulsars=[{"name": "J0030"}]),
            RadioLikelihoodConfig(
                pulsars=[{"name": "J0740+6620", "mass_mean": 2.08, "mass_std": 0.07}]
            ),
        ],
        ids=["zero", "gw", "nicer", "radio"],
    )
    def test_accepted_likelihoods(self, likelihood):
        cfg = EOSReweightingInferenceConfig(
            likelihoods=[likelihood], sampler=self._sampler_config()
        )
        assert cfg.likelihoods[0].type == likelihood.type

    @pytest.mark.parametrize(
        "likelihood",
        [
            GWResampledLikelihoodConfig(events=[{"name": "GW170817"}]),
            NICERKDELikelihoodConfig(
                pulsars=[
                    {
                        "name": "J0030",
                        "amsterdam_samples_file": "a.npz",
                        "maryland_samples_file": "m.npz",
                    }
                ]
            ),
            ChiEFTLikelihoodConfig(),
            EOSConstraintsLikelihoodConfig(),
            TOVConstraintsLikelihoodConfig(),
            EsymConstraintsLikelihoodConfig(),
            GammaConstraintsLikelihoodConfig(),
        ],
        ids=[
            "gw_resampled",
            "nicer_kde",
            "chieft",
            "constraints_eos",
            "constraints_tov",
            "constraints_esym",
            "constraints_gamma",
        ],
    )
    def test_rejected_likelihoods(self, likelihood):
        with pytest.raises(ValueError, match="not supported by EOS reweighting"):
            EOSReweightingInferenceConfig(
                likelihoods=[likelihood], sampler=self._sampler_config()
            )

    def test_disabled_incompatible_likelihood_is_not_rejected(self):
        """A disabled likelihood should not trigger the type-compatibility check."""
        cfg = EOSReweightingInferenceConfig(
            likelihoods=[
                ZeroLikelihoodConfig(),
                ChiEFTLikelihoodConfig(enabled=False),
            ],
            sampler=self._sampler_config(),
        )
        assert len(cfg.likelihoods) == 2

    def test_rejects_multiple_incompatible_types_together(self):
        with pytest.raises(ValueError, match="not supported by EOS reweighting"):
            EOSReweightingInferenceConfig(
                likelihoods=[
                    ChiEFTLikelihoodConfig(),
                    EOSConstraintsLikelihoodConfig(),
                ],
                sampler=self._sampler_config(),
            )


class TestEOSReweightingHDF5:
    def test_evidence_saved_and_loaded(self, tmp_path, tmp_npz, zero_likelihood):
        """Evidence scalars must survive the HDF5 save/load round-trip."""
        from jesterTOV.inference.result import InferenceResult

        sampler = _make_sampler(tmp_npz, zero_likelihood)
        output = sampler.sample(jax.random.key(0))

        # Build a minimal config-like object
        class _Cfg:
            seed = 0

            def model_dump(self):
                return {"seed": 0}

        result = InferenceResult.from_eos_reweighting(output, _Cfg(), runtime=1.0)
        h5_path = tmp_path / "result.h5"
        result.save(h5_path)

        loaded = InferenceResult.load(h5_path)
        assert np.isfinite(loaded.metadata["log_Z"]), "log_Z must be saved and finite"
        assert loaded.metadata["N_eff"] > 0, "N_eff must be saved and positive"
        assert np.isfinite(loaded.metadata["log_Z_std"])
        assert np.isfinite(loaded.metadata["N_eff_fraction"])
