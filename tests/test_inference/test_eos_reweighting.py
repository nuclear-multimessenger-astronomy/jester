"""Unit tests for EOSReweightingSampler."""

from pathlib import Path

import jax
import numpy as np
import pytest

from jesterTOV.inference.likelihoods.combined import ZeroLikelihood
from jesterTOV.inference.samplers.eos_reweighting import EOSReweightingSampler
from jesterTOV.inference.config.schemas.samplers import EOSReweightingConfig

jax.config.update("jax_enable_x64", True)

N_EOS = 5
N_GRID = 30


def _make_dummy_npz(tmp_dir: Path, n_eos: int = N_EOS, n_pts: int = 50) -> str:
    """Write a minimal NPZ file with toy EOS curves."""
    masses = np.tile(np.linspace(0.5, 2.0, n_pts), (n_eos, 1))
    lambdas = np.random.default_rng(0).uniform(100, 1000, (n_eos, 1)) * (
        (2.0 - masses) / 1.5
    ) ** 3
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
        batch_size=None,
        n_bootstrap=20,
    )
    return EOSReweightingSampler(likelihood=likelihood, config=cfg)


class TestEOSReweightingSamplerBasic:
    def test_output_shapes(self, tmp_npz, zero_likelihood):
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        key = jax.random.key(0)
        out = sampler.sample(key)

        assert out.log_prob.shape == (N_EOS,), "log_prob must be [N_eos]"
        assert out.samples["eos_index"].shape == (N_EOS,)
        assert out.samples["log_weight"].shape == (N_EOS,)
        assert out.samples["posterior_weight"].shape == (N_EOS,)

    def test_evidence_finite(self, tmp_npz, zero_likelihood):
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        out = sampler.sample(jax.random.key(0))

        ev = out.metadata["evidence"]
        assert np.isfinite(ev["log_Z"]), "log_Z must be finite"
        assert ev["N_eff"] > 0, "N_eff must be positive"
        assert 0.0 < ev["N_eff_fraction"] <= 1.0

    def test_zero_likelihood_uniform_weights(self, tmp_npz, zero_likelihood):
        """ZeroLikelihood → all log-weights equal → N_eff = N."""
        sampler = _make_sampler(tmp_npz, zero_likelihood)
        out = sampler.sample(jax.random.key(0))

        lw = np.array(out.samples["log_weight"])
        assert np.allclose(lw, lw[0], atol=1e-6), "ZeroLikelihood must give equal weights"
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

class TestEOSReweightingConfig:
    def test_config_validation_bad_batch_size(self):
        with pytest.raises(Exception):
            EOSReweightingConfig(eos_file="x.npz", batch_size=-1, n_grid=50, m_min=0.5)

    def test_config_validation_bad_progress_interval(self):
        with pytest.raises(Exception):
            EOSReweightingConfig(eos_file="x.npz", progress_interval=1.5, n_grid=50, m_min=0.5)
        with pytest.raises(Exception):
            EOSReweightingConfig(eos_file="x.npz", progress_interval=-0.1, n_grid=50, m_min=0.5)

    def test_progress_interval_zero_disables_progress(self, tmp_npz, zero_likelihood):
        """progress_interval=0 should still produce correct results."""
        cfg = EOSReweightingConfig(
            eos_file=tmp_npz, n_grid=N_GRID, m_min=0.5, n_bootstrap=5, progress_interval=0.0
        )
        sampler = EOSReweightingSampler(likelihood=zero_likelihood, config=cfg)
        out = sampler.sample(jax.random.key(0))
        assert out.log_prob.shape == (N_EOS,)

    def test_missing_radii_raises(self, tmp_path, zero_likelihood):
        """NPZ file without 'radii' key must raise a clear ValueError."""
        p = str(tmp_path / "no_radii.npz")
        masses = np.tile(np.linspace(0.5, 2.0, 30), (3, 1))
        np.savez(p, masses=masses, lambdas=np.ones_like(masses) * 500)
        cfg = EOSReweightingConfig(eos_file=p, n_grid=20, m_min=0.5, n_bootstrap=5)
        sampler = EOSReweightingSampler(likelihood=zero_likelihood, config=cfg)
        with pytest.raises(ValueError, match="radii"):
            sampler.sample(jax.random.key(0))


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
