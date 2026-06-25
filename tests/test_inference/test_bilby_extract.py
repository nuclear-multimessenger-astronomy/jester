"""Tests for jesterTOV.inference.flows.bilby_extract."""

import numpy as np
import pytest
import h5py
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers to create synthetic bilby-like HDF5 files
# ---------------------------------------------------------------------------


def _make_new_format_hdf5(
    path: Path,
    params: dict[str, np.ndarray],
) -> None:
    """Write a new-format bilby HDF5 (posterior group) to *path*."""
    with h5py.File(path, "w") as f:
        grp = f.create_group("posterior")
        for key, values in params.items():
            grp.create_dataset(key, data=values)


# Shared posterior data used across several tests
N = 200
_RNG = np.random.default_rng(0)

_M1_SOURCE = _RNG.uniform(1.1, 1.6, N)
_M2_SOURCE = _RNG.uniform(0.9, _M1_SOURCE)
_LAMBDA_1 = _RNG.uniform(0.0, 2000.0, N)
_LAMBDA_2 = _RNG.uniform(0.0, 2000.0, N)

_FULL_PARAMS = {
    "mass_1_source": _M1_SOURCE,
    "mass_2_source": _M2_SOURCE,
    "lambda_1": _LAMBDA_1,
    "lambda_2": _LAMBDA_2,
}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestReadHDF5:
    """Tests for HDF5 file reading."""

    def test_read_posterior_group(self, tmp_path: Path) -> None:
        """posterior group is read correctly into a parameter dict."""
        hdf5_path = tmp_path / "result.hdf5"
        _make_new_format_hdf5(hdf5_path, _FULL_PARAMS)

        from jesterTOV.inference.flows.bilby_extract import _read_bilby_hdf5

        params = _read_bilby_hdf5(str(hdf5_path))
        assert set(params.keys()) == set(_FULL_PARAMS.keys())
        np.testing.assert_allclose(params["mass_1_source"], _M1_SOURCE)
        np.testing.assert_allclose(params["lambda_2"], _LAMBDA_2)

    def test_missing_posterior_group_raises(self, tmp_path: Path) -> None:
        """File without a posterior group raises ValueError."""
        hdf5_path = tmp_path / "no_posterior.hdf5"
        # Write only a samples dataset (no posterior group)
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("some_data", data=np.zeros(10))

        from jesterTOV.inference.flows.bilby_extract import _read_bilby_hdf5

        with pytest.raises(ValueError, match="posterior"):
            _read_bilby_hdf5(str(hdf5_path))


class TestExtractAPI:
    """Tests for the extract_gw_posterior_from_bilby public function."""

    def test_extract_saves_npz_with_correct_keys(self, tmp_path: Path) -> None:
        """Output NPZ contains exactly mass_1_source, mass_2_source, lambda_1, lambda_2."""
        hdf5_path = tmp_path / "result.hdf5"
        _make_new_format_hdf5(hdf5_path, _FULL_PARAMS)

        from jesterTOV.inference.flows.bilby_extract import (
            extract_gw_posterior_from_bilby,
        )

        out_path = tmp_path / "posterior.npz"
        extract_gw_posterior_from_bilby(str(hdf5_path), output_file=str(out_path))

        assert out_path.exists()
        data = np.load(out_path)
        assert set(data.files) == {
            "mass_1_source",
            "mass_2_source",
            "lambda_1",
            "lambda_2",
        }

    def test_extract_values_match_input(self, tmp_path: Path) -> None:
        """Values in the output NPZ exactly match the input arrays."""
        hdf5_path = tmp_path / "result.hdf5"
        _make_new_format_hdf5(hdf5_path, _FULL_PARAMS)

        from jesterTOV.inference.flows.bilby_extract import (
            extract_gw_posterior_from_bilby,
        )

        out_path = tmp_path / "posterior.npz"
        extract_gw_posterior_from_bilby(str(hdf5_path), output_file=str(out_path))

        data = np.load(out_path)
        np.testing.assert_allclose(data["mass_1_source"], _M1_SOURCE)
        np.testing.assert_allclose(data["mass_2_source"], _M2_SOURCE)
        np.testing.assert_allclose(data["lambda_1"], _LAMBDA_1)
        np.testing.assert_allclose(data["lambda_2"], _LAMBDA_2)

    def test_extract_default_output_path(self, tmp_path: Path) -> None:
        """Omitting output_file generates a path in the same dir with correct suffix."""
        hdf5_path = tmp_path / "GW170817_result.hdf5"
        _make_new_format_hdf5(hdf5_path, _FULL_PARAMS)

        from jesterTOV.inference.flows.bilby_extract import (
            extract_gw_posterior_from_bilby,
        )

        returned_path = extract_gw_posterior_from_bilby(str(hdf5_path))

        expected = str(tmp_path / "GW170817_result_gw_jester_posterior.npz")
        assert returned_path == expected
        assert Path(returned_path).exists()

    def test_extract_missing_required_param_raises(self, tmp_path: Path) -> None:
        """Missing lambda_1 raises KeyError with a descriptive message."""
        params_no_lambda1 = {k: v for k, v in _FULL_PARAMS.items() if k != "lambda_1"}
        hdf5_path = tmp_path / "missing.hdf5"
        _make_new_format_hdf5(hdf5_path, params_no_lambda1)

        from jesterTOV.inference.flows.bilby_extract import (
            extract_gw_posterior_from_bilby,
        )

        out_path = tmp_path / "out.npz"
        with pytest.raises(KeyError, match="lambda_1"):
            extract_gw_posterior_from_bilby(str(hdf5_path), output_file=str(out_path))
