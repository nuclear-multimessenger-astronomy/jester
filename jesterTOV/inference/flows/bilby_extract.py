"""Extract GW posterior samples from bilby result HDF5 files.

This module provides utilities to read bilby result files and extract
the parameters needed for jester's GW likelihood: ``mass_1_source``,
``mass_2_source``, ``lambda_1``, and ``lambda_2``.

Bilby serialises its ``Result`` object to HDF5 using a recursive
dict-to-group mapping (``recursively_save_dict_contents_to_group``).
The ``posterior`` attribute — a pandas DataFrame
of reweighted samples — is stored as an HDF5 group with one dataset per
parameter column.  This is the canonical source for derived parameters such
as ``mass_1_source`` and ``lambda_1``.

Note: the file also contains a ``samples`` dataset (raw nested-sampling live
points) and ``search_parameter_keys``, but those are unweighted sampler outputs
and do not include derived quantities.  This module reads only the ``posterior``
group.

No bilby installation is required.
"""

import argparse
from pathlib import Path

import numpy as np

from jesterTOV.logging_config import get_logger


logger = get_logger("jester")

# Parameters required in the final .npz output
_REQUIRED_OUTPUT_PARAMS: list[str] = [
    "mass_1_source",
    "mass_2_source",
    "lambda_1",
    "lambda_2",
]


# ---------------------------------------------------------------------------
# HDF5 reading
# ---------------------------------------------------------------------------


def _read_bilby_hdf5(filepath: str) -> dict[str, np.ndarray]:
    """Read the ``posterior`` group from a bilby result HDF5 file.

    Bilby stores the reweighted posterior as an HDF5 group with one dataset
    per parameter column (``f["posterior"]["<param>"]``).  This is the layout
    produced by ``recursively_save_dict_contents_to_group`` since bilby 1.1.0.

    Parameters
    ----------
    filepath : str
        Path to the bilby result ``.hdf5`` file.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from parameter name to 1-D array of posterior samples.

    Raises
    ------
    ValueError
        If the file does not contain a ``posterior`` group.
    """
    import h5py

    with h5py.File(filepath, "r") as f:
        if "posterior" not in f or not isinstance(f["posterior"], h5py.Group):
            raise ValueError(
                f"No 'posterior' group found in '{filepath}'. "
                "Expected a bilby result file saved with bilby >= 1.1.0. "
                f"Available top-level keys: {list(f.keys())}"
            )
        posterior = f["posterior"]
        assert isinstance(posterior, h5py.Group)
        return {key: np.array(posterior[key]) for key in posterior.keys()}  # type: ignore[index]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_gw_posterior_from_bilby(
    bilby_result_file: str,
    output_file: str | None = None,
) -> str:
    """Extract GW posterior samples from a bilby result HDF5 file.

    Reads ``mass_1_source``, ``mass_2_source``, ``lambda_1``, and
    ``lambda_2`` from a bilby result file and saves them as a ``.npz`` file
    suitable for use with :class:`~jesterTOV.inference.flows.config.FlowTrainingConfig`.

    All four parameters must be present in the bilby result file.  Bilby
    writes them directly for BNS analyses, so no parameter conversion is
    performed here.

    Parameters
    ----------
    bilby_result_file : str
        Path to bilby result ``.hdf5`` file.
    output_file : str | None
        Output ``.npz`` path.  Defaults to the same directory as the input
        with a ``_gw_jester_posterior.npz`` suffix appended to the stem.

    Returns
    -------
    str
        Path to the saved ``.npz`` file.

    Raises
    ------
    KeyError
        If a required parameter is absent from the bilby result file.
    ValueError
        If the HDF5 file does not contain a ``posterior`` group.
    """
    bilby_result_file = str(bilby_result_file)

    # Determine default output path
    if output_file is None:
        stem = Path(bilby_result_file).stem
        output_file = str(
            Path(bilby_result_file).parent / f"{stem}_gw_jester_posterior.npz"
        )

    logger.info(f"Reading bilby result from {bilby_result_file}")
    params = _read_bilby_hdf5(bilby_result_file)
    logger.info(f"Found {len(next(iter(params.values())))} posterior samples")
    logger.info(f"Available parameters: {sorted(params.keys())}")

    # Validate required output parameters
    for key in _REQUIRED_OUTPUT_PARAMS:
        if key not in params:
            raise KeyError(
                f"Required parameter '{key}' not found in bilby result file "
                f"'{bilby_result_file}'. "
                f"Available parameters: {sorted(params.keys())}"
            )

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save NPZ with exactly the required keys
    np.savez(
        output_file,
        mass_1_source=params["mass_1_source"],
        mass_2_source=params["mass_2_source"],
        lambda_1=params["lambda_1"],
        lambda_2=params["lambda_2"],
    )

    logger.info(f"Saved GW posterior samples to {output_file}")
    return str(output_file)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for extracting GW posteriors from bilby results.

    Usage::

        jester_extract_gw_posterior_bilby result.hdf5 [--output out.npz]
    """
    parser = argparse.ArgumentParser(
        prog="jester_extract_gw_posterior_bilby",
        description=(
            "Extract mass_1_source, mass_2_source, lambda_1, lambda_2 from a "
            "bilby result HDF5 file and save them as a .npz file for use with "
            "jester's GW flow training pipeline."
        ),
    )
    parser.add_argument(
        "bilby_result_file",
        type=str,
        help="Path to bilby HDF5 result file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npz file path.  Defaults to the same directory as the "
            "input with '_gw_jester_posterior.npz' appended to the stem."
        ),
    )
    args = parser.parse_args()
    output_path = extract_gw_posterior_from_bilby(
        bilby_result_file=args.bilby_result_file,
        output_file=args.output,
    )
    logger.info(f"Saved: {output_path}")
