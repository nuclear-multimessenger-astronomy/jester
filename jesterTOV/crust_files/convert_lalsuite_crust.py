"""
Extract a crust table from a LALSuite-tabulated full-EOS ``.npz`` file (as found under
``jesterTOV/tabulated_eos/lalsuite/``) and convert it to the ``.npz`` format used by
:class:`jesterTOV.eos.crust.Crust`.

Unit conventions
-----------------
The LALSuite tabulated-EOS files store ``n``, ``p``, ``e`` in jester's internal geometric
units (see ``jesterTOV/utils.py``), not in fm^-3 / MeV/fm^3 like the crust files. This
script converts via ``utils.geometric_to_fm_inv3`` / ``utils.geometric_to_MeV_fm_inv3`` and
truncates the table at a maximum density (by default 0.75 nsat, with nsat = 0.16 fm^-3)
low enough that it comfortably covers the crust-core matching window used by the meta-model
EOS (default ``max_n_crust_nsat = 0.5``, see ``jesterTOV/eos/metamodel/base.py``).

Usage::

    python convert_lalsuite_crust.py HQC18
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from jesterTOV import utils

LALSUITE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tabulated_eos",
    "lalsuite",
)
CRUST_DIR = os.path.dirname(os.path.abspath(__file__))

NSAT = 0.16  # fm^-3, standard nuclear saturation density convention used throughout jester


def convert_model(
    source_path: str, max_n_nsat: float = 0.75
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a LALSuite full-EOS table and return the crust part as (n, e, p) in jester
    crust units (fm^-3, MeV/fm^3, MeV/fm^3)."""
    data = np.load(source_path)
    n = np.asarray(data["n"], dtype=float) * utils.geometric_to_fm_inv3
    p = np.asarray(data["p"], dtype=float) * utils.geometric_to_MeV_fm_inv3
    e = np.asarray(data["e"], dtype=float) * utils.geometric_to_MeV_fm_inv3

    mask = (p > 0) & (e > 0) & (n <= max_n_nsat * NSAT)
    n, e, p = n[mask], e[mask], p[mask]

    if n.size == 0:
        raise ValueError("No crust points remain after filtering")
    if not np.all(np.diff(n) > 0):
        raise ValueError("Density is not strictly monotonic after filtering")

    return n, e, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name", help="Base name of the source file under tabulated_eos/lalsuite/, e.g. HQC18"
    )
    parser.add_argument(
        "--max-n-nsat",
        type=float,
        default=0.75,
        help="Maximum density to keep, in units of nsat=0.16 fm^-3 (default: 0.75).",
    )
    parser.add_argument(
        "--output-dir",
        default=CRUST_DIR,
        help="Directory to write the .npz file to (default: this script's directory).",
    )
    args = parser.parse_args()

    source_path = os.path.join(LALSUITE_DIR, f"{args.name}.npz")
    n, e, p = convert_model(source_path, max_n_nsat=args.max_n_nsat)

    out_path = os.path.join(args.output_dir, f"{args.name}.npz")
    np.savez(out_path, n=n, e=e, p=p)
    print(
        f"saved {out_path} ({len(n)} points, n in [{n[0]:.3e}, {n[-1]:.3e}] fm^-3, "
        f"i.e. up to {n[-1] / NSAT:.3f} nsat)"
    )


if __name__ == "__main__":
    main()
