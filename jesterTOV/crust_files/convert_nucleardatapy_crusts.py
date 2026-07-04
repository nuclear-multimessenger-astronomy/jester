"""
Convert crust models from the `nucleardatapy` toolkit into the ``.npz`` format used by
:class:`jesterTOV.eos.crust.Crust`.

``nucleardatapy`` is **not** a jester dependency: it is a large toolkit for nuclear
data/meta-analyses, only needed here as a one-off to regenerate these crust files. To run
this script, install it directly from its main branch (validated here against the state
of that branch tagged as version 1.1.0)::

    pip install "nucleardatapy @ git+https://github.com/jeromemargueron/nucleardatapy.git@main"

Then run, e.g.::

    python convert_nucleardatapy_crusts.py

Unit conventions
-----------------
``nucleardatapy`` reports number density ``den`` in fm^-3 and energy density
``eps_tot = e2a_tot * den`` in MeV/fm^3 directly from the tabulated per-nucleon energy, so
no further unit conversion is required for those two quantities.

Pressure requires more care. ``nucleardatapy`` recomputes a pressure ``pre_tot`` for
*every* model via a cubic-spline derivative of the energy vs. density curve, regardless
of whether the underlying data file already tabulates pressure directly. Comparing the
two for models that do tabulate pressure (e.g. 2022-GMRS-BSK14) shows ``pre_tot`` is
non-monotonic near the crust-core transition with relative deviations up to ~90% from the
raw tabulated column, presumably from spline noise given the number/spacing of points.
For any model that tabulates pressure directly (the ``pre`` attribute), this script uses
that raw column instead of ``pre_tot``. Models lacking a tabulated pressure column (only
Negele-Vautherin, in practice) have no choice but to fall back to the derived ``pre_tot``,
so they are excluded by default (see ``MODELS`` below) and should be treated as lower
confidence if enabled.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.interpolate import interp1d

CRUST_DIR = os.path.dirname(os.path.abspath(__file__))

# nucleardatapy model name -> (output name used for the .npz file, enabled by default).
# Toggle individual entries on/off here as needed; disabled models are skipped unless
# explicitly requested via --models.
MODELS: dict[str, tuple[str, bool]] = {
    # Coarse (11-point), no tabulated pressure column -> falls back to the unreliable
    # spline-derived pre_tot. Kept off by default; opt in via --models if needed.
    "1973-Negele-Vautherin": ("NV_1973", False),
    # Broken in nucleardatapy 1.1: raises "invalid column index 28 at row 1 with 28
    # columns" while parsing the upstream data file. Left here so the fix is trivial to
    # pick up if/when it's fixed upstream.
    "2018-PCPFDDG-BSK22": ("PCPFDDG_BSK22", False),
    "2018-PCPFDDG-BSK24": ("PCPFDDG_BSK24", False),
    "2018-PCPFDDG-BSK25": ("PCPFDDG_BSK25", False),
    "2018-PCPFDDG-BSK26": ("PCPFDDG_BSK26", False),
    # Coarse (11-13 point) tables, but tabulate pressure directly.
    "2020-MVCD-D1S": ("MVCD_D1S", True),
    "2020-MVCD-D1M": ("MVCD_D1M", True),
    "2020-MVCD-D1MS": ("MVCD_D1MS", True),
    # Dense (~1300-1650 point) tables, tabulate pressure directly.
    "2022-GMRS-BSK14": ("GMRS_BSK14", True),
    "2022-GMRS-BSK16": ("GMRS_BSK16", True),
    "2022-GMRS-DHSL59": ("GMRS_DHSL59", True),
    "2022-GMRS-DHSL69": ("GMRS_DHSL69", True),
    "2022-GMRS-F0": ("GMRS_F0", True),
    "2022-GMRS-H1": ("GMRS_H1", True),
    "2022-GMRS-H2": ("GMRS_H2", True),
    "2022-GMRS-H3": ("GMRS_H3", True),
    "2022-GMRS-H4": ("GMRS_H4", True),
    "2022-GMRS-H5": ("GMRS_H5", True),
    "2022-GMRS-H7": ("GMRS_H7", True),
    "2022-GMRS-LNS5": ("GMRS_LNS5", True),
    "2022-GMRS-RATP": ("GMRS_RATP", True),
    "2022-GMRS-SGII": ("GMRS_SGII", True),
    "2022-GMRS-SLY5": ("GMRS_SLY5", True),
}

# Flag causality/kink issues, but only hard-fail the whole conversion for a model if the
# minimal structural requirements (positive, monotonic, non-empty) are violated.
CS2_MAX = 1.0
GAMMA_JUMP_THRESHOLD = 5.0  # flag |change in log-log slope| beyond this between points


def _import_nucleardatapy():
    try:
        import nucleardatapy as nuda
    except ImportError as exc:
        raise ImportError(
            "This script requires the 'nucleardatapy' package, which is not a jester "
            "dependency and must be installed separately as a one-off:\n"
            '  pip install "nucleardatapy @ '
            'git+https://github.com/jeromemargueron/nucleardatapy.git@main"\n'
            "(validated here against the state of that branch tagged as version 1.1.0)"
        ) from exc
    return nuda


def _report_gamma_jumps(n: np.ndarray, p: np.ndarray, label: str) -> None:
    log_n = np.log(n)
    log_p = np.log(p)
    gamma = np.diff(log_p) / np.diff(log_n)
    jumps = np.abs(np.diff(gamma))
    bad = np.where(jumps > GAMMA_JUMP_THRESHOLD)[0]
    for i in bad:
        print(
            f"  [warn] {label}: steep change in log-log slope (Gamma) near "
            f"n={n[i + 1]:.4e} fm^-3 (Delta Gamma={jumps[i]:.2f})"
        )


def _report_causality(n: np.ndarray, p: np.ndarray, e: np.ndarray, label: str) -> None:
    cs2 = np.gradient(p, e)
    bad = np.where((cs2 <= 0) | (cs2 > CS2_MAX))[0]
    for i in bad:
        print(
            f"  [warn] {label}: cs2={cs2[i]:.4e} out of (0, {CS2_MAX}] at "
            f"n={n[i]:.4e} fm^-3"
        )


def convert_model(
    nuda,
    nuda_name: str,
    interpolate: bool = True,
    ndat: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single crust model from nucleardatapy and return (n, e, p) in jester units."""
    crust = nuda.crust.setupCrust(model=nuda_name)

    n = np.asarray(crust.den, dtype=float)  # fm^-3
    e = np.asarray(crust.eps_tot, dtype=float)  # MeV/fm^3, direct e2a_tot * den product

    if hasattr(crust, "pre"):
        p = np.asarray(crust.pre, dtype=float)  # MeV/fm^3, tabulated directly
    else:
        print(
            f"  [warn] {nuda_name}: no tabulated pressure column; falling back to "
            "nucleardatapy's spline-derived pre_tot, which is known to be noisy"
        )
        p = np.asarray(crust.pre_tot, dtype=float)

    # Sort and de-duplicate by density, drop non-positive pressure/energy points (mirrors
    # jesterTOV.eos.crust.Crust._preprocess so bad data fails here, not at inference time).
    order = np.argsort(n)
    n, e, p = n[order], e[order], p[order]
    mask = (p > 0) & (e > 0)
    n, e, p = n[mask], e[mask], p[mask]
    n, unique_idx = np.unique(n, return_index=True)
    e, p = e[unique_idx], p[unique_idx]

    if n.size == 0:
        raise ValueError(f"{nuda_name}: no valid crust points remain after filtering")
    if not np.all(np.diff(n) > 0):
        raise ValueError(
            f"{nuda_name}: density is not strictly monotonic after filtering"
        )

    _report_gamma_jumps(n, p, f"{nuda_name} (raw)")
    _report_causality(n, p, e, f"{nuda_name} (raw)")

    if interpolate:
        n_interp = np.logspace(np.log10(n[0]), np.log10(n[-1]), ndat)
        # Clip away floating-point round-trip error through log10/logspace so the
        # endpoints stay strictly within [n[0], n[-1]] for interp1d's bounds check.
        n_interp = np.clip(n_interp, n[0], n[-1])
        e = interp1d(n, e, kind="cubic")(n_interp)
        p = interp1d(n, p, kind="cubic")(n_interp)
        n = n_interp

        if not np.all(np.diff(p) > 0):
            print(
                f"  [warn] {nuda_name}: pressure is not monotonic after interpolation"
            )
        _report_gamma_jumps(n, p, f"{nuda_name} (interpolated)")
        _report_causality(n, p, e, f"{nuda_name} (interpolated)")

    return n, e, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of nucleardatapy model names to convert (default: all enabled "
        "entries in MODELS).",
    )
    parser.add_argument(
        "--output-dir",
        default=CRUST_DIR,
        help="Directory to write .npz files to (default: this script's directory).",
    )
    parser.add_argument(
        "--ndat", type=int, default=100, help="Interpolation grid size."
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Save the filtered raw tables without re-interpolating onto a smooth grid.",
    )
    args = parser.parse_args()

    nuda = _import_nucleardatapy()

    if args.models is not None:
        selected = {
            name: MODELS.get(name, (name.replace("-", "_"), True))[0]
            for name in args.models
        }
    else:
        selected = {name: out for name, (out, enabled) in MODELS.items() if enabled}

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Converting {len(selected)} crust model(s)...")
    results = []
    for nuda_name, out_name in selected.items():
        print(f"\n{nuda_name} -> {out_name}.npz")
        try:
            n, e, p = convert_model(
                nuda, nuda_name, interpolate=not args.no_interpolate, ndat=args.ndat
            )
        except Exception as exc:  # noqa: BLE001 - report and continue with other models
            print(f"  [error] skipping {nuda_name}: {exc}")
            continue

        out_path = os.path.join(args.output_dir, f"{out_name}.npz")
        np.savez(out_path, n=n, e=e, p=p)
        results.append((out_name, n[0], n[-1], len(n)))
        print(
            f"  saved {out_path} ({len(n)} points, n in [{n[0]:.3e}, {n[-1]:.3e}] fm^-3)"
        )

    print("\nSummary:")
    for out_name, n_min, n_max, npts in results:
        print(f"  {out_name:15s} npts={npts:4d} n=[{n_min:.3e}, {n_max:.3e}] fm^-3")


if __name__ == "__main__":
    main()
