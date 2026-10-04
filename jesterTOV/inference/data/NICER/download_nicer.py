#!/usr/bin/env python3
"""
Download, extract, and downsample NICER mass-radius posterior samples.

Pipeline (all controlled by constants below):

  Step 1  Download Zenodo archives for Maryland and Amsterdam groups.
          (requires ``zenodo_get``: ``uv pip install zenodo-get``)
  Step 2  Extract Maryland text files → .npz with mass/radius/metadata.
          J0437 and J0614 (Miller, Dittmann, Holt et al. 2026) and J0740
          (Dittmann et al. 2024) are importance-resampled using their weight
          column to obtain an equal-weight posterior.
  Step 3  Extract Amsterdam tar.gz archives → .npz, plus the direct-download
          Kini, Mauviard, Salmi et al. 2026 J0030 (PDT-U) equal-weight file,
          the Riley et al. 2021 J0740 (ST-U) MultiNest run archive, and the
          Vinciguerra et al. 2023 J0030 (ST-U/ST+PST/ST+PDT/PDT-U) reference
          runs from the full reproduction-package archive, and the Mauviard
          et al. 2026 J1614-2230 (ST-U) headline run.
  Step 4  Downsample all .npz files to MAX_SAMPLES.

Outputs are written to the same directory as this script (NICER/).
Raw Zenodo archives are cached in NICER/zenodo_data/ (gitignored).
"""

import tarfile
import tempfile
from urllib.request import urlretrieve
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from zenodo_downloader import ZenodoDownloader

# ============================================================
# USER CONFIGURATION - Tweak these constants as needed
# ============================================================

# Re-download/re-extract even if output files already exist
IGNORE_CACHE: bool = False

# Download Zenodo archives (large GB-scale files).
# Set False only if you have already downloaded them.
DOWNLOAD_ZENODO: bool = True

# Extract Amsterdam tar.gz archives (large, may take several minutes).
EXTRACT_AMSTERDAM: bool = True

# Deliberately NOT fetched (known gaps in the set of public NICER M-R samples):
#   - PSR J1231-1411 (Salmi et al. 2024, Zenodo 13358349): known from private
#     communication that it is probably not a good idea to use these samples
#     for EOS inference, so we leave them out. (The posteriors also depend
#     strongly on the radius prior; see the paper, "A Complex Case".)
#   - PSR J2124-3358 (Gonzalez-Caniulef et al. 2026, Zenodo 20640366): only
#     credible-region contours are public so far; the posterior samples are to
#     be released upon acceptance of the paper.

# Downsample each .npz to at most this many samples (None = no limit)
MAX_SAMPLES: int | None = 100_000

# ============================================================

SCRIPT_DIR = Path(__file__).parent
ZENODO_DIR = SCRIPT_DIR / "zenodo_data"
OUTPUT_DIR = SCRIPT_DIR


# ============================================================
# Zenodo downloading
# ============================================================


def download_zenodo_data() -> None:
    """Download all required Zenodo archives.

    Most datasets use ``zenodo_get`` (downloads the full record).
    J0614 (both groups) is an exception: only one small file is needed per
    group, so those are fetched directly from the Zenodo file URL instead.
    """
    downloader = ZenodoDownloader(base_dir=ZENODO_DIR)

    # Full-record downloads via zenodo_get
    datasets_to_download = [
        ("J0030", "maryland", "original"),
        ("J0740", "maryland", "original"),
        ("J0030", "amsterdam", "original"),
        ("J0740", "amsterdam", "recent"),
        ("J0030", "amsterdam", "intermediate"),
    ]
    for psr, group, version in datasets_to_download:
        print(f"\nZenodo: {psr}/{group}/{version}")
        downloader.download_dataset(psr, group, version, force=IGNORE_CACHE)

    # J0437 and J0614: download only the small headline archive (direct URL, no zenodo_get)
    _direct_downloads = [
        (
            "J0437/amsterdam/original",
            "headline_result_samples_and_contours.tar.gz",
            "https://zenodo.org/records/13766753/files/headline_result_samples_and_contours.tar.gz",
        ),
        (
            "J0614/amsterdam/original",
            "Headline_Contours_and_Samples.tar.gz",
            "https://zenodo.org/records/17380576/files/Headline_Contours_and_Samples.tar.gz",
        ),
        (
            "J0437/maryland/original",
            "J0437_NICER_RM.txt",
            "https://zenodo.org/records/17833896/files/J0437_NICER_RM.txt",
        ),
        (
            "J0614/maryland/original",
            "J0614_NICER_rm.txt",
            "https://zenodo.org/records/22131748/files/J0614_NICER_rm.txt",
        ),
        (
            "J0030/amsterdam/recent",
            "equal_weight_samples_PDTU.txt",
            "https://zenodo.org/records/18741942/files/equal_weight_samples_PDTU.txt",
        ),
        (
            "J1614/amsterdam/original",
            "MR_samples_and_contours_J1614.tar.gz",
            "https://zenodo.org/records/22163155/files/MR_samples_and_contours_J1614.tar.gz",
        ),
        (
            "J0740/amsterdam/original",
            "STU_NICERxXMM_FIH_run11.tar.gz",
            "https://zenodo.org/records/7096886/files/STU_NICERxXMM_FIH_run11.tar.gz",
        ),
        # Only the headline M-R file (~70 MB) of the ~3.5 GB Dittmann et al. record
        (
            "J0740/maryland/recent",
            "J0740_NICERXMM_full_mr.txt",
            "https://zenodo.org/records/10215109/files/J0740_NICERXMM_full_mr.txt",
        ),
    ]
    for rel_dir, filename, url in _direct_downloads:
        dest_dir = ZENODO_DIR / rel_dir
        dest_file = dest_dir / filename
        if not dest_file.exists() or IGNORE_CACHE:
            dest_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nDownloading {dest_file.parent.parent.parent.name} archive: {url}")
            urlretrieve(url, dest_file)
            print(
                f"  Saved: {dest_file.name} ({dest_file.stat().st_size / 1024:.1f} KB)"
            )
        else:
            print(
                f"\n{dest_file.parent.parent.parent.name} archive already cached: {dest_file.name}"
            )


# ============================================================
# Maryland extraction
# ============================================================


def parse_maryland_txt(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Parse a Maryland group text file (columns: radius km, mass Msun, weight)."""
    print(f"\n  Parsing: {filepath.name}")
    data = np.loadtxt(filepath, comments="#")

    radius = data[:, 0]
    mass = data[:, 1]

    header_lines: list[str] = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.strip())
            else:
                break

    fname = filepath.stem
    psr = (
        "J0030+0451"
        if "J0030" in fname
        else ("J0740+6620" if "J0740" in fname else "Unknown")
    )

    if "2spot" in fname:
        hotspot = "2spot"
    elif "3spot" in fname:
        hotspot = "3spot"
    else:
        hotspot = "unknown"

    if "NICER+XMM-relative" in fname:
        data_used = "NICER+XMM-relative"
    elif "NICER+XMM" in fname:
        data_used = "NICER+XMM"
    elif "NICER-only" in fname:
        data_used = "NICER-only"
    else:
        data_used = "NICER-only"

    variant = "RM" if "RM" in fname else ("full" if "full" in fname else "unknown")

    metadata: Dict = {
        "psr": psr,
        "group": "maryland",
        "hotspot_model": hotspot,
        "data_used": data_used,
        "model_variant": variant,
        "n_samples": len(radius),
        "source_file": filepath.name,
        "header": "\n".join(header_lines),
        "zenodo_record": (
            "https://zenodo.org/records/3473466"
            if psr == "J0030+0451"
            else "https://zenodo.org/records/4670689"
        ),
        "paper": (
            "Miller et al. 2019 (ApJL 887 L24)"
            if psr == "J0030+0451"
            else "Miller et al. 2021 (ApJL 918 L28)"
        ),
    }

    print(
        f"    PSR {psr}, hotspot={hotspot}, data={data_used}, variant={variant}, n={len(radius):,}"
    )
    return radius, mass, metadata


def process_maryland_data() -> list[Path]:
    """Extract all Maryland text files to .npz."""
    print("\n" + "=" * 70)
    print("MARYLAND DATA")
    print("=" * 70)

    source_files = [
        ZENODO_DIR / "J0030/maryland/original/J0030_2spot_RM.txt",
        ZENODO_DIR / "J0030/maryland/original/J0030_2spot_full.txt",
        ZENODO_DIR / "J0030/maryland/original/J0030_3spot_RM.txt",
        ZENODO_DIR / "J0030/maryland/original/J0030_3spot_full.txt",
        ZENODO_DIR / "J0740/maryland/original/NICER-only_J0740_RM.txt",
        ZENODO_DIR / "J0740/maryland/original/NICER-only_J0740_full.txt",
        ZENODO_DIR / "J0740/maryland/original/NICER+XMM_J0740_RM.txt",
        ZENODO_DIR / "J0740/maryland/original/NICER+XMM_J0740_full.txt",
        ZENODO_DIR / "J0740/maryland/original/NICER+XMM-relative_J0740_RM.txt",
        ZENODO_DIR / "J0740/maryland/original/NICER+XMM-relative_J0740_full.txt",
    ]

    results: list[Path] = []
    for src in source_files:
        if not src.exists():
            print(f"\n  Not found (download Zenodo first): {src.name}")
            continue

        radius, mass, meta = parse_maryland_txt(src)
        psr_clean = meta["psr"].replace("+", "")
        data_clean = meta["data_used"].replace("+", "").replace("-", "_")
        out_name = f"{psr_clean}_maryland_{meta['hotspot_model']}_{data_clean}_{meta['model_variant']}.npz"
        out_path = OUTPUT_DIR / out_name

        if out_path.exists() and not IGNORE_CACHE:
            print(f"    Cached: {out_name}")
            results.append(out_path)
            continue

        np.savez(out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
        print(f"    Saved: {out_name} ({out_path.stat().st_size / 1024:.1f} KB)")
        results.append(out_path)

    return results


def parse_dittmann2024_j0740_txt(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Parse Dittmann et al. 2024 J0740+6620 NICER+XMM full-atmosphere M-R file.

    Format: columns are radius [km], mass [Msun], weight. The rows are raw
    ``emcee`` MCMC output with a (discrete) multiplicity weight, so the file is
    not equal-weight: the unweighted median radius is 12.70 km, while the
    weighted one is 12.92 km as in the paper. We importance-resample using the
    weight column, as for the Miller et al. 2026 files. The weighted 2-sigma
    to +2-sigma radius percentiles (10.99, 11.79, 12.92, 15.01, 18.57 km)
    reproduce the paper's summary table of equatorial radii (``tab:radii``),
    NICER+XMM with the fully ionized hydrogen atmosphere (``H_full``) -- do not
    drop the weights without re-verifying.
    """
    print(f"\n  Parsing: {filepath.name}")
    data = np.loadtxt(filepath, comments="#")
    radius_all = data[:, 0]
    mass_all = data[:, 1]
    weights = data[:, 2]

    rng = np.random.default_rng(seed=42)
    w_norm = weights / weights.sum()
    n_resample = min(
        MAX_SAMPLES if MAX_SAMPLES is not None else len(weights), len(weights)
    )
    idx = rng.choice(len(weights), size=n_resample, replace=True, p=w_norm)
    radius = radius_all[idx]
    mass = mass_all[idx]

    metadata: Dict = {
        "psr": "J0740+6620",
        "group": "maryland",
        "analysis": "Dittmann et al. 2024",
        "hotspot_model": "unknown",
        "data_used": "NICER+XMM",
        "model_variant": "full",
        "n_samples": len(radius),
        "weighted": False,
        "resampled_from_weights": True,
        "source_file": filepath.name,
        "zenodo_record": "https://zenodo.org/records/10215109",
        "paper": (
            "Dittmann et al. 2024 (A More Precise Measurement of the Radius of "
            "PSR J0740+6620 Using Updated NICER Data, ApJ 974, 295, arXiv:2406.14467)"
        ),
        "format": (
            "equal-weight resampled from raw weighted emcee posterior "
            "(original format: radius, mass, weight)"
        ),
        "notes": (
            "Fully ionized hydrogen atmosphere (H_full), NICER data through 2022 April "
            "plus XMM-Newton; the paper's headline result, R = 12.92 +2.09/-1.13 km. "
            "No R < 16 km cut is applied."
        ),
    }
    print(
        f"    PSR J0740+6620, data=NICER+XMM (H_full), n={len(radius):,} "
        f"(resampled from {len(weights):,} weighted samples)"
    )
    return radius, mass, metadata


def process_j0740_maryland_recent_data() -> list[Path]:
    """Extract the Dittmann et al. 2024 J0740 Maryland NICER+XMM file to .npz."""
    src = ZENODO_DIR / "J0740/maryland/recent/J0740_NICERXMM_full_mr.txt"
    if not src.exists():
        print(f"\n  Not found (download Zenodo first): {src.name}")
        return []

    out_name = "J07406620_maryland_NICERXMM_full_Dittmann2024.npz"
    out_path = OUTPUT_DIR / out_name

    if out_path.exists() and not IGNORE_CACHE:
        print(f"    Cached: {out_name}")
        return [out_path]

    radius, mass, meta = parse_dittmann2024_j0740_txt(src)
    np.savez(out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
    print(
        f"    Saved: {out_name} ({out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
    )
    return [out_path]


def parse_miller2026_j0437_txt(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Parse Miller, Dittmann, Holt et al. 2026 J0437 NICER RM file.

    Format: columns are radius [km], mass [Msun], weight — raw weighted
    posterior samples (not equal-weight). We importance-resample using the
    weight column to obtain an equal-weight posterior, following the same
    treatment as the Riley et al. 2019 MultiNest chains in
    ``parse_riley2019_mr_file``.
    """
    print(f"\n  Parsing: {filepath.name}")
    data = np.loadtxt(filepath, comments="#")
    radius_all = data[:, 0]
    mass_all = data[:, 1]
    weights = data[:, 2]

    rng = np.random.default_rng(seed=42)
    w_norm = weights / weights.sum()
    n_resample = min(
        MAX_SAMPLES if MAX_SAMPLES is not None else len(weights), len(weights)
    )
    idx = rng.choice(len(weights), size=n_resample, replace=True, p=w_norm)
    radius = radius_all[idx]
    mass = mass_all[idx]

    metadata: Dict = {
        "psr": "J0437-4715",
        "group": "maryland",
        "analysis": "Miller, Dittmann, Holt, et al. 2026",
        "hotspot_model": "3spot+GPL",
        "data_used": "NICER-only",
        "model_variant": "RM",
        "n_samples": len(radius),
        "weighted": False,
        "resampled_from_weights": True,
        "source_file": filepath.name,
        "zenodo_record": "https://zenodo.org/records/17833896",
        "paper": "Miller, Dittmann, Holt, et al. 2026 (ApJL 1000, L48, arXiv:2512.08790)",
        "format": (
            "equal-weight resampled from raw weighted posterior "
            "(original format: radius, mass, weight)"
        ),
    }
    print(f"    PSR J0437-4715, hotspot=3spot+GPL, data=NICER-only, n={len(radius):,}")
    return radius, mass, metadata


def process_j0437_maryland_data() -> list[Path]:
    """Extract the Miller et al. 2026 J0437 Maryland RM file to .npz."""
    src = ZENODO_DIR / "J0437/maryland/original/J0437_NICER_RM.txt"
    if not src.exists():
        print(f"\n  Not found (download Zenodo first): {src.name}")
        return []

    out_name = "J04374715_maryland_3spotGPL_NICER_only_RM.npz"
    out_path = OUTPUT_DIR / out_name

    if out_path.exists() and not IGNORE_CACHE:
        print(f"    Cached: {out_name}")
        return [out_path]

    radius, mass, meta = parse_miller2026_j0437_txt(src)
    np.savez(out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
    print(
        f"    Saved: {out_name} ({out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
    )
    return [out_path]


def parse_miller2026_j0614_txt(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Parse Miller, Dittmann, Holt, et al. 2026 J0614-3329 NICER RM file.

    Format: columns are radius [km], mass [Msun], weight — raw weighted
    posterior samples (not equal-weight) from the headline three-circular-spot
    model fit to NICER-only data. We importance-resample using the weight
    column to obtain an equal-weight posterior, following the same treatment
    as the J0437 Maryland file in ``parse_miller2026_j0437_txt``.
    """
    print(f"\n  Parsing: {filepath.name}")
    data = np.loadtxt(filepath, comments="#")
    radius_all = data[:, 0]
    mass_all = data[:, 1]
    weights = data[:, 2]

    rng = np.random.default_rng(seed=42)
    w_norm = weights / weights.sum()
    n_resample = min(
        MAX_SAMPLES if MAX_SAMPLES is not None else len(weights), len(weights)
    )
    idx = rng.choice(len(weights), size=n_resample, replace=True, p=w_norm)
    radius = radius_all[idx]
    mass = mass_all[idx]

    metadata: Dict = {
        "psr": "J0614-3329",
        "group": "maryland",
        "analysis": "Miller, Dittmann, Holt, et al. 2026",
        "hotspot_model": "3circle",
        "data_used": "NICER-only",
        "model_variant": "RM",
        "n_samples": len(radius),
        "weighted": False,
        "resampled_from_weights": True,
        "source_file": filepath.name,
        "zenodo_record": "https://zenodo.org/records/22131748",
        "paper": "Miller, Dittmann, Holt, et al. 2026 (arXiv:2609.00965)",
        "format": (
            "equal-weight resampled from raw weighted posterior "
            "(original format: radius, mass, weight)"
        ),
    }
    print(f"    PSR J0614-3329, hotspot=3circle, data=NICER-only, n={len(radius):,}")
    return radius, mass, metadata


def process_j0614_maryland_data() -> list[Path]:
    """Extract the Miller et al. 2026 J0614-3329 Maryland RM file to .npz."""
    src = ZENODO_DIR / "J0614/maryland/original/J0614_NICER_rm.txt"
    if not src.exists():
        print(f"\n  Not found (download Zenodo first): {src.name}")
        return []

    out_name = "J06143329_maryland_3circle_NICER_only_RM.npz"
    out_path = OUTPUT_DIR / out_name

    if out_path.exists() and not IGNORE_CACHE:
        print(f"    Cached: {out_name}")
        return [out_path]

    radius, mass, meta = parse_miller2026_j0614_txt(src)
    np.savez(out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
    print(
        f"    Saved: {out_name} ({out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
    )
    return [out_path]


# ============================================================
# Amsterdam extraction
# ============================================================


def parse_riley2019_mr_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Parse Riley et al. 2019 M-R files and resample to equal-weight posterior.

    The M_R.txt files are raw MultiNest chains with format:
        col 0: importance weight
        col 1: -2 * log(likelihood)
        col 2: mass [Msun]
        col 3: radius [km]

    Many rows are dead points with low or zero weight that reflect the prior
    rather than the posterior.  We importance-resample using the weights to
    obtain a proper equal-weight posterior sample.

    Note: investigation in internal-jester-review/nicer_check showed that
    ST_PST is the headline model matching Riley+2019 (M=1.34, R=12.71 km),
    while ST_U and ST_S give anomalous results and should be used with caution.
    """
    data = np.loadtxt(filepath, comments="#")
    weights = data[:, 0]
    mass_all = data[:, 2]
    radius_all = data[:, 3]

    # Importance-resample to equal-weight posterior
    rng = np.random.default_rng(seed=42)
    w_norm = weights / weights.sum()
    n_resample = min(
        MAX_SAMPLES if MAX_SAMPLES is not None else len(weights), len(weights)
    )
    idx = rng.choice(len(weights), size=n_resample, replace=True, p=w_norm)
    mass = mass_all[idx]
    radius = radius_all[idx]

    model = filepath.parent.name
    metadata: Dict = {
        "psr": "J0030+0451",
        "group": "amsterdam",
        "analysis": "Riley et al. 2019",
        "hotspot_model": model,
        "data_used": "NICER-only",
        "n_samples": len(mass),
        "weighted": False,
        "resampled_from_weights": True,
        "source_file": filepath.name,
        "zenodo_record": "https://zenodo.org/records/7096789",
        "paper": "Riley et al. 2019 (ApJL 887 L21)",
        "format": (
            "equal-weight resampled from MultiNest chain "
            "(original format: weight, -2*log(L), mass, radius)"
        ),
    }
    return radius, mass, metadata


# Vinciguerra et al. 2023 (J0030) NICER-only reference-run archive members.
# Settings: SE 0.3/0.8, ET 0.1, LP 1e4, MM on -- the paper's designated
# "reference run" for each model (Section 5, Table \ref{tab:compare_models}).
# Paths and the specific resume-stage subdirectory (there can be several
# incremental MultiNest resumes per model) were confirmed by cross-checking
# the median/68% credible interval of columns 0-1 against the paper's
# Table \ref{tab:compare_models} NICER-only row for each model:
#   ST-U:    M=1.12+0.13-0.08, R=10.53+1.15-0.89
#   ST+PST:  M=1.37+/-0.17,    R=13.11+/-1.30
#   ST+PDT:  M=1.20+0.14-0.11, R=11.16+0.90-0.80
#   PDT-U:   M=1.41+0.20-0.19, R=13.12+1.35-1.21
# All four matched to within Monte Carlo noise -- do not change the member
# paths (in particular the PDT-U resume stage) without re-verifying against
# this table.
VINCIGUERRA2023_MR_MEMBERS: dict[str, str] = {
    "ST_U": (
        "updated_analyses_PSRJ0030_up_to_2018_NICER_data/ST_U/NICER/"
        "STU_10klp_et0p1_se0p3_MMon/STU_outputs/run1_resume/"
        "run1_resume_nlive10k_eff0.3_noCONST_noMM_noIS_tol-1post_equal_weights.dat"
    ),
    "ST_PST": (
        "updated_analyses_PSRJ0030_up_to_2018_NICER_data/ST_PST/NICER/"
        "STPST_LR_10klp_et0p1_se0p3_MMon/STPST_outputs/run1_resume/"
        "run1_nlive10k_eff0.3_noCONST_MMon_noIS_tol-1post_equal_weights.dat"
    ),
    "ST_PDT": (
        "updated_analyses_PSRJ0030_up_to_2018_NICER_data/ST_PDT/NICER/"
        "STPDT_LR_10klp_et0p1_se0p3_MMon/STPDT_outputs/run1/"
        "stpdt_run1_10klp_eff0.8_noCONST_MMon_noIS_tol-1post_equal_weights.dat"
    ),
    "PDT_U": (
        "updated_analyses_PSRJ0030_up_to_2018_NICER_data/PDT_U/NICER/"
        "PDTU_LR_10klp_et0p1_se0p8_MMon/PDTU_outputs/run1_resume/"
        "pdtu_run1_nlive10klp_eff0.8_noCONST_MMon_noIS_tol-1post_equal_weights.dat"
    ),
}


def parse_vinciguerra2023_mr_file(
    filepath: Path, model: str
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Parse a Vinciguerra et al. 2023 NICER-only reference-run equal-weight file.

    The ``post_equal_weights.dat`` file produced by MultiNest is already an
    equal-weight posterior (no importance resampling needed): column 0 is
    mass (Msun) and column 1 is radius (km), same convention as the other
    X-PSI archives in this pipeline (see ``VINCIGUERRA2023_MR_MEMBERS`` for
    the cross-check against the paper's quoted headline numbers).
    """
    print(f"\n  Parsing: {filepath.name} ({model})")
    data = np.loadtxt(filepath, comments="#")
    mass = data[:, 0]
    radius = data[:, 1]

    metadata: Dict = {
        "psr": "J0030+0451",
        "group": "amsterdam",
        "analysis": "Vinciguerra et al. 2023",
        "hotspot_model": model.replace("_", "+", 1) if model != "ST_U" else "ST-U",
        "data_used": "NICER-only",
        "n_samples": len(mass),
        "weighted": False,
        "source_file": filepath.name,
        "zenodo_record": "https://zenodo.org/records/8239000",
        "paper": (
            "Vinciguerra et al. 2023 (An updated mass-radius analysis of the "
            "2017-2018 NICER data set of PSR J0030+0451, ApJ 961, 62, "
            "arXiv:2308.09469)"
        ),
        "settings": "reference run: SE 0.3/0.8, ET 0.1, LP 1e4, MM on",
    }
    print(f"    PSR J0030+0451, hotspot={model}, data=NICER-only, n={len(mass):,}")
    return radius, mass, metadata


def process_j0030_amsterdam_vinciguerra_data() -> list[Path]:
    """Extract Vinciguerra et al. 2023 NICER-only reference-run posteriors to .npz."""
    archive = (
        ZENODO_DIR
        / "J0030/amsterdam/intermediate/updated_analyses_PSRJ0030_up_to_2018_NICER_data.tar.gz"
    )
    if not archive.exists():
        print(f"\n  Not found (download Zenodo first): {archive.name}")
        return []

    results: list[Path] = []
    to_extract = {
        model: member
        for model, member in VINCIGUERRA2023_MR_MEMBERS.items()
        if not (
            OUTPUT_DIR / f"J00300451_amsterdam_{model}_NICER_only_Vinciguerra2023.npz"
        ).exists()
        or IGNORE_CACHE
    }

    cached_models = set(VINCIGUERRA2023_MR_MEMBERS) - set(to_extract)
    for model in cached_models:
        out_name = f"J00300451_amsterdam_{model}_NICER_only_Vinciguerra2023.npz"
        print(f"  Cached: {out_name}")
        results.append(OUTPUT_DIR / out_name)

    if not to_extract:
        return results

    print(f"\nVinciguerra et al. 2023 — {archive.name} (this is a ~7 GB archive)")
    with tarfile.open(archive, "r:gz") as tar:
        for model, member_path in to_extract.items():
            out_name = f"J00300451_amsterdam_{model}_NICER_only_Vinciguerra2023.npz"
            out_path = OUTPUT_DIR / out_name
            try:
                member = tar.getmember(member_path)
                raw = tar.extractfile(member)
                if raw is None:
                    raise ValueError(f"Could not read {member_path}")
                with tempfile.NamedTemporaryFile(
                    mode="wb", delete=False, suffix=".dat"
                ) as tmp:
                    tmp.write(raw.read())
                    tmp_path = Path(tmp.name)

                radius, mass, meta = parse_vinciguerra2023_mr_file(tmp_path, model)
                tmp_path.unlink()

                np.savez(out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
                print(
                    f"  Saved: {out_name} ({out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
                )
                results.append(out_path)
            except KeyError:
                print(f"  Not found in archive: {member_path}")
            except Exception as e:
                print(f"  Error extracting {member_path}: {e}")

    return results


def parse_salmi_recent_mr_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Parse Salmi et al. recent M-R equal-weight samples.

    Format: mass (Msun), radius (km)
    """
    data = np.loadtxt(filepath, comments="#")
    mass = data[:, 0]
    radius = data[:, 1]

    metadata: Dict = {
        "psr": "J0740+6620",
        "group": "amsterdam",
        "analysis": "Salmi et al. 2024",
        "hotspot_model": "gamma",
        "data_used": "NICER+XMM",
        "n_samples": len(mass),
        "weighted": False,
        "source_file": filepath.name,
        "zenodo_record": "https://zenodo.org/records/10519473",
        "paper": "Salmi et al. 2024 (The Radius of the High-mass Pulsar PSR J0740+6620 with 3.6 yr of NICER Data, ApJ 974, 294)",
        "settings": "lp40k_se001",
    }
    return radius, mass, metadata


def parse_riley2021_j0740_mr_file(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Parse Riley et al. 2021 J0740+6620 ST-U equal-weight posterior.

    The ``post_equal_weights.dat`` file produced by MultiNest is already an
    equal-weight posterior (no importance resampling needed), with the X-PSI
    ST-U free-parameter vector as columns followed by -2*log(likelihood) as
    the last column: column 0 is mass (Msun) and column 1 is (equatorial,
    Schwarzschild-coordinate) radius (km). This ordering was confirmed by
    matching the mean/sigma of columns 0-1 in the run's ``stats.dat`` against
    the headline result quoted in the paper (M = 2.072 +0.067/-0.066 Msun,
    R_eq = 12.39 +1.30/-0.98 km) -- do not reorder without re-verifying
    against ``stats.dat`` in the archive.
    """
    print(f"\n  Parsing: {filepath.name}")
    data = np.loadtxt(filepath, comments="#")
    mass = data[:, 0]
    radius = data[:, 1]

    metadata: Dict = {
        "psr": "J0740+6620",
        "group": "amsterdam",
        "analysis": "Riley et al. 2021",
        "hotspot_model": "ST-U",
        "data_used": "NICER+XMM",
        "n_samples": len(mass),
        "weighted": False,
        "source_file": filepath.name,
        "zenodo_record": "https://zenodo.org/records/7096886",
        "paper": "Riley et al. 2021 (A NICER View of the Massive Pulsar PSR J0740+6620 Informed by Radio Timing and XMM-Newton Spectroscopy, ApJL 918, L27, arXiv:2105.06980)",
        "settings": "nlive4000_eff0.1_noCONST_noMM_noIS_tol-1",
    }
    print(f"    PSR J0740+6620, hotspot=ST-U, data=NICER+XMM, n={len(mass):,}")
    return radius, mass, metadata


def process_j0740_amsterdam_original_data() -> list[Path]:
    """Extract Riley et al. 2021 J0740 ST-U equal-weight samples to .npz."""
    archive = ZENODO_DIR / "J0740/amsterdam/original/STU_NICERxXMM_FIH_run11.tar.gz"
    mr_member = (
        "STU_NICERxXMM_FIH_run11/samples/"
        "nlive4000_eff0.1_noCONST_noMM_noIS_tol-1post_equal_weights.dat"
    )
    out_name = "J07406620_amsterdam_STU_NICERXMM_Riley2021.npz"
    out_path = OUTPUT_DIR / out_name

    if not archive.exists():
        print(f"\n  Not found (download Zenodo first): {archive.name}")
        return []

    if out_path.exists() and not IGNORE_CACHE:
        print(f"    Cached: {out_name}")
        return [out_path]

    print(f"\nRiley et al. 2021 — {archive.name}")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            member = tar.getmember(mr_member)
            raw = tar.extractfile(member)
            if raw is None:
                raise ValueError(f"Could not read {mr_member}")
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, suffix=".dat"
            ) as tmp:
                tmp.write(raw.read())
                tmp_path = Path(tmp.name)

        radius, mass, meta = parse_riley2021_j0740_mr_file(tmp_path)
        tmp_path.unlink()

        np.savez(out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
        print(
            f"  Saved: {out_name} ({out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
        )
        return [out_path]
    except KeyError:
        print(f"  File not found in archive: {mr_member}")
        return []
    except Exception as e:
        print(f"  Error: {e}")
        return []


def parse_kini2026_j0030_txt(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Parse Kini, Mauviard, Salmi, et al. 2026 J0030 equal-weight M-R file.

    Format: columns are mass (Msun), radius (km) — already an equal-weight
    posterior (no importance resampling needed), unlike the raw weighted
    ``weighted_samples_PDTU.txt`` file also available on the same Zenodo
    record.
    """
    print(f"\n  Parsing: {filepath.name}")
    data = np.loadtxt(filepath, comments="#")
    mass = data[:, 0]
    radius = data[:, 1]

    metadata: Dict = {
        "psr": "J0030+0451",
        "group": "amsterdam",
        "analysis": "Kini, Mauviard, Salmi, et al. 2026",
        "hotspot_model": "PDT-U",
        "data_used": "NICER+XMM",
        "n_samples": len(mass),
        "weighted": False,
        "source_file": filepath.name,
        "zenodo_record": "https://zenodo.org/records/18741942",
        "paper": (
            "Kini, Mauviard, Salmi, et al. 2026 (A NICER View of PSR J0030+0451: "
            "Updated Constraints from Six Years of NICER Observations, arXiv:2602.23743)"
        ),
        "notes": (
            "Bayes-preferred hotspot model (PDT-U over ST+PDT) from six years of "
            "NICER data (2017 Jul - 2023 Jan) jointly analyzed with archival XMM-Newton "
            "data. Recommended Amsterdam model for J0030+0451, superseding the "
            "Riley et al. 2019 ST+PST result."
        ),
    }
    print(f"    PSR J0030+0451, hotspot=PDT-U, data=NICER+XMM, n={len(mass):,}")
    return radius, mass, metadata


def process_j0030_amsterdam_recent_data() -> list[Path]:
    """Extract the Kini et al. 2026 J0030 Amsterdam equal-weight file to .npz."""
    src = ZENODO_DIR / "J0030/amsterdam/recent/equal_weight_samples_PDTU.txt"
    if not src.exists():
        print(f"\n  Not found (download Zenodo first): {src.name}")
        return []

    out_name = "J00300451_amsterdam_PDTU_NICERXMM_Kini2026.npz"
    out_path = OUTPUT_DIR / out_name

    if out_path.exists() and not IGNORE_CACHE:
        print(f"    Cached: {out_name}")
        return [out_path]

    radius, mass, meta = parse_kini2026_j0030_txt(src)
    np.savez(out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
    print(
        f"    Saved: {out_name} ({out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
    )
    return [out_path]


def process_j1614_amsterdam_data() -> list[Path]:
    """Extract Mauviard et al. 2026 J1614-2230 ST-U equal-weight samples to .npz."""
    archive = (
        ZENODO_DIR / "J1614/amsterdam/original/MR_samples_and_contours_J1614.tar.gz"
    )
    out_name = "J16142230_amsterdam_STU_NICER_only_Mauviard2026.npz"
    out_path = OUTPUT_DIR / out_name
    if not archive.exists():
        print(f"\n  Not found (download Zenodo first): {archive.name}")
        return []
    if out_path.exists() and not IGNORE_CACHE:
        print(f"  Cached: {out_name}")
        return [out_path]

    print(f"\nMauviard et al. 2026 — {archive.name}")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if "post_equal_weights" in m.name]
            if not members:
                raise FileNotFoundError(
                    f"No *post_equal_weights* file in archive: {[m.name for m in tar.getmembers()]}"
                )
            member = members[0]
            raw = tar.extractfile(member)
            if raw is None:
                raise ValueError(f"Could not read {member.name}")
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, suffix=".dat"
            ) as tmp:
                tmp.write(raw.read())
                tmp_path = Path(tmp.name)

        # Column 0 is mass (Msun), column 1 is equatorial radius (km); this was
        # confirmed against the paper's headline result (M = 1.937 Msun,
        # R_eq = 10.06 +1.25/-0.87 km) and the Zenodo README.
        data = np.loadtxt(tmp_path, comments="#")
        tmp_path.unlink()
        mass = data[:, 0]
        radius = data[:, 1]

        meta: Dict = {
            "psr": "J1614-2230",
            "group": "amsterdam",
            "analysis": "Mauviard et al. 2026",
            "hotspot_model": "ST-U",
            "data_used": "NICER+XMM+Chandra",
            "n_samples": len(mass),
            "weighted": False,
            "source_file": member.name,
            "zenodo_record": "https://zenodo.org/records/22163155",
            "paper": "Mauviard et al. 2026 (A NICER view of PSR J1614-2230: a massive and compact millisecond pulsar)",
            "settings": "40kLP_0p03SE_0p1ET",
        }
        np.savez(out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
        print(
            f"  Saved: {out_name} ({out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
        )
        return [out_path]
    except Exception as e:
        print(f"  Error: {e}")
        return []


def extract_amsterdam_data() -> list[Path]:
    """Extract Amsterdam M-R samples from tar.gz archives."""
    print("\n" + "=" * 70)
    print("AMSTERDAM DATA")
    print("=" * 70)

    results: list[Path] = []

    # 1. Riley et al. 2019 (J0030)
    riley_archive = (
        ZENODO_DIR / "J0030/amsterdam/original/A_NICER_VIEW_OF_PSR_J0030p0451.tar.gz"
    )
    riley_mr_files = [
        "A_NICER_VIEW_OF_PSR_J0030p0451/ST_S/ST_S__M_R.txt",
        "A_NICER_VIEW_OF_PSR_J0030p0451/ST_U/ST_U__M_R.txt",
        "A_NICER_VIEW_OF_PSR_J0030p0451/CDT_U/CDT_U__M_R.txt",
        "A_NICER_VIEW_OF_PSR_J0030p0451/ST_EST/ST_EST__M_R.txt",
        "A_NICER_VIEW_OF_PSR_J0030p0451/ST_PST/ST_PST__M_R.txt",
    ]

    if riley_archive.exists():
        print(f"\nRiley et al. 2019 — {riley_archive.name}")
        with tarfile.open(riley_archive, "r:gz") as tar:
            for mr_path in riley_mr_files:
                model = mr_path.split("/")[1]
                out_name = f"J00300451_amsterdam_{model}_NICER_only_Riley2019.npz"
                out_path = OUTPUT_DIR / out_name

                if out_path.exists() and not IGNORE_CACHE:
                    print(f"  Cached: {out_name}")
                    results.append(out_path)
                    continue

                try:
                    member = tar.getmember(mr_path)
                    raw = tar.extractfile(member)
                    if raw is None:
                        raise ValueError(f"Could not read {mr_path}")
                    with tempfile.NamedTemporaryFile(
                        mode="wb", delete=False, suffix=".txt"
                    ) as tmp:
                        tmp.write(raw.read())
                        tmp_path = Path(tmp.name)

                    radius, mass, meta = parse_riley2019_mr_file(tmp_path)
                    tmp_path.unlink()

                    np.savez(out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
                    print(
                        f"  Saved: {out_name} ({out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
                    )
                    results.append(out_path)
                except KeyError:
                    print(f"  Not found in archive: {mr_path}")
                except Exception as e:
                    print(f"  Error extracting {mr_path}: {e}")
    else:
        print(f"\nArchive not found (download Zenodo first): {riley_archive.name}")

    # 2. Salmi et al. recent (J0740)
    salmi_archive = ZENODO_DIR / "J0740/amsterdam/recent/mr_samples_and_contours.tar.gz"
    salmi_mr_file = "mr_samples_and_contours/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat"
    out_name = "J07406620_amsterdam_gamma_NICERXMM_equal_weights_recent.npz"
    out_path = OUTPUT_DIR / out_name

    if salmi_archive.exists():
        print(f"\nSalmi et al. recent — {salmi_archive.name}")
        if out_path.exists() and not IGNORE_CACHE:
            print(f"  Cached: {out_name}")
            results.append(out_path)
        else:
            try:
                with tarfile.open(salmi_archive, "r:gz") as tar:
                    member = tar.getmember(salmi_mr_file)
                    raw = tar.extractfile(member)
                    if raw is None:
                        raise ValueError(f"Could not read {salmi_mr_file}")
                    with tempfile.NamedTemporaryFile(
                        mode="wb", delete=False, suffix=".dat"
                    ) as tmp:
                        tmp.write(raw.read())
                        tmp_path = Path(tmp.name)

                radius, mass, meta = parse_salmi_recent_mr_file(tmp_path)
                tmp_path.unlink()

                np.savez(out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
                print(
                    f"  Saved: {out_name} ({out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
                )
                results.append(out_path)
            except KeyError:
                print(f"  File not found in archive: {salmi_mr_file}")
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print(f"\nArchive not found (download Zenodo first): {salmi_archive.name}")

    # 3. Choudhury et al. 2024 (J0437-4715)
    j0437_archive = (
        ZENODO_DIR
        / "J0437/amsterdam/original/headline_result_samples_and_contours.tar.gz"
    )
    j0437_out_name = "J04374715_amsterdam_CST_PDT_NICER_only_Choudhury2024.npz"
    j0437_out_path = OUTPUT_DIR / j0437_out_name

    if j0437_archive.exists():
        print(f"\nChoudhury et al. 2024 — {j0437_archive.name}")
        if j0437_out_path.exists() and not IGNORE_CACHE:
            print(f"  Cached: {j0437_out_name}")
            results.append(j0437_out_path)
        else:
            try:
                with tarfile.open(j0437_archive, "r:gz") as tar:
                    sample_members = [
                        m
                        for m in tar.getmembers()
                        if "post_equal_weights" in m.name and m.name.endswith(".dat")
                    ]
                    if not sample_members:
                        raise FileNotFoundError(
                            "No *post_equal_weights*.dat found. "
                            f"Files: {[m.name for m in tar.getmembers()]}"
                        )
                    member = sample_members[0]
                    print(f"  Extracting: {member.name}")
                    raw = tar.extractfile(member)
                    if raw is None:
                        raise ValueError(f"Could not read {member.name}")
                    with tempfile.NamedTemporaryFile(
                        mode="wb", delete=False, suffix=".dat"
                    ) as tmp:
                        tmp.write(raw.read())
                        tmp_path = Path(tmp.name)

                # Format: mass (Msun), radius (km) — equal weights
                data = np.loadtxt(tmp_path, comments="#")
                tmp_path.unlink()
                mass = data[:, 0]
                radius = data[:, 1]

                meta: Dict = {
                    "psr": "J0437-4715",
                    "group": "amsterdam",
                    "analysis": "Choudhury et al. 2024",
                    "hotspot_model": "CST+PDT",
                    "data_used": "NICER-only",
                    "n_samples": len(mass),
                    "weighted": False,
                    "source_file": member.name,
                    "zenodo_record": "https://zenodo.org/records/13766753",
                    "paper": "Choudhury et al. 2024 (A NICER View of the Nearest and Brightest Millisecond Pulsar: PSR J0437-4715, ApJL 971, L20)",
                }
                np.savez(j0437_out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
                print(
                    f"  Saved: {j0437_out_name} ({j0437_out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
                )
                results.append(j0437_out_path)
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print(f"\nArchive not found (download Zenodo first): {j0437_archive.name}")

    # 4. Mauviard et al. 2025 (J0614-3329)
    j0614_archive = (
        ZENODO_DIR / "J0614/amsterdam/original/Headline_Contours_and_Samples.tar.gz"
    )
    j0614_out_name = "J06143329_amsterdam_ST_PDT_NICER_only_Mauviard2025.npz"
    j0614_out_path = OUTPUT_DIR / j0614_out_name

    if j0614_archive.exists():
        print(f"\nMauviard et al. 2025 — {j0614_archive.name}")
        if j0614_out_path.exists() and not IGNORE_CACHE:
            print(f"  Cached: {j0614_out_name}")
            results.append(j0614_out_path)
        else:
            try:
                with tarfile.open(j0614_archive, "r:gz") as tar:
                    # Find the equal-weights samples file (search by name pattern)
                    sample_members = [
                        m
                        for m in tar.getmembers()
                        if "post_equal_weights" in m.name and m.name.endswith(".dat")
                    ]
                    if not sample_members:
                        raise FileNotFoundError(
                            "No *post_equal_weights*.dat found in archive. "
                            f"Available files: {[m.name for m in tar.getmembers()]}"
                        )
                    member = sample_members[0]
                    print(f"  Extracting: {member.name}")
                    raw = tar.extractfile(member)
                    if raw is None:
                        raise ValueError(f"Could not read {member.name}")
                    with tempfile.NamedTemporaryFile(
                        mode="wb", delete=False, suffix=".dat"
                    ) as tmp:
                        tmp.write(raw.read())
                        tmp_path = Path(tmp.name)

                # Format: mass (Msun), radius (km) — equal weights
                data = np.loadtxt(tmp_path, comments="#")
                tmp_path.unlink()
                mass = data[:, 0]
                radius = data[:, 1]

                meta: Dict = {
                    "psr": "J0614-3329",
                    "group": "amsterdam",
                    "analysis": "Mauviard et al. 2025",
                    "hotspot_model": "ST+PDT",
                    "data_used": "NICER-only",
                    "n_samples": len(mass),
                    "weighted": False,
                    "source_file": member.name,
                    "zenodo_record": "https://zenodo.org/records/17380576",
                    "paper": "Mauviard et al. 2025 (A NICER View of the 1.4 Msun Edge-on Pulsar PSR J0614-3329, ApJ 995, 60, arXiv:2506.14883)",
                }
                np.savez(j0614_out_path, radius=radius, mass=mass, metadata=meta)  # type: ignore[arg-type]
                print(
                    f"  Saved: {j0614_out_name} ({j0614_out_path.stat().st_size / 1024:.1f} KB, {len(radius):,} samples)"
                )
                results.append(j0614_out_path)
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print(f"\nArchive not found (download Zenodo first): {j0614_archive.name}")

    # 5. Kini, Mauviard, Salmi, et al. 2026 (J0030, recent — PDT-U)
    print("\nKini et al. 2026 — equal_weight_samples_PDTU.txt")
    results.extend(process_j0030_amsterdam_recent_data())

    # 6. Riley et al. 2021 (J0740, original — ST-U)
    results.extend(process_j0740_amsterdam_original_data())

    # 7. Vinciguerra et al. 2023 (J0030, intermediate — 4 NICER-only reference runs)
    results.extend(process_j0030_amsterdam_vinciguerra_data())

    # 8. Mauviard et al. 2026 (J1614-2230 — ST-U headline run)
    results.extend(process_j1614_amsterdam_data())

    return results


# ============================================================
# Downsampling
# ============================================================


def downsample_all_npz() -> None:
    """Downsample every .npz in OUTPUT_DIR to at most MAX_SAMPLES samples."""
    if MAX_SAMPLES is None:
        return

    print("\n" + "=" * 70)
    print(f"DOWNSAMPLING (max {MAX_SAMPLES:,} samples per file)")
    print("=" * 70)

    npz_files = sorted(OUTPUT_DIR.glob("*.npz"))
    if not npz_files:
        print("No .npz files found.")
        return

    rng = np.random.default_rng(seed=42)
    total_before = total_after = 0.0

    for filepath in npz_files:
        data = np.load(filepath, allow_pickle=True)
        radius: np.ndarray = data["radius"]
        mass: np.ndarray = data["mass"]
        meta: dict = data["metadata"].item()

        n = len(radius)
        size_before = filepath.stat().st_size / (1024**2)
        total_before += size_before

        if n <= MAX_SAMPLES:
            total_after += size_before
            continue

        idx = np.sort(rng.choice(n, size=MAX_SAMPLES, replace=False))
        meta["original_n_samples"] = n
        meta["downsampled_to"] = MAX_SAMPLES
        meta["downsampling_seed"] = 42

        np.savez(filepath, radius=radius[idx], mass=mass[idx], metadata=meta)  # type: ignore[arg-type]
        size_after = filepath.stat().st_size / (1024**2)
        total_after += size_after
        print(
            f"  {filepath.name}: {n:,} → {MAX_SAMPLES:,} samples  ({size_before:.2f} → {size_after:.2f} MB)"
        )

    saved = total_before - total_after
    if saved > 0:
        print(
            f"\n  Total saved: {saved:.2f} MB  ({total_before:.2f} → {total_after:.2f} MB)"
        )


# ============================================================
# Summary
# ============================================================


def print_summary() -> None:
    npz_files = sorted(OUTPUT_DIR.glob("*.npz"))
    print("\n" + "=" * 70)
    print(f"SUMMARY — {len(npz_files)} files in {OUTPUT_DIR}")
    print("=" * 70)
    j0030 = [f for f in npz_files if "J0030" in f.name]
    j0437 = [f for f in npz_files if "J0437" in f.name]
    j0614 = [f for f in npz_files if "J0614" in f.name]
    j0740 = [f for f in npz_files if "J0740" in f.name]
    for label, files in [
        ("J0030+0451", j0030),
        ("J0437-4715", j0437),
        ("J0614-3329", j0614),
        ("J0740+6620", j0740),
    ]:
        if files:
            print(f"\n{label}:")
            for f in files:
                print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
    total = sum(f.stat().st_size for f in npz_files) / (1024**2)
    print(f"\nTotal size: {total:.2f} MB")


# ============================================================
# Main
# ============================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ZENODO_DIR.mkdir(parents=True, exist_ok=True)

    if DOWNLOAD_ZENODO:
        print("\n" + "=" * 70)
        print("STEP 1: DOWNLOAD ZENODO ARCHIVES")
        print("=" * 70)
        download_zenodo_data()
    else:
        print("\nStep 1 skipped (DOWNLOAD_ZENODO = False)")

    print("\n" + "=" * 70)
    print("STEP 2: EXTRACT MARYLAND DATA")
    print("=" * 70)
    process_maryland_data()
    process_j0437_maryland_data()
    process_j0614_maryland_data()
    process_j0740_maryland_recent_data()

    if EXTRACT_AMSTERDAM:
        print("\n" + "=" * 70)
        print("STEP 3: EXTRACT AMSTERDAM DATA")
        print("=" * 70)
        extract_amsterdam_data()
    else:
        print("\nStep 3 skipped (EXTRACT_AMSTERDAM = False)")

    downsample_all_npz()
    print_summary()
    print("\nDone.")


if __name__ == "__main__":
    main()
