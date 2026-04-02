#!/usr/bin/env python3
"""
Download, extract, and downsample NICER mass-radius posterior samples.

Pipeline (all controlled by constants below):

  Step 1  Download Zenodo archives for Maryland and Amsterdam groups.
          (requires ``zenodo_get``: ``uv pip install zenodo-get``)
  Step 2  Extract Maryland text files → .npz with mass/radius/metadata.
  Step 3  Extract Amsterdam tar.gz archives → .npz.
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
    J0614 is an exception: only one small file is needed, so it is fetched
    directly from the Zenodo file URL instead.
    """
    downloader = ZenodoDownloader(base_dir=ZENODO_DIR)

    # Full-record downloads via zenodo_get
    datasets_to_download = [
        ("J0030", "maryland", "original"),
        ("J0740", "maryland", "original"),
        ("J0030", "amsterdam", "original"),
        ("J0740", "amsterdam", "recent"),
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
            "https://zenodo.org/records/3473464"
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
        "zenodo_record": "https://zenodo.org/records/3524457",
        "paper": "Riley et al. 2019 (ApJL 887 L21)",
        "format": (
            "equal-weight resampled from MultiNest chain "
            "(original format: weight, -2*log(L), mass, radius)"
        ),
    }
    return radius, mass, metadata


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
        "paper": "Salmi et al. 2024",
        "settings": "lp40k_se001",
    }
    return radius, mass, metadata


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
                    "paper": "Choudhury et al. 2024 (A NICER View of the Nearest and Brightest Millisecond Pulsar: PSR J0437-4715)",
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

    # 4. Dittmann et al. 2025 (J0614-3329)
    j0614_archive = (
        ZENODO_DIR / "J0614/amsterdam/original/Headline_Contours_and_Samples.tar.gz"
    )
    j0614_out_name = "J06143329_amsterdam_ST_PDT_NICER_only_Dittmann2025.npz"
    j0614_out_path = OUTPUT_DIR / j0614_out_name

    if j0614_archive.exists():
        print(f"\nDittmann et al. 2025 — {j0614_archive.name}")
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
                    "analysis": "Dittmann et al. 2025",
                    "hotspot_model": "ST+PDT",
                    "data_used": "NICER-only",
                    "n_samples": len(mass),
                    "weighted": False,
                    "source_file": member.name,
                    "zenodo_record": "https://zenodo.org/records/17380576",
                    "paper": "Dittmann et al. 2025 (A NICER view of the 1.4 solar-mass edge-on pulsar PSR J0614-3329)",
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
