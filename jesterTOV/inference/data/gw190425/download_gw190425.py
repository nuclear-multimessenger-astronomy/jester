#!/usr/bin/env python3
"""
Download and process GW190425 posterior samples from LIGO DCC.

Extracts: mass_1_source, mass_2_source, lambda_1, lambda_2
Converts detector-frame masses to source frame using bilby.
Also downloads XP-NRTV3 samples from the neural_priors repository.

Data sources:
  https://dcc.ligo.org/P2000026/public  (GW190425)
  https://github.com/ThibeauWouters/neural_priors  (XP-NRTV3)
"""

import urllib.request
from pathlib import Path

import h5py
import numpy as np
import requests  # type: ignore[import-not-found]
from tqdm import tqdm  # type: ignore[import-not-found]

try:
    from bilby.gw.conversion import luminosity_distance_to_redshift  # type: ignore[import-not-found]
except ImportError:
    raise ImportError("bilby is required. Install with: pip install bilby")

# ============================================================
# USER CONFIGURATION - Tweak these constants as needed
# ============================================================

# Set True to force re-download even if files already exist
IGNORE_CACHE: bool = False

# Set True to also download XP-NRTV3 samples
DOWNLOAD_XPNRTV3: bool = True

# Downsample GW posteriors to at most this many samples (None = keep all)
MAX_SAMPLES: int | None = None

# ============================================================

DATA_DIR = Path(__file__).parent

URL = "https://dcc.ligo.org/public/0165/P2000026/002/posterior_samples.h5"

XPNRTV3_URL = "https://raw.githubusercontent.com/ThibeauWouters/neural_priors/b0ae4235f0c74a6f9e2f6cc4c3385a3ac780d4f8/final_results/GW190425/bns/default/samples.npz"
XPNRTV3_PARAMS = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]


# ============================================================
# Download helpers
# ============================================================


def download_file(url: str, output_path: Path) -> None:
    """Download with a progress bar (uses requests for streaming)."""
    print(f"Downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with (
            open(output_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def download_file_simple(url: str, output_path: Path) -> None:
    """Lightweight download without progress bar (for small files)."""
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


# ============================================================
# HDF5 exploration and loading
# ============================================================


def find_posterior_paths(hdf5_path: Path) -> list[str]:
    """Return all posterior_samples dataset paths in the HDF5 file."""
    paths = []
    with h5py.File(hdf5_path, "r") as f:
        for key in f.keys():
            if key == "version":
                continue
            if isinstance(f[key], h5py.Group) and "posterior_samples" in f[key]:  # type: ignore[operator]
                paths.append(f"{key}/posterior_samples")
            elif "posterior" in key.lower() and isinstance(f[key], h5py.Dataset):
                paths.append(key)
    return paths


def load_posterior_by_path(
    hdf5_path: Path, posterior_path: str
) -> tuple[dict, list[str]]:
    """Load a single posterior dataset from the HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        obj = f[posterior_path]
        if isinstance(obj, h5py.Dataset) and obj.dtype.names:  # type: ignore[union-attr]
            header = list(obj.dtype.names)  # type: ignore[arg-type]
            data = np.array(obj)
            posterior = {n: data[n] for n in header}  # type: ignore[call-overload]
        elif isinstance(obj, h5py.Group):
            header = list(obj.keys())
            posterior = {k: np.array(obj[k]) for k in header}
        else:
            raise ValueError(f"Cannot parse dataset at '{posterior_path}'")
    print(f"  {len(next(iter(posterior.values())))} samples, {len(header)} parameters")
    return posterior, header


# ============================================================
# Mass frame conversion
# ============================================================


def detector_to_source_frame(
    m1_det: np.ndarray, m2_det: np.ndarray, d_L: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    z = luminosity_distance_to_redshift(d_L)
    return m1_det / (1 + z), m2_det / (1 + z)


# ============================================================
# Extraction and saving
# ============================================================


def extract_and_save(
    posterior: dict,
    header: list[str],
    dataset_name: str,
    waveform_model: str,
) -> Path | None:
    """Extract the four required parameters and save as .npz."""
    print(f"\nExtracting parameters for '{dataset_name}'")

    has_source = "mass_1_source" in header or "m1_source" in header
    has_detector = "m1_detector_frame_Msun" in header or "mass_1" in header

    if has_source:
        key = "mass_1_source" if "mass_1_source" in header else "m1_source"
        m1 = posterior[key]
        m2 = posterior[key.replace("1", "2")]
    elif has_detector:
        if "m1_detector_frame_Msun" in header:
            m1_det = posterior["m1_detector_frame_Msun"]
            m2_det = posterior["m2_detector_frame_Msun"]
            d_L = posterior["luminosity_distance_Mpc"]
        else:
            m1_det = posterior["mass_1"]
            m2_det = posterior["mass_2"]
            d_L = posterior.get("luminosity_distance") or posterior.get("distance")
            if d_L is None:
                raise ValueError("Cannot find luminosity distance for frame conversion")
        m1, m2 = detector_to_source_frame(m1_det, m2_det, d_L)
    else:
        raise ValueError("No mass parameters found in posterior")

    if "lambda_1" in header:
        lam1, lam2 = posterior["lambda_1"], posterior["lambda_2"]
    elif "lambda1" in header:
        lam1, lam2 = posterior["lambda1"], posterior["lambda2"]
    else:
        print(f"  WARNING: no lambda_1/lambda_2 found — skipping '{dataset_name}'")
        return None

    extracted = {
        "mass_1_source": m1,
        "mass_2_source": m2,
        "lambda_1": lam1,
        "lambda_2": lam2,
    }

    if MAX_SAMPLES is not None and len(m1) > MAX_SAMPLES:
        rng = np.random.default_rng(seed=42)
        idx = np.sort(rng.choice(len(m1), size=MAX_SAMPLES, replace=False))
        extracted = {k: v[idx] for k, v in extracted.items()}
        print(f"  Downsampled from {len(m1)} to {MAX_SAMPLES} samples")

    metadata = {
        "event": "GW190425",
        "waveform_model": waveform_model,
        "dataset": dataset_name,
        "data_source": "LIGO-P2000026",
        "dcc_url": URL,
        "n_samples": len(extracted["mass_1_source"]),
        "parameters": ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
        "conversion_tool": "N/A" if has_source else "bilby.gw.conversion",
        "notes": (
            "Source frame masses provided"
            if has_source
            else "Converted from detector frame"
        ),
    }

    output_file = DATA_DIR / f"gw190425_{dataset_name}_posterior.npz"
    np.savez(output_file, **extracted, metadata=metadata)  # type: ignore[arg-type]
    print(
        f"  Saved: {output_file.name} ({output_file.stat().st_size / 1024:.1f} KB, {metadata['n_samples']} samples)"
    )
    return output_file


# ============================================================
# XP-NRTV3 downloader
# ============================================================


def download_xpnrtv3() -> None:
    """Download XP-NRTV3 samples and extract the four relevant parameters."""
    output_file = DATA_DIR / "gw190425_xp_nrtv3.npz"
    if not IGNORE_CACHE and output_file.exists():
        print(f"\nXP-NRTV3: {output_file.name} already exists — skipping")
        return

    print("\nDownloading XP-NRTV3 samples for GW190425...")
    tmp_file = DATA_DIR / "_xpnrtv3_full_tmp.npz"
    download_file_simple(XPNRTV3_URL, tmp_file)

    data = np.load(tmp_file)
    extracted = {p: data[p] for p in XPNRTV3_PARAMS if p in data}
    missing = set(XPNRTV3_PARAMS) - set(extracted)
    if missing:
        print(f"  WARNING: missing parameters: {missing}")
    np.savez(output_file, **extracted)
    tmp_file.unlink()
    print(f"  Saved: {output_file.name} ({output_file.stat().st_size / 1024:.1f} KB)")


# ============================================================
# Main
# ============================================================


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download the HDF5 file
    h5_file = DATA_DIR / "posterior_samples.h5"
    if IGNORE_CACHE or not h5_file.exists():
        download_file(URL, h5_file)
    else:
        print(f"Cached: {h5_file.name}")

    # Discover and process each posterior in the file
    posterior_paths = find_posterior_paths(h5_file)
    print(f"\nFound {len(posterior_paths)} posteriors: {posterior_paths}")

    for post_path in posterior_paths:
        if "/" not in post_path:
            continue
        print("\n" + "=" * 70)
        print(f"Processing: {post_path}")
        print("=" * 70)

        try:
            posterior, header = load_posterior_by_path(h5_file, post_path)

            # Derive names: "C01:IMRPhenomPv2/posterior_samples" -> waveform="IMRPhenomPv2", variant="C01"
            parts = post_path.split("/")
            if ":" in parts[0]:
                variant, waveform = parts[0].split(":", 1)
                dataset_name = f"{variant.lower()}_{waveform.lower()}"
            else:
                dataset_name = parts[0].lower()
                waveform = parts[0]

            extract_and_save(posterior, header, dataset_name, waveform)

        except Exception as e:
            print(f"  ERROR processing {post_path}: {e}")

    if DOWNLOAD_XPNRTV3:
        print("\n" + "=" * 70)
        print("XP-NRTV3")
        print("=" * 70)
        download_xpnrtv3()

    print("\n" + "=" * 70)
    print("All GW190425 data ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
