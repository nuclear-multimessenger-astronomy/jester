#!/usr/bin/env python3
"""
Download and process GW170817 posterior samples from LIGO DCC.

Extracts: mass_1_source, mass_2_source, lambda_1, lambda_2
Converts detector-frame masses to source frame using bilby.
Also downloads XP-NRTV3 samples from the neural_priors repository.

Supports two data formats:
  - .dat.gz (PhenomPNRT posteriors from P1800061)
  - .hdf5   (GWTC-1 posteriors from P1800370)

Data sources:
  https://dcc.ligo.org/P1800061/public  (PhenomPNRT)
  https://dcc.ligo.org/P1800370/public  (GWTC-1)
  https://github.com/ThibeauWouters/neural_priors  (XP-NRTV3)
"""

import gzip
import urllib.request
from pathlib import Path

import h5py
import numpy as np

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

URLS = {
    # GW170817 - PhenomPNRT posteriors (P1800061)
    "gw170817_low_spin": "https://dcc.ligo.org/public/0150/P1800061/011/low_spin_PhenomPNRT_posterior_samples.dat.gz",
    "gw170817_high_spin": "https://dcc.ligo.org/public/0150/P1800061/011/high_spin_PhenomPNRT_posterior_samples.dat.gz",
    # GW170817 - GWTC-1 posteriors (P1800370)
    "gw170817_gwtc1": "https://dcc.ligo.org/public/0157/P1800370/005/GW170817_GWTC-1.hdf5",
}

XPNRTV3_URL = "https://raw.githubusercontent.com/ThibeauWouters/neural_priors/b0ae4235f0c74a6f9e2f6cc4c3385a3ac780d4f8/final_results/GW170817/bns/default/samples.npz"
XPNRTV3_PARAMS = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]


# ============================================================
# Download helpers
# ============================================================


def download_file(url: str, output_path: Path) -> None:
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


# ============================================================
# Posterior loaders
# ============================================================


def load_datgz_posterior(gz_path: Path) -> tuple[dict, list[str]]:
    """Load posterior samples from a gzipped ASCII file."""
    print(f"\nLoading {gz_path.name}")
    with gzip.open(gz_path, "rt") as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    data = np.array(
        [
            [float(x) for x in line.strip().split()]
            for line in lines[1:]
            if line.strip() and not line.startswith("#")
        ]
    )
    print(f"  {len(data)} samples, {len(header)} parameters")
    return {col: data[:, i] for i, col in enumerate(header)}, header


def load_hdf5_posterior(hdf5_path: Path) -> tuple[dict | None, list[str] | None]:
    """Load posterior from HDF5 file (single-posterior files only).

    Returns (None, None) if the file contains multiple posteriors,
    signalling that the caller should use load_hdf5_posterior_by_name instead.
    """
    print(f"\nLoading {hdf5_path.name}")
    with h5py.File(hdf5_path, "r") as f:
        posterior_datasets = [k for k in f.keys() if k.endswith("_posterior")]
        if len(posterior_datasets) > 1:
            print(f"  Multiple posteriors found: {posterior_datasets}")
            return None, None

        for group_name in [
            "posterior_samples",
            "posterior",
            "samples",
            "IMRPhenomPv2NRT_highSpin_posterior",
            "Overall_posterior",
        ]:
            if group_name in f:
                group = f[group_name]
                if isinstance(group, h5py.Group):
                    header = list(group.keys())
                    posterior = {k: np.array(group[k]) for k in header}
                    print(f"  {len(next(iter(posterior.values())))} samples")
                    return posterior, header
                elif isinstance(group, h5py.Dataset):
                    if group.dtype.names:
                        header = list(group.dtype.names)  # type: ignore[arg-type]
                        data = np.array(group)
                        return {n: data[n] for n in header}, header  # type: ignore[call-overload]

        # Fall back to root-level datasets
        root_datasets = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        if root_datasets:
            header = root_datasets
            posterior = {k: np.array(f[k]) for k in header}
            return posterior, header

    raise ValueError(f"Could not find posterior samples in {hdf5_path}")


def load_hdf5_posterior_by_name(
    hdf5_path: Path, posterior_name: str
) -> tuple[dict, list[str]]:
    """Load a named structured-array posterior from an HDF5 file."""
    print(f"\nLoading '{posterior_name}' from {hdf5_path.name}")
    with h5py.File(hdf5_path, "r") as f:
        dataset = f[posterior_name]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"'{posterior_name}' is a group, not a dataset")
        if not dataset.dtype.names:  # type: ignore[union-attr]
            raise ValueError(f"Dataset '{posterior_name}' is not a structured array")
        header = list(dataset.dtype.names)  # type: ignore[arg-type]
        data = np.array(dataset)
        posterior = {n: data[n] for n in header}  # type: ignore[call-overload]
    print(f"  {len(data)} samples, {len(header)} parameters")
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
    data_source: str,
    dcc_url: str,
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
        print(f"  WARNING: no lambda_1/lambda_2 found — skipping {dataset_name}")
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
        "event": "GW170817",
        "waveform_model": waveform_model,
        "dataset": dataset_name,
        "data_source": data_source,
        "dcc_url": dcc_url,
        "n_samples": len(extracted["mass_1_source"]),
        "parameters": ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
        "conversion_tool": "N/A" if has_source else "bilby.gw.conversion",
        "notes": (
            "Source frame masses provided"
            if has_source
            else "Converted from detector frame"
        ),
    }

    output_file = DATA_DIR / f"gw170817_{dataset_name}_posterior.npz"
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
    output_file = DATA_DIR / "gw170817_xp_nrtv3.npz"
    if not IGNORE_CACHE and output_file.exists():
        print(f"\nXP-NRTV3: {output_file.name} already exists — skipping")
        return

    print("\nDownloading XP-NRTV3 samples for GW170817...")
    tmp_file = DATA_DIR / "_xpnrtv3_full_tmp.npz"
    download_file(XPNRTV3_URL, tmp_file)

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

    for dataset_name, url in URLS.items():
        print("\n" + "=" * 70)
        print(f"Processing: {dataset_name.upper()}")
        print("=" * 70)

        if url.endswith(".dat.gz"):
            cache_file = (
                DATA_DIR / f"{dataset_name}_PhenomPNRT_posterior_samples.dat.gz"
            )
            if IGNORE_CACHE or not cache_file.exists():
                download_file(url, cache_file)
            else:
                print(f"Cached: {cache_file.name}")

            posterior, header = load_datgz_posterior(cache_file)
            extract_and_save(
                posterior, header, dataset_name, "PhenomPNRT", "LIGO-P1800061", url
            )

        elif url.endswith(".hdf5") or url.endswith(".h5"):
            ext = ".hdf5" if url.endswith(".hdf5") else ".h5"
            cache_file = DATA_DIR / f"{dataset_name}_posterior{ext}"
            if IGNORE_CACHE or not cache_file.exists():
                download_file(url, cache_file)
            else:
                print(f"Cached: {cache_file.name}")

            posterior, header = load_hdf5_posterior(cache_file)

            if posterior is None:
                # Multiple posteriors in this file — process each one
                with h5py.File(cache_file, "r") as f:
                    posterior_names = [k for k in f.keys() if k.endswith("_posterior")]

                for pname in posterior_names:
                    posterior, header = load_hdf5_posterior_by_name(cache_file, pname)
                    parts = pname.replace("_posterior", "").split("_")
                    spin_type = parts[-1].lower() if len(parts) >= 2 else "unknown"
                    waveform = (
                        "_".join(parts[:-1]) + f"_{parts[-1]}"
                        if len(parts) >= 2
                        else pname
                    )
                    full_name = f"{dataset_name}_{spin_type}"
                    extract_and_save(
                        posterior, header, full_name, waveform, "LIGO-P1800370", url
                    )
            else:
                assert header is not None
                extract_and_save(
                    posterior,
                    header,
                    dataset_name,
                    "IMRPhenomPv2NRT",
                    "LIGO-P1800370",
                    url,
                )

    if DOWNLOAD_XPNRTV3:
        print("\n" + "=" * 70)
        print("XP-NRTV3")
        print("=" * 70)
        download_xpnrtv3()

    print("\n" + "=" * 70)
    print("All GW170817 data ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
