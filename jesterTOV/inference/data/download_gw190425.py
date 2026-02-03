#!/usr/bin/env python3
"""
Download and process GW190425 posterior samples from LIGO DCC.

Extracts: mass_1_source, mass_2_source, lambda_1, lambda_2
Converts detector frame masses to source frame using bilby (if needed).
Saves as npz file with metadata.

Data source: https://dcc.ligo.org/P2000026/public

Usage:
    python download_gw190425.py              # Skip download if files exist
    python download_gw190425.py --ignore-cache  # Force re-download
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from bilby.gw.conversion import luminosity_distance_to_redshift  # type: ignore[import-not-found]

# URL for GW190425 posterior samples
URL = "https://dcc.ligo.org/public/0165/P2000026/002/posterior_samples.h5"

# Output directory
DATA_DIR = Path(__file__).parent / "gw190425"
DATA_DIR.mkdir(parents=True, exist_ok=True)

import requests  # type: ignore[import-not-found]
from tqdm import tqdm


def download_file(url, output_path):
    """Download file from URL. This is using requests since urllib.request.urlretrieve is pretty slow"""
    print(f"Downloading from {url}")
    print(f"Saving to {output_path}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))

        with (
            open(output_path, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):  # 16 MB chunks
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    size_kb = output_path.stat().st_size / 1024
    print(f"Download complete: {size_kb:.1f} KB")


def explore_hdf5_structure(hdf5_path):
    """Explore complete HDF5 file structure."""
    print("\n" + "=" * 80)
    print("EXPLORING HDF5 STRUCTURE")
    print("=" * 80)

    with h5py.File(hdf5_path, "r") as f:
        print(f"\nRoot level objects: {list(f.keys())}")

        def print_item(name, obj, indent=0):
            prefix = "  " * indent
            if isinstance(obj, h5py.Group):
                print(f"{prefix}📁 GROUP: {name}")
                if obj.keys():
                    print(f"{prefix}   Contains: {list(obj.keys())}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{prefix}📄 DATASET: {name}")
                print(f"{prefix}   Shape: {obj.shape}, Dtype: {obj.dtype}")
                if obj.dtype.names:
                    print(
                        f"{prefix}   Fields: {obj.dtype.names[:5]}..."
                        if len(obj.dtype.names) > 5
                        else f"{prefix}   Fields: {obj.dtype.names}"
                    )
                print(f"{prefix}   N samples: {len(obj)}")

        f.visititems(print_item)

        # Look for posteriors
        print("\n" + "-" * 80)
        print("SEARCHING FOR POSTERIORS")
        print("-" * 80)

        posteriors = []

        # Search for posterior_samples datasets in all top-level groups
        for key in f.keys():
            if key == "version":
                continue  # Skip version metadata

            if isinstance(f[key], h5py.Group):
                # Check if this group contains posterior_samples
                group = f[key]
                if "posterior_samples" in group:  # type: ignore[operator]
                    full_path = f"{key}/posterior_samples"
                    posteriors.append(full_path)
                    print(f"Found: {full_path}")
            elif "posterior" in key.lower():
                # Also catch root-level posterior datasets (like GW170817)
                posteriors.append(key)
                print(f"Found: {key}")

        print(f"\nFound {len(posteriors)} posterior datasets:")
        for p in posteriors:
            obj = f[p]
            if isinstance(obj, h5py.Dataset):
                print(f"  - {p}: {len(obj)} samples")
            else:
                print(f"  - {p}: GROUP")

        return posteriors


def load_posterior_by_path(hdf5_path, posterior_path):
    """Load a specific posterior from HDF5 file."""
    print(f"\nLoading {posterior_path}")

    with h5py.File(hdf5_path, "r") as f:
        if posterior_path not in f:
            raise ValueError(f"Posterior '{posterior_path}' not found")

        dataset = f[posterior_path]

        # Extract data - check if it's a structured dataset or group
        if isinstance(dataset, h5py.Dataset) and dataset.dtype.names:  # type: ignore[union-attr]
            # Structured array
            header = list(dataset.dtype.names)  # type: ignore[arg-type]
            data = np.array(dataset)
            posterior = {name: data[name] for name in header}  # type: ignore[call-overload]
        elif isinstance(dataset, h5py.Group):
            # Group with individual datasets
            header = list(dataset.keys())
            posterior = {key: np.array(dataset[key]) for key in header}
        else:
            raise ValueError(f"Cannot parse dataset format: {dataset}")

        print(
            f"Loaded {len(header)} parameters, {len(next(iter(posterior.values())))} samples"
        )

        return posterior, header


def explore_parameters(posterior, header):
    """Show all available parameters."""
    print("\n" + "=" * 80)
    print("AVAILABLE PARAMETERS")
    print("=" * 80)

    print(f"\nTotal parameters: {len(header)}")
    print("\nParameter ranges:")
    for p in sorted(header):
        if p in posterior:
            vals = posterior[p]
            print(
                f"  {p:<40} min={vals.min():.3e}, max={vals.max():.3e}, mean={vals.mean():.3e}"
            )


def detector_to_source_frame(m1_det, m2_det, d_L):
    """Convert detector frame masses to source frame using bilby."""
    z = luminosity_distance_to_redshift(d_L)
    m1_source = m1_det / (1 + z)
    m2_source = m2_det / (1 + z)
    return m1_source, m2_source


def extract_and_save(
    posterior, header, dataset_name, waveform_model, data_source="LIGO-P2000026"
):
    """Extract mass_1_source, mass_2_source, lambda_1, lambda_2 and save as npz."""
    print("\n" + "=" * 80)
    print(f"EXTRACTING PARAMETERS: {dataset_name}")
    print("=" * 80)

    # Check what's available
    has_source_masses = "mass_1_source" in header or "m1_source" in header
    has_detector_masses = "m1_detector_frame_Msun" in header or "mass_1" in header

    print("\nParameter detection:")
    print(f"  Source frame masses: {has_source_masses}")
    print(f"  Detector frame masses: {has_detector_masses}")

    # Extract or convert masses
    if has_source_masses:
        print("\n✓ Using source frame masses directly")
        if "mass_1_source" in header:
            m1_source = posterior["mass_1_source"]
            m2_source = posterior["mass_2_source"]
        else:
            m1_source = posterior["m1_source"]
            m2_source = posterior["m2_source"]

    elif has_detector_masses:
        print("\n→ Converting detector frame to source frame using bilby")

        # Try different naming conventions
        if "m1_detector_frame_Msun" in header:
            m1_det = posterior["m1_detector_frame_Msun"]
            m2_det = posterior["m2_detector_frame_Msun"]
            d_L = posterior["luminosity_distance_Mpc"]
        elif "mass_1" in header:
            m1_det = posterior["mass_1"]
            m2_det = posterior["mass_2"]
            d_L = posterior.get("luminosity_distance", posterior.get("distance"))
            if d_L is None:
                raise ValueError("Cannot find luminosity distance")
        else:
            raise ValueError("Cannot identify mass parameters")

        m1_source, m2_source = detector_to_source_frame(m1_det, m2_det, d_L)

        print(f"  m1_detector: {m1_det.mean():.3f} ± {m1_det.std():.3f} Msun")
        print(f"  m1_source:   {m1_source.mean():.3f} ± {m1_source.std():.3f} Msun")
        print(f"  m2_detector: {m2_det.mean():.3f} ± {m2_det.std():.3f} Msun")
        print(f"  m2_source:   {m2_source.mean():.3f} ± {m2_source.std():.3f} Msun")
    else:
        raise ValueError("Cannot find mass parameters in any known format")

    # Extract lambdas
    if "lambda_1" in header:
        lambda_1 = posterior["lambda_1"]
        lambda_2 = posterior["lambda_2"]
    elif "lambda1" in header:
        lambda_1 = posterior["lambda1"]
        lambda_2 = posterior["lambda2"]
    else:
        print("WARNING: No lambda_1/lambda_2 found")
        print(
            f"Available lambda-like parameters: {[p for p in header if 'lambda' in p.lower() or 'tidal' in p.lower()]}"
        )
        return None

    print(f"  lambda_1: {lambda_1.mean():.3f} ± {lambda_1.std():.3f}")
    print(f"  lambda_2: {lambda_2.mean():.3f} ± {lambda_2.std():.3f}")

    # Save
    extracted = {
        "mass_1_source": m1_source,
        "mass_2_source": m2_source,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
    }

    metadata = {
        "event": "GW190425",
        "waveform_model": waveform_model,
        "dataset": dataset_name,
        "data_source": data_source,
        "dcc_url": URL,
        "n_samples": len(m1_source),
        "parameters": ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
        "conversion_tool": "N/A" if has_source_masses else "bilby.gw.conversion",
        "notes": (
            "Source frame masses provided"
            if has_source_masses
            else "Converted from detector frame"
        ),
    }

    output_file = DATA_DIR / f"gw190425_{dataset_name}_posterior.npz"
    # Metadata is stored as numpy object array
    np.savez(output_file, **extracted, metadata=metadata)  # type: ignore[arg-type]

    print(f"\n✓ Saved to: {output_file}")
    print(f"  Parameters: {metadata['parameters']}")
    print(f"  N samples: {metadata['n_samples']}")

    return output_file


def main():
    """Main processing function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Download and process GW190425 posterior samples"
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Force re-download even if files exist",
    )
    args = parser.parse_args()

    # Download
    h5_file = DATA_DIR / "posterior_samples.h5"
    if args.ignore_cache or not h5_file.exists():
        if args.ignore_cache and h5_file.exists():
            print("--ignore-cache: Re-downloading despite existing file")
        download_file(URL, h5_file)
    else:
        print(f"File already exists: {h5_file} (use --ignore-cache to re-download)")

    # Explore structure
    posteriors = explore_hdf5_structure(h5_file)

    # Process each posterior found
    print("\n" + "=" * 80)
    print("PROCESSING POSTERIORS")
    print("=" * 80)

    for post_path in posteriors:
        if not isinstance(post_path, str) or "/" not in post_path:
            continue  # Skip groups

        print(f"\n{'='*80}")
        print(f"PROCESSING: {post_path}")
        print("=" * 80)

        try:
            # Load
            posterior, header = load_posterior_by_path(h5_file, post_path)

            # Explore
            explore_parameters(posterior, header)

            # Extract waveform model and variant from path
            # e.g., "C01:IMRPhenomPv2/posterior_samples" -> waveform="IMRPhenomPv2", variant="C01"
            parts = post_path.split("/")
            if len(parts) == 2:
                variant = parts[0].replace(":", "_")
                waveform_base = (
                    parts[0].split(":")[-1] if ":" in parts[0] else "Unknown"
                )
                dataset_name = variant.lower()
                waveform_model = waveform_base
            else:
                dataset_name = post_path.replace("/", "_").lower()
                waveform_model = "Unknown"

            # Extract and save
            output_file = extract_and_save(
                posterior, header, dataset_name, waveform_model
            )

            # Verify
            if output_file and output_file.exists():
                print("\n" + "-" * 80)
                print("VERIFICATION")
                print("-" * 80)
                loaded = np.load(output_file, allow_pickle=True)
                for key in ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]:
                    if key in loaded:
                        print(
                            f"  {key}: shape={loaded[key].shape}, mean={loaded[key].mean():.3f}"
                        )
                print(f"\nMetadata: {loaded['metadata'].item()}")

        except Exception as e:
            print(f"ERROR processing {post_path}: {e}")
            continue


if __name__ == "__main__":
    main()
