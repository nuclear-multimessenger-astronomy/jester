#!/usr/bin/env python3
"""
Download XP-NRTV3 samples for GW170817 and GW190425 from neural_priors repository.

This script downloads the full samples.npz files, extracts only the relevant
tidal parameters, and saves them to the data directory.
"""

import urllib.request
from pathlib import Path
import numpy as np

# Base URL for neural_priors repository
BASE_URL = "https://raw.githubusercontent.com/ThibeauWouters/neural_priors/b0ae4235f0c74a6f9e2f6cc4c3385a3ac780d4f8/final_results"

# Events to download
EVENTS = ["GW170817", "GW190425"]

# Parameters to extract
PARAMS_TO_EXTRACT = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]


def download_samples(event: str, data_dir: Path) -> Path:
    """Download full samples file for a given event."""
    url = f"{BASE_URL}/{event}/bns/default/samples.npz"
    output_dir = data_dir / event.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{event.lower()}_xp_nrtv3_full.npz"

    print(f"Downloading {event} samples from {url}...")
    urllib.request.urlretrieve(url, output_file)
    print(f"Saved to {output_file}")

    return output_file


def extract_parameters(full_file: Path) -> Path:
    """Extract relevant parameters and save to new file."""
    print(f"Loading {full_file}...")
    data = np.load(full_file)

    # Extract only the parameters we need
    extracted = {}
    for param in PARAMS_TO_EXTRACT:
        if param in data:
            extracted[param] = data[param]
            print(f"  Extracted {param}: shape {data[param].shape}")
        else:
            print(f"  WARNING: {param} not found in file!")

    # Save extracted data
    output_file = full_file.parent / full_file.name.replace("_full.npz", ".npz")
    np.savez(output_file, **extracted)
    print(f"Saved extracted parameters to {output_file}")

    return output_file


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


def main():
    """Main execution function."""
    # Get data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir

    print("=" * 80)
    print("Downloading XP-NRTV3 samples from neural_priors repository")
    print("=" * 80)

    for event in EVENTS:
        print(f"\n{'='*80}")
        print(f"Processing {event}")
        print(f"{'='*80}")

        # Download full file
        full_file = download_samples(event, data_dir)
        full_size = get_file_size_mb(full_file)
        print(f"Full file size: {full_size:.2f} MB")

        # Extract parameters
        extracted_file = extract_parameters(full_file)
        extracted_size = get_file_size_mb(extracted_file)
        print(f"Extracted file size: {extracted_size:.2f} MB")

        # Delete full file
        print(f"Deleting full file {full_file}...")
        full_file.unlink()
        print("Done!")

        # Verify extracted file
        print(f"\nVerifying {extracted_file}...")
        data = np.load(extracted_file)
        print(f"Parameters in file: {list(data.keys())}")
        for param in data.keys():
            print(f"  {param}: shape {data[param].shape}, dtype {data[param].dtype}")

        print(f"\n✓ {event} processing complete!")
        print(f"  Final file: {extracted_file}")
        print(f"  Size: {extracted_size:.2f} MB")
        print(f"  Size reduction: {full_size - extracted_size:.2f} MB ({(1 - extracted_size/full_size)*100:.1f}% smaller)")

    print("\n" + "=" * 80)
    print("All downloads complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
