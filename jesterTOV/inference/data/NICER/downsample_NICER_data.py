#!/usr/bin/env python3
"""
Downsample NICER mass-radius posterior samples to reduce file sizes.

This script randomly downsamples all NICER datasets to a maximum of 100,000
samples per file, which is sufficient for EOS inference while significantly
reducing storage requirements.

Usage:
    python downsample_NICER_data.py [--dry-run]

Options:
    --dry-run  Show what would be downsampled without actually doing it
"""

import argparse
from pathlib import Path

import numpy as np


def downsample_file(filepath: Path, max_samples: int = 100000, dry_run: bool = False):
    """
    Downsample a single NICER npz file.

    Args:
        filepath: Path to npz file
        max_samples: Maximum number of samples to keep
        dry_run: If True, don't actually modify files

    Returns:
        Tuple of (original_size_mb, new_size_mb, was_downsampled)
    """
    # Load data
    data = np.load(filepath, allow_pickle=True)
    radius = data['radius']
    mass = data['mass']
    metadata = data['metadata'].item()

    n_samples = len(radius)
    original_size_mb = filepath.stat().st_size / (1024**2)

    # Check if downsampling needed
    if n_samples <= max_samples:
        return original_size_mb, original_size_mb, False

    if dry_run:
        # Estimate new size
        new_size_mb = original_size_mb * (max_samples / n_samples)
        return original_size_mb, new_size_mb, True

    # Random downsampling
    rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
    indices = rng.choice(n_samples, size=max_samples, replace=False)
    indices = np.sort(indices)  # Keep original ordering

    radius_downsampled = radius[indices]
    mass_downsampled = mass[indices]

    # Update metadata
    metadata['original_n_samples'] = n_samples
    metadata['downsampled_to'] = max_samples
    metadata['downsampling_method'] = 'random_choice'
    metadata['downsampling_seed'] = 42

    # Save downsampled data
    np.savez(
        filepath,
        radius=radius_downsampled,
        mass=mass_downsampled,
        metadata=metadata
    )

    new_size_mb = filepath.stat().st_size / (1024**2)

    return original_size_mb, new_size_mb, True


def main():
    parser = argparse.ArgumentParser(
        description='Downsample NICER mass-radius posterior samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downsampled without modifying files'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100000,
        help='Maximum number of samples per file (default: 100000)'
    )

    args = parser.parse_args()

    # Get all npz files
    nicer_dir = Path(__file__).parent
    npz_files = sorted(nicer_dir.glob('*.npz'))

    if not npz_files:
        print("No npz files found in NICER directory")
        return

    print("NICER Data Downsampling")
    print("=" * 80)
    print(f"Target: {args.max_samples:,} samples per file")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    total_original = 0
    total_new = 0
    files_downsampled = 0
    files_skipped = 0

    for filepath in npz_files:
        original_size, new_size, was_downsampled = downsample_file(
            filepath, args.max_samples, args.dry_run
        )

        total_original += original_size
        total_new += new_size

        if was_downsampled:
            files_downsampled += 1
            reduction = (1 - new_size / original_size) * 100
            status = "would downsample" if args.dry_run else "downsampled"
            print(f"✓ {filepath.name}")
            print(f"  {status}: {original_size:.2f} MB → {new_size:.2f} MB "
                  f"({reduction:.1f}% reduction)")
        else:
            files_skipped += 1

    print()
    print("=" * 80)
    print("Summary:")
    print(f"  Files downsampled: {files_downsampled}")
    print(f"  Files skipped: {files_skipped} (already ≤{args.max_samples:,} samples)")
    print(f"  Total size: {total_original:.2f} MB → {total_new:.2f} MB")
    print(f"  Total reduction: {total_original - total_new:.2f} MB "
          f"({(1 - total_new/total_original)*100:.1f}%)")

    if args.dry_run:
        print()
        print("⚠ This was a dry run. No files were modified.")
        print("  Run without --dry-run to actually downsample.")


if __name__ == "__main__":
    main()
