"""
Exploration script for NICER datasets from Zenodo

This script downloads NICER datasets and explores their structure to understand:
- What files are included
- What hotspot models are available
- Data format and columns
- Number of samples

By default, the script skips downloads if files already exist (caching enabled).

Usage:
    uv run python explore_nicer_datasets.py --psr J0030 --group amsterdam --version recent
    uv run python explore_nicer_datasets.py --list  # List all available datasets
    uv run python explore_nicer_datasets.py --explore-all  # Explore all datasets (WARNING: large downloads!)
    uv run python explore_nicer_datasets.py --psr J0030 --group amsterdam --version recent --ignore-cache  # Force re-download
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from zenodo_downloader import ZenodoDownloader, ZENODO_DATASETS


def list_files_in_directory(directory: Path, indent: int = 0) -> None:
    """
    Recursively list all files in a directory

    Parameters
    ----------
    directory : Path
        Directory to list
    indent : int
        Indentation level for nested directories
    """
    if not directory.exists():
        print(f"{'  ' * indent}[Directory does not exist: {directory}]")
        return

    items = sorted(directory.iterdir())
    for item in items:
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"{'  ' * indent}📄 {item.name} ({size_mb:.2f} MB)")
        elif item.is_dir():
            print(f"{'  ' * indent}📁 {item.name}/")
            list_files_in_directory(item, indent + 1)


def explore_text_file(file_path: Path, max_lines: int = 20) -> dict:
    """
    Explore a text/data file to understand its structure

    Parameters
    ----------
    file_path : Path
        Path to the file
    max_lines : int
        Maximum number of lines to display

    Returns
    -------
    dict
        Information about the file
    """
    info = {
        "path": file_path,
        "size_mb": file_path.stat().st_size / (1024 * 1024),
        "readable": False,
        "n_lines": 0,
        "columns": [],
        "n_samples": 0,
        "sample_data": None,
    }

    try:
        # Try reading as text first to see header
        with open(file_path, 'r') as f:
            lines = f.readlines()
            info["n_lines"] = len(lines)
            info["readable"] = True

        print(f"\n{'='*80}")
        print(f"File: {file_path.name}")
        print(f"Size: {info['size_mb']:.2f} MB")
        print(f"Total lines: {info['n_lines']}")
        print(f"{'='*80}")

        # Show first few lines
        print(f"\nFirst {min(max_lines, len(lines))} lines:")
        print("-" * 80)
        for i, line in enumerate(lines[:max_lines]):
            print(f"Line {i+1:3d}: {line.rstrip()}")

        # Try to parse as CSV/data file with different separators
        print(f"\n{'='*80}")
        print("Attempting to parse as data file...")
        print(f"{'='*80}")

        # Try different separators
        for sep in [' ', '\t', ',', '  ', '   ']:
            try:
                # Try reading with pandas, skipping different numbers of header lines
                for skiprows in [0, 1, 6]:  # Common header sizes
                    try:
                        df = pd.read_csv(file_path, sep=sep, skiprows=skiprows, header=None)

                        # Check if we got reasonable data (more than 1 column, reasonable number of rows)
                        if df.shape[1] > 1 and df.shape[0] > 10:
                            print(f"\n✓ Successfully parsed with sep='{sep}', skiprows={skiprows}")
                            print(f"  Shape: {df.shape} (rows × columns)")
                            print(f"  Columns: {list(df.columns)}")

                            # Show data types
                            print(f"\n  Data types:")
                            for col in df.columns:
                                dtype = df[col].dtype
                                n_unique = df[col].nunique()
                                print(f"    Column {col}: {dtype}, {n_unique} unique values")

                            # Show basic statistics for numeric columns
                            print(f"\n  Statistics for first few columns:")
                            print(df.iloc[:, :min(5, df.shape[1])].describe())

                            # Show first few rows
                            print(f"\n  First 10 rows:")
                            print(df.head(10))

                            # Store info
                            info["columns"] = list(df.columns)
                            info["n_samples"] = df.shape[0]
                            info["sample_data"] = df.head(10)

                            # Try to identify M and R columns
                            print(f"\n  Attempting to identify Mass and Radius columns...")
                            for col in df.columns:
                                data = df[col]
                                if data.dtype in [np.float64, np.float32, float]:
                                    mean_val = data.mean()
                                    std_val = data.std()
                                    min_val = data.min()
                                    max_val = data.max()

                                    # Mass typically 1-3 Msun
                                    if 1.0 < mean_val < 3.0 and 0.1 < std_val < 0.5:
                                        print(f"    Column {col} might be MASS (mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}])")

                                    # Radius typically 10-15 km
                                    if 8.0 < mean_val < 18.0 and 0.5 < std_val < 5.0:
                                        print(f"    Column {col} might be RADIUS (mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}])")

                                    # Weight typically 0-1 or all 1s
                                    if 0.0 <= min_val and max_val <= 1.1:
                                        n_unique = data.nunique()
                                        if n_unique == 1 or (0.0 < std_val < 0.5):
                                            print(f"    Column {col} might be WEIGHT (mean={mean_val:.3f}, std={std_val:.3f}, n_unique={n_unique})")

                            return info
                    except Exception:
                        continue
            except Exception:
                continue

        print("\n✗ Could not parse file with standard separators")

    except Exception as e:
        print(f"\n✗ Error reading file: {e}")

    return info


def explore_dataset(psr_name: str, group: str, version: str, download_dir: Path) -> None:
    """
    Explore a NICER dataset

    Parameters
    ----------
    psr_name : str
        Pulsar name
    group : str
        Analysis group
    version : str
        Dataset version
    download_dir : Path
        Directory containing the dataset
    """
    print(f"\n{'#'*80}")
    print(f"# EXPLORING: {psr_name} / {group} / {version}")
    print(f"# Directory: {download_dir}")
    print(f"{'#'*80}\n")

    # List directory structure
    print("Directory structure:")
    print("-" * 80)
    list_files_in_directory(download_dir)
    print()

    # Find and explore data files
    print("\nSearching for data files (.txt, .dat, .csv, .h5)...")
    print("=" * 80)

    data_extensions = ['.txt', '.dat', '.csv', '.h5', '.fits']
    data_files = []
    for ext in data_extensions:
        data_files.extend(download_dir.rglob(f"*{ext}"))

    if not data_files:
        print("No data files found!")
        return

    print(f"Found {len(data_files)} data file(s):\n")
    for f in data_files:
        print(f"  - {f.relative_to(download_dir)}")

    # Explore each data file
    for data_file in data_files:
        if data_file.suffix in ['.txt', '.dat', '.csv']:
            explore_text_file(data_file)
        else:
            print(f"\nSkipping binary file: {data_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore NICER datasets from Zenodo"
    )
    parser.add_argument(
        "--psr",
        type=str,
        choices=["J0030", "J0740"],
        help="Pulsar name",
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=["amsterdam", "maryland"],
        help="Analysis group",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["recent", "intermediate", "original"],
        help="Dataset version",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets",
    )
    parser.add_argument(
        "--explore-all",
        action="store_true",
        help="Explore all available datasets (WARNING: large downloads!)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download, only explore if data already exists",
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Force re-download even if files already exist",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        help="Override download directory",
    )

    args = parser.parse_args()

    # Initialize downloader
    base_dir = Path(args.download_dir) if args.download_dir else None
    downloader = ZenodoDownloader(base_dir=base_dir)

    # List datasets
    if args.list:
        downloader.list_available_datasets()
        return

    # Explore all datasets
    if args.explore_all:
        print("\n" + "!"*80)
        print("! WARNING: This will download ALL NICER datasets!")
        print("! This may take a long time and use significant disk space.")
        print("!"*80)
        response = input("\nContinue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

        for psr_name in ZENODO_DATASETS:
            for group in ZENODO_DATASETS[psr_name]:
                for version in ZENODO_DATASETS[psr_name][group]:
                    if not args.no_download:
                        download_dir = downloader.download_dataset(
                            psr_name, group, version, force=args.ignore_cache
                        )
                    else:
                        download_dir = downloader.base_dir / psr_name / group / version

                    if download_dir and download_dir.exists():
                        explore_dataset(psr_name, group, version, download_dir)
        return

    # Explore single dataset
    if not args.psr or not args.group or not args.version:
        parser.error("Must specify --psr, --group, and --version (or use --list/--explore-all)")

    # Download dataset
    if not args.no_download:
        download_dir = downloader.download_dataset(
            args.psr, args.group, args.version, force=args.ignore_cache
        )
    else:
        download_dir = downloader.base_dir / args.psr / args.group / args.version

    if download_dir and download_dir.exists():
        explore_dataset(args.psr, args.group, args.version, download_dir)
    else:
        print(f"Dataset directory not found: {download_dir}")


if __name__ == "__main__":
    main()
