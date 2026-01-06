#!/usr/bin/env python3
"""
Explore and extract NICER mass-radius posterior samples.

This script processes NICER datasets for PSR J0030+0451 and PSR J0740+6620
from both Amsterdam and Maryland analysis groups, extracting mass-radius
samples into lightweight npz files for use in JESTER inference.

Usage:
    python explore_and_extract_nicer.py [--extract-amsterdam] [--ignore-cache]

Options:
    --extract-amsterdam  Extract Amsterdam tar.gz files (WARNING: large files)
    --ignore-cache       Force re-extraction even if output files exist
"""

import argparse
import gzip
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_maryland_txt(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Parse Maryland group txt files (format: columns with # headers).

    Args:
        filepath: Path to txt file

    Returns:
        radius: Radius samples in km
        mass: Mass samples in solar masses
        metadata: Dictionary with file information
    """
    print(f"\nProcessing Maryland file: {filepath.name}")

    # Read file, skipping comment lines
    data = np.loadtxt(filepath, comments='#')

    # Extract columns
    radius = data[:, 0]  # Column 1: Radius (km)
    mass = data[:, 1]    # Column 2: Mass (Msun)

    # Read header to get metadata
    header_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line.strip())
            else:
                break

    # Parse filename to get configuration info
    filename = filepath.stem

    # Determine PSR
    if 'J0030' in filename:
        psr = 'J0030+0451'
    elif 'J0740' in filename:
        psr = 'J0740+6620'
    else:
        psr = 'Unknown'

    # Determine hotspot model
    if '2spot' in filename:
        hotspot = '2spot'
    elif '3spot' in filename:
        hotspot = '3spot'
    else:
        hotspot = 'unknown'

    # Determine data used
    if 'NICER+XMM-relative' in filename:
        data_used = 'NICER+XMM-relative'
    elif 'NICER+XMM' in filename:
        data_used = 'NICER+XMM'
    elif 'NICER-only' in filename:
        data_used = 'NICER-only'
    else:
        data_used = 'NICER-only'  # Default for J0030

    # Determine model variant
    if 'RM' in filename:
        variant = 'RM'
    elif 'full' in filename:
        variant = 'full'
    else:
        variant = 'unknown'

    metadata = {
        'psr': psr,
        'group': 'maryland',
        'hotspot_model': hotspot,
        'data_used': data_used,
        'model_variant': variant,
        'n_samples': len(radius),
        'source_file': filepath.name,
        'header': '\n'.join(header_lines),
        'zenodo_record': 'https://zenodo.org/records/3473466' if psr == 'J0030+0451' else 'https://zenodo.org/records/4670689',
        'paper': 'Miller et al. 2019 (ApJL 887 L24)' if psr == 'J0030+0451' else 'Miller et al. 2021 (ApJL 918 L28)',
    }

    print(f"  PSR: {psr}")
    print(f"  Group: maryland")
    print(f"  Hotspot model: {hotspot}")
    print(f"  Data used: {data_used}")
    print(f"  Variant: {variant}")
    print(f"  Samples: {len(radius):,}")
    print(f"  Radius range: [{radius.min():.2f}, {radius.max():.2f}] km")
    print(f"  Mass range: [{mass.min():.3f}, {mass.max():.3f}] Msun")

    return radius, mass, metadata


def save_nicer_samples(
    output_dir: Path,
    psr: str,
    group: str,
    hotspot: str,
    data_used: str,
    variant: str,
    radius: np.ndarray,
    mass: np.ndarray,
    metadata: Dict
):
    """
    Save NICER M-R samples to npz file with descriptive naming.

    Filename format: {psr}_{group}_{hotspot}_{data_used}_{variant}.npz
    Example: J0030_maryland_2spot_NICER-only_full.npz
    """
    # Clean up names for filename
    psr_clean = psr.replace('+', '')  # J0030+0451 -> J0030
    data_clean = data_used.replace('+', '').replace('-', '_')  # NICER+XMM -> NICERXMM

    filename = f"{psr_clean}_{group}_{hotspot}_{data_clean}_{variant}.npz"
    filepath = output_dir / filename

    # Save
    # Metadata is stored as numpy object array
    np.savez(
        filepath,
        radius=radius,
        mass=mass,
        metadata=metadata  # type: ignore[arg-type]
    )

    print(f"  ✓ Saved: {filename}")
    print(f"    Size: {filepath.stat().st_size / 1024:.1f} KB")

    return filepath


def process_maryland_data(zenodo_dir: Path, output_dir: Path, ignore_cache: bool = False):
    """Process all Maryland txt files."""
    print("\n" + "="*80)
    print("PROCESSING MARYLAND DATA")
    print("="*80)

    maryland_files = [
        zenodo_dir / "J0030/maryland/original/J0030_2spot_RM.txt",
        zenodo_dir / "J0030/maryland/original/J0030_2spot_full.txt",
        zenodo_dir / "J0030/maryland/original/J0030_3spot_RM.txt",
        zenodo_dir / "J0030/maryland/original/J0030_3spot_full.txt",
        zenodo_dir / "J0740/maryland/original/NICER-only_J0740_RM.txt",
        zenodo_dir / "J0740/maryland/original/NICER-only_J0740_full.txt",
        zenodo_dir / "J0740/maryland/original/NICER+XMM_J0740_RM.txt",
        zenodo_dir / "J0740/maryland/original/NICER+XMM_J0740_full.txt",
        zenodo_dir / "J0740/maryland/original/NICER+XMM-relative_J0740_RM.txt",
        zenodo_dir / "J0740/maryland/original/NICER+XMM-relative_J0740_full.txt",
    ]

    processed_files = []

    for filepath in maryland_files:
        if not filepath.exists():
            print(f"\n⚠ File not found: {filepath.name}")
            continue

        # Parse data
        radius, mass, metadata = parse_maryland_txt(filepath)

        # Generate output filename
        psr = metadata['psr']
        group = metadata['group']
        hotspot = metadata['hotspot_model']
        data_used = metadata['data_used']
        variant = metadata['model_variant']

        # Clean up names for filename
        psr_clean = psr.replace('+', '')
        data_clean = data_used.replace('+', '').replace('-', '_')

        output_filename = f"{psr_clean}_{group}_{hotspot}_{data_clean}_{variant}.npz"
        output_filepath = output_dir / output_filename

        # Check cache
        if output_filepath.exists() and not ignore_cache:
            print(f"  ℹ Output already exists: {output_filename}")
            print(f"    (use --ignore-cache to re-extract)")
            processed_files.append(output_filepath)
            continue

        # Save
        saved_path = save_nicer_samples(
            output_dir, psr, group, hotspot, data_used, variant,
            radius, mass, metadata
        )
        processed_files.append(saved_path)

    return processed_files


def explore_amsterdam_archives(zenodo_dir: Path):
    """
    Explore Amsterdam tar.gz archives without extracting.

    Based on README files, we know:
    - J0030 Riley 2019: M-R files are {MODEL}/{MODEL}__M_R.txt (ST-S, ST-U, ST+PST, etc.)
    - J0030 Vinciguerra 2023: MultiNest outputs in *_outputs/run1*/ directories
    - J0740 Salmi 2022: MultiNest outputs, recommended: STU_NICER_3c50bkgsms_hr_df3X
    - J0740 recent: mr_samples_and_contours.tar.gz (likely ready-to-use M-R samples)
    """
    print("\n" + "="*80)
    print("EXPLORING AMSTERDAM ARCHIVES (not extracting)")
    print("="*80)

    archives = {
        'J0030 Amsterdam (Riley et al. 2019)': {
            'path': zenodo_dir / "J0030/amsterdam/original/A_NICER_VIEW_OF_PSR_J0030p0451.tar.gz",
            'mr_files': ['ST_S/ST_S__M_R.txt', 'ST_U/ST_U__M_R.txt', 'CDT_U/CDT_U__M_R.txt',
                         'ST_EST/ST_EST__M_R.txt', 'ST_PST/ST_PST__M_R.txt'],
            'format': 'weight, -2*log(L), mass (Msun), radius (km)',
        },
        'J0030 Amsterdam (Vinciguerra et al. 2023)': {
            'path': zenodo_dir / "J0030/amsterdam/intermediate/updated_analyses_PSRJ0030_up_to_2018_NICER_data.tar.gz",
            'mr_files': ['ST_U/*_outputs/run1*/', 'ST_PST/*_outputs/run1*/',
                         'ST_PDT/*_outputs/run1*/', 'PDT_U/*_outputs/run1*/'],
            'format': 'MultiNest output files',
        },
        'J0740 Amsterdam (Salmi et al. 2022)': {
            'path': zenodo_dir / "J0740/amsterdam/intermediate/psr_J0740+6620_with_NICER_background_estimates.tar.gz",
            'mr_files': ['STU/3C50/STU_NICER_3c50bkgsms_hr_df3X/STU_outputs/run1/',
                         'STU/W21_SW/STU_NICER_old/', 'STU/3C50_XMM/*_outputs/run1/'],
            'format': 'MultiNest output files, recommend: STU_NICER_3c50bkgsms_hr_df3X',
        },
        'J0740 Amsterdam (Salmi et al. recent)': {
            'path': zenodo_dir / "J0740/amsterdam/recent/mr_samples_and_contours.tar.gz",
            'mr_files': ['mr_samples/', 'contours/'],
            'format': 'Ready-to-use M-R samples (likely)',
        },
    }

    for name, info in archives.items():
        archive_path = info['path']
        if not archive_path.exists():
            print(f"\n⚠ Archive not found: {name}")
            continue

        print(f"\n{name}")
        print(f"  Path: {archive_path}")
        size_gb = archive_path.stat().st_size / (1024**3)
        print(f"  Size: {size_gb:.2f} GB")
        print(f"  Expected M-R files: {', '.join(info['mr_files'][:3])}")
        print(f"  Format: {info['format']}")

        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                members = tar.getmembers()
                print(f"  Total files in archive: {len(members)}")

                # Search for the expected M-R files
                found_files = []
                for pattern in info['mr_files']:
                    # Convert pattern to simple matching
                    matching = [m for m in members if any(part in m.name for part in pattern.split('/'))]
                    found_files.extend(matching)

                if found_files:
                    print(f"  ✓ Found {len(found_files)} matching M-R files:")
                    for m in found_files[:10]:
                        print(f"    - {m.name} ({m.size / 1024:.1f} KB)")
                    if len(found_files) > 10:
                        print(f"    ... and {len(found_files) - 10} more")
                else:
                    print("  ⚠ No M-R files found matching expected patterns")
                    print("  Searching for any M_R or mass/radius files:")
                    mr_files = [m for m in members if 'M_R' in m.name or ('mass' in m.name.lower() and 'radius' in m.name.lower())]
                    if mr_files:
                        for m in mr_files[:10]:
                            print(f"    - {m.name} ({m.size / 1024:.1f} KB)")

        except Exception as e:
            print(f"  ✗ Error reading archive: {e}")


def parse_riley2019_mr_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Parse Riley et al. 2019 M-R files.

    Format: weight, -2*log(L), mass (Msun), radius (km)
    """
    data = np.loadtxt(filepath, comments='#')

    # Columns: weight, -2*log(L), mass, radius
    weights = data[:, 0]
    mass = data[:, 2]
    radius = data[:, 3]

    # Extract model name from path (e.g., ST_PST/ST_PST__M_R.txt -> ST_PST)
    model = filepath.parent.name if filepath.parent.name != 'amsterdam' else filepath.stem.replace('__M_R', '')

    metadata = {
        'psr': 'J0030+0451',
        'group': 'amsterdam',
        'analysis': 'Riley et al. 2019',
        'hotspot_model': model,
        'data_used': 'NICER-only',
        'n_samples': len(mass),
        'weighted': True,
        'source_file': filepath.name,
        'zenodo_record': 'https://zenodo.org/records/3524457',
        'paper': 'Riley et al. 2019 (ApJL 887 L21)',
        'format': 'columns: weight, -2*log(L), mass (Msun), radius (km)',
    }

    return radius, mass, metadata


def parse_salmi_recent_mr_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Parse Salmi et al. recent M-R files.

    These are ready-to-use M-R samples from the most recent analysis.
    """
    data = np.loadtxt(filepath, comments='#')

    # Determine format based on filename
    if 'wmrsamples' in filepath.name:
        # Weighted samples: columns are weight, mass, radius
        weights = data[:, 0]
        radius = data[:, 2]  # Radius in km (third column)
        mass = data[:, 1]    # Mass in Msun (second column)
        weighted = True
    elif 'post_equal_weights' in filepath.name:
        # Equal weight samples: columns are mass, radius
        radius = data[:, 1]  # Radius in km (second column)
        mass = data[:, 0]    # Mass in Msun (first column)
        weighted = False
    else:
        # Default: assume mass, radius
        radius = data[:, 1]
        mass = data[:, 0]
        weighted = False

    metadata = {
        'psr': 'J0740+6620',
        'group': 'amsterdam',
        'analysis': 'Salmi et al. (recent)',
        'hotspot_model': 'gamma',  # From filename J0740_gamma_NxX
        'data_used': 'NICER+XMM',  # NxX = NICER+XMM
        'n_samples': len(mass),
        'weighted': weighted,
        'source_file': filepath.name,
        'zenodo_record': 'https://zenodo.org/records/10519473',
        'paper': 'Salmi et al. (in prep)',
        'settings': 'lp40k_se001',  # 40k live points, sampling efficiency 0.01
    }

    return radius, mass, metadata


def extract_amsterdam_data(zenodo_dir: Path, output_dir: Path, ignore_cache: bool = False):
    """
    Extract Amsterdam M-R samples from tar.gz archives.

    Extracts:
    1. Riley et al. 2019: ST-S, ST-U, ST+PST, etc. M-R files
    2. Salmi et al. recent: Ready-to-use M-R samples
    3. (Optional) Vinciguerra 2023 and Salmi 2022 MultiNest outputs
    """
    print("\n" + "="*80)
    print("EXTRACTING AMSTERDAM DATA")
    print("="*80)
    print("\n⚠ This will extract tar.gz files and may take several minutes...")

    extracted_files = []

    # 1. Extract Riley et al. 2019 M-R files
    print("\n--- J0030 Riley et al. 2019 ---")
    riley2019_archive = zenodo_dir / "J0030/amsterdam/original/A_NICER_VIEW_OF_PSR_J0030p0451.tar.gz"
    riley2019_mr_files = [
        'A_NICER_VIEW_OF_PSR_J0030p0451/ST_S/ST_S__M_R.txt',
        'A_NICER_VIEW_OF_PSR_J0030p0451/ST_U/ST_U__M_R.txt',
        'A_NICER_VIEW_OF_PSR_J0030p0451/CDT_U/CDT_U__M_R.txt',
        'A_NICER_VIEW_OF_PSR_J0030p0451/ST_EST/ST_EST__M_R.txt',
        'A_NICER_VIEW_OF_PSR_J0030p0451/ST_PST/ST_PST__M_R.txt',
    ]

    if riley2019_archive.exists():
        print(f"Extracting from: {riley2019_archive.name}")
        with tarfile.open(riley2019_archive, 'r:gz') as tar:
            for mr_file_path in riley2019_mr_files:
                try:
                    # Extract model name (e.g., ST_PST)
                    model = mr_file_path.split('/')[1]

                    # Check if already extracted
                    output_filename = f"J00300451_amsterdam_{model}_NICER_only_Riley2019.npz"
                    output_filepath = output_dir / output_filename

                    if output_filepath.exists() and not ignore_cache:
                        print(f"  ℹ Already exists: {output_filename}")
                        extracted_files.append(output_filepath)
                        continue

                    # Extract to temporary location
                    print(f"  Extracting: {model}...")
                    member = tar.getmember(mr_file_path)

                    # Extract to a temporary file
                    extracted_file = tar.extractfile(member)
                    if extracted_file is None:
                        raise ValueError(f"Could not extract {mr_file_path}")

                    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as tmp:
                        tmp.write(extracted_file.read())
                        tmp_path = Path(tmp.name)

                    # Parse the data
                    radius, mass, metadata = parse_riley2019_mr_file(tmp_path)

                    # Clean up temp file
                    tmp_path.unlink()

                    # Save (metadata as numpy object array)
                    np.savez(
                        output_filepath,
                        radius=radius,
                        mass=mass,
                        metadata=metadata  # type: ignore[arg-type]
                    )

                    print(f"  ✓ Saved: {output_filename}")
                    print(f"    Samples: {len(radius):,}, Size: {output_filepath.stat().st_size / 1024:.1f} KB")
                    extracted_files.append(output_filepath)

                except KeyError:
                    print(f"  ⚠ File not found in archive: {mr_file_path}")
                except Exception as e:
                    print(f"  ✗ Error extracting {mr_file_path}: {e}")
    else:
        print(f"⚠ Archive not found: {riley2019_archive}")

    # 2. Extract Salmi et al. recent M-R files (equal-weight samples only)
    print("\n--- J0740 Salmi et al. (recent) ---")
    salmi_recent_archive = zenodo_dir / "J0740/amsterdam/recent/mr_samples_and_contours.tar.gz"
    salmi_recent_mr_files = [
        'mr_samples_and_contours/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat',
    ]

    if salmi_recent_archive.exists():
        print(f"Extracting from: {salmi_recent_archive.name}")
        with tarfile.open(salmi_recent_archive, 'r:gz') as tar:
            for mr_file_path in salmi_recent_mr_files:
                try:
                    # Use equal-weight samples (sufficient for EOS inference)
                    variant = 'equal_weights'

                    # Check if already extracted
                    output_filename = f"J07406620_amsterdam_gamma_NICERXMM_{variant}_recent.npz"
                    output_filepath = output_dir / output_filename

                    if output_filepath.exists() and not ignore_cache:
                        print(f"  ℹ Already exists: {output_filename}")
                        extracted_files.append(output_filepath)
                        continue

                    # Extract to temporary location
                    print(f"  Extracting equal-weight samples...")
                    member = tar.getmember(mr_file_path)

                    # Extract to a temporary file
                    extracted_file = tar.extractfile(member)
                    if extracted_file is None:
                        raise ValueError(f"Could not extract {mr_file_path}")

                    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.dat') as tmp:
                        tmp.write(extracted_file.read())
                        tmp_path = Path(tmp.name)

                    # Parse the data
                    radius, mass, metadata = parse_salmi_recent_mr_file(tmp_path)

                    # Clean up temp file
                    tmp_path.unlink()

                    # Save (metadata as numpy object array)
                    np.savez(
                        output_filepath,
                        radius=radius,
                        mass=mass,
                        metadata=metadata  # type: ignore[arg-type]
                    )

                    print(f"  ✓ Saved: {output_filename}")
                    print(f"    Samples: {len(radius):,}, Size: {output_filepath.stat().st_size / 1024:.1f} KB")
                    extracted_files.append(output_filepath)

                except KeyError:
                    print(f"  ⚠ File not found in archive: {mr_file_path}")
                except Exception as e:
                    print(f"  ✗ Error extracting {mr_file_path}: {e}")
    else:
        print(f"⚠ Archive not found: {salmi_recent_archive}")

    print(f"\n✓ Extracted {len(extracted_files)} Amsterdam files")
    return extracted_files


def print_summary(output_dir: Path):
    """Print summary of extracted files."""
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)

    npz_files = sorted(output_dir.glob("*.npz"))

    if not npz_files:
        print("\nNo files extracted yet.")
        return

    print(f"\nExtracted {len(npz_files)} files to {output_dir}\n")

    # Group by PSR
    j0030_files = [f for f in npz_files if 'J0030' in f.name]
    j0740_files = [f for f in npz_files if 'J0740' in f.name]

    if j0030_files:
        print("J0030+0451:")
        for f in j0030_files:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")

    if j0740_files:
        print("\nJ0740+6620:")
        for f in j0740_files:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")

    # Print total size
    total_size = sum(f.stat().st_size for f in npz_files) / (1024**2)
    print(f"\nTotal size: {total_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Explore and extract NICER mass-radius posterior samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--extract-amsterdam',
        action='store_true',
        help='Extract Amsterdam tar.gz files (WARNING: large files, may take time)'
    )
    parser.add_argument(
        '--ignore-cache',
        action='store_true',
        help='Force re-extraction even if output files exist'
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    zenodo_dir = script_dir / "zenodo_data"
    output_dir = script_dir / "NICER"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print("NICER Data Exploration and Extraction")
    print("="*80)
    print(f"Zenodo data directory: {zenodo_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ignore cache: {args.ignore_cache}")

    # Process Maryland data (always available)
    process_maryland_data(zenodo_dir, output_dir, args.ignore_cache)

    # Explore Amsterdam archives
    explore_amsterdam_archives(zenodo_dir)

    # Extract Amsterdam data if requested
    if args.extract_amsterdam:
        extract_amsterdam_data(zenodo_dir, output_dir, args.ignore_cache)

    # Print summary
    print_summary(output_dir)

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
