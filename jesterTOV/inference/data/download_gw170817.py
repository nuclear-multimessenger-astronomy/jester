#!/usr/bin/env python3
"""
Download and process GW170817 posterior samples from LIGO DCC.

Extracts: mass_1_source, mass_2_source, lambda_1, lambda_2
Converts detector frame masses to source frame using bilby (if needed).
Saves as npz file with metadata.

Supports two formats:
1. .dat.gz (PhenomPNRT posteriors) - ASCII format
2. .hdf5/.h5 (GWTC posteriors) - HDF5 format

Usage:
    python download_gw170817.py              # Skip download if files exist
    python download_gw170817.py --ignore-cache  # Force re-download
"""

import argparse
import urllib.request
import gzip
import numpy as np
import h5py
from pathlib import Path
try:
    from bilby.gw.conversion import luminosity_distance_to_redshift  # type: ignore[import-not-found]
except ImportError:
    raise ImportError("bilby is required for this script. Please install bilby via 'pip install bilby'.")

# URLs for the posterior samples
URLS = {
    # GW170817 - PhenomPNRT posteriors (P1800061)
    "gw170817_low_spin": "https://dcc.ligo.org/public/0150/P1800061/011/low_spin_PhenomPNRT_posterior_samples.dat.gz",
    "gw170817_high_spin": "https://dcc.ligo.org/public/0150/P1800061/011/high_spin_PhenomPNRT_posterior_samples.dat.gz",
    # GW170817 - GWTC-1 posteriors (P1800370)
    "gw170817_gwtc1": "https://dcc.ligo.org/public/0157/P1800370/005/GW170817_GWTC-1.hdf5",
}

# Event-specific output directories
def get_data_dir(event_name):
    """Get output directory for a specific event."""
    base_dir = Path(__file__).parent
    if "gw170817" in event_name.lower():
        return base_dir / "gw170817"
    elif "gw190425" in event_name.lower():
        return base_dir / "gw190425"
    else:
        return base_dir / event_name.lower()

def download_file(url, output_path):
    """Download file from URL."""
    print(f"Downloading from {url}")
    print(f"Saving to {output_path}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Download complete: {output_path.stat().st_size / 1024:.1f} KB")

def load_posterior_samples(gz_path):
    """Load posterior samples from gzipped dat file."""
    print(f"\nLoading {gz_path}")

    # Read the gzipped file
    with gzip.open(gz_path, 'rt') as f:
        lines = f.readlines()

    # First line should be column names
    header = lines[0].strip().split()
    print(f"Found {len(header)} columns")

    # Load data
    data = []
    for line in lines[1:]:
        if line.strip() and not line.startswith('#'):
            data.append([float(x) for x in line.strip().split()])

    data = np.array(data)
    print(f"Loaded {len(data)} samples")

    # Create dictionary with column names
    posterior = {col: data[:, i] for i, col in enumerate(header)}

    return posterior, header

def load_hdf5_posterior(hdf5_path):
    """Load posterior samples from HDF5 file."""
    print(f"\nLoading {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Explore HDF5 structure
        print(f"\nHDF5 structure:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")

        f.visititems(print_structure)

        # Try to find posterior samples
        # Common locations: 'posterior_samples', 'posterior', 'samples', or at root
        posterior_data = None
        header = []

        # Check for multiple posteriors in the file
        # Look for datasets ending with '_posterior'
        posterior_datasets = [key for key in f.keys() if key.endswith('_posterior')]

        if len(posterior_datasets) > 1:
            print(f"\nFound multiple posteriors: {posterior_datasets}")
            print("This file will be processed separately to extract each posterior")
            return None, None  # Signal that this needs special handling

        # Check common group names for single posterior files
        for group_name in ['posterior_samples', 'posterior', 'samples', 'IMRPhenomPv2NRT_highSpin_posterior', 'Overall_posterior']:
            if group_name in f:
                print(f"\nFound posterior group: {group_name}")
                group = f[group_name]

                # If it's a group with datasets, extract them
                if isinstance(group, h5py.Group):
                    header = list(group.keys())
                    posterior_data = {key: np.array(group[key]) for key in header}
                    print(f"Loaded {len(header)} parameters from group")
                    break
                # If it's a dataset, try to extract column names
                elif isinstance(group, h5py.Dataset):
                    data = np.array(group)
                    if 'columns' in group.attrs:
                        # HDF5 attrs can return various array-like types
                        header = list(group.attrs['columns'])  # type: ignore[arg-type]
                    elif data.dtype.names:  # Structured array
                        header = list(data.dtype.names)
                        # Structured array field access is valid but type checker needs help
                        posterior_data = {name: data[name] for name in header}  # type: ignore[call-overload]
                    break

        # If no specific group found, check if datasets are at root level
        if posterior_data is None:
            print("\nNo standard posterior group found, checking root level datasets...")
            root_datasets = [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]
            if root_datasets:
                header = root_datasets
                posterior_data = {key: np.array(f[key]) for key in header}
                print(f"Loaded {len(header)} parameters from root level")

    if posterior_data is None:
        raise ValueError("Could not find posterior samples in HDF5 file")

    if len(next(iter(posterior_data.values()))) > 0:
        print(f"Loaded {len(next(iter(posterior_data.values())))} samples")
    else:
        print("WARNING: No samples found")

    return posterior_data, header

def load_hdf5_posterior_by_name(hdf5_path, posterior_name):
    """Load a specific posterior from HDF5 file by dataset name."""
    print(f"\nLoading {posterior_name} from {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        if posterior_name not in f:
            raise ValueError(f"Posterior '{posterior_name}' not found in file")

        dataset = f[posterior_name]

        # Ensure it's a Dataset not a Group
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"'{posterior_name}' is not a dataset")

        # Extract structured array
        if dataset.dtype.names:  # type: ignore[union-attr]
            header = list(dataset.dtype.names)  # type: ignore[arg-type]
            data = np.array(dataset)
            posterior_data = {name: data[name] for name in header}  # type: ignore[call-overload]
            print(f"Loaded {len(header)} parameters, {len(data)} samples")
        else:
            raise ValueError(f"Dataset {posterior_name} is not a structured array")

    return posterior_data, header

def explore_parameters(posterior, header):
    """Explore available parameters."""
    print("\n" + "="*80)
    print("EXPLORING PARAMETERS")
    print("="*80)

    print(f"\nAll available parameters ({len(header)}):")
    for p in header:
        if p in posterior:
            print(f"  {p}: min={posterior[p].min():.3f}, max={posterior[p].max():.3f}, mean={posterior[p].mean():.3f}")

def detector_to_source_frame(m1_det, m2_det, d_L):
    """
    Convert detector frame masses to source frame using bilby.

    Parameters
    ----------
    m1_det : array
        Primary mass in detector frame (solar masses)
    m2_det : array
        Secondary mass in detector frame (solar masses)
    d_L : array
        Luminosity distance (Mpc)

    Returns
    -------
    m1_source, m2_source : arrays
        Masses in source frame (solar masses)
    """
    # Convert luminosity distance to redshift using bilby (with default cosmology)
    z = luminosity_distance_to_redshift(d_L)

    # Convert to source frame: m_source = m_det / (1 + z)
    m1_source = m1_det / (1 + z)
    m2_source = m2_det / (1 + z)

    return m1_source, m2_source

def extract_and_save(posterior, header, dataset_name, waveform_model="PhenomPNRT", data_source="LIGO-P1800061", event_name="GW170817"):
    """Extract required parameters and save as npz."""
    print("\n" + "="*80)
    print(f"EXTRACTING PARAMETERS ({dataset_name})")
    print("="*80)

    # Get event-specific output directory
    data_dir = get_data_dir(event_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if source frame masses are already present
    has_source_masses = 'mass_1_source' in header or 'm1_source' in header
    has_detector_masses = 'm1_detector_frame_Msun' in header or 'mass_1' in header

    print(f"\nParameter availability:")
    print(f"  Source frame masses: {has_source_masses}")
    print(f"  Detector frame masses: {has_detector_masses}")

    # Extract or convert masses
    if has_source_masses:
        # Already in source frame - just extract
        print("\n✓ Source frame masses already present, using directly...")

        # Try different naming conventions
        if 'mass_1_source' in header:
            m1_source = posterior['mass_1_source']
            m2_source = posterior['mass_2_source']
        elif 'm1_source' in header:
            m1_source = posterior['m1_source']
            m2_source = posterior['m2_source']
        else:
            raise ValueError("Source frame masses found but cannot identify parameter names")

        print(f"  m1_source: {m1_source.mean():.3f} ± {m1_source.std():.3f} Msun")
        print(f"  m2_source: {m2_source.mean():.3f} ± {m2_source.std():.3f} Msun")

    elif has_detector_masses:
        # Need to convert from detector frame
        print("\n→ Converting detector frame masses to source frame using bilby...")

        # Try different naming conventions for detector frame
        if 'm1_detector_frame_Msun' in header:
            m1_det = posterior['m1_detector_frame_Msun']
            m2_det = posterior['m2_detector_frame_Msun']
            d_L = posterior['luminosity_distance_Mpc']
        elif 'mass_1' in header:
            m1_det = posterior['mass_1']
            m2_det = posterior['mass_2']
            d_L = posterior.get('luminosity_distance', posterior.get('distance', None))
            if d_L is None:
                raise ValueError("Cannot find luminosity distance for conversion")
        else:
            raise ValueError("Detector frame masses found but cannot identify parameter names")

        m1_source, m2_source = detector_to_source_frame(m1_det, m2_det, d_L)

        print(f"  m1_detector: {m1_det.mean():.3f} ± {m1_det.std():.3f} Msun")
        print(f"  m1_source:   {m1_source.mean():.3f} ± {m1_source.std():.3f} Msun")
        print(f"  m2_detector: {m2_det.mean():.3f} ± {m2_det.std():.3f} Msun")
        print(f"  m2_source:   {m2_source.mean():.3f} ± {m2_source.std():.3f} Msun")
    else:
        raise ValueError("Cannot find mass parameters in either source or detector frame")

    # Extract lambdas (try different naming conventions)
    if 'lambda_1' in header:
        lambda_1 = posterior['lambda_1']
        lambda_2 = posterior['lambda_2']
    elif 'lambda1' in header:
        lambda_1 = posterior['lambda1']
        lambda_2 = posterior['lambda2']
    elif 'lambdatilde' in header:
        # Some datasets only have lambda_tilde and delta_lambda
        print("WARNING: Only tidal parameters lambda_tilde/delta_lambda found, not individual lambdas")
        print("Skipping this dataset for now - need component lambdas")
        return None
    else:
        print("WARNING: No tidal deformability parameters found")
        print("Available parameters:", header)
        return None

    print(f"  lambda_1: {lambda_1.mean():.3f} ± {lambda_1.std():.3f}")
    print(f"  lambda_2: {lambda_2.mean():.3f} ± {lambda_2.std():.3f}")

    # Prepare output dictionary
    extracted = {
        'mass_1_source': m1_source,
        'mass_2_source': m2_source,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
    }

    # Add metadata
    conversion_note = 'Source frame masses already provided' if has_source_masses else 'Masses converted from detector frame to source frame using bilby'

    metadata = {
        'event': event_name,
        'waveform_model': waveform_model,
        'dataset': dataset_name,
        'data_source': data_source,
        'dcc_url': URLS.get(dataset_name.split('_')[0] if '_' in dataset_name else dataset_name, 'Unknown'),
        'n_samples': len(m1_source),
        'parameters': ['mass_1_source', 'mass_2_source', 'lambda_1', 'lambda_2'],
        'conversion_tool': 'bilby.gw.conversion' if not has_source_masses else 'N/A',
        'notes': conversion_note,
    }

    # Save as npz
    output_file = data_dir / f"{event_name.lower()}_{dataset_name}_posterior.npz"
    # Metadata is stored as a numpy object array to work around npz limitations
    np.savez(
        output_file,
        **extracted,
        metadata=metadata  # type: ignore[arg-type]
    )

    print(f"\nSaved to: {output_file}")
    print(f"Parameters: {metadata['parameters']}")
    print(f"N samples: {metadata['n_samples']}")

    return output_file

def main():
    """Main processing function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Download and process GW170817 posterior samples')
    parser.add_argument('--ignore-cache', action='store_true',
                        help='Force re-download even if files exist')
    args = parser.parse_args()

    # Process all datasets
    for dataset_name, url in URLS.items():
        print("\n" + "="*80)
        print(f"PROCESSING {dataset_name.upper()} POSTERIOR")
        print("="*80)

        # Determine event name
        if "gw170817" in dataset_name.lower():
            event_name = "GW170817"
        elif "gw190425" in dataset_name.lower():
            event_name = "GW190425"
        else:
            event_name = dataset_name.upper()

        # Get event-specific data directory
        data_dir = get_data_dir(event_name)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Determine file format and download
        if url.endswith('.hdf5') or url.endswith('.h5'):
            # HDF5 format (GWTC)
            ext = '.h5' if url.endswith('.h5') else '.hdf5'
            data_file = data_dir / f"{dataset_name}_posterior{ext}"
            if args.ignore_cache or not data_file.exists():
                if args.ignore_cache and data_file.exists():
                    print(f"--ignore-cache: Re-downloading despite existing file")
                download_file(url, data_file)
            else:
                print(f"File already exists: {data_file} (use --ignore-cache to re-download)")

            # Try to load - check if it has multiple posteriors
            posterior, header = load_hdf5_posterior(data_file)

            if posterior is None:
                # Multiple posteriors detected - process each one
                print("\nProcessing multiple posteriors from HDF5 file...")

                import h5py
                with h5py.File(data_file, 'r') as f:
                    posterior_datasets = [key for key in f.keys() if key.endswith('_posterior')]

                for posterior_name in posterior_datasets:
                    print("\n" + "="*80)
                    print(f"PROCESSING {posterior_name}")
                    print("="*80)

                    # Load specific posterior
                    posterior, header = load_hdf5_posterior_by_name(data_file, posterior_name)
                    explore_parameters(posterior, header)

                    # Extract waveform model and spin type from name
                    # e.g., "IMRPhenomPv2NRT_highSpin_posterior" -> model="IMRPhenomPv2NRT", spin="highSpin"
                    parts = posterior_name.replace('_posterior', '').split('_')
                    if len(parts) >= 2:
                        spin_type = parts[-1].lower()  # "highSpin" -> "highspin"
                        waveform_base = '_'.join(parts[:-1])  # "IMRPhenomPv2NRT"
                        waveform_model = f"{waveform_base}_{parts[-1]}"  # "IMRPhenomPv2NRT_highSpin"
                    else:
                        spin_type = "unknown"
                        waveform_model = posterior_name.replace('_posterior', '')

                    # Create dataset name combining dataset and spin type
                    full_dataset_name = f"{dataset_name}_{spin_type}"

                    # Determine data source based on event
                    if "gw170817" in dataset_name.lower():
                        data_source = "LIGO-P1800370"
                    elif "gw190425" in dataset_name.lower():
                        data_source = "LIGO-P2000026"
                    else:
                        data_source = "Unknown"

                    # Extract and save
                    output_file = extract_and_save(posterior, header, full_dataset_name, waveform_model, data_source, event_name)

                    # Verify saved file
                    if output_file and output_file.exists():
                        print("\n" + "-"*80)
                        print("VERIFICATION")
                        print("-"*80)
                        loaded = np.load(output_file, allow_pickle=True)
                        print(f"Saved arrays: {[k for k in loaded.keys() if k != 'metadata']}")
                        for key in ['mass_1_source', 'mass_2_source', 'lambda_1', 'lambda_2']:
                            if key in loaded:
                                print(f"  {key}: shape={loaded[key].shape}, dtype={loaded[key].dtype}")
                                print(f"    mean={loaded[key].mean():.3f}, std={loaded[key].std():.3f}")
                        if 'metadata' in loaded:
                            print(f"\nMetadata:")
                            for k, v in loaded['metadata'].item().items():
                                print(f"  {k}: {v}")

                continue  # Skip to next dataset

            # Single posterior - process normally
            explore_parameters(posterior, header)

            # Extract and save
            waveform_model = "IMRPhenomPv2NRT_highSpin" if "gwtc" in dataset_name else "Unknown"

            # Determine data source
            if "gw170817" in dataset_name.lower():
                data_source = "LIGO-P1800370"
            elif "gw190425" in dataset_name.lower():
                data_source = "LIGO-P2000026"
            else:
                data_source = "Unknown"

            output_file = extract_and_save(posterior, header, dataset_name, waveform_model, data_source, event_name)

        elif url.endswith('.dat.gz'):
            # ASCII format (PhenomPNRT)
            data_file = data_dir / f"{dataset_name}_PhenomPNRT_posterior_samples.dat.gz"
            if args.ignore_cache or not data_file.exists():
                if args.ignore_cache and data_file.exists():
                    print(f"--ignore-cache: Re-downloading despite existing file")
                download_file(url, data_file)
            else:
                print(f"File already exists: {data_file} (use --ignore-cache to re-download)")

            # Load and explore ASCII
            posterior, header = load_posterior_samples(data_file)
            explore_parameters(posterior, header)

            # Extract and save
            waveform_model = "PhenomPNRT"

            # Determine data source
            if "gw170817" in dataset_name.lower():
                data_source = "LIGO-P1800061"
            else:
                data_source = "Unknown"

            output_file = extract_and_save(posterior, header, dataset_name, waveform_model, data_source, event_name)

        else:
            print(f"Unknown file format for URL: {url}")
            continue

        # Verify saved file
        if output_file and output_file.exists():
            print("\n" + "-"*80)
            print("VERIFICATION")
            print("-"*80)
            loaded = np.load(output_file, allow_pickle=True)
            print(f"Saved arrays: {[k for k in loaded.keys() if k != 'metadata']}")
            for key in ['mass_1_source', 'mass_2_source', 'lambda_1', 'lambda_2']:
                if key in loaded:
                    print(f"  {key}: shape={loaded[key].shape}, dtype={loaded[key].dtype}")
                    print(f"    mean={loaded[key].mean():.3f}, std={loaded[key].std():.3f}")
            if 'metadata' in loaded:
                print(f"\nMetadata:")
                for k, v in loaded['metadata'].item().items():
                    print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
