# Data Download Scripts - Development Guidelines

This file provides guidance for creating and maintaining data download scripts in this directory.

## Core Principles

### 1. Caching by Default

**CRITICAL**: All download scripts MUST implement caching to avoid unnecessary re-downloads.

- **Default behavior**: Skip download if file already exists
- **Override**: Provide `--ignore-cache` flag to force re-download

**Implementation pattern**:
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description='Download and process [EVENT] data')
    parser.add_argument('--ignore-cache', action='store_true',
                        help='Force re-download even if files exist')
    args = parser.parse_args()

    # Download logic
    if args.ignore_cache or not data_file.exists():
        if args.ignore_cache and data_file.exists():
            print(f"--ignore-cache: Re-downloading despite existing file")
        download_file(url, data_file)
    else:
        print(f"File already exists: {data_file} (use --ignore-cache to re-download)")
```

**Rationale**:
- Data files are large (hundreds of MB to GB)
- LIGO/NICER servers should not be hammered unnecessarily
- Reproducibility: cached files ensure consistent data across runs
- Development efficiency: faster iteration when working with scripts

### 2. Script Organization

**One script per event/dataset**:
- ✅ `download_gw170817.py` - GW170817 event
- ✅ `download_gw190425.py` - GW190425 event
- ✅ `download_nicer_j0030.py` - NICER PSR J0030+0451 (future)
- ✅ `download_nicer_j0740.py` - NICER PSR J0740+6620 (future)

**DO NOT**:
- ❌ Mix multiple events in one script
- ❌ Create overly generic "download_all.py" scripts
- ❌ Hardcode event-specific logic without clear separation

**Rationale**:
- Each event has unique data formats, waveform models, and conventions
- Easier to maintain and debug
- Users can download only what they need

### 3. Data Directory Structure

```
data/
├── CLAUDE.md                    # This file
├── README.md                    # User-facing documentation
├── download_gw170817.py         # GW170817 download script
├── download_gw190425.py         # GW190425 download script
├── gw170817/                    # GW170817 data
│   ├── *.npz                    # Processed posteriors
│   ├── *.h5, *.hdf5, *.dat.gz  # Original data (cached)
├── gw190425/                    # GW190425 data
│   └── ...
└── nicer/                       # NICER data (future)
    ├── j0030/
    └── j0740/
```

### 4. Extracted Parameter Consistency

**All scripts MUST extract these parameters**:
- `mass_1_source` - Primary mass in source frame (solar masses)
- `mass_2_source` - Secondary mass in source frame (solar masses)
- `lambda_1` - Tidal deformability of primary
- `lambda_2` - Tidal deformability of secondary

**Frame conversion**:
- If data provides detector frame masses, convert to source frame using `bilby.gw.conversion.luminosity_distance_to_redshift`
- If data provides source frame masses directly, use them as-is
- Always document which approach was used in metadata

**Output format**: NumPy `.npz` files with:
```python
{
    'mass_1_source': ndarray,
    'mass_2_source': ndarray,
    'lambda_1': ndarray,
    'lambda_2': ndarray,
    'metadata': dict  # Event info, waveform model, conversion method, etc.
}
```

### 5. Exploration Before Extraction

**ALWAYS**:
1. Download the original data file
2. Explore its structure (print all groups, datasets, parameters)
3. Check parameter naming conventions
4. Detect whether masses are in source or detector frame
5. Only then extract and convert

**DO NOT**:
- Assume parameter names without checking
- Blindly apply conversions
- Skip metadata exploration

### 6. Metadata Requirements

**Every `.npz` file MUST include**:
```python
metadata = {
    'event': 'GW170817',                    # Event name
    'waveform_model': 'IMRPhenomPv2NRT',    # Waveform approximant
    'dataset': 'gwtc1_highspin',            # Dataset identifier
    'data_source': 'LIGO-P1800370',         # DCC document number
    'dcc_url': 'https://...',               # Original data URL
    'n_samples': 4041,                      # Number of posterior samples
    'parameters': ['mass_1_source', ...],   # Extracted parameters
    'conversion_tool': 'bilby.gw.conversion',  # Tool used for conversions
    'notes': 'Converted from detector frame',  # Any relevant notes
}
```

### 7. Error Handling

**Scripts MUST**:
- Handle missing parameters gracefully (print available parameters, suggest alternatives)
- Catch and report download errors
- Validate extracted data (check for NaN, inf, negative masses, etc.)
- Provide clear error messages

**Example**:
```python
if 'lambda_1' not in header and 'lambda1' not in header:
    print("ERROR: No lambda parameters found")
    print(f"Available parameters: {header}")
    print("Lambda-like parameters: {[p for p in header if 'lambda' in p.lower()]}")
    return None
```

### 8. Documentation

**Each script MUST include**:
- Docstring with event name, data source, extracted parameters
- Usage examples in docstring
- Comments explaining non-obvious logic
- Links to DCC pages and publications

**Update README.md** when:
- Adding new datasets
- Changing waveform models
- Updating parameter extraction logic

## NICER Data Guidelines

When implementing NICER download scripts:

1. **Follow the same caching pattern** with `--ignore-cache`
2. **Handle multiple analysis groups**: Amsterdam vs Maryland posteriors
3. **Handle multiple hotspot models**: Different geometric assumptions
4. **Extract M-R posteriors**: Focus on mass and radius, not full parameter space
5. **Document which analysis group and hotspot model**: Critical for reproducibility
6. **Handle version evolution**: Multiple Zenodo releases per pulsar

**Example structure for NICER**:
```
nicer/
├── j0030/
│   ├── amsterdam_rileys2019_*.npz      # Amsterdam group, Riley et al. 2019
│   ├── amsterdam_rileys2021_*.npz      # Amsterdam group, Riley et al. 2021
│   └── ...
└── j0740/
    ├── amsterdam_rileys2021_*.npz
    ├── maryland_millers2021_*.npz
    └── ...
```

## Testing Checklist

Before committing a new download script:

- [ ] Script has `--ignore-cache` flag
- [ ] Default behavior: skip download if file exists
- [ ] Script explores data structure before extraction
- [ ] Detects source vs detector frame masses correctly
- [ ] Extracts all 4 required parameters (mass_1_source, mass_2_source, lambda_1, lambda_2)
- [ ] Metadata is complete and accurate
- [ ] Script handles errors gracefully
- [ ] README.md is updated with new dataset info
- [ ] Output files are saved in event-specific subdirectory
- [ ] Verification step prints sample statistics

## Examples

See existing scripts:
- `download_gw170817.py` - Handles multiple waveform models, both .dat.gz and .hdf5 formats
- `download_gw190425.py` - Clean single-event script, HDF5 format with structure exploration

## Questions?

If uncertain about implementation details:
1. Check existing scripts for patterns
2. Review this file for guidelines
3. Consult DCC documentation pages
4. Test with `--ignore-cache` to ensure caching works
