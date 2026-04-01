# Data Download Scripts - Development Guidelines

## Directory Structure

```
data/
├── gw170817/
│   ├── .gitignore              # Ignores *.hdf5, *.h5, *.dat.gz
│   ├── download_gw170817.py    # Unified: download + process + xpnrtv3
│   └── *.npz                   # Processed posteriors (version-controlled)
├── gw190425/
│   ├── .gitignore              # Ignores *.hdf5, *.h5
│   ├── download_gw190425.py    # Unified: download + process + xpnrtv3
│   └── *.npz
├── NICER/
│   ├── .gitignore              # Ignores zenodo_data/
│   ├── zenodo_downloader.py    # ZenodoDownloader utility class
│   ├── download_nicer.py       # Unified: download + extract + downsample
│   ├── zenodo_data/            # Raw Zenodo archives (gitignored, large)
│   └── *.npz                   # Extracted M-R posteriors (version-controlled)
└── chiEFT/
    └── 2402.04172/             # Koehn et al. 2025 pressure-density bands
```

## Script Design Philosophy

Each subdirectory has **one unified script** that handles the full pipeline from raw
download to processed `.npz` output. Scripts use **constants at the top** (no argparse)
so users tweak `IGNORE_CACHE`, `MAX_SAMPLES`, etc. directly in the file and run:

```bash
uv run python gw170817/download_gw170817.py
uv run python gw190425/download_gw190425.py
uv run python NICER/download_nicer.py
```

## Core Principles

### 1. Caching by Default

All scripts cache downloads and skip existing files by default. Set `IGNORE_CACHE = True`
at the top of a script to force a full re-run.

**Rationale**: data files are large (hundreds of MB to GB); LIGO/NICER servers should
not be hit unnecessarily; reproducibility requires stable cached files.

### 2. Constants Instead of argparse

User-tunable parameters are defined as module-level constants with clear comments:

```python
# Set True to force re-download even if files already exist
IGNORE_CACHE: bool = False

# Downsample to at most this many samples (None = keep all)
MAX_SAMPLES: int | None = 100_000
```

This keeps scripts simple to run (`python download_nicer.py`) while making the
configuration easy to find and adjust.

### 3. Extracted Parameter Consistency (GW events)

All GW posterior scripts MUST extract:
- `mass_1_source` — primary mass in source frame (solar masses)
- `mass_2_source` — secondary mass in source frame (solar masses)
- `lambda_1` — tidal deformability of primary
- `lambda_2` — tidal deformability of secondary

If the source provides detector-frame masses, convert using
`bilby.gw.conversion.luminosity_distance_to_redshift`. Document the conversion in metadata.

### 4. Metadata Requirements

Every `.npz` file MUST include a `metadata` dict:

```python
metadata = {
    'event': 'GW170817',
    'waveform_model': 'IMRPhenomPv2NRT',
    'dataset': 'gwtc1_highspin',
    'data_source': 'LIGO-P1800370',
    'dcc_url': 'https://...',
    'n_samples': 4041,
    'parameters': ['mass_1_source', ...],
    'conversion_tool': 'bilby.gw.conversion',
    'notes': 'Converted from detector frame',
}
```

NICER files include: `psr`, `group`, `hotspot_model`, `data_used`, `n_samples`,
`zenodo_record`, `paper`.

### 5. Downsampling

Downsampling is integrated into each script, not a separate step.
For NICER, the `MAX_SAMPLES` constant controls the target size (default `100_000`).
For GW events, `MAX_SAMPLES = None` (posteriors are already small after 4-parameter extraction).

## Adding a New Dataset

1. Create a subdirectory: `data/<event_name>/`
2. Add a `.gitignore` for raw downloaded files
3. Write `download_<event_name>.py` with:
   - Constants at the top (`IGNORE_CACHE`, `MAX_SAMPLES`, …)
   - `main()` that downloads, processes, and optionally downsamples
   - Complete metadata in every `.npz` output
4. Update `README.md` with dataset description
