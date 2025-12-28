# KDE and Normalizing Flow Validation

This directory contains validation scripts that compare:
- **Gravitational Wave (GW) likelihoods**: Original posterior samples vs Normalizing Flow (NF) approximations
- **NICER likelihoods**: Original posterior samples vs Kernel Density Estimate (KDE) approximations

## Purpose

The JESTER inference system uses:
1. **Normalizing flows** to approximate GW event posteriors (efficient evaluation of p(m₁, m₂, Λ₁, Λ₂))
2. **Kernel density estimation** to approximate NICER M-R posteriors (efficient marginalization over mass)

This validation ensures that these approximations faithfully represent the original posterior samples.

## Directory Structure

```
kde_nf_validation/
├── README.md                  # This file
├── validate_kde_nf.py         # Main validation script
└── figures/                   # Output directory for validation plots
    ├── gw170817_nf_validation.png
    ├── nicer_j0030_kde_validation.png
    └── nicer_j0740_kde_validation.png
```

## Usage

### Run Full Validation

```bash
cd /Users/Woute029/Documents/Code/projects/jester_review/jester/examples/kde_nf_validation
uv run python validate_kde_nf.py
```

This will:
1. Load original posterior samples for each dataset
2. Load trained NF models (GW) or construct KDEs (NICER)
3. Sample from NF/KDE
4. Generate corner plots comparing original vs approximation
5. Save figures to `figures/` directory

### Expected Output

The script generates one figure per dataset:

- **`gw170817_nf_validation.png`**: Corner plot of (m₁, m₂, Λ₁, Λ₂) for GW170817
  - Blue contours: Original GWTC-1 low-spin posterior samples
  - Red contours: Normalizing flow samples

- **`nicer_j0030_kde_validation.png`**: Corner plot of (M, R) for PSR J0030+0451
  - Blue contours: Original Amsterdam + Maryland samples (equal weight)
  - Red contours: KDE samples (equal weight from both groups)

- **`nicer_j0740_kde_validation.png`**: Corner plot of (M, R) for PSR J0740+6620
  - Blue contours: Original Amsterdam + Maryland samples (equal weight)
  - Red contours: KDE samples (equal weight from both groups)

## Configuration

### GW Events

Configured in `GW_EVENTS` dictionary:

```python
GW_EVENTS = {
    "GW170817": {
        "data_file": "jesterTOV/inference/data/gw170817/gw170817_gwtc1_lowspin_posterior.npz",
        "model_dir": "jesterTOV/inference/flows/models/gw_maf/gw170817/gw170817_gwtc1_lowspin_posterior",
        "parameters": ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
        "labels": [r"$m_1$ [$M_\odot$]", r"$m_2$ [$M_\odot$]", r"$\Lambda_1$", r"$\Lambda_2$"],
        "n_samples": 10000,
    },
}
```

- `data_file`: Original posterior samples (.npz file)
- `model_dir`: Trained normalizing flow model directory
- `parameters`: Keys to extract from .npz file
- `labels`: Plot labels with LaTeX formatting
- `n_samples`: Number of samples to draw from NF

### NICER Pulsars

Configured in `NICER_PULSARS` dictionary:

```python
NICER_PULSARS = {
    "J0030": {
        "amsterdam": "jesterTOV/inference/data/NICER/J00300451_amsterdam_ST_U_NICER_only_Riley2019.npz",
        "maryland": "jesterTOV/inference/data/NICER/J00300451_maryland_3spot_NICER_only_RM.npz",
        "parameters": ["mass", "radius"],
        "labels": [r"$M$ [$M_\odot$]", r"$R$ [km]"],
        "n_samples": 10000,
    },
}
```

- `amsterdam`: Amsterdam group posterior samples (.npz file)
- `maryland`: Maryland group posterior samples (.npz file)
- `parameters`: Keys to extract from .npz files
- `labels`: Plot labels with LaTeX formatting
- `n_samples`: Number of samples to draw from KDE (split equally between groups)

## Implementation Details

### GW Validation (`validate_gw_event`)

1. **Load original samples**: Extract (m₁, m₂, Λ₁, Λ₂) from .npz file
2. **Load NF model**: Use `Flow.from_directory()` to load trained model
3. **Sample from NF**: Use `flow.sample(key, (n_samples,))` to generate samples
4. **Create corner plot**: Overlay original (blue) and NF (red) contours

### NICER Validation (`validate_nicer_pulsar`)

1. **Load original samples**: Extract (M, R) from Amsterdam and Maryland .npz files
2. **Construct KDEs**: Use `scipy.stats.gaussian_kde` (same as `NICERLikelihood`)
3. **Sample from KDEs**: Draw equal number of samples from each group's KDE
4. **Create corner plot**: Overlay original (blue) and KDE (red) contours

### Corner Plot (`plot_corner_comparison`)

- Uses `corner` package for corner plots
- Plots density contours (68% and 95% credible regions)
- Original samples: Blue, filled contours
- Approximation samples: Red, filled contours
- Smoothing: 1.0 (default)
- No individual data points (for clarity)

## Adding New Datasets

### Add a GW Event

```python
GW_EVENTS["GW190425"] = {
    "data_file": DATA_DIR / "gw190425" / "gw190425_phenompnrt-ls_posterior.npz",
    "model_dir": MODELS_DIR / "gw190425" / "gw190425_phenompnrt-ls_posterior",
    "parameters": ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
    "labels": [r"$m_1$ [$M_\odot$]", r"$m_2$ [$M_\odot$]", r"$\Lambda_1$", r"$\Lambda_2$"],
    "n_samples": 10000,
}
```

### Add a NICER Pulsar

```python
NICER_PULSARS["J1234"] = {
    "amsterdam": DATA_DIR / "NICER" / "J1234_amsterdam_samples.npz",
    "maryland": DATA_DIR / "NICER" / "J1234_maryland_samples.npz",
    "parameters": ["mass", "radius"],
    "labels": [r"$M$ [$M_\odot$]", r"$R$ [km]"],
    "n_samples": 10000,
}
```

## Dependencies

- `jax`: JAX framework for sampling from NF
- `numpy`: Array operations
- `scipy`: `gaussian_kde` for KDE construction and sampling
- `matplotlib`: Base plotting library
- `corner`: Corner plot generation

## Reproducibility

- Random seed is fixed to `SEED = 42` for all sampling operations
- Original data files are version-controlled in `jesterTOV/inference/data/`
- NF models are version-controlled in `jesterTOV/inference/flows/models/`
- All paths are relative to JESTER repository root

## Interpretation

**Good approximation** should show:
- Strong overlap between blue (original) and red (approximation) contours
- Similar 68% and 95% credible region shapes
- Minimal bias in marginal distributions

**Poor approximation** would show:
- Systematic shifts between contours
- Different credible region shapes (e.g., broader or narrower)
- Missing features (e.g., multimodality)

## References

- **GW170817 data**: GWTC-1 catalog, low-spin prior
- **NICER data**: Riley et al. 2019 (Amsterdam), Miller et al. (Maryland)
- **Normalizing flows**: `flowjax` package for flow models
- **KDE**: `scipy.stats.gaussian_kde` with Scott's rule for bandwidth
