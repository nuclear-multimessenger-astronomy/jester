# JESTER Inference Data

This directory contains observational data for constraining neutron star equation of state (EOS) parameters through Bayesian inference.

## Directory Structure

```
data/
├── README.md                    # This file
├── download_gw170817.py         # Script to download and process GW170817 data (supports .dat.gz and .hdf5)
├── gw170817/                    # GW170817 binary neutron star merger
│   ├── gw170817_low_spin_posterior.npz          # PhenomPNRT low spin (3,952 samples)
│   ├── gw170817_high_spin_posterior.npz         # PhenomPNRT high spin (9,117 samples)
│   ├── gw170817_gwtc1_highspin_posterior.npz    # GWTC-1 high spin (4,041 samples)
│   ├── gw170817_gwtc1_lowspin_posterior.npz     # GWTC-1 low spin (8,078 samples)
│   ├── low_spin_PhenomPNRT_posterior_samples.dat.gz     # Original P1800061 data
│   ├── high_spin_PhenomPNRT_posterior_samples.dat.gz    # Original P1800061 data
│   └── gwtc1_posterior.hdf5                     # Original P1800370 data (both spins)
└── old_utils.py                 # Legacy utilities (to be refactored)
```

## Gravitational Wave Events

### GW170817 - Binary Neutron Star Merger

TODO: clean up this AI mess.

Links: https://dcc.ligo.org/LIGO-P1800061/public for Properties samples, https://dcc.ligo.org/LIGO-P1800370/public for GWTC-1 results

**Event**: First detected binary neutron star merger (August 17, 2017)

**Data Source**: LIGO-Virgo Collaboration Data Release
- **DCC Page**: https://dcc.ligo.org/LIGO-P1800061/public
- **Tutorial**: https://nbviewer.org/urls/dcc.ligo.org/public/0150/P1800061/011/Data%20Release%20Tutorial.ipynb

**Publication**:
- Abbott et al. (2019), "Properties of the Binary Neutron Star Merger GW170817"
- Physical Review X 9, 011001
- arXiv:1805.11579

**Available Datasets**:

We provide **four** posterior sample sets for GW170817, from two different data releases:

#### 1. PhenomPNRT Posteriors (P1800061)

**Waveform Model**: PhenomPNRT (Phenomenological Post-Newtonian-Ringdown-Tides)
- Frequency-domain model for spinning, precessing binary neutron stars
- Includes tidal deformability effects (lambda1, lambda2)

**Spin Priors**:

TODO: need to double check these values

- **Low Spin** (3,952 samples): Dimensionless spin magnitudes restricted to [0, 0.05]
  - More conservative assumption based on observed pulsars
  - M1 = 1.488 ± 0.081 Msun, M2 = 1.257 ± 0.065 Msun
  - Lambda_1 = 341.0 ± 310.6, Lambda_2 = 553.3 ± 466.0
  - File: `gw170817_low_spin_posterior.npz`

- **High Spin** (9,117 samples): Dimensionless spin magnitudes allowed up to 0.89
  - Broader parameter space exploration
  - M1 = 1.642 ± 0.199 Msun, M2 = 1.156 ± 0.118 Msun
  - Lambda_1 = 251.9 ± 265.9, Lambda_2 = 681.0 ± 648.0
  - File: `gw170817_high_spin_posterior.npz`

#### 2. GWTC-1 Posteriors (P1800370)

**Waveform Model**: IMRPhenomPv2NRT
- Different waveform approximant used for GWTC-1 catalog
- Also includes tidal effects

**Spin Priors**:

- **High Spin** (4,041 samples): Higher spin magnitudes allowed
  - M1 = 1.617 ± 0.174 Msun, M2 = 1.169 ± 0.110 Msun
  - Lambda_1 = 284.6 ± 295.6, Lambda_2 = 708.1 ± 661.6
  - File: `gw170817_gwtc1_highspin_posterior.npz`

- **Low Spin** (8,078 samples): Restricted spin magnitudes [0, 0.05]
  - M1 = 1.477 ± 0.078 Msun, M2 = 1.265 ± 0.063 Msun
  - Lambda_1 = 368.2 ± 335.8, Lambda_2 = 586.5 ± 509.7
  - File: `gw170817_gwtc1_lowspin_posterior.npz`

**Note**: The GWTC-1 HDF5 file (`gwtc1_posterior.hdf5`) contains both spin priors, and the script automatically extracts both into separate files.

**Extracted Parameters**:
- `mass_1_source`: Primary mass in source frame (solar masses)
- `mass_2_source`: Secondary mass in source frame (solar masses)
- `lambda_1`: Tidal deformability of primary
- `lambda_2`: Tidal deformability of secondary

**Frame Conversion**:
- Original data provides masses in **detector frame** (redshifted)
- Converted to **source frame** using `bilby.gw.conversion.luminosity_distance_to_redshift`
- Relation: `m_source = m_detector / (1 + z)`
- For GW170817 at ~40 Mpc, redshift z ~ 0.009 (small correction)

**File Format** (`.npz` - NumPy compressed):
```python
import numpy as np

data = np.load('gw170817/gw170817_low_spin_posterior.npz', allow_pickle=True)

# Access parameters
mass_1_source = data['mass_1_source']  # shape: (n_samples,)
mass_2_source = data['mass_2_source']
lambda_1 = data['lambda_1']
lambda_2 = data['lambda_2']

# Access metadata
metadata = data['metadata'].item()  # Convert numpy scalar to dict
print(metadata['event'])            # 'GW170817'
print(metadata['waveform_model'])   # 'PhenomPNRT'
print(metadata['n_samples'])        # 3952 or 9117
```

**Download Script**:
```bash
# Download and process GW170817 data
uv run python jesterTOV/inference/data/download_gw170817.py
```

The script:
1. Downloads compressed posterior samples from LIGO DCC
2. Extracts relevant parameters (masses, lambdas)
3. Converts masses from detector frame to source frame using bilby
4. Saves as `.npz` files with comprehensive metadata

## NICER Data (TODO)

### PSR J0030+0451

**Data Sources**:
- **Amsterdam Group (most recent)**: https://zenodo.org/records/8239000
- **Amsterdam Group (earlier)**: https://zenodo.org/records/7096789
- **Second analysis group**: https://zenodo.org/records/3473466

**Status**: To be implemented

### PSR J0740+6620

**Data Sources**:
- **Amsterdam (most recent)**: https://zenodo.org/records/10519473
- **Amsterdam (earlier)**: https://zenodo.org/records/6827537, https://zenodo.org/records/5735003
- **Maryland**: https://zenodo.org/records/4670689

**Status**: To be implemented

## Usage in JESTER Inference

The processed data files are used by the likelihood classes in `jesterTOV/inference/likelihoods/`:

```python
from jesterTOV.inference.likelihoods.gw import GWLikelihood

# Load GW170817 data (automatically handled by likelihood)
likelihood = GWLikelihood(
    event_name="GW170817",
    spin_prior="low_spin",  # or "high_spin"
    data_path="jesterTOV/inference/data/gw170817/gw170817_low_spin_posterior.npz"
)

# Evaluate likelihood for EOS parameters
log_likelihood = likelihood.evaluate(eos_params, data={})
```

## Data Maintenance

### Updating GW170817 Data

If LIGO releases updated posteriors:
1. Update URLs in `download_gw170817.py`
2. Delete old `.npz` files (keep original `.dat.gz` for reference)
3. Re-run: `uv run python jesterTOV/inference/data/download_gw170817.py`
4. Update this README with new publication info

### Adding New Events

For new gravitational wave events (e.g., GW190425):
1. Create `download_gw<event>.py` script following GW170817 pattern
2. Create subdirectory `gw<event>/`
3. Document in this README with metadata
4. Add likelihood support in `likelihoods/gw.py`

## Reproducibility

All data processing scripts are version-controlled to ensure reproducibility:
- Original data URLs are preserved in download scripts
- Conversion methods are documented (bilby versions in metadata)
- Intermediate files (`.dat.gz`) are retained for verification

## References

### GW170817
- Abbott et al. (2017), "GW170817: Observation of Gravitational Waves from a Binary Neutron Star Inspiral", Phys. Rev. Lett. 119, 161101, arXiv:1710.05832
- Abbott et al. (2019), "Properties of the Binary Neutron Star Merger GW170817", Phys. Rev. X 9, 011001, arXiv:1805.11579
- Abbott et al. (2019), "GWTC-1: A Gravitational-Wave Transient Catalog of Compact Binary Mergers", Phys. Rev. X 9, 031040, arXiv:1811.12907
- LIGO P1800061 Data Release: https://dcc.ligo.org/LIGO-P1800061/public
- LIGO P1800370 Data Release (GWTC-1): https://dcc.ligo.org/LIGO-P1800370/public

### NICER
- Riley et al. (2019), "A NICER View of PSR J0030+0451", ApJL 887, L21, arXiv:1912.05702
- Miller et al. (2019), "PSR J0030+0451 Mass and Radius from NICER Data", ApJL 887, L24, arXiv:1912.05705
- Riley et al. (2021), "A NICER View of the Massive Pulsar PSR J0740+6620", ApJL 918, L27, arXiv:2105.06980
- Miller et al. (2021), "The Radius of PSR J0740+6620 from NICER", ApJL 918, L28, arXiv:2105.06979
