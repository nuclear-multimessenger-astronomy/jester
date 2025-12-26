# JESTER Inference System Documentation

**Version**: 1.0 (December 2024)
**Status**: Production-ready

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Quick Start](#quick-start)
4. [Configuration System](#configuration-system)
5. [Prior Specification](#prior-specification)
6. [Transforms](#transforms)
7. [Likelihoods](#likelihoods)
8. [Data Management](#data-management)
9. [Sampling](#sampling)
10. [Complete Workflow](#complete-workflow)
11. [Advanced Usage](#advanced-usage)
12. [File Structure Reference](#file-structure-reference)

---

## Overview

The JESTER inference system provides Bayesian parameter estimation for neutron star equation of state (EOS) models using multiple astrophysical observations. The system is built on a modular, configuration-driven architecture that supports:

- **Multiple EOS parameterizations**: MetaModel (nuclear empirical parameters) and MetaModel+CSE (with Constant Speed Extension)
- **Multiple observational constraints**: Gravitational waves, X-ray timing (NICER), radio pulsar timing, chiral EFT, and nuclear experiments
- **Advanced sampling**: Normalizing flow-enhanced MCMC via flowMC backend
- **Full configurability**: YAML-based configuration with validation
- **Type safety**: Pydantic models ensure configuration correctness

### Key Features

- ✅ **Configuration-driven**: Define entire inference run in a single YAML file
- ✅ **Modular design**: Independent components for priors, transforms, likelihoods, data
- ✅ **Reproducible**: Version-controlled configs, automatic seed management
- ✅ **Extensible**: Easy to add new likelihoods, transforms, or EOS models
- ✅ **Type-safe**: Early error detection via Pydantic validation

---

## System Architecture

### Module Structure

```
jesterTOV/inference/
├── config/              # Configuration system
│   ├── parser.py        # YAML loading with path resolution
│   └── schema.py        # Pydantic validation models
│
├── priors/              # Prior system
│   ├── parser.py        # .prior file parsing (executes Python code)
│   ├── simple_priors.py # SimpleUniformPrior implementation
│   └── library.py       # Common prior definitions
│
├── transforms/          # EOS parameter transforms
│   ├── base.py          # JesterTransformBase ABC
│   ├── metamodel.py     # MetaModel transform (NEP → M-R-Λ)
│   ├── metamodel_cse.py # MetaModel+CSE transform
│   └── factory.py       # Transform creation from config
│
├── likelihoods/         # Observational constraints
│   ├── gw.py            # Gravitational wave events
│   ├── nicer.py         # NICER X-ray timing
│   ├── radio.py         # Radio pulsar timing
│   ├── chieft.py        # Chiral EFT constraints
│   ├── rex.py           # PREX/CREX experiments
│   ├── combined.py      # Combined and zero likelihoods
│   └── factory.py       # Likelihood creation from config
│
├── data/                # Data loading and caching
│   ├── loader.py        # Lazy loading with KDE construction
│   └── paths.py         # Path management (not yet used)
│
├── samplers/            # MCMC sampler wrappers
│   ├── jester_sampler.py # Standalone flowMC wrapper (Jim-like)
│   └── flowmc.py        # Sampler factory
│
├── base/                # Base classes (copied from Jim)
│   ├── likelihood.py    # LikelihoodBase ABC
│   ├── prior.py         # Prior, CombinePrior ABCs
│   └── transform.py     # NtoMTransform, BijectiveTransform ABCs
│
├── run_inference.py     # Main entry point (config-driven)
└── cli.py               # Command-line interface
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION PHASE                        │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         config.yaml      prior.prior    data files
              │               │               │
              v               v               v
      ┌─────────────┐  ┌──────────┐  ┌──────────────┐
      │ load_config │  │  parse   │  │  DataLoader  │
      │  (parser)   │  │  prior   │  │   (lazy)     │
      └──────┬──────┘  └────┬─────┘  └──────┬───────┘
             │              │                │
             v              v                v
      InferenceConfig  CombinePrior    KDEs/Models
             │              │                │
┌────────────┴──────────────┴────────────────┴─────────────────┐
│                    COMPONENT CREATION                         │
└───────────────────────────────────────────────────────────────┘
             │              │                │
             v              v                v
      ┌──────────┐   ┌──────────┐   ┌──────────────┐
      │Transform │   │   Prior  │   │  Likelihood  │
      │ Factory  │   │ (parsed) │   │   Factory    │
      └────┬─────┘   └────┬─────┘   └──────┬───────┘
           │              │                 │
           v              v                 v
    JesterTransform  CombinePrior   CombinedLikelihood
           │              │                 │
┌──────────┴──────────────┴─────────────────┴──────────────────┐
│                      SAMPLER SETUP                            │
└───────────────────────────────────────────────────────────────┘
                              │
                              v
                      ┌───────────────┐
                      │ JesterSampler │
                      │  (flowMC)     │
                      └───────┬───────┘
                              │
┌─────────────────────────────┴─────────────────────────────────┐
│                    SAMPLING PHASE                             │
└───────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
          Training loops           Production loops
          (NF learning)            (final samples)
                 │                         │
                 └────────────┬────────────┘
                              │
                              v
                    ┌─────────────────┐
                    │  Save results   │
                    │  - samples      │
                    │  - log_prob     │
                    │  - runtime      │
                    └────────┬────────┘
                             │
┌────────────────────────────┴───────────────────────────────────┐
│                   POSTPROCESSING                               │
└────────────────────────────────────────────────────────────────┘
                              │
                              v
                    ┌─────────────────┐
                    │ Generate EOS    │
                    │ samples         │
                    │ (TOV solve)     │
                    └────────┬────────┘
                             │
                             v
                    ┌─────────────────┐
                    │ Save EOS data   │
                    │ - M, R, Λ       │
                    │ - n, p, e, h    │
                    └─────────────────┘
```

---

## Quick Start

### Installation

```bash
# Install JESTER with inference dependencies
uv pip install -e ".[inference]"
```

### Running Your First Inference

1. **Create a configuration file** (`config.yaml`):

```yaml
seed: 43

transform:
  type: "metamodel"
  ndat_metamodel: 100
  nmax_nsat: 25.0

prior:
  specification_file: "prior.prior"

likelihoods:
  - type: "zero"
    enabled: true

sampler:
  n_chains: 10
  n_loop_training: 2
  n_loop_production: 2
  n_local_steps: 50
  n_global_steps: 50
  n_epochs: 20
  learning_rate: 0.001
  output_dir: "./outdir/"
```

2. **Create a prior file** (`prior.prior`):

```python
# Nuclear Empirical Parameters
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])
```

3. **Run inference**:

```bash
# Using the console script (recommended)
run_jester_inference config.yaml

# Or using module syntax
uv run python -m jesterTOV.inference.run_inference --config config.yaml
```

4. **Check results**:

```bash
ls outdir/
# results_production.npz  - MCMC samples
# eos_samples.npz         - EOS curves (M, R, Λ)
# runtime.txt             - Timing information
```

---

## Configuration System

### YAML Configuration Structure

The configuration file is the central control point for inference runs. All settings are validated using Pydantic models defined in `config/schema.py`.

#### Complete Configuration Template

```yaml
# Random seed for reproducibility
seed: 43

# Transform configuration: How to convert parameters to observables
transform:
  type: "metamodel"  # or "metamodel_cse"
  ndat_metamodel: 100       # Number of EOS grid points
  nmax_nsat: 25.0           # Maximum density (units of nsat)
  nb_CSE: 0                 # Number of CSE grid points (only for metamodel_cse)
  min_nsat_TOV: 0.75        # Minimum density for TOV integration
  ndat_TOV: 100             # TOV integration grid size
  nb_masses: 100            # Number of masses for marginalization
  crust_name: "DH"          # Crust model: "DH", "BPS", or "DH_fixed"

# Prior specification file (bilby-style Python format)
prior:
  specification_file: "prior.prior"  # Path relative to config file

# List of observational constraints
likelihoods:
  - type: "gw"                # Gravitational wave
    enabled: true
    parameters:
      events:
        - name: "GW170817"
          model_dir: "./data/GW/GW170817/model/"
      penalty_value: -99999.0
      N_masses_evaluation: 20

  - type: "nicer"             # X-ray timing
    enabled: true
    parameters:
      pulsars:
        - name: "J0030"
          amsterdam_samples_file: "./data/NICER/J0030_amsterdam.npz"
          maryland_samples_file: "./data/NICER/J0030_maryland.npz"
      N_masses_evaluation: 100

  - type: "radio"             # Radio pulsar timing
    enabled: true
    parameters:
      pulsars:
        - name: "J0740+6620"
          mass_mean: 2.08
          mass_std: 0.07
      nb_masses: 100

  - type: "chieft"            # Chiral EFT
    enabled: true
    parameters:
      low_filename: "./data/chiEFT/low.dat"
      high_filename: "./data/chiEFT/high.dat"
      nb_n: 100

  - type: "constraints"       # Physical constraints
    enabled: true
    parameters:
      penalty_tov: -1e10

  - type: "zero"              # Prior-only sampling
    enabled: false

# MCMC sampler configuration
sampler:
  n_chains: 20                # Parallel chains
  n_loop_training: 3          # Training phase loops
  n_loop_production: 3        # Production phase loops
  n_local_steps: 100          # MCMC steps per loop
  n_global_steps: 100         # Global (NF) steps per loop
  n_epochs: 30                # NF training epochs per loop
  learning_rate: 0.001        # NF learning rate
  train_thinning: 1           # Training sample thinning
  output_thinning: 5          # Output sample thinning
  n_eos_samples: 10000        # EOS curves to generate
  output_dir: "./outdir/"     # Results directory

# Postprocessing configuration
postprocessing:
  enabled: true               # Run postprocessing after inference
  make_cornerplot: true       # Generate corner plot
  make_massradius: true       # Generate M-R diagram
  make_pressuredensity: true  # Generate P-ρ plot
  make_histograms: true       # Generate parameter histograms
  make_contours: true         # Generate contour plots
  prior_dir: null             # Optional: path to prior samples for comparison

# Execution options
dry_run: false                # Setup everything but don't run sampler
validate_only: false          # Only validate configuration

# Data paths (optional - likelihoods specify their data files directly)
data_paths: {}
```

### Configuration Validation

The system uses Pydantic for automatic validation:

```python
# Example: Invalid configuration
transform:
  type: "metamodel"
  nb_CSE: 8  # ERROR: nb_CSE must be 0 for metamodel

# Example: Missing required field
likelihoods: []  # ERROR: At least one likelihood must be enabled

# Example: Invalid file extension
prior:
  specification_file: "prior.yaml"  # ERROR: Must be .prior file
```

**Validation checks**:
- Transform type matches nb_CSE value
- At least one likelihood enabled
- Prior file has `.prior` extension
- Positive values for n_chains, n_loop_*, etc.
- Learning rate in (0, 1]
- Valid crust name: "DH", "BPS", or "DH_fixed"

### Path Resolution

Paths in configuration are resolved relative to the config file directory:

```yaml
# If config.yaml is in /path/to/project/runs/run1/
prior:
  specification_file: "prior.prior"
  # Resolves to: /path/to/project/runs/run1/prior.prior

prior:
  specification_file: "../../priors/standard.prior"
  # Resolves to: /path/to/project/priors/standard.prior
```

---

## Prior Specification

### Prior File Format

Prior files use a **bilby-style Python format** with direct variable assignment. The parser executes the Python code to extract Prior objects.

#### Basic Example

```python
# prior.prior
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
```

### Prior Inclusion Rules

The parser automatically includes/excludes parameters based on configuration:

| Parameter Pattern | Inclusion Rule |
|------------------|----------------|
| `*_sat`, `*_sym` | Always included (NEP parameters) |
| `nbreak` | Only if `nb_CSE > 0` |
| `mass_1_*`, `mass_2_*` | Only if GW event is enabled with `sample_masses: true` |
| CSE grid parameters | Auto-generated if `nb_CSE > 0` |

#### Example: Conditional Parameters

```python
# prior.prior

# Always included (NEP parameters)
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])

# Only included if nb_CSE > 0 in config
nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])

# Only included if GW170817 is sampled with sample_masses: true
mass_1_GW170817 = UniformPrior(1.5, 2.1, parameter_names=["mass_1_GW170817"])
mass_2_GW170817 = UniformPrior(1.0, 1.5, parameter_names=["mass_2_GW170817"])

# CSE grid parameters (n_CSE_i_u, cs2_CSE_i) are added automatically
# No need to define them in the prior file
```

### CSE Grid Parameters

When `nb_CSE > 0`, the parser automatically adds CSE grid parameters:

```python
# For nb_CSE = 8, these are added automatically:
# n_CSE_0_u = UniformPrior(0.0, 1.0, parameter_names=["n_CSE_0_u"])
# cs2_CSE_0 = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_0"])
# n_CSE_1_u = UniformPrior(0.0, 1.0, parameter_names=["n_CSE_1_u"])
# cs2_CSE_1 = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_1"])
# ... (continues for i = 0 to 7)
# cs2_CSE_8 = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_8"])
```

**Total parameters for MetaModel+CSE with nb_CSE=8**:
- 8 NEP parameters (`*_sat`, `*_sym`)
- 1 `nbreak`
- 8 × 2 CSE grid parameters (`n_CSE_i_u`, `cs2_CSE_i`)
- 1 final `cs2_CSE_8`
- **Total: 26 parameters**

### Available Prior Types

Currently supported: `UniformPrior`

```python
from jesterTOV.inference.priors.simple_priors import SimpleUniformPrior

# Uniform prior over [min, max]
param = UniformPrior(
    minimum=0.0,
    maximum=1.0,
    parameter_names=["param_name"]
)
```

**Future support planned**: `LogUniformPrior`, `GaussianPrior`, `TruncatedGaussianPrior`

---

## Transforms

Transforms convert EOS **parameters** (microscopic) to **observables** (macroscopic).

### Transform Types

#### 1. MetaModel Transform

Converts Nuclear Empirical Parameters (NEP) to Mass-Radius-Lambda curves.

**Input parameters** (8):
- `K_sat`, `Q_sat`, `Z_sat` (symmetric matter)
- `E_sym`, `L_sym`, `K_sym`, `Q_sym`, `Z_sym` (asymmetric matter)

**Output observables**:
- `logpc_EOS`: Log of central pressure grid
- `masses_EOS`: Neutron star masses (geometric units)
- `radii_EOS`: Neutron star radii (geometric units)
- `Lambdas_EOS`: Tidal deformabilities
- `n`, `p`, `e`, `h`: EOS thermodynamic quantities
- `cs2`: Sound speed squared
- `dloge_dlogp`: Logarithmic derivative

**Configuration**:
```yaml
transform:
  type: "metamodel"
  ndat_metamodel: 100    # EOS grid resolution
  nmax_nsat: 25.0        # Maximum density
  min_nsat_TOV: 0.75     # Minimum TOV density
  ndat_TOV: 100          # TOV grid resolution
  nb_masses: 100         # Masses for marginalization
  crust_name: "DH"       # Crust model
```

**Physics**: MetaModel constructs EOS using Taylor expansion around saturation density:
$$E(\rho, \delta) = E_0 + \sum_{n=2}^{4} \frac{K_n}{n!}x^n + \delta^2 \sum_{n=1}^{4} \frac{S_n}{n!}x^{n-1}$$

where $x = (\rho - \rho_0)/3\rho_0$, $\delta = (\rho_n - \rho_p)/\rho$.

#### 2. MetaModel+CSE Transform

Extends MetaModel with a Constant Speed of Sound region at high densities.

**Input parameters** (8 NEP + 1 + 2×nb_CSE + 1):
- 8 NEP parameters (same as MetaModel)
- `nbreak`: Density where CSE starts (in units of nsat)
- `n_CSE_i_u`: Normalized density grid points (uniform [0, 1])
- `cs2_CSE_i`: Sound speed squared at each grid point (uniform [0, 1])

**Configuration**:
```yaml
transform:
  type: "metamodel_cse"
  nb_CSE: 8              # Number of CSE grid segments
  # ... other parameters same as metamodel
```

**Physics**:
- Below `nbreak`: MetaModel physics
- Above `nbreak`: Piecewise constant sound speed interpolation
- Grid points: `n_CSE_i = nbreak + (nmax - nbreak) * n_CSE_i_u`

**When to use**:
- **MetaModel**: Standard inference, smooth EOS to ~2-3 nsat
- **MetaModel+CSE**: High-density physics, stiff EOS, phase transitions

### Transform Workflow

```
Parameters (dict)
       ↓
[MetaModel/CSE]
       ↓
EOS: (n, p, e, h, cs2)
       ↓
[TOV Solver]
       ↓
Observables: (M, R, Λ)
```

**Code path**:
```python
# transforms/factory.py
transform = create_transform(config.transform)

# During sampling
transformed_params = transform.forward(sampled_params)
# Returns: {"masses_EOS": ..., "radii_EOS": ..., "Lambdas_EOS": ...}
```

---

## Likelihoods

Likelihoods quantify agreement between predicted observables and data.

### Available Likelihood Types

#### 1. Gravitational Wave (`gw`)

Compares predicted tidal deformabilities against GW event posteriors.

**Parameters**:
```yaml
- type: "gw"
  enabled: true
  parameters:
    event_name: "GW170817"
    model_path: "./NFs/GW170817/model.eqx"
    sample_masses: true  # Include m1, m2 as parameters
    very_negative_value: -9999999.0  # Return value for invalid M-R
```

**Physics**: Uses normalizing flow (NF) trained on LIGO/Virgo posterior samples. Evaluates:
$$\log p(m_1, m_2, \Lambda_1, \Lambda_2 | \text{data})$$

**Inputs from transform**:
- `masses_EOS`, `radii_EOS`, `Lambdas_EOS`
- `mass_1_GW170817`, `mass_2_GW170817` (from prior if `sample_masses: true`)

**Implementation**: `likelihoods/gw.py` - `GWlikelihood_with_masses`

#### 2. NICER X-ray Timing (`nicer`)

Compares predicted M-R curve against NICER pulse profile modeling.

**Parameters**:
```yaml
- type: "nicer"
  enabled: true
  parameters:
    targets: ["J0030", "J0740"]  # PSR J0030+0451, PSR J0740+6620
    analysis_groups: ["amsterdam", "maryland"]
    sample_masses: false
    m_min: 1.0
    m_max: 2.5
    nb_masses: 100  # Marginalization grid
```

**Physics**: Evaluates KDE of M-R posterior from pulse profile analysis:
$$\log p(M, R | \text{NICER data})$$

If `sample_masses: false`, marginalizes over mass grid.

**Implementation**: `likelihoods/nicer.py` - `NICERLikelihood`, `NICERLikelihood_with_masses`

#### 3. Radio Pulsar Timing (`radio`)

Gaussian constraint on maximum neutron star mass from timing measurements.

**Parameters**:
```yaml
- type: "radio"
  enabled: true
  parameters:
    psr_name: "J0740+6620"
    mass_mean: 2.08  # Solar masses
    mass_std: 0.07
    nb_masses: 100
```

**Physics**:
$$\log p(M_\text{max} | \text{timing}) = -\frac{(M_\text{max} - \mu)^2}{2\sigma^2}$$

**Implementation**: `likelihoods/radio.py` - `RadioTimingLikelihood`

#### 4. Chiral Effective Field Theory (`chieft`)

Ensures EOS pressure lies within ChiEFT uncertainty bands at low density.

**Parameters**:
```yaml
- type: "chieft"
  enabled: true
  parameters:
    nb_n: 100  # Number of density points to check
```

**Physics**: At densities $n < 2n_\text{sat}$, requires:
$$p_\text{low}(n) \leq p_\text{EOS}(n) \leq p_\text{high}(n)$$

Returns $-\infty$ if EOS violates bands.

**Implementation**: `likelihoods/chieft.py` - `ChiEFTLikelihood`

#### 5. PREX/CREX (`rex`)

Neutron skin thickness constraint from lead nucleus experiments.

**Parameters**:
```yaml
- type: "rex"
  enabled: true
  parameters:
    experiment_name: "PREX"  # or "CREX"
```

**Physics**: Relates $(E_\text{sym}, L_\text{sym})$ to neutron skin thickness via KDE.

**Implementation**: `likelihoods/rex.py` - `REXLikelihood`

⚠️ **Status**: Not yet fully implemented (placeholder).

#### 6. Zero Likelihood (`zero`)

For prior-only sampling (no observational constraints).

**Parameters**:
```yaml
- type: "zero"
  enabled: true
```

**Returns**: Always returns `0.0` (log-likelihood).

**Use case**: Testing transform, prior sampling, prior predictive checks.

**Implementation**: `likelihoods/combined.py` - `ZeroLikelihood`

### Combined Likelihood

When multiple likelihoods are enabled, they are combined via summation (log space):

$$\log p(\theta | \text{all data}) = \sum_i \log p(\theta | \text{data}_i)$$

**Automatic combination**: Handled by `likelihoods/factory.py` → `create_combined_likelihood()`

---

## Data Management

### DataLoader Class

The `DataLoader` class (`data/loader.py`) handles:
- Lazy loading (load only when needed)
- Caching (load once, reuse)
- KDE construction from posterior samples
- Path management

**Initialization**:
```python
from jesterTOV.inference.data.loader import DataLoader

# Use default paths
loader = DataLoader()

# Override paths
loader = DataLoader(data_paths={
    "nicer_j0030_amsterdam": "./my_data/J0030.txt",
    "chieft_low": "./my_data/chieft_low.dat",
})
```

### Supported Data Types

| Data Type | Loader Method | Format | Returns |
|-----------|---------------|--------|---------|
| NICER KDE | `load_nicer_kde(psr, group)` | CSV (M, R, weight) | `gaussian_kde` |
| ChiEFT bands | `load_chieft_bands()` | TXT (n, p) | `(n_low, p_low, n_high, p_high)` |
| GW NF model | `load_gw_nf_model(event)` | Equinox `.eqx` | NF model object |
| REX posterior | `load_rex_posterior(exp)` | NPZ | `gaussian_kde` |

### Data Path Configuration

**Method 1**: Use default paths (in `DataLoader._get_default_paths()`)

**Method 2**: Override via config file
```yaml
data_paths:
  nicer_j0030_amsterdam: "/absolute/path/to/data.txt"
  chieft_low: "./relative/path/to/low.dat"
```

**Method 3**: Pass to DataLoader directly
```python
loader = DataLoader(data_paths={"key": "path"})
```

### Caching Mechanism

DataLoader caches all loaded data in `self._cache`:

```python
# First call: loads and caches
kde1 = loader.load_nicer_kde("J0030", "amsterdam")

# Second call: returns cached version (instant)
kde2 = loader.load_nicer_kde("J0030", "amsterdam")

assert kde1 is kde2  # Same object
```

**Cache keys**: Automatically generated (e.g., `"nicer_j0030_amsterdam"`, `"chieft_bands"`)

---

## Sampling

### JesterSampler

The `JesterSampler` class (`samplers/jester_sampler.py`) is a standalone implementation wrapping flowMC. It provides a Jim-like interface with critical bug fixes.

**Architecture**:
```
JesterSampler
    ├─ Prior (CombinePrior)
    ├─ Likelihood (LikelihoodBase)
    ├─ Transforms (list of NtoMTransform)
    └─ flowMC.Sampler
        ├─ Local sampler (MALA or GaussianRandomWalk)
        └─ Global sampler (MaskedCouplingRQSpline NF)
```

### Sampler Parameters

**Core parameters** (from `SamplerConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_chains` | 20 | Number of parallel MCMC chains |
| `n_loop_training` | 3 | Number of training loops |
| `n_loop_production` | 3 | Number of production loops |
| `n_local_steps` | 100 | MCMC steps per loop |
| `n_global_steps` | 100 | Global (NF) steps per loop |
| `n_epochs` | 30 | NF training epochs per loop |
| `learning_rate` | 0.001 | NF learning rate |
| `train_thinning` | 1 | Keep every Nth training sample |
| `output_thinning` | 5 | Keep every Nth production sample |

**Advanced parameters**:
- `local_sampler_name`: `"MALA"` or `"GaussianRandomWalk"` (default)
- `local_sampler_arg`: `{"step_size": mass_matrix * eps}`
- `use_global`: `True` (enable NF proposals)
- `num_layers`: 10 (NF architecture)
- `hidden_size`: [128, 128] (NF hidden layers)
- `num_bins`: 8 (RQ spline bins)

### Sampling Phases

#### Phase 1: Training

**Goal**: Train normalizing flow to approximate posterior

**Process**:
1. Run `n_local_steps` of local MCMC (MALA/GRW)
2. Collect samples
3. Train NF for `n_epochs` on collected samples
4. Repeat for `n_loop_training` loops

**Output**: Trained NF model, training samples (usually discarded)

#### Phase 2: Production

**Goal**: Generate final posterior samples

**Process**:
1. Run `n_local_steps` of local MCMC
2. Run `n_global_steps` using NF proposals
3. Repeat for `n_loop_production` loops

**Output**: Final samples (saved to `results_production.npz`)

### Posterior Evaluation

JesterSampler constructs the posterior through:

```python
def posterior(params_array, data):
    # 1. Convert array to named dict
    params = add_name(params_array, prior.parameter_names)

    # 2. Evaluate prior
    log_prior = prior.log_prob(params)

    # 3. Apply likelihood transforms
    for transform in likelihood_transforms:
        params = transform.forward(params)

    # 4. Evaluate likelihood
    log_likelihood = likelihood.evaluate(params, data)

    # 5. Return log posterior
    return log_likelihood + log_prior
```

**Important**: The `data` argument is always passed as an empty dict `{}` in JESTER, since data is encapsulated in likelihood objects.

---

## Complete Workflow

### Step-by-Step Execution

#### 1. Configuration Loading

```python
from jesterTOV.inference.config.parser import load_config

config = load_config("config.yaml")
# Returns: InferenceConfig object (validated)
```

**What happens**:
- Reads YAML file
- Validates structure with Pydantic
- Resolves relative paths (prior file, data paths)
- Raises `ValidationError` if invalid

#### 2. Prior Setup

```python
from jesterTOV.inference.priors.parser import parse_prior_file

# Determine conditional parameters
nb_CSE = config.transform.nb_CSE if config.transform.type == "metamodel_cse" else 0
# Parse prior file
prior = parse_prior_file(
    config.prior.specification_file,
    nb_CSE=nb_CSE
)
# Returns: CombinePrior object
```

**What happens**:
- Reads `.prior` file
- Executes Python code to create Prior objects
- Filters parameters based on config (NEP, nbreak, GW masses)
- Auto-generates CSE grid parameters if `nb_CSE > 0`
- Combines into `CombinePrior`

#### 3. Transform Creation

```python
from jesterTOV.inference.transforms.factory import create_transform

transform = create_transform(config.transform)
# Returns: MetaModelTransform or MetaModelCSETransform
```

**What happens**:
- Factory selects transform class based on `config.transform.type`
- Initializes with TOV solver configuration
- Sets up EOS → M-R-Λ conversion

#### 4. Data Loading

```python
from jesterTOV.inference.data.loader import DataLoader

data_loader = DataLoader(data_paths=config.data_paths)
# Returns: DataLoader instance (lazy, nothing loaded yet)
```

**What happens**:
- Stores data paths
- Initializes empty cache
- Actual loading happens on-demand in likelihood creation

#### 5. Likelihood Creation

```python
from jesterTOV.inference.likelihoods.factory import create_combined_likelihood

likelihood = create_combined_likelihood(config.likelihoods, data_loader)
# Returns: CombinedLikelihood or single likelihood
```

**What happens**:
- Iterates through enabled likelihoods in config
- Calls data_loader to load required data (KDEs, models)
- Creates individual likelihood objects
- Combines into `CombinedLikelihood` if multiple

#### 6. Sampler Setup

```python
from jesterTOV.inference.samplers.flowmc import setup_flowmc_sampler

sampler = setup_flowmc_sampler(
    config.sampler,
    prior,
    likelihood,
    transform,
    seed=config.seed
)
# Returns: JesterSampler instance
```

**What happens**:
- Creates mass matrix (identity by default)
- Initializes JesterSampler with:
  - Prior, likelihood, transform
  - flowMC backend (MALA/GRW + NF)
  - All sampler parameters from config

#### 7. Sampling Execution

```python
import jax

# Initialize chains from prior
sampler.sample(jax.random.PRNGKey(config.seed))

# Print summary
sampler.print_summary()
```

**What happens**:
- Samples initial positions from prior
- Runs training loops (local MCMC + NF training)
- Runs production loops (local + global MCMC)
- Stores samples in internal state

#### 8. Results Retrieval

```python
# Get samples as dict
samples = sampler.get_samples(training=False)
# Returns: {"K_sat": array, "Q_sat": array, ...}

# Get sampler state
state = sampler.sampler.get_sampler_state(training=False)
# Returns: {"chains": array, "log_prob": array, "local_accs": array, ...}
```

#### 9. Save Results

```python
import numpy as np
import os

outdir = config.sampler.output_dir
os.makedirs(outdir, exist_ok=True)

# Save production samples
np.savez(
    os.path.join(outdir, "results_production.npz"),
    log_prob=state["log_prob"].flatten(),
    **{k: np.array(v) for k, v in samples.items()}
)
```

#### 10. Generate EOS Samples

```python
import jax.numpy as jnp

# Select random subset
n_eos = 10000
idx = np.random.choice(len(log_prob), size=n_eos, replace=False)
chosen_samples = {k: jnp.array(v[idx]) for k, v in samples.items()}

# Run TOV solver on selected samples
eos_transform = create_transform(config.transform)  # Fresh transform
transformed = jax.vmap(eos_transform.forward)(chosen_samples)

# Save EOS data
chosen_samples.update(transformed)
np.savez(
    os.path.join(outdir, "eos_samples.npz"),
    log_prob=log_prob[idx],
    **chosen_samples
)
```

### Full Script Example

```python
#!/usr/bin/env python
"""Complete inference workflow example"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from jesterTOV.inference.config.parser import load_config
from jesterTOV.inference.priors.parser import parse_prior_file
from jesterTOV.inference.transforms.factory import create_transform
from jesterTOV.inference.likelihoods.factory import create_combined_likelihood
from jesterTOV.inference.samplers.flowmc import setup_flowmc_sampler
from jesterTOV.inference.data.loader import DataLoader

# 1. Load configuration
config = load_config("config.yaml")
print(f"Loaded config with {len(config.likelihoods)} likelihoods")

# 2. Setup components
nb_CSE = config.transform.nb_CSE if config.transform.type == "metamodel_cse" else 0

prior = parse_prior_file(config.prior.specification_file, nb_CSE=nb_CSE)
print(f"Prior has {prior.n_dim} dimensions")

transform = create_transform(config.transform)
data_loader = DataLoader(data_paths=config.data_paths)
likelihood = create_combined_likelihood(config.likelihoods, data_loader)

# 3. Setup sampler
sampler = setup_flowmc_sampler(config.sampler, prior, likelihood, transform, seed=config.seed)

# 4. Run sampling
print("Starting MCMC sampling...")
sampler.sample(jax.random.PRNGKey(config.seed))
sampler.print_summary()

# 5. Save results
samples = sampler.get_samples(training=False)
state = sampler.sampler.get_sampler_state(training=False)
log_prob = state["log_prob"].flatten()

outdir = Path(config.sampler.output_dir)
outdir.mkdir(parents=True, exist_ok=True)

np.savez(outdir / "results_production.npz", log_prob=log_prob, **samples)
print(f"Results saved to {outdir}")
```

---

## Advanced Usage

### Custom Likelihoods

To add a new likelihood type:

1. **Create likelihood class** in `likelihoods/my_likelihood.py`:

```python
from jesterTOV.inference.base import LikelihoodBase
from jaxtyping import Array, Float

class MyLikelihood(LikelihoodBase):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def evaluate(self, params: dict, data: dict) -> Float[Array, ""]:
        """Evaluate log likelihood"""
        # Extract needed parameters
        masses = params["masses_EOS"]
        radii = params["radii_EOS"]

        # Compute log likelihood
        log_like = ...  # Your computation here

        return log_like
```

2. **Add to factory** in `likelihoods/factory.py`:

```python
from .my_likelihood import MyLikelihood

def create_likelihood(config, data_loader):
    # ... existing code ...

    elif config.type == "my_type":
        return MyLikelihood(
            param1=config.parameters.get("param1"),
            param2=config.parameters.get("param2")
        )
```

3. **Update schema** in `config/schema.py`:

```python
class LikelihoodConfig(BaseModel):
    type: Literal["gw", "nicer", "radio", "chieft", "rex", "zero", "my_type"]
    # ...
```

4. **Use in config**:

```yaml
likelihoods:
  - type: "my_type"
    enabled: true
    parameters:
      param1: 42
      param2: 3.14
```

### Custom Priors

To add a new prior distribution:

1. **Create prior class** in `priors/simple_priors.py`:

```python
from jesterTOV.inference.base import Prior
import jax.numpy as jnp

class MyPrior(Prior):
    def __init__(self, param1, param2, parameter_names):
        self.param1 = param1
        self.param2 = param2
        self.parameter_names = parameter_names

    def sample(self, rng_key, n_samples):
        """Sample from prior"""
        # Return dict: {"param_name": jnp.array([...])}
        ...

    def log_prob(self, params):
        """Evaluate log probability"""
        # params is dict: {"param_name": value}
        ...
```

2. **Register in parser** in `priors/parser.py`:

```python
namespace = {
    "UniformPrior": SimpleUniformPrior,
    "MyPrior": MyPrior,  # Add here
}
```

3. **Use in `.prior` file**:

```python
K_sat = MyPrior(param1=1.0, param2=2.0, parameter_names=["K_sat"])
```

### Adjusting Step Sizes

Tuning MCMC step sizes for better acceptance rates:

```python
# In samplers/flowmc.py or your script

# Option 1: Global scaling
eps_mass_matrix = 1e-3  # Decrease for higher acceptance

# Option 2: Per-parameter scaling
mass_matrix = jnp.diag(jnp.array([
    1.0,   # K_sat
    0.5,   # Q_sat (smaller steps)
    2.0,   # Z_sat (larger steps)
    # ... for all parameters
]))
local_sampler_arg = {"step_size": mass_matrix * eps_mass_matrix}
```

**Target acceptance rates**:
- MALA: 50-70%
- Gaussian Random Walk: 20-40%

### Resuming Runs

Currently, resuming is not implemented. To resume a run:

1. Save sampler state manually:
```python
state = sampler.sampler.get_sampler_state(training=False)
np.savez("checkpoint.npz", **state)
```

2. Load and continue:
```python
# This is a placeholder - full implementation needed
checkpoint = np.load("checkpoint.npz")
# Reinitialize sampler with checkpoint state
# Continue sampling
```

### Parallel Runs

Run multiple inference jobs in parallel:

```bash
# Method 1: Bash loop
for seed in 1 2 3 4 5; do
    sed "s/seed: 43/seed: $seed/" config.yaml > config_$seed.yaml
    run_jester_inference config_$seed.yaml --output-dir ./run_$seed &
done
wait

# Method 2: GNU parallel
parallel run_jester_inference config_{}.yaml --output-dir ./run_{} ::: 1 2 3 4 5
```

---

## File Structure Reference

### Input Files

```
your_project/
├── config.yaml              # Main configuration
├── prior.prior              # Prior specification
└── data/                    # Observational data
    ├── NICER/
    │   ├── J0030/
    │   │   ├── amsterdam.txt
    │   │   └── maryland.txt
    │   └── J0740/
    │       ├── amsterdam.dat
    │       └── maryland.txt
    ├── chieft/
    │   ├── low.dat
    │   └── high.dat
    └── NFs/
        └── GW170817/
            └── model.eqx
```

### Output Files

```
outdir/
├── results_production.npz   # MCMC samples
│   ├── log_prob            # Log posterior values
│   ├── K_sat               # Parameter samples
│   ├── Q_sat
│   └── ... (all parameters)
│
├── eos_samples.npz          # EOS curves (subset of samples)
│   ├── log_prob
│   ├── K_sat, Q_sat, ...   # Parameters
│   ├── masses_EOS          # M-R-Λ curves
│   ├── radii_EOS
│   ├── Lambdas_EOS
│   ├── n, p, e, h          # Thermodynamic quantities
│   └── cs2                 # Sound speed
│
└── runtime.txt              # Timing information
```

### Loading Results

```python
import numpy as np

# Load MCMC samples
results = np.load("outdir/results_production.npz")
log_prob = results["log_prob"]
K_sat_samples = results["K_sat"]

# Load EOS curves
eos = np.load("outdir/eos_samples.npz")
masses = eos["masses_EOS"]  # Shape: (n_samples, nb_masses)
radii = eos["radii_EOS"]
```

---

## Summary

### Key Takeaways

1. **Configuration-driven**: Everything controlled via YAML files
2. **Modular architecture**: Independent components for flexibility
3. **Type-safe**: Pydantic validation prevents configuration errors
4. **Extensible**: Easy to add new likelihoods, priors, transforms
5. **Reproducible**: Seed management, config versioning

### Recommended Workflow

1. **Start with examples**: Copy `examples/inference/*/config.yaml`
2. **Modify for your case**: Adjust likelihoods, priors, sampler settings
3. **Test with small run**: Use `n_chains: 5`, `n_loop_training: 1`
4. **Scale up**: Increase chains and loops for production
5. **Analyze results**: Load `.npz` files, make corner plots, M-R diagrams

### Getting Help

- **Examples**: See `jester/examples/inference/` for working configurations
- **Source code**: Read docstrings in module files for detailed API documentation
- **Issues**: Report bugs or request features on GitHub

---

**Document Version**: 1.0
**Last Updated**: December 2024
**Maintainers**: JESTER Development Team
