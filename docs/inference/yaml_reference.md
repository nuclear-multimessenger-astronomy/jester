(yaml-reference)=
# YAML configuration reference

This page is the authoritative reference for all supported YAML configuration options in the JESTER inference system. When you add or change configuration fields, update this file by hand to keep it accurate.

## Overview

JESTER uses YAML configuration files validated by Pydantic models (see {class}`~jesterTOV.inference.config.schema.InferenceConfig`). This reference documents every supported field, its type, default value, and purpose. For a worked example from start to finish, see the {doc}`quickstart` guide. For a conceptual explanation of the full inference pipeline, see {ref}`inference-workflow`.

---

## Run options

Control runtime behavior for validation, debugging, and random seed configuration.

::::{dropdown} **Configuration options**
:open:

```yaml
seed: 43              # Random seed for reproducibility
dry_run: false        # Validate configuration without running inference
validate_only: false  # Only validate configuration and exit
debug_nans: false     # Enable JAX NaN debugging for numerical issues
```

**Field Details:**

- **`seed`** (`int`, default: `43`) - Random seed for reproducibility across runs
- **`dry_run`** (`bool`, default: `false`) - Parse and validate configuration without running inference
- **`validate_only`** (`bool`, default: `false`) - Validate configuration and prior file, then exit
- **`debug_nans`** (`bool`, default: `false`) - Enable JAX NaN debugging for catching numerical issues during inference

::::

---

## EOS configuration

The `eos` section specifies which equation of state (EOS) parametrization to use. For a conceptual overview of all available EOS models, see {ref}`overview-eos`. Each model transforms a set of physically motivated parameters into a pressure–density relation that is then fed to the TOV solver.

### Metamodel

The metamodel parametrizes the EOS using nuclear empirical parameters (NEPs) around nuclear saturation density. For the physical motivation and parameter definitions, see {ref}`eos-metamodel`. The corresponding Python class is {class}`~jesterTOV.eos.metamodel.MetaModel_EOS_model`. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.MetamodelEOSConfig`.

::::{dropdown} **Metamodel configuration**

```yaml
eos:
  type: "metamodel"    # Required: EOS parametrization type
  ndat_metamodel: 100  # Number of points for EOS table
  nmax_nsat: 25.0      # Maximum density (in units of saturation density)
  nmin_MM_nsat: 0.75   # Minimum density for metamodel (in units of n_sat)
  crust_name: "DH"     # Crust model: "DH", "BPS", or "SLy"
  nb_CSE: 0            # Must be 0 for standard metamodel
```

**Requirements:**
- `nb_CSE` must be 0 (or omitted) for this parametrization

::::

### Metamodel CSE

The metamodel with speed-of-sound extension (CSE) replaces the metamodel above a breakdown density with a flexible speed-of-sound parametrization, allowing for more freedom at high densities. For the physical motivation and parameter definitions, see {ref}`eos-metamodel-cse`. The corresponding Python class is {class}`~jesterTOV.eos.metamodel.MetaModel_with_CSE_EOS_model`. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.MetamodelCSEEOSConfig`.

::::{dropdown} **Metamodel CSE configuration**

```yaml
eos:
  type: "metamodel_cse"       # Required: EOS parametrization type
  nb_CSE: 8                   # Number of CSE enforcement points (must be > 0)
  ndat_CSE: 100               # Number of grid points for the CSE density region
  max_nbreak_nsat: null       # Maximum breaking density in units of n_sat (optional)
  ndat_metamodel: 100         # Number of points for EOS table
  nmax_nsat: 25.0             # Maximum density (in units of saturation density)
  nmin_MM_nsat: 0.75          # Minimum density for metamodel (in units of n_sat)
  crust_name: "DH"            # Crust model: "DH", "BPS", or "SLy"
```

**Field Details:**

- **`nb_CSE`** (`int`, default: `8`) - Number of CSE enforcement grid points (must be > 0)
- **`ndat_CSE`** (`int`, default: `100`) - Number of density grid points for the CSE region
- **`max_nbreak_nsat`** (`float | null`, default: `null`) - Maximum allowed breaking density in units of saturation density. If set, must be consistent with the upper bound of the `nbreak` prior.
- **`ndat_metamodel`** (`int`, default: `100`) - Number of points for metamodel EOS table
- **`nmax_nsat`** (`float`, default: `25.0`) - Maximum density in units of saturation density
- **`nmin_MM_nsat`** (`float`, default: `0.75`) - Starting density for metamodel grid as fraction of saturation density
- **`crust_name`** (`str`, default: `"DH"`) - Crust model: `"DH"`, `"BPS"`, or `"SLy"`

**Requirements:**
- `nb_CSE` must be > 0 for this parametrization

::::

### Metamodel PeakCSE

The metamodel with peak speed-of-sound extension (PeakCSE) is a variant of the CSE parametrization that models a peak feature in the speed of sound at high densities. The corresponding Python class is {class}`~jesterTOV.eos.metamodel.MetaModel_with_PeakCSE_EOS_model`. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.MetamodelPeakCSEEOSConfig`.

::::{dropdown} **Metamodel PeakCSE configuration**

```yaml
eos:
  type: "metamodel_peak_cse"  # Required: EOS parametrization type
  ndat_CSE: 100               # Number of grid points for the PeakCSE region
  max_nbreak_nsat: null       # Maximum breaking density in units of n_sat (optional)
  ndat_metamodel: 100         # Number of points for EOS table
  nmax_nsat: 25.0             # Maximum density (in units of saturation density)
  nmin_MM_nsat: 0.75          # Minimum density for metamodel (in units of n_sat)
  crust_name: "DH"            # Crust model: "DH", "BPS", or "SLy"
```

**Field Details:**

- **`ndat_CSE`** (`int`, default: `100`) - Number of density grid points for the PeakCSE region
- **`max_nbreak_nsat`** (`float | null`, default: `null`) - Maximum allowed breaking density in units of saturation density. If set, the metamodel grid is only computed up to this density, which can speed up inference.
- **`ndat_metamodel`** (`int`, default: `100`) - Number of points for metamodel EOS table
- **`nmax_nsat`** (`float`, default: `25.0`) - Maximum density in units of saturation density
- **`nmin_MM_nsat`** (`float`, default: `0.75`) - Starting density for metamodel grid as fraction of saturation density
- **`crust_name`** (`str`, default: `"DH"`) - Crust model: `"DH"`, `"BPS"`, or `"SLy"`

::::

### Spectral

The spectral decomposition parametrizes the adiabatic index as a function of pressure using a series of basis functions, following Lindblom (2010). It is compatible with LALSimulation for GW analysis. For the physical motivation and parameter definitions, see {ref}`eos-spectral`. The corresponding Python class is {class}`~jesterTOV.eos.spectral.SpectralDecomposition_EOS_model`. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.SpectralEOSConfig`.

::::{dropdown} **Spectral configuration**

```yaml
eos:
  type: "spectral"         # Required: EOS parametrization type
  n_points_high: 500       # Number of points for high-density spectral region
  crust_name: "SLy"        # Crust model (recommended "SLy" for LALSuite compatibility)
  reparametrized: false    # Use whitened reparametrization of gamma coefficients
  sigma_scale: 1.0         # Width scaling for reparametrized prior (only used when reparametrized: true)
```

**Field Details:**

- **`n_points_high`** (`int`, default: `500`) - Number of points for high-density spectral region
- **`crust_name`** (`str`, default: `"DH"`) - Crust model: `"DH"`, `"BPS"`, or `"SLy"`. Use `"SLy"` for LALSuite compatibility.
- **`reparametrized`** (`bool`, default: `false`) - If `false`, sample directly in $(\gamma_0, \gamma_1, \gamma_2, \gamma_3)$. If `true`, sample in a whitened space $(\tilde{\gamma}_0, \tilde{\gamma}_1, \tilde{\gamma}_2, \tilde{\gamma}_3)$ centred on a Gaussian fit to a radio-timing inference result. Use a `MultivariateGaussianPrior` in the prior file when this is enabled.
- **`sigma_scale`** (`float`, default: `1.0`) - Multiplicative scaling applied to the Cholesky factor to widen the prior around the radio posterior. Only used when `reparametrized: true`. Increase above 1.0 to broaden the prior.

**Requirements:**
- `nb_CSE` must be 0 (or omitted)

**Recommended:**
- Use `constraints_gamma` likelihood to bound Gamma parameters (optional but recommended)

::::

---

## TOV configuration

The `tov` section configures the Tolman–Oppenheimer–Volkoff (TOV) equation solver used to compute neutron star structure (mass, radius, tidal deformability). For a conceptual overview of all available solvers, see {ref}`overview-tov`. The Pydantic base schema is {class}`~jesterTOV.inference.config.schema.BaseTOVConfig`.

Three solver types are supported: `"gr"` (standard GR), `"anisotropy"` (phenomenological parametrizations to include pressure anisotropy), and `"scalar_tensor"` (scalar-tensor gravity). All solvers share the base fields `min_nsat_TOV`, `ndat_TOV`, and `nb_masses`; the beyond-GR solvers additionally require theory parameters that must be included in the prior file.

### General Relativity

The standard GR TOV solver integrates the Tolman–Oppenheimer–Volkoff equations in full General Relativity. It requires no additional theory parameters. The corresponding Python class is {class}`~jesterTOV.tov.gr.GRTOVSolver`. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.GRTOVConfig`.

::::{dropdown} **GR TOV Solver configuration**
:open:

```yaml
tov:
  type: "gr"          # TOV solver type
  min_nsat_TOV: 0.75  # Minimum central density for TOV integration (in units of n_sat)
  ndat_TOV: 100       # Number of data points for TOV integration
  nb_masses: 100      # Number of masses for M-R-Λ family construction
```

**Field Details:**

- **`type`** (`str`) - Must be `"gr"` for this solver
- **`min_nsat_TOV`** (`float`, default: `0.75`) - Minimum central density for TOV integration in units of saturation density
- **`ndat_TOV`** (`int`, default: `100`) - Number of data points for TOV integration
- **`nb_masses`** (`int`, default: `100`) - Number of masses to sample when constructing the M-R-Λ family (see {class}`~jesterTOV.tov.data_classes.FamilyData`)

::::

### Pressure anisotropy 

The anisotropy solver extends the standard TOV equations with phenomenological beyond-GR corrections through an additional sigma term in the pressure gradient equation. For the underlying physics, see {ref}`tov-anisotropy`. The corresponding Python class is {class}`~jesterTOV.tov.anisotropy.AnisotropyTOVSolver`.

::::{dropdown} **Anisotropy TOV Solver configuration**

```yaml
tov:
  type: "anisotropy"  # TOV solver type
  min_nsat_TOV: 0.75  # Minimum central density for TOV integration (in units of n_sat)
  ndat_TOV: 100       # Number of data points for TOV integration
  nb_masses: 100      # Number of masses for M-R-Λ family construction
```

**Field Details:**

- **`type`** (`str`) - Must be `"anisotropy"` for this solver
- **`min_nsat_TOV`** (`float`, default: `0.75`) - Minimum central density for TOV integration in units of saturation density
- **`ndat_TOV`** (`int`, default: `100`) - Number of data points for TOV integration
- **`nb_masses`** (`int`, default: `100`) - Number of masses to sample when constructing the M-R-Λ family

**Required prior parameters:**

The following theory parameters must be included in the `.prior` file:

| Parameter | Description |
|-----------|-------------|
| `lambda_BL` | Bowers–Liang coupling constant |
| `lambda_DY` | Doneva–Yazadjiev (Horvat et al.) coupling constant |
| `lambda_HB` | Herrera–Barreto (Cosenza et al.) coupling constant |
| `gamma` | Post-Newtonian gamma coupling |
| `alpha` | Post-Newtonian alpha coupling |
| `beta` | Post-Newtonian beta coupling |

Any subset of these parameters may be sampled; those omitted default to the GR limit (zero). Setting all coupling constants to zero recovers standard GR.

::::

---

## Prior configuration

Specify prior distributions for EOS parameters using a `.prior` specification file. The prior file uses a Python-based syntax (bilby-style) and is parsed by {func}`~jesterTOV.inference.priors.parser.parse_prior_file` into a {class}`~jesterTOV.inference.base.prior.CombinePrior` object. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.PriorConfig`.

::::{dropdown} **Prior configuration**

```yaml
prior:
  specification_file: "prior.prior"  # Path to prior specification file (required)
```

**Field Details:**

- **`specification_file`** (`str`, **required**) - Path to prior specification file (must end with `.prior`)

Individual priors in the `.prior` file are specified as {class}`~jesterTOV.inference.base.prior.UniformPrior` instances or other prior types. See the {doc}`quickstart` for an example prior file.

::::

---

## Likelihoods

The `likelihoods` section specifies observational constraints to include in the inference. Multiple likelihoods can be combined for multi-messenger analysis; they are assembled into a {class}`~jesterTOV.inference.likelihoods.combined.CombinedLikelihood`. For a conceptual overview of all available likelihoods and their physical motivation, see {ref}`overview-likelihoods`.

### Gravitational wave observations

Constrain the EOS using gravitational wave observations of binary neutron star mergers. For the physics and methodology, see {ref}`likelihood-gw`.

#### Standard GW likelihood (presampled)

::::{dropdown} **Standard GW Likelihood (presampled)**

```yaml
- type: "gw"
  enabled: true
  events:                       # List of GW events (see GWEventConfig below)
    - name: "GW170817"
      nf_model_dir: "./NFs/GW170817"
  penalty_value: 0.0            # Log-likelihood penalty for M > M_TOV (default: 0.0)
  N_masses_evaluation: 2000     # Number of mass samples to pre-sample (optional, default: 2000)
  N_masses_batch_size: 1000     # Batch size for processing (optional, default: 1000)
  seed: 42                      # Random seed for mass sampling (optional, default: 42)
```

**Field Details:**

- **`events`** (`list[GWEventConfig]`) - List of GW event configs (see **GWEventConfig** below). Each entry must have `name`. Three modes are supported:
  - **Pre-trained flow**: set `nf_model_dir` to point to a trained flow, or omit it to use a built-in preset.
  - **From bilby result**: set `from_bilby_result` to the path of a bilby HDF5 result file; jester will extract posterior samples and train a flow automatically before inference.
  - **From NPZ file**: set `from_npz_file` to an existing `.npz` file with posterior samples; jester will train a flow directly from it, skipping the bilby extraction step.
- **`penalty_value`** (`float`, default: `0.0`) - Log-likelihood penalty for masses exceeding TOV maximum mass (default: 0.0, i.e. no penalty)
- **`N_masses_evaluation`** (`int`, default: `2000`) - Number of mass samples to pre-sample from the GW posterior
- **`N_masses_batch_size`** (`int`, default: `1000`) - Batch size for jax.lax.map processing of mass grid
- **`seed`** (`int`, default: `42`) - Random seed for mass pre-sampling from GW posterior

**Description:**

**Default GW likelihood** (presampled version): pre-samples masses from the GW posterior for efficient evaluation. Recommended for production use. See {class}`~jesterTOV.inference.likelihoods.gw.GWLikelihood` for the full API. For information on training flows from bilby results or NPZ files, see {doc}`training_flows`.

**GWEventConfig fields** (each entry in `events`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Event name, e.g. `GW170817` |
| `nf_model_dir` | str\|null | null | Path to a pre-trained normalizing flow directory. Mutually exclusive with `from_bilby_result` and `from_npz_file`. |
| `from_bilby_result` | str\|null | null | Path to a bilby result `.hdf5` file. jester will extract posterior samples and train a flow automatically. Mutually exclusive with `nf_model_dir` and `from_npz_file`. |
| `from_npz_file` | str\|null | null | Path to an existing `.npz` file with posterior samples (`mass_1_source`, `mass_2_source`, `lambda_1`, `lambda_2`). jester will train a flow directly from this file, skipping bilby extraction. Mutually exclusive with `nf_model_dir` and `from_bilby_result`. |
| `flow_config` | str\|null | null | Path to a {class}`~jesterTOV.inference.flows.config.FlowTrainingConfig` YAML file for custom flow training (only valid with `from_bilby_result` or `from_npz_file`). |
| `retrain_flow` | bool | false | Force re-training even if a cached flow exists (only valid with `from_bilby_result` or `from_npz_file`). |

**Examples**:

```yaml
# Pre-trained flow (preset):
events:
  - name: GW170817

# Pre-trained flow (custom path):
events:
  - name: GW170817
    nf_model_dir: ./my_flow

# From bilby result (auto-train):
events:
  - name: GW170817
    from_bilby_result: ./GW170817_result.hdf5

# From existing NPZ file (skip bilby extraction):
events:
  - name: GW170817
    from_npz_file: ./GW170817_posterior.npz
```

::::

#### Resampled GW likelihood (legacy)

::::{dropdown} **Resampled GW likelihood (legacy)**

```yaml
- type: "gw_resampled"
  enabled: true
  events:                       # List of GW events
    - name: "GW170817"
      nf_model_dir: "./NFs/GW170817"
  penalty_value: 0.0            # Log-likelihood penalty for M > M_TOV (default: 0.0)
  N_masses_evaluation: 20       # Number of masses per evaluation (optional, default: 20)
  N_masses_batch_size: 10       # Batch size for sampling (optional, default: 10)
```

**Field Details:**

- **`events`** (`list[dict]`) - List of GW events with `name` and optional `nf_model_dir` keys
- **`penalty_value`** (`float`, default: `0.0`) - Log-likelihood penalty for masses exceeding TOV maximum mass (default: 0.0, i.e. no penalty)
- **`N_masses_evaluation`** (`int`, default: `20`) - Number of mass samples to draw on-the-fly per likelihood evaluation
- **`N_masses_batch_size`** (`int`, default: `10`) - Batch size for mass sampling and processing

**Description:**

**Legacy GW likelihood**: resamples masses from the GW posterior on-the-fly during each likelihood evaluation. Slower than the presampled version. See {class}`~jesterTOV.inference.likelihoods.gw.GWLikelihoodResampled` for the full API.

::::

### NICER observations

Constrain the mass–radius relation using NICER X-ray timing observations of millisecond pulsars. For the physics and methodology, see {ref}`likelihood-nicer`.

#### NICER flow likelihood (default)

::::{dropdown} **NICER Flow Likelihood (default)**

```yaml
- type: "nicer"
  enabled: true
  pulsars:                      # List of pulsars with flow model directories
    - name: "J0030"
      amsterdam_model_dir: "./flows/models/nicer_maf/J0030/amsterdam"
      maryland_model_dir: "./flows/models/nicer_maf/J0030/maryland"
  N_masses_evaluation: 100      # Number of mass samples (optional, default: 100)
  N_masses_batch_size: 20       # Batch size for processing (optional, default: 20)
  seed: 42                      # Random seed for mass pre-sampling (optional, default: 42)
```

**Field Details:**

- **`pulsars`** (`list[dict]`) - List of pulsars with `name`, `amsterdam_model_dir`, and `maryland_model_dir` keys. Model directories must point to trained normalizing flow models.
- **`N_masses_evaluation`** (`int`, default: `100`) - Number of mass samples to pre-sample from flow for deterministic evaluation
- **`N_masses_batch_size`** (`int`, default: `20`) - Batch size for processing mass samples with jax.lax.map
- **`seed`** (`int`, default: `42`) - Random seed for reproducible mass pre-sampling from flow

**Description:**

**Default NICER likelihood** using pre-trained normalizing flows on M-R posteriors. Pre-samples masses once at initialization for efficient, deterministic evaluation. Recommended for production use. See {class}`~jesterTOV.inference.likelihoods.nicer.NICERLikelihood` for the full API. For information on training custom flows from NICER posterior samples, see {doc}`training_flows`.

::::

#### NICER KDE likelihood (legacy)

::::{dropdown} **NICER KDE Likelihood (legacy)**

```yaml
- type: "nicer_kde"
  enabled: true
  pulsars:                      # List of pulsars with sample files
    - name: "J0030"
      amsterdam_samples_file: "./data/NICER/J0030/amsterdam.npz"
      maryland_samples_file: "./data/NICER/J0030/maryland.npz"
  N_masses_evaluation: 100      # Number of masses per evaluation (optional, default: 100)
  N_masses_batch_size: 20       # Batch size for sampling (optional, default: 20)
```

**Field Details:**

- **`pulsars`** (`list[dict]`) - List of pulsars with `name`, `amsterdam_samples_file`, and `maryland_samples_file` keys pointing to M-R posterior samples (npz format).
- **`N_masses_evaluation`** (`int`, default: `100`) - Number of mass samples to draw on-the-fly from posterior samples per evaluation
- **`N_masses_batch_size`** (`int`, default: `20`) - Batch size for mass sampling and KDE evaluation

**Description:**

**Legacy NICER likelihood** using kernel density estimation on M-R posterior samples. Resamples masses during each evaluation (slower, non-deterministic). For backward compatibility only — use the flow-based version for new analyses. See {ref}`likelihood-nicer` for a comparison of the two approaches.

::::

### Radio pulsar observations

Constrain neutron star masses using radio pulsar timing measurements. For the physics and methodology, see {ref}`likelihood-radio`. The Python class is {class}`~jesterTOV.inference.likelihoods.radio.RadioTimingLikelihood`.

::::{dropdown} **Radio Pulsar Likelihood**

```yaml
- type: "radio"
  enabled: true
  pulsars:                      # List of pulsars
    - name: "J0740+6620"
      mass_mean: 2.08
      mass_std: 0.07
  penalty_value: 0.0            # Penalty for M_TOV ≤ m_min (optional, default: 0.0)
```

**Field Details:**

- **`pulsars`** (`list[dict]`) - List of pulsars with `name`, `mass_mean`, and `mass_std` keys for Gaussian mass constraints
- **`penalty_value`** (`float`, default: `0.0`) - Log-likelihood penalty for invalid TOV solutions (M_TOV ≤ m_min)

::::

### Nuclear theory constraints

Constrain the low-density EOS using nuclear theory calculations and laboratory measurements.

#### ChiEFT likelihood

Constrains the EOS at densities below ~2 $n_\text{sat}$ using chiral effective field theory (ChiEFT) calculations. The likelihood checks that the predicted pressure–density relation falls within the ChiEFT uncertainty bands. For the physics and the specific bands used, see {ref}`likelihood-chieft`. The Python class is {class}`~jesterTOV.inference.likelihoods.chieft.ChiEFTLikelihood`.

::::{dropdown} **ChiEFT Likelihood**

```yaml
- type: "chieft"
  enabled: true
  low_filename: null   # Path to lower bound ChiEFT data file (optional, default: built-in)
  high_filename: null  # Path to upper bound ChiEFT data file (optional, default: built-in)
  nb_n: 100            # Number of density points to check against bands
```

**Field Details:**

- **`low_filename`** (`str | null`, default: `null`) - Path to lower bound ChiEFT data file. If `null`, uses the built-in default (`data/chiEFT/2402.04172/low.dat`).
- **`high_filename`** (`str | null`, default: `null`) - Path to upper bound ChiEFT data file. If `null`, uses the built-in default (`data/chiEFT/2402.04172/high.dat`).
- **`nb_n`** (`int`, default: `100`) - Number of density points to evaluate against ChiEFT uncertainty bands

::::

#### REX likelihood

::::{dropdown} **REX Likelihood**

```{warning}
This is not fully implemented yet.
```

```yaml
- type: "rex"
  enabled: true
  experiment_name: "PREX"  # Experiment: "PREX" or "CREX" (default: "PREX")
```

**Field Details:**

- **`experiment_name`** (`str`) - Nuclear experiment identifier: `"PREX"` or `"CREX"`

**Description:**

Constrains nuclear symmetry energy parameters using neutron skin thickness measurements from electron scattering experiments. The Python class is {class}`~jesterTOV.inference.likelihoods.rex.REXLikelihood`.

- **PREX** — Lead Radius Experiment (²⁰⁸Pb)
- **CREX** — Calcium Radius Experiment (⁴⁸Ca)

::::

### Generic constraints

Apply custom physics-motivated constraints on EOS and TOV observables. These are implemented as likelihoods that return large negative values (i.e., zero probability) when a constraint is violated, and zero otherwise. For the base class interface, see {class}`~jesterTOV.inference.likelihoods.constraints.ConstraintEOSLikelihood`.

#### EOS constraints

::::{dropdown} **EOS constraints**

```yaml
- type: "constraints_eos"
  enabled: true
  penalty_causality: -1e10  # Penalty for causality violation cs² > 1 (default: -1e10)
  penalty_stability: -1e10  # Penalty for thermodynamic instability cs² < 0 (default: -1e10)
  penalty_pressure: -1e10   # Penalty for non-monotonic pressure (default: -1e10)
```

**Field Details:**

- **`penalty_causality`** (`float`, default: `-1e10`) - Log-likelihood penalty applied when causality is violated ($c_s^2 > 1$)
- **`penalty_stability`** (`float`, default: `-1e10`) - Log-likelihood penalty applied when thermodynamic stability is violated ($c_s^2 < 0$)
- **`penalty_pressure`** (`float`, default: `-1e10`) - Log-likelihood penalty applied when pressure is non-monotonic

**Description:**

Apply hard constraints on equation of state properties (pressure, energy density, sound speed). See {class}`~jesterTOV.inference.likelihoods.constraints.ConstraintEOSLikelihood` for the full API.

::::

#### TOV constraints

::::{dropdown} **TOV constraints**

```yaml
- type: "constraints_tov"
  enabled: true
  penalty_tov: -1e10  # Penalty for TOV integration failure (default: -1e10)
```

**Field Details:**

- **`penalty_tov`** (`float`, default: `-1e10`) - Log-likelihood penalty applied when the TOV integration fails

**Description:**

Apply hard constraints on TOV solution properties (maximum mass, radius bounds, etc.). See {class}`~jesterTOV.inference.likelihoods.constraints.ConstraintTOVLikelihood` for the full API.

::::

#### Symmetry energy constraints

::::{dropdown} **Symmetry energy constraints**

```yaml
- type: "constraints_esym"
  enabled: true
  penalty_esym: -1e10  # Penalty per density point where e_sym < 0 (default: -1e10)
```

**Field Details:**

- **`penalty_esym`** (`float`, default: `-1e10`) - Log-likelihood penalty applied per density grid point where the symmetry energy $e_\mathrm{sym}(n) < 0$

**Description:**

Penalises metamodel configurations where the symmetry energy $e_\mathrm{sym}(n) = e(n, \delta=1) - e(n, \delta=0)$ becomes negative, which is unphysical because it would imply pure neutron matter is more bound than symmetric nuclear matter. The penalty is proportional to the number of density points in violation, so a single mild excursion incurs a smaller (but still very large) penalty than a broad violation. This likelihood is only meaningful for the `metamodel`, `metamodel_cse`, and `metamodel_peakCSE` EOS types; it returns 0.0 gracefully for all other EOS parametrisations. See {class}`~jesterTOV.inference.likelihoods.constraints.ConstraintEsymLikelihood` for the full API.

::::

#### Gamma constraints

::::{dropdown} **Gamma constraints**

```yaml
- type: "constraints_gamma"
  enabled: true
  penalty_gamma: -1e10  # Penalty for Gamma bound violation (default: -1e10)
```

**Field Details:**

- **`penalty_gamma`** (`float`, default: `-1e10`) - Log-likelihood penalty applied when any spectral Gamma parameter falls outside $[0.6, 4.5]$

**Description:**

Apply bounds on spectral decomposition Gamma parameters, enforcing causality and thermodynamic stability. Recommended when using `type: "spectral"` (see {ref}`eos-spectral`). See {class}`~jesterTOV.inference.likelihoods.constraints.ConstraintGammaLikelihood` for the full API.

::::

### Prior-only sampling

Sample from the prior without applying observational constraints.

::::{dropdown} **Zero Likelihood**

```yaml
- type: "zero"
  enabled: true
  parameters: {}  # No parameters needed
```

**Description:**

Returns zero log-likelihood (uniform likelihood) for all EOS configurations. Use this for prior-only sampling to explore the prior volume without observational constraints. See {class}`~jesterTOV.inference.likelihoods.combined.ZeroLikelihood` for the full API.

::::

---

## Samplers

Choose a sampling algorithm for Bayesian inference. JESTER supports four backends with different strengths. For a conceptual comparison, see {ref}`overview-samplers`. The sampler base class is {class}`~jesterTOV.inference.samplers.jester_sampler.JesterSampler`. All samplers produce a {class}`~jesterTOV.inference.samplers.jester_sampler.SamplerOutput` with posterior samples, log-probabilities, and metadata. The base Pydantic schema is {class}`~jesterTOV.inference.config.schema.BaseSamplerConfig`.

### Sequential Monte Carlo with random walk

BlackJAX SMC with adaptive tempering and Gaussian Random Walk kernel. **Production-ready and recommended for most analyses.** For a detailed explanation of the algorithm, see {ref}`sampler-smc`. The Python class is {class}`~jesterTOV.inference.samplers.blackjax.smc.random_walk.BlackJAXSMCRandomWalkSampler`. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.SMCRandomWalkSamplerConfig`.

::::{dropdown} **Sequential Monte Carlo with Random Walk Configuration**

```yaml
sampler:
  type: "smc-rw"             # Sampler type identifier
  output_dir: "./outdir/"    # Output directory for results
  n_eos_samples: 10000       # Number of final posterior samples
  log_prob_batch_size: 1000  # Batch size for log-probability evaluation

  n_particles: 10000         # Number of SMC particles
  n_mcmc_steps: 1            # MCMC steps per tempering stage
  target_ess: 0.9            # Target effective sample size (ESS) fraction
  random_walk_sigma: 1.0     # Gaussian random walk step size
```

**Field Details:**

- **`n_particles`** (`int`, default: `10000`) - Number of particles for SMC
- **`n_mcmc_steps`** (`int`, default: `1`) - MCMC rejuvenation steps per tempering stage
- **`target_ess`** (`float`, default: `0.9`) - Target ESS fraction for adaptive tempering (0.0–1.0)
- **`random_walk_sigma`** (`float`, default: `1.0`) - Step size for Gaussian random walk kernel

**Output:**
- Posterior samples with equal weights
- Effective sample size (ESS) statistics per tempering stage

**When to Use:**
- General-purpose Bayesian inference (**recommended default**)
- Fast inference on CPU or GPU
- When derivative information is unavailable or expensive

::::

### Nested sampling (BlackJAX NS-AW)

BlackJAX nested sampling with acceptance walk for Bayesian evidence estimation and posterior sampling. For a detailed explanation of the algorithm, see {ref}`sampler-nested`. The Python class is {class}`~jesterTOV.inference.samplers.blackjax.nested_sampling.acceptance_walk.BlackJAXNSAWSampler`. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.BlackJAXNSAWConfig`.

::::{dropdown} **Nested Sampling (BlackJAX NS-AW) Configuration (type: "blackjax-ns-aw")**

```yaml
sampler:
  type: "blackjax-ns-aw"     # Sampler type identifier
  output_dir: "./outdir/"    # Output directory for results
  n_eos_samples: 10000       # Number of final posterior samples
  log_prob_batch_size: 1000  # Batch size for log-probability evaluation

  n_live: 1000               # Number of live points
  n_delete_frac: 0.5         # Fraction of live points to delete per iteration
  n_target: 60               # Target number of MCMC steps
  max_mcmc: 5000             # Maximum MCMC steps per iteration
  max_proposals: 1000        # Maximum proposals per live point update
  termination_dlogz: 0.1     # Termination criterion (log evidence uncertainty)
```

**Field Details:**

- **`n_live`** (`int`, default: `1000`) - Number of live points for nested sampling
- **`n_delete_frac`** (`float`, default: `0.5`) - Fraction of live points to delete per iteration
- **`n_target`** (`int`, default: `60`) - Target number of MCMC steps for acceptance walk
- **`max_mcmc`** (`int`, default: `5000`) - Maximum MCMC steps per iteration
- **`max_proposals`** (`int`, default: `1000`) - Maximum proposal attempts per live point update
- **`termination_dlogz`** (`float`, default: `0.1`) - Terminate when log-evidence uncertainty < this value

**Output:**
- Log-evidence (logZ) with uncertainty estimate
- Posterior samples with importance weights (see {class}`~jesterTOV.inference.result.InferenceResult`)

**When to Use:**
- Model comparison requiring Bayesian evidence
- Exploring multi-modal posteriors
- When evidence estimation is the primary goal

::::

### FlowMC (normalizing flow MCMC)

Normalizing flow-enhanced MCMC combining local MCMC proposals with global normalizing flow proposals. For a detailed explanation of the algorithm, see {ref}`sampler-flowmc`. The Python class is {class}`~jesterTOV.inference.samplers.flowmc.FlowMCSampler`. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.FlowMCSamplerConfig`.

::::{dropdown} **FlowMC (Normalizing Flow MCMC) Configuration**

```yaml
sampler:
  type: "flowmc"             # Sampler type identifier
  output_dir: "./outdir/"    # Output directory for results
  n_eos_samples: 10000       # Number of final posterior samples
  log_prob_batch_size: 1000  # Batch size for log-probability evaluation

  n_chains: 20               # Number of parallel MCMC chains
  n_loop_training: 3         # Number of training loops
  n_local_steps: 100         # Local MCMC steps per training loop
  n_epochs: 30               # NF training epochs per loop
  learning_rate: 0.001       # NF optimizer learning rate
  train_thinning: 1          # Thinning factor for training samples

  n_loop_production: 3       # Number of production loops
  n_global_steps: 100        # Global NF proposal steps per production loop
  output_thinning: 5         # Thinning factor for output samples
```

**Sampling Phases:**

1. **Training Phase** — `n_loop_training` loops of:
   - `n_local_steps` MCMC steps using local proposals
   - Train normalizing flow for `n_epochs` on collected samples
2. **Production Phase** — `n_loop_production` loops of:
   - `n_local_steps` MCMC steps using local proposals
   - `n_global_steps` using normalizing flow proposals

**When to Use:**
- Multi-modal or high-dimensional posteriors
- Long production runs requiring efficient exploration
- When training overhead is acceptable

::::

### Sequential Monte Carlo with NUTS

BlackJAX SMC with adaptive tempering and No-U-Turn Sampler (NUTS) kernel. **EXPERIMENTAL — use with caution.** For a detailed explanation of the SMC framework, see {ref}`sampler-smc`. The Python class is {class}`~jesterTOV.inference.samplers.blackjax.smc.nuts.BlackJAXSMCNUTSSampler`. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.SMCNUTSSamplerConfig`.

::::{dropdown} **Sequential Monte Carlo with NUTS Configuration**

```{warning}
This sampler is experimental and may produce unstable results, use at own risk. Use the SMC with a random walk sampler for stable production analyses.
```

```yaml
sampler:
  type: "smc-nuts"              # Sampler type identifier (EXPERIMENTAL)
  output_dir: "./outdir/"       # Output directory for results
  n_eos_samples: 10000          # Number of final posterior samples
  log_prob_batch_size: 1000     # Batch size for log-probability evaluation

  n_particles: 10000            # Number of SMC particles
  n_mcmc_steps: 1               # NUTS steps per tempering stage
  target_ess: 0.9               # Target effective sample size (ESS) fraction

  init_step_size: 0.01          # Initial NUTS step size
  mass_matrix_base: 0.2         # Base value for mass matrix diagonal
  mass_matrix_param_scales: {}  # Per-parameter mass matrix scaling
  target_acceptance: 0.7        # Target acceptance rate for step size adaptation
  adaptation_rate: 0.3          # Rate of step size adaptation
```

**Field Details:**

- **`init_step_size`** (`float`, default: `0.01`) - Initial step size for NUTS integrator
- **`mass_matrix_base`** (`float`, default: `0.2`) - Base diagonal value for mass matrix
- **`mass_matrix_param_scales`** (`dict`, default: `{}`) - Per-parameter scaling factors for mass matrix
- **`target_acceptance`** (`float`, default: `0.7`) - Target acceptance probability for step size tuning
- **`adaptation_rate`** (`float`, default: `0.3`) - Adaptation rate for step size controller

**Output:**
- Posterior samples with equal weights
- Effective sample size (ESS) statistics per tempering stage

**When to Use:**
- **EXPERIMENTAL** — not recommended for production use
- High-dimensional posteriors where gradient information helps
- When NUTS kernel stability can be verified

**Warning:** This sampler is experimental. Use SMC Random Walk for production analyses.

::::

::::{dropdown} **EOS Reweighting** (`type: "eos-reweighting"`)

Evaluates jester's GPU-accelerated likelihoods on a discrete set of tabulated EOS curves (M, $\Lambda$, R tables) rather than sampling a parametric EOS model. Returns the marginal log-likelihood per EOS, the Bayesian evidence $\log Z$, and optionally a Bayes factor between two EOS sets. The Python class is {class}`~jesterTOV.inference.samplers.eos_reweighting.EOSReweightingSampler`. The Pydantic config schema is {class}`~jesterTOV.inference.config.schema.EOSReweightingConfig`.

This sampler does **not** require an `eos`, `tov`, or `prior` section in the YAML — the EOS is provided as tabulated curves. The top-level config schema is {class}`~jesterTOV.inference.config.schemas.eos_reweighting.EOSReweightingInferenceConfig`.

```yaml
sampler:
  type: "eos-reweighting"

  # EOS input — NPZ files with keys: masses, lambdas, radii (all 1-D float64)
  # For a file containing N curves: arrays shaped [N, n_points]
  eos_set_A:
    - "path/to/eos_set_A.npz"       # one or more NPZ files
  eos_set_B:                         # optional — enables Bayes factor B_AB
    - "path/to/eos_set_B.npz"

  # Mass interpolation grid
  n_grid: 200        # number of grid points (default: 200)
  m_min: 0.5         # lower bound in M_sun (default: 0.5)
  m_max: null        # upper bound in M_sun; null → min(M_TOV) across set A

  # JAX batch size for lax.map over EOS curves
  # null → auto-tune starting from N (all at once ≈ vmap), halving on OOM
  batch_size: null

  n_bootstrap: 500   # bootstrap resamples for log Z uncertainty (default: 500)

  output_dir: "outdir/eos_reweighting/"
```

**EOS table format.** Each NPZ file must contain:

| Key | Shape | Description |
|---|---|---|
| `masses` | `[n_points]` or `[N, n_points]` | Gravitational mass in $M_\odot$, monotone increasing |
| `lambdas` | same as `masses` | Dimensionless tidal deformability $\Lambda$ |
| `radii` | same as `masses` | Radius in km (optional; omit if no NICER likelihood) |

**Output.** The sampler writes `result.h5` to `output_dir` containing:
- `posterior/parameters/eos_index` — integer index per EOS in set A
- `posterior/parameters/log_weight` — log-likelihood per EOS
- `posterior/parameters/posterior_weight` — normalised posterior weight
- `metadata/evidence` — `log_Z`, `log_Z_std`, `N_eff`, `N_eff_fraction` for set A
- `metadata/evidence_B`, `metadata/bayes_factor` — set B evidence and $\log_{10} B_{AB}$ (if `eos_set_B` provided)

**When to use:**
- When collaborators provide tabulated EOS sets from e.g. nuclear-theory calculations
- When computing the Bayesian evidence $\log Z$ or Bayes factor between two EOS families
- When you want to reweight an existing EOS prior sample with jester's multi-messenger likelihoods

::::

---

## Data paths (optional)

Override default data file locations for likelihoods.

::::{dropdown} **Data Path Overrides**

```yaml
data_paths:
  # NICER data files
  nicer_j0030_amsterdam: "./data/NICER/J0030/amsterdam.txt"
  nicer_j0030_maryland: "./data/NICER/J0030/maryland.txt"
  nicer_j0740_amsterdam: "./data/NICER/J0740/amsterdam.dat"
  nicer_j0740_maryland: "./data/NICER/J0740/maryland.txt"

  # ChiEFT uncertainty bands
  chieft_low: "./data/chieft/low_density.txt"
  chieft_high: "./data/chieft/high_density.txt"

  # Gravitational wave normalizing flow models
  gw170817_model: "./NFs/GW170817/model.eqx"

  # REX posteriors
  prex_posterior: "./data/REX/PREX_posterior.npz"
  crex_posterior: "./data/REX/CREX_posterior.npz"
```

**Description:**

The `data_paths` section allows overriding default data file locations. If omitted, JESTER uses built-in default paths from the package installation. Data path resolution is handled by the {mod}`jesterTOV.inference.data` module.

::::

---

## Postprocessing

Configure automatic plot generation and posterior analysis after inference completes. Results are stored in HDF5 format (see {class}`~jesterTOV.inference.result.InferenceResult`) and postprocessing reads directly from that file.

::::{dropdown} **Postprocessing configuration**

```yaml
postprocessing:
  enabled: true               # Enable postprocessing
  make_cornerplot: true       # Generate corner plot of posterior
  make_massradius: true       # Generate M-R diagram
  make_masslambda: true       # Generate M-Λ diagram
  make_pressuredensity: true  # Generate P-ε diagram
  make_histograms: true       # Generate 1D posterior histograms
  make_cs2: true              # Generate speed-of-sound plot
  make_contours: false        # Generate radii and pressure credible-interval contour plots
  prior_dir: null             # Optional: directory with prior samples
  injection_eos_path: null    # Optional: path to true EOS for injection studies
  plot_format: "pdf"          # Optional: file format for saving plots. Either 'png' or 'pdf'
```

**Field Details:**

- **`enabled`** (`bool`, default: `true`) - Enable/disable all postprocessing
- **`make_cornerplot`** (`bool`, default: `true`) - Generate corner plot of EOS parameters
- **`make_massradius`** (`bool`, default: `true`) - Generate mass-radius diagram with posterior families
- **`make_masslambda`** (`bool`, default: `true`) - Generate mass-tidal deformability diagram
- **`make_pressuredensity`** (`bool`, default: `true`) - Generate pressure-energy density relation
- **`make_histograms`** (`bool`, default: `true`) - Generate 1D marginalized posterior histograms
- **`make_cs2`** (`bool`, default: `true`) - Generate speed-of-sound as function of density
- **`make_contours`** (`bool`, default: `false`) - Generate radii vs mass and pressure vs density credible-interval contour plots
- **`prior_dir`** (`str | None`, default: `null`) - Directory containing prior samples for comparison
- **`injection_eos_path`** (`str | None`, default: `null`) - Path to true EOS for injection studies

The Pydantic schema is {class}`~jesterTOV.inference.config.schema.PostprocessingConfig`. For an example of how to read and inspect the output file, see {doc}`../examples/inference/result`. For the full API of all postprocessing functions, see {mod}`jesterTOV.inference.postprocessing`.

::::

---

## Complete examples

### Minimal configuration (prior-only)

A miniam config showing how to sample from the prior distribution without observational constraints.

::::{dropdown} **Minimal Configuration (Prior-Only)**

```yaml
seed: 43

eos:
  type: "metamodel"

tov:
  type: "gr"

prior:
  specification_file: "prior.prior"

likelihoods:
  - type: "zero"
    enabled: true

sampler:
  type: "smc-rw"
  n_particles: 5000
  output_dir: "./outdir/"
```

::::

### Advanced multi-messenger configuration

Several full configurations are available in the `examples` directory in jester:
* `examples/inference/anisotropy`: Using the metamodel+CSE equation of state and the anisotropy solver, with several of the likelihoods available.
* `examples/inference/blackjax-ns-aw`: Examples of the nested sampler implemented in `blackjax`
* `examples/inference/flowmc`: Examples of the flowMC sampler
* `examples/inference/mm_peakcse`: Examples of the metamodel+CSE analysis
* `examples/inference/smc_random_walk`: Examples of the SMC sampler with random walk kernel
* `examples/inference/spectral`: Examples of the spectral EOS parameterization
* `examples/inference/spectral_reparam`: Examples of the spectral EOS parameterization, after the reparametrization described in TODO: add the new docs page once it exists