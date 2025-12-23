# JESTER Inference System Architecture

**Component connections, data flow, and implementation details**

## Table of Contents

1. [Module Dependency Graph](#module-dependency-graph)
2. [Execution Flow](#execution-flow)
3. [Component Interfaces](#component-interfaces)
4. [Data Transformations](#data-transformations)
5. [Class Hierarchy](#class-hierarchy)
6. [Configuration Pipeline](#configuration-pipeline)

---

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUTS                             │
└─────────────────────────────────────────────────────────────────┘
    config.yaml          prior.prior          data files
        │                    │                     │
        └────────────────────┴─────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   run_inference │
                    │      .py        │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌──────────────┐
│ config/       │    │ priors/       │    │ data/        │
│   parser.py   │    │   parser.py   │    │   loader.py  │
│   schema.py   │    │   library.py  │    │   paths.py   │
└───────┬───────┘    └───────┬───────┘    └──────┬───────┘
        │                    │                    │
        ▼                    ▼                    ▼
  InferenceConfig      CombinePrior         DataLoader
        │                    │                    │
        ├────────────────────┼────────────────────┤
        │                    │                    │
        ▼                    │                    ▼
┌───────────────┐            │            ┌──────────────┐
│ transforms/   │            │            │ likelihoods/ │
│   factory.py  │            │            │   factory.py │
│   metamodel.py│            │            │   gw.py      │
│   ...cse.py   │            │            │   nicer.py   │
└───────┬───────┘            │            └──────┬───────┘
        │                    │                    │
        ▼                    │                    ▼
  JesterTransform            │           CombinedLikelihood
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ samplers/       │
                    │   flowmc.py     │
                    │   jester_sampler│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  JesterSampler  │
                    │   (flowMC)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   MCMC Sampling │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Results     │
                    │   .npz files    │
                    └─────────────────┘
```

### Module Dependencies

**No circular dependencies**:
- `base/` has no dependencies (foundation)
- `config/` depends only on Pydantic
- `priors/` depends on `base/`
- `transforms/` depends on `base/`, `jesterTOV.eos`
- `likelihoods/` depends on `base/`, `data/`
- `data/` has minimal dependencies (NumPy, JAX)
- `samplers/` depends on everything above
- `run_inference.py` orchestrates all modules

---

## Execution Flow

### Phase 1: Initialization

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Configuration                                       │
└─────────────────────────────────────────────────────────────┘

config_path (str)
    │
    ├─> config/parser.py::load_config()
    │       ├─> Read YAML file
    │       ├─> Resolve relative paths (prior file)
    │       └─> Validate with Pydantic
    │
    └─> InferenceConfig (validated object)
            ├─> transform: TransformConfig
            ├─> prior: PriorConfig
            ├─> likelihoods: list[LikelihoodConfig]
            ├─> sampler: SamplerConfig
            └─> data_paths: dict[str, str]

┌─────────────────────────────────────────────────────────────┐
│ 2. Parse Prior Specification                                │
└─────────────────────────────────────────────────────────────┘

prior.prior (Python file)
    │
    ├─> priors/parser.py::parse_prior_file()
    │       ├─> Execute Python code in namespace
    │       ├─> Extract Prior objects
    │       ├─> Filter based on config (nb_CSE, sample_gw_events)
    │       └─> Auto-generate CSE grid priors if needed
    │
    └─> CombinePrior (list of Prior objects)
            ├─> parameter_names: list[str]
            ├─> n_dim: int
            ├─> sample(rng_key, n) -> dict
            └─> log_prob(params) -> float

┌─────────────────────────────────────────────────────────────┐
│ 3. Create Transform                                         │
└─────────────────────────────────────────────────────────────┘

TransformConfig
    │
    ├─> transforms/factory.py::create_transform()
    │       ├─> Determine input parameter names
    │       ├─> Create name_mapping: (input_names, output_names)
    │       └─> Instantiate transform class
    │
    └─> JesterTransformBase (MetaModelTransform or MetaModelCSETransform)
            ├─> forward(params: dict) -> dict
            ├─> get_eos_type() -> str
            └─> get_parameter_names() -> list[str]

┌─────────────────────────────────────────────────────────────┐
│ 4. Load Data and Create Likelihoods                         │
└─────────────────────────────────────────────────────────────┘

data_paths: dict + list[LikelihoodConfig]
    │
    ├─> data/loader.py::DataLoader(data_paths)
    │       ├─> Store paths
    │       ├─> Initialize empty cache
    │       └─> Methods: load_nicer_kde(), load_chieft_bands(), etc.
    │
    ├─> likelihoods/factory.py::create_combined_likelihood()
    │       ├─> For each enabled likelihood:
    │       │       ├─> Load required data via DataLoader
    │       │       └─> Instantiate likelihood class
    │       └─> Combine into CombinedLikelihood if multiple
    │
    └─> CombinedLikelihood
            ├─> likelihoods: list[LikelihoodBase]
            └─> evaluate(params: dict, data: dict) -> float

┌─────────────────────────────────────────────────────────────┐
│ 5. Setup Sampler                                            │
└─────────────────────────────────────────────────────────────┘

SamplerConfig + Prior + Likelihood + Transform
    │
    ├─> samplers/flowmc.py::setup_flowmc_sampler()
    │       ├─> Create mass matrix (identity)
    │       ├─> Setup local sampler args (step_size)
    │       └─> Instantiate JesterSampler
    │
    └─> JesterSampler
            ├─> prior: CombinePrior
            ├─> likelihood: LikelihoodBase
            ├─> likelihood_transforms: list[NtoMTransform]
            ├─> sampler: flowMC.Sampler
            │       ├─> local_sampler: MALA or GaussianRandomWalk
            │       └─> nf_model: MaskedCouplingRQSpline
            └─> Methods:
                    ├─> sample(rng_key)
                    ├─> get_samples(training=bool) -> dict
                    └─> print_summary()
```

### Phase 2: Sampling

```
┌─────────────────────────────────────────────────────────────┐
│ 6. MCMC Sampling                                            │
└─────────────────────────────────────────────────────────────┘

JesterSampler.sample(rng_key)
    │
    ├─> Initialize chains from prior
    │       ├─> Sample until all chains have finite log_prob
    │       └─> Initial positions: (n_chains, n_dim)
    │
    ├─> TRAINING PHASE (n_loop_training loops)
    │   │
    │   └─> For each loop:
    │           ├─> Run n_local_steps of local MCMC
    │           │       └─> Evaluate posterior(params)
    │           │               ├─> prior.log_prob(params)
    │           │               ├─> transform.forward(params)
    │           │               └─> likelihood.evaluate(transformed, {})
    │           │
    │           ├─> Collect samples
    │           └─> Train NF for n_epochs on samples
    │
    ├─> PRODUCTION PHASE (n_loop_production loops)
    │   │
    │   └─> For each loop:
    │           ├─> Run n_local_steps of local MCMC
    │           └─> Run n_global_steps using NF proposals
    │
    └─> Store final samples in sampler state
            ├─> chains: (n_loop_production, n_chains, n_local_steps, n_dim)
            ├─> log_prob: (n_loop_production, n_chains, n_local_steps)
            └─> local_accs, global_accs, loss_vals

┌─────────────────────────────────────────────────────────────┐
│ 7. Results Extraction                                       │
└─────────────────────────────────────────────────────────────┘

JesterSampler.get_samples(training=False)
    │
    ├─> Get sampler state
    ├─> Flatten chains to 1D
    ├─> Apply thinning (output_thinning)
    ├─> Convert arrays to parameter dict
    │       ├─> Use parameter_names from prior
    │       └─> Returns: {"K_sat": array, "Q_sat": array, ...}
    │
    └─> Save to outdir/results_production.npz
            ├─> log_prob: (n_samples,)
            └─> K_sat, Q_sat, ...: (n_samples,)
```

### Phase 3: Postprocessing

```
┌─────────────────────────────────────────────────────────────┐
│ 8. Generate EOS Samples                                     │
└─────────────────────────────────────────────────────────────┘

Samples + Transform
    │
    ├─> Select random subset (n_eos_samples)
    │
    ├─> JIT compile transform.forward()
    │       └─> Warm-up with small batch
    │
    ├─> Vectorized TOV solve: vmap(transform.forward)(samples)
    │       ├─> Input: {"K_sat": (n,), "Q_sat": (n,), ...}
    │       └─> Output: {"masses_EOS": (n, nb_masses),
    │                     "radii_EOS": (n, nb_masses),
    │                     "Lambdas_EOS": (n, nb_masses),
    │                     "n": (n, ndat), "p": (n, ndat), ...}
    │
    └─> Save to outdir/eos_samples.npz
            ├─> log_prob: (n_eos_samples,)
            ├─> K_sat, Q_sat, ...: (n_eos_samples,)
            └─> masses_EOS, radii_EOS, ...: (n_eos_samples, nb_masses)
```

---

## Component Interfaces

### Configuration System

**Input**: YAML file
**Output**: `InferenceConfig` (Pydantic model)

```python
class InferenceConfig(BaseModel):
    seed: int
    transform: TransformConfig
    prior: PriorConfig
    likelihoods: list[LikelihoodConfig]
    sampler: SamplerConfig
    data_paths: dict[str, str]

# Usage
config = load_config("config.yaml")
assert isinstance(config, InferenceConfig)
```

### Prior System

**Input**: `.prior` file (Python code)
**Output**: `CombinePrior`

```python
class CombinePrior:
    def __init__(self, prior_list: list[Prior]):
        self.prior_list = prior_list
        self.parameter_names = [...]  # Flattened names
        self.n_dim = len(self.parameter_names)

    def sample(self, rng_key, n_samples: int) -> dict[str, Array]:
        """Sample from prior"""
        # Returns: {"param_name": jnp.array([...]), ...}

    def log_prob(self, params: dict[str, Array]) -> Array:
        """Evaluate log probability"""
        # Returns: scalar or array

# Usage
prior = parse_prior_file("prior.prior", nb_CSE=8, sample_gw_events=["GW170817"])
samples = prior.sample(rng_key, 1000)
log_p = prior.log_prob(samples)
```

### Transform System

**Input**: Parameter dict (microscopic)
**Output**: Observable dict (macroscopic)

```python
class JesterTransformBase(NtoMTransform):
    def __init__(self, name_mapping: tuple[list[str], list[str]], **kwargs):
        input_names, output_names = name_mapping
        # ...

    def forward(self, params: dict[str, Array]) -> dict[str, Array]:
        """Transform parameters to observables"""
        # Input: {"K_sat": value, "Q_sat": value, ...}
        # Output: {"masses_EOS": array, "radii_EOS": array, ...}

    @abstractmethod
    def get_eos_type(self) -> str:
        """Return EOS type identifier"""

# Usage
transform = create_transform(config.transform)
params = {"K_sat": 200.0, "Q_sat": 300.0, ...}
observables = transform.forward(params)
print(observables.keys())  # ["masses_EOS", "radii_EOS", "Lambdas_EOS", ...]
```

### Likelihood System

**Input**: Observable dict + data dict
**Output**: Log likelihood (scalar)

```python
class LikelihoodBase(ABC):
    @abstractmethod
    def evaluate(self, params: dict[str, Array], data: dict) -> Array:
        """Evaluate log likelihood"""
        # Input: {"masses_EOS": array, "radii_EOS": array, ...}, {}
        # Output: scalar log likelihood

class CombinedLikelihood(LikelihoodBase):
    def __init__(self, likelihoods: list[LikelihoodBase]):
        self.likelihoods = likelihoods

    def evaluate(self, params: dict, data: dict) -> Array:
        """Sum log likelihoods"""
        return sum(lk.evaluate(params, data) for lk in self.likelihoods)

# Usage
likelihood = create_combined_likelihood(config.likelihoods, data_loader)
observables = transform.forward(params)
log_like = likelihood.evaluate(observables, {})
```

### Data Loading

**Input**: Data paths
**Output**: Lazy-loaded, cached data objects

```python
class DataLoader:
    def __init__(self, data_paths: dict[str, str | Path] | None = None):
        self.data_paths = data_paths or self._get_default_paths()
        self._cache = {}

    def load_nicer_kde(self, psr_name: str, analysis_group: str) -> gaussian_kde:
        """Load and cache NICER KDE"""
        cache_key = f"nicer_{psr_name}_{analysis_group}"
        if cache_key not in self._cache:
            # Load, construct KDE, cache
            self._cache[cache_key] = kde
        return self._cache[cache_key]

# Usage
loader = DataLoader(data_paths={"nicer_j0030_amsterdam": "./data/J0030.txt"})
kde = loader.load_nicer_kde("J0030", "amsterdam")  # Loads and caches
kde2 = loader.load_nicer_kde("J0030", "amsterdam")  # Returns cached (instant)
assert kde is kde2
```

### Sampler Interface

**Input**: Prior, Likelihood, Transform, Config
**Output**: MCMC samples

```python
class JesterSampler:
    def __init__(self, likelihood, prior, likelihood_transforms=[], **kwargs):
        self.likelihood = likelihood
        self.prior = prior
        self.likelihood_transforms = likelihood_transforms
        # Setup flowMC backend
        self.sampler = flowMC.Sampler(...)

    def sample(self, rng_key: PRNGKeyArray):
        """Run MCMC sampling"""
        # Initialize, run training, run production

    def get_samples(self, training: bool = False) -> dict[str, Array]:
        """Get samples as parameter dict"""
        # Returns: {"K_sat": array, "Q_sat": array, ...}

    def print_summary(self):
        """Print sampling statistics"""

# Usage
sampler = setup_flowmc_sampler(config.sampler, prior, likelihood, transform)
sampler.sample(jax.random.PRNGKey(43))
samples = sampler.get_samples(training=False)
```

---

## Data Transformations

### Parameter → EOS → Observables

```
┌──────────────────────────────────────────────────────────────┐
│ Input: Microscopic Parameters (NEP)                         │
└──────────────────────────────────────────────────────────────┘

{"K_sat": 200.0,      # Incompressibility (MeV)
 "Q_sat": 300.0,      # Skewness (MeV)
 "Z_sat": -500.0,     # Kurtosis (MeV)
 "E_sym": 32.0,       # Symmetry energy (MeV)
 "L_sym": 60.0,       # Symmetry slope (MeV)
 "K_sym": -100.0,     # Symmetry incompressibility (MeV)
 "Q_sym": 200.0,      # Symmetry skewness (MeV)
 "Z_sym": 0.0}        # Symmetry kurtosis (MeV)

    │
    │ JesterTransformBase.forward()
    ├──> MetaModel construction
    │       ├─> Build energy density expansion
    │       ├─> Compute pressure, enthalpy
    │       └─> Attach crust model (DH/BPS)
    │
    ▼

┌──────────────────────────────────────────────────────────────┐
│ Intermediate: EOS Thermodynamics                            │
└──────────────────────────────────────────────────────────────┘

{"n": jnp.array([0.01, 0.02, ..., 4.0]),     # Density (fm^-3)
 "p": jnp.array([...]),                      # Pressure (geometric units)
 "e": jnp.array([...]),                      # Energy density (geom. units)
 "h": jnp.array([...]),                      # Enthalpy (geom. units)
 "cs2": jnp.array([...]),                    # Sound speed squared
 "dloge_dlogp": jnp.array([...])}            # Logarithmic derivative

    │
    │ construct_family() - TOV solver
    ├──> Integrate TOV equations for mass grid
    │       ├─> For each central pressure in grid
    │       ├─> Solve dm/dr, dP/dr from center to surface
    │       └─> Extract M, R, Λ at surface
    │
    ▼

┌──────────────────────────────────────────────────────────────┐
│ Output: Macroscopic Observables                             │
└──────────────────────────────────────────────────────────────┘

{"logpc_EOS": jnp.array([...]),              # Log central pressures
 "masses_EOS": jnp.array([1.2, 1.4, ..., 2.5]),   # Masses (geom. units)
 "radii_EOS": jnp.array([11.5, 12.0, ..., 13.2]), # Radii (geom. units)
 "Lambdas_EOS": jnp.array([...]),            # Tidal deformabilities
 "n": ..., "p": ..., ...}                    # (EOS quantities included)
```

**Units**:
- Geometric units: G = c = 1
- Convert to SI: `M_SI = M_geom * 4.926e-6` (solar masses)
- Convert to SI: `R_SI = R_geom * 1.477` (km)

### Posterior Evaluation Chain

```
┌─────────────────────────────────────────────────────────────┐
│ Sampler proposes parameters (array)                         │
└─────────────────────────────────────────────────────────────┘

params_array: jnp.array([200.0, 300.0, -500.0, ...])  # Shape: (n_dim,)

    │
    │ add_name() - Convert to dict
    ▼

params_dict: {"K_sat": 200.0, "Q_sat": 300.0, ...}

    │
    │ prior.log_prob()
    ├──> Evaluate prior probability
    ▼

log_prior: -15.3  # Scalar

    │
    │ For each transform in likelihood_transforms:
    │   params_dict = transform.forward(params_dict)
    ▼

transformed_params: {"masses_EOS": [...], "radii_EOS": [...], ...}

    │
    │ likelihood.evaluate(transformed_params, {})
    ├──> Evaluate data likelihood
    ▼

log_likelihood: -234.5  # Scalar

    │
    │ Sum log probabilities
    ▼

log_posterior = log_prior + log_likelihood: -249.8

    │
    └──> Return to flowMC for accept/reject decision
```

---

## Class Hierarchy

### Base Classes (Copied from Jim)

```
┌─────────────────────────────────────────────────────────────┐
│ base/                                                        │
└─────────────────────────────────────────────────────────────┘

Prior (ABC)
    ├─ sample(rng_key, n) -> dict
    └─ log_prob(params) -> float
        │
        ├─ UniformPrior
        └─ CombinePrior
                └─ Combines list[Prior]

Transform (ABC)
    ├─ forward(params) -> params
    └─ propagate_name(names) -> names
        │
        ├─ BijectiveTransform (1-to-1)
        │       └─ inverse(params) -> (params, jacobian)
        │
        └─ NtoMTransform (N-to-M)
                └─ forward(params) -> params

LikelihoodBase (ABC)
    └─ evaluate(params, data) -> float
        │
        ├─ GWlikelihood_with_masses
        ├─ NICERLikelihood
        ├─ RadioTimingLikelihood
        ├─ ChiEFTLikelihood
        ├─ REXLikelihood
        ├─ ZeroLikelihood
        └─ CombinedLikelihood
```

### JESTER-Specific Classes

```
┌─────────────────────────────────────────────────────────────┐
│ priors/                                                      │
└─────────────────────────────────────────────────────────────┘

SimpleUniformPrior(Prior)
    └─ Replacement for jimgw's UniformPrior

┌─────────────────────────────────────────────────────────────┐
│ transforms/                                                  │
└─────────────────────────────────────────────────────────────┘

JesterTransformBase(NtoMTransform)
    ├─ Common TOV integration logic
    └─ Abstract: get_eos_type(), get_parameter_names()
        │
        ├─ MetaModelTransform
        │       └─ NEP → M-R-Λ (8 params)
        │
        └─ MetaModelCSETransform
                └─ NEP + CSE → M-R-Λ (8 + 1 + 2*nb_CSE + 1 params)

┌─────────────────────────────────────────────────────────────┐
│ samplers/                                                    │
└─────────────────────────────────────────────────────────────┘

JesterSampler
    ├─ Wraps flowMC.Sampler
    ├─ Jim-like interface with bug fixes
    └─ Constructs posterior from prior + likelihood + transforms
```

### Configuration Classes (Pydantic)

```
┌─────────────────────────────────────────────────────────────┐
│ config/schema.py                                             │
└─────────────────────────────────────────────────────────────┘

InferenceConfig(BaseModel)
    ├─ seed: int
    ├─ transform: TransformConfig
    ├─ prior: PriorConfig
    ├─ likelihoods: list[LikelihoodConfig]
    ├─ sampler: SamplerConfig
    └─ data_paths: dict[str, str]

TransformConfig(BaseModel)
    ├─ type: Literal["metamodel", "metamodel_cse"]
    ├─ ndat_metamodel: int
    ├─ nb_CSE: int
    └─ ... (TOV integration params)

LikelihoodConfig(BaseModel)
    ├─ type: Literal["gw", "nicer", "radio", ...]
    ├─ enabled: bool
    └─ parameters: dict[str, Any]

SamplerConfig(BaseModel)
    ├─ n_chains: int
    ├─ n_loop_training: int
    ├─ n_epochs: int
    └─ ... (flowMC params)
```

---

## Configuration Pipeline

### YAML → Pydantic → Components

```
┌──────────────────────────────────────────────────────────────┐
│ config.yaml (User Input)                                     │
└──────────────────────────────────────────────────────────────┘

seed: 43
transform:
  type: "metamodel"
  ndat_metamodel: 100
prior:
  specification_file: "prior.prior"
likelihoods:
  - type: "nicer"
    enabled: true
    parameters:
      targets: ["J0030"]
sampler:
  n_chains: 20
  output_dir: "./outdir/"

    │
    │ yaml.safe_load()
    ▼

config_dict: {
    "seed": 43,
    "transform": {"type": "metamodel", "ndat_metamodel": 100},
    "prior": {"specification_file": "prior.prior"},
    ...
}

    │
    │ Path resolution
    ├──> Resolve relative paths to absolute
    │    "prior.prior" → "/abs/path/to/prior.prior"
    ▼

resolved_dict: {
    "prior": {"specification_file": "/abs/path/to/prior.prior"},
    ...
}

    │
    │ Pydantic validation
    ├──> InferenceConfig(**resolved_dict)
    ├──> Type checking, range validation, custom validators
    └──> Raises ValidationError if invalid
    ▼

config: InferenceConfig
    ├─ config.seed = 43
    ├─ config.transform.type = "metamodel"
    ├─ config.prior.specification_file = "/abs/path/to/prior.prior"
    └─ config.sampler.n_chains = 20

    │
    │ Component creation
    ├──> create_transform(config.transform)
    ├──> parse_prior_file(config.prior.specification_file)
    ├──> create_combined_likelihood(config.likelihoods, ...)
    └──> setup_flowmc_sampler(config.sampler, ...)
    ▼

Ready-to-use components
```

### Validation Flow

```
InferenceConfig validation
    │
    ├─ TransformConfig validation
    │   ├─ type in ["metamodel", "metamodel_cse"] ✓
    │   ├─ nb_CSE == 0 if type == "metamodel" ✓
    │   └─ ndat_metamodel > 0 ✓
    │
    ├─ PriorConfig validation
    │   └─ specification_file ends with ".prior" ✓
    │
    ├─ LikelihoodConfig validation (for each)
    │   ├─ type in ["gw", "nicer", ...] ✓
    │   └─ At least one enabled ✓
    │
    └─ SamplerConfig validation
        ├─ n_chains > 0 ✓
        ├─ learning_rate in (0, 1] ✓
        └─ All positive values ✓

If any check fails → Pydantic ValidationError with detailed message
```

---

## Summary

### Key Architectural Principles

1. **Separation of Concerns**
   - Configuration (YAML) separate from implementation (Python)
   - Data loading separate from likelihood evaluation
   - Transform logic separate from sampling

2. **Lazy Evaluation**
   - Data loaded only when needed
   - Cached after first load
   - JIT compilation for TOV solver

3. **Type Safety**
   - Pydantic models validate all inputs
   - JAX typing for arrays
   - Abstract base classes enforce interfaces

4. **Modularity**
   - Easy to add new likelihoods (factory pattern)
   - Easy to add new priors (parser namespace)
   - Easy to add new transforms (inheritance from base)

5. **Reproducibility**
   - All settings in version-controlled YAML
   - Explicit seed management
   - Deterministic JAX operations

### Design Patterns Used

- **Factory Pattern**: Transform and likelihood creation
- **Strategy Pattern**: Different likelihood types with same interface
- **Lazy Initialization**: Data loading on demand
- **Dependency Injection**: Components passed to samplers
- **Template Method**: JesterTransformBase defines workflow, subclasses implement details

---

**Architecture Version**: 1.0
**Last Updated**: December 2024
