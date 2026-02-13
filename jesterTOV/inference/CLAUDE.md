# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important developer guidelines

- You do not know everything about samplers. Instead of just doing something that "seems right", please ask more information about samplers and best practices. We can provide src code. *Better to ask for help than to make wrong assumptions and write sloppy code!*
- **blackjax**: For this, the src code is available at `/Users/Woute029/Documents/Code/projects/jester_review/blackjax`: use this to understand how to properly use blackjax samplers and best practices!

## Module Overview

The `jesterTOV/inference/` module provides Bayesian inference for constraining neutron star equation of state (EOS) parameters using multi-messenger observations. It implements a modular, configuration-driven architecture with normalizing flow-enhanced MCMC sampling.

### Key Concepts

**Transforms**: Convert parameter spaces
- Sample transforms: Applied during sampling with Jacobian (bijective)
- Likelihood transforms: Applied before likelihood evaluation (N-to-M)
- JESTER uses single unified `JesterTransform` class: NEP → M-R-Λ via EOS + TOV
  - EOS classes know their required parameters
  - TOV solvers know their required parameters
  - Transform coordinates the full pipeline and validates parameters

**Priors**: Bilby-style Python syntax in `.prior` files
```python
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
```

**Samplers**: Four backends available
- `type: "flowmc"` - Flow-enhanced MCMC (production ready)
  - Normalizing flow guidance for efficient sampling
  - Training + production phases
- `type: "smc-rw"` - Sequential Monte Carlo with Random Walk kernel (production ready, **DEFAULT**)
  - Gaussian Random Walk with sigma adaptation
  - Target ESS: 0.9, ~10-30 MCMC steps per tempering level
- `type: "smc-nuts"` - Sequential Monte Carlo with NUTS kernel (production ready)
  - NUTS kernel with Hessian-based mass matrix adaptation
  - More efficient for complex posteriors
- `type: "blackjax-ns-aw"` - Nested Sampling with Acceptance Walk (experimental)
  - For model comparison and evidence estimation
  - Needs additional testing/fixes

### Inference Documentation
- `docs/inference_index.md` - Navigation hub
- `docs/inference_quickstart.md` - Quick start guide
- `docs/inference.md` - Complete reference
- `docs/inference_yaml_reference.md` - Auto-generated YAML reference

Full details in `jesterTOV/inference/CLAUDE.md`

---

## Architecture

### Testing inference locally

The example in `examples/inference/smc_random_walk/chiEFT` finishes in less than 1 minute: good for testing some changes locally for inference/postprocessing.

### Modular Structure

```
jesterTOV/inference/
├── config/              # YAML parsing and Pydantic validation
│   ├── schema.py        # Configuration data models
│   ├── parser.py        # YAML loading
│   └── generate_yaml_reference.py  # Auto-generate docs
├── priors/              # Prior specification system
│   └── parser.py        # Parse .prior files (bilby-style Python format)
├── transforms/          # Unified transform system
│   ├── transform.py     # JesterTransform - single class for all EOS+TOV combinations
│   └── __init__.py      # Exports JesterTransform
├── likelihoods/         # Observational constraints
│   ├── gw.py            # Gravitational wave events (GW170817, GW190425)
│   ├── nicer.py         # X-ray timing (J0030, J0740, B0437)
│   ├── radio.py         # Radio pulsar timing (FIDUCEO/FIDUCEO2)
│   ├── chieft.py        # Chiral EFT low-density constraints
│   ├── rex.py           # PREX/CREX neutron skin experiments
│   ├── constraints.py   # Physical constraints (EOS/TOV/Gamma)
│   ├── combined.py      # CombinedLikelihood wrapper
│   └── factory.py       # Likelihood creation from config
├── data/                # Data loading and preprocessing
│   ├── __init__.py      # Data loading functions (NICER, GW posteriors, ChiEFT)
│   └── paths.py         # Path management and Zenodo caching
├── samplers/            # Sampler implementations
│   ├── jester_sampler.py  # Base JesterSampler + SAMPLER_REGISTRY
│   ├── flowmc.py        # FlowMC backend
│   └── blackjax/        # BlackJAX backends
│       ├── base.py      # BlackjaxSampler base class
│       ├── smc/         # Sequential Monte Carlo framework
│       │   ├── base.py  # BlackjaxSMCSampler
│       │   ├── random_walk.py  # SMC-RW (production ready)
│       │   └── nuts.py  # SMC-NUTS (production ready)
│       └── nested_sampling/
│           └── ns_aw.py # NS with Acceptance Walk (experimental)
├── base/                # Base classes (copied from Jim v0.2.0)
│   ├── likelihood.py    # LikelihoodBase ABC
│   ├── prior.py         # Prior, CombinePrior, UniformPrior
│   └── transform.py     # NtoMTransform, BijectiveTransform
├── run_inference.py     # Main entry point
└── cli.py               # Command-line interface
```

### Execution Flow

```
config.yaml + prior.prior
    ↓
parse_config() → InferenceConfig (Pydantic validated)
    ↓
parse_prior_file() → CombinePrior object
    ↓
JesterTransform.from_config(config.transform)
  ├─ Instantiate EOS (MetaModel/MetaModelCSE/Spectral)
  └─ Instantiate TOV solver (GR/Post/ScalarTensor)
    ↓
Validate parameters
  ├─ Check all required EOS params in prior → raise error if missing
  └─ Check all required TOV params in prior → warn if unused
    ↓
Load data (NICER, GW posteriors, ChiEFT, etc.)
  ├─ Cache downloads from Zenodo
  └─ Construct KDEs for GW posteriors
    ↓
create_likelihood() → CombinedLikelihood
  ├─ Individual likelihoods from factory
  └─ Equal weighting (1/N_likelihoods per likelihood)
    ↓
create_sampler() → Sampler from SAMPLER_REGISTRY
  ├─ FlowMCSampler (flowmc)
  ├─ BlackJAXSMCRandomWalkSampler (smc-rw)
  ├─ BlackJAXSMCNUTSSampler (smc-nuts)
  └─ BlackJAXNSAWSampler (blackjax-ns-aw)
    ↓
sampler.sample(prng_key) → SamplerOutput
  ├─ samples: dict[str, Array]
  ├─ log_prob: Array
  └─ metadata: dict[str, Any] (ESS, weights, acceptance rates, etc.)
    ↓
InferenceResult.from_sampler() → HDF5 format
  ├─ posterior (parameters + derived EOS quantities)
  ├─ metadata (config + run statistics)
  └─ histories (diagnostics: log_prob, ESS, etc.)
    ↓
Save to outdir/{result_id}.h5
```

### EOS/TOV Architecture

**Key Design Principle**: Modular separation of concerns

1. **EOS Classes** (`jesterTOV/eos/`):
   - Base: `Interpolate_EOS_model` (abstract base class)
   - Available models:
     - `MetaModel_EOS_model` - Nuclear empirical parameters (9 NEPs)
       - Reference: Margueron et al. (PRD 103, 045803, 2021)
       - Required: E_sat, K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym
       - Crust options: BPS, DH, DH_fixed, SLy
     - `MetaModel_with_CSE_EOS_model` - MetaModel + crust-core-saturation extension
       - Required: 9 NEPs + nbreak + nb_CSE grid parameters (typically 4-8)
     - `SpectralDecomposition_EOS_model` - Spectral representation
       - Reference: Lindblom 2010 (PRD 82, 103011)
       - Required: gamma_0, gamma_1, gamma_2, gamma_3
       - Uses 10-point Gauss-Legendre quadrature
   - Each implements:
     - `construct_eos(params: dict) -> EOSData` - Build EOS from parameters
     - `get_required_parameters() -> list[str]` - List parameter names

2. **TOV Solvers** (`jesterTOV/tov/`):
   - Base: `TOVSolverBase` (abstract base class)
   - Available solvers:
     - `GRTOVSolver` - General Relativity
       - Standard TOV equations, no additional parameters
       - Uses Dopri5 (Dormand-Prince 5th order)
     - `PostTOVSolver` - Beyond-GR modifications
       - Phenomenological sigma terms (Yagi & Yunes 2013)
       - Models: Bowers-Liang, Doneva-Yazadjiev, Herrera-Barreto, Post-Newtonian
       - Required: coupling constants (lambda_BL, lambda_DY, etc.)
     - `ScalarTensorTOVSolver` - Scalar-tensor gravity
       - Jordan frame (Brown 2023, ApJ 958 125)
       - Required: beta_ST, phi_c, nu_c
   - Each implements:
     - `solve(eos_data, pc, **kwargs) -> TOVSolution` - Single star
     - `construct_family(eos_data, ndat, min_nsat, **kwargs) -> FamilyData` - M-R-Λ family
     - `get_required_parameters() -> list[str]` - List additional parameters
   - Key features:
     - Uses Diffrax ODE solver with adaptive step size
     - Computes Love number k2 for tidal deformability
     - Parallelized via `jax.vmap()` over central pressures

3. **JAX Dataclasses** (`jesterTOV/tov/data_classes.py`):
   - All use NamedTuple for automatic JAX pytree compatibility
   - `EOSData` - EOS quantities (8 fields)
     - ns, ps, hs, es, dloge_dlogps, cs2, mu (optional), extra_constraints (optional)
   - `TOVSolution` - Single star solution
     - M (mass), R (radius), k2 (Love number)
   - `FamilyData` - M-R-Λ family curves
     - log10pcs, masses (M☉), radii (km), lambdas (dimensionless)

4. **JesterTransform** (`jesterTOV/inference/transforms/transform.py`):
   - Single unified class for all EOS+TOV combinations
   - Created via `JesterTransform.from_config(config)`
   - Coordinates: params → EOS.construct_eos() → TOV.construct_family() → observables
   - Validates: all required params are in prior (raises error if missing)
   - Logs warning: if prior contains unused parameters

**JAX Compatibility Requirements**:
- No Python `if` statements on traced values (use `jnp.where()`)
- No `float()` casts on traced arrays
- Dataclasses must be JAX pytrees (use NamedTuple, not @dataclass)

### Parameter Validation

**Automatic validation at transform setup** (in `run_inference.py`):

After creating `JesterTransform`, the code validates that all required parameters are present in the prior:

```python
transform = JesterTransform.from_config(config.transform, ...)
required_params = set(transform.get_parameter_names())
prior_params = set(prior.parameter_names)

# Check for missing parameters
missing_params = required_params - prior_params
if missing_params:
    raise ValueError(
        f"Transform with EOS = {eos_name} and TOV = {tov_name} is missing "
        f"params = {sorted(missing_params)} from the prior file"
    )

# Warn about unused parameters
unused_params = prior_params - required_params
if unused_params:
    logger.warning(f"Prior contains unused parameters: {sorted(unused_params)}")
```

**Benefits**:
- Catch configuration errors before sampling starts (fail-fast)
- Clear error messages identifying which parameters are missing
- Helpful for debugging when switching between EOS types

**Tests**: See `tests/test_inference/test_transform_validation.py` for unit tests

### Sampler Architecture

**Base Class: JesterSampler** (`samplers/jester_sampler.py`)
- Handles parameter transforms (sample + likelihood)
- Manages posterior evaluation with Jacobian corrections
- Provides standardized `SamplerOutput` interface

**Sampler Registry:**
```python
SAMPLER_REGISTRY = {
    "flowmc": FlowMCSampler,
    "smc-rw": BlackJAXSMCRandomWalkSampler,
    "smc-nuts": BlackJAXSMCNUTSSampler,
    "blackjax-ns-aw": BlackJAXNSAWSampler,
}
```

**BlackJAX Sampler Hierarchy:**
```
JesterSampler (base)
    ├─ FlowMCSampler (flowmc.py)
    └─ BlackjaxSampler (blackjax/base.py) - Shared transform logic
        ├─ BlackjaxSMCSampler (blackjax/smc/base.py) - SMC framework
        │   ├─ BlackJAXSMCRandomWalkSampler (blackjax/smc/random_walk.py)
        │   └─ BlackJAXSMCNUTSSampler (blackjax/smc/nuts.py)
        └─ BlackJAXNSAWSampler (blackjax/nested_sampling/ns_aw.py)
```

**SamplerOutput Structure:**
```python
class SamplerOutput:
    samples: dict[str, Array]        # Parameter samples (N_samples × N_params)
    log_prob: Array                  # Log probability (posterior for MCMC, likelihood for NS)
    metadata: dict[str, Any]         # Sampler-specific data
```

**Metadata Contents** (sampler-specific):
- **SMC samplers**: ESS (effective sample size), acceptance rates, weights, tempering schedule
- **FlowMC**: flow training history, MCMC acceptance rates
- **Nested sampling**: evidence (log Z), evidence error, iteration counts

**Key Design Features:**
- Automatic transform application (prior → sampling space)
- Jacobian correction for bijective transforms
- JAX-compatible (JIT compilation, vmap, grad)
- Deterministic sampling via `jax.random.PRNGKey`

## Configuration System

### YAML Configuration

Configuration files use YAML with Pydantic validation. See `examples/inference/*/config.yaml` for examples.

**Key sections:**
- `seed`: Random seed for reproducibility (JAX PRNGKey)
- `transform`: EOS transform configuration
  - `type`: EOS model (metamodel, metamodel_cse, spectral)
  - `nb_CSE`: Number of CSE parameters (only for metamodel_cse)
  - `tov_solver`: TOV solver type (gr, post, scalar_tensor)
  - Grid parameters: ndat, min_nsat, etc.
- `prior`: Path to `.prior` specification file (bilby-style Python)
- `likelihoods`: List of observational constraints (discriminated union)
  - Available types: gw, gw_resampled, nicer, radio, chieft, rex, eos_constraints, tov_constraints, gamma_constraints, zero
  - Each likelihood has `enabled` flag and type-specific parameters
- `sampler`: Sampler configuration (discriminated union by type)
  - FlowMC: n_chains, n_loop_training, n_loop_production, learning_rate, etc.
  - SMC-RW: n_particles, n_mcmc_steps, target_ess, etc.
  - SMC-NUTS: n_particles, n_mcmc_steps, target_ess, etc.
  - NS-AW: n_live_points, max_samples, etc.
- `data_paths`: Override default data file locations (optional)
- `outdir`: Output directory for results (default: "outdir")

**Likelihood Types** (all in `config/schema.py`):
1. `GWLikelihoodConfig` - Gravitational wave events (pre-sampled)
   - events: list of event names (e.g., ["GW170817"])
2. `GWResampledLikelihoodConfig` - GW with resampling during MCMC
3. `NICERLikelihoodConfig` - X-ray timing
   - sources: list of sources (e.g., ["J0030", "J0740"])
4. `RadioLikelihoodConfig` - Radio pulsar timing
   - database: "FIDUCEO" or "FIDUCEO2"
5. `ChiEFTLikelihoodConfig` - Chiral EFT constraints
   - nb_n: number of density points
6. `REXLikelihoodConfig` - PREX/CREX neutron skin
7. `EOSConstraintsLikelihoodConfig` - EOS physical validity (causality, stability)
8. `TOVConstraintsLikelihoodConfig` - TOV solver success
9. `GammaConstraintsLikelihoodConfig` - Spectral gamma bounds
10. `ZeroLikelihoodConfig` - Prior-only sampling (no data)

**IMPORTANT**: When modifying `config/schema.py`, regenerate YAML documentation:
```bash
uv run python -m jesterTOV.inference.config.generate_yaml_reference
```

### Prior Specification

Priors are specified in `.prior` files using bilby-style Python syntax: (note: the following example is specific for `metamodel` or `metamodel_cse`)

```python
# Nuclear Empirical Parameters (required for MetaModel/MetaModelCSE)
E_sat = UniformPrior(-16.1, -15.9, parameter_names=["E_sat"])
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])

# CSE breaking density (for metamodel_cse transform only)
nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])
```

**Important notes**:
- All 9 NEP parameters must be present for MetaModel/MetaModelCSE EOS types
- E_sat is now a free parameter (no longer fixed to -16.0 by default)
- CSE grid parameters (p_0, ..., p_N) are added programmatically if `nb_CSE > 0`
- Parameter validation will raise an error if any required parameter is missing from prior

## Key Design Principles

### Transform System

Transforms convert between parameter spaces. Two types:

1. **Sample Transforms** (BijectiveTransform):
   - Applied during sampling with Jacobian corrections
   - Must be invertible (1-to-1 mapping)
   - Examples: LogitTransform for bounded parameters

2. **Likelihood Transforms** (NtoMTransform):
   - Applied before likelihood evaluation
   - Can be N-to-M mapping (e.g., NEP → M-R-Λ curves)
   - No Jacobian corrections
   - **JesterTransform is the single unified likelihood transform**:
     - Handles all EOS types (metamodel, metamodel_cse, spectral)
     - Handles all TOV solver types (gr, post, scalar_tensor)
     - Use `JesterTransform.from_config(config)` to instantiate

### Sampler Architecture

`JesterSampler` is a base class, with subclasses for different sampler algorithms implemented as subclasses of `JesterSampler`.

## Common Development Tasks

### Adding a New Likelihood

1. **Create likelihood class** in `likelihoods/` inheriting from `LikelihoodBase`
   ```python
   from jesterTOV.inference.base.likelihood import LikelihoodBase

   class MyNewLikelihood(LikelihoodBase):
       def evaluate(self, params: dict, data: Any) -> float:
           """Compute log probability for parameters."""
           # Your implementation here
           return log_prob
   ```

2. **Add Pydantic config model** to `config/schema.py`:
   ```python
   class MyNewLikelihoodConfig(BaseModel):
       type: Literal["my_new_likelihood"]
       enabled: bool = True
       # Add your likelihood-specific parameters here
       param1: float
       param2: int
   ```

3. **Update LikelihoodConfig discriminated union** in `config/schema.py`:
   ```python
   LikelihoodConfig = Annotated[
       Union[
           # ... existing configs ...
           MyNewLikelihoodConfig,
       ],
       Field(discriminator="type"),
   ]
   ```

4. **Add to factory** in `likelihoods/factory.py`:
   ```python
   elif config.type == "my_new_likelihood":
       from .my_new import MyNewLikelihood
       data = load_my_data()  # If needed
       likelihood = MyNewLikelihood(data)
   ```

5. **Regenerate YAML docs**:
   ```bash
   uv run python -m jesterTOV.inference.config.generate_yaml_reference
   ```

6. **Add tests** in `tests/test_inference/test_likelihoods.py`

### Adding a New EOS Model

**Steps**:
1. **Create EOS class** in `jesterTOV/eos/` inheriting from `Interpolate_EOS_model`
   ```python
   from jesterTOV.eos.base import Interpolate_EOS_model
   from jesterTOV.tov.data_classes import EOSData

   class MyNewEOS(Interpolate_EOS_model):
       def construct_eos(self, params: dict[str, float]) -> EOSData:
           """Build EOS from parameters."""
           # Your implementation here
           return EOSData(ns=..., ps=..., hs=..., es=..., ...)

       def get_required_parameters(self) -> list[str]:
           """Return list of required parameter names."""
           return ["param1", "param2", ...]
   ```

2. **Add to JesterTransform factory** in `transforms/transform.py`:
   ```python
   def _create_eos(config: TransformConfig) -> Interpolate_EOS_model:
       if config.type == "my_new_eos":
           from jesterTOV.eos.my_new import MyNewEOS
           return MyNewEOS(...)
   ```

3. **Update TransformConfig** in `config/schema.py`:
   ```python
   class TransformConfig(BaseModel):
       type: Literal["metamodel", "metamodel_cse", "spectral", "my_new_eos"]
       # Add any EOS-specific config fields
   ```

4. **Regenerate YAML docs** and **add tests**

**No need to create new transform classes** - `JesterTransform` handles all EOS × TOV combinations automatically!

### Adding a New TOV Solver

**Steps**:
1. **Create solver class** in `jesterTOV/tov/` inheriting from `TOVSolverBase`
   ```python
   from jesterTOV.tov.base import TOVSolverBase
   from jesterTOV.tov.data_classes import EOSData, TOVSolution, FamilyData

   class MyNewTOVSolver(TOVSolverBase):
       def solve(self, eos_data: EOSData, pc: float, **kwargs) -> TOVSolution:
           """Solve TOV for single central pressure."""
           # Your implementation here
           return TOVSolution(M=..., R=..., k2=...)

       def construct_family(self, eos_data: EOSData, ndat: int,
                           min_nsat: float, **kwargs) -> FamilyData:
           """Build M-R-Λ family curves."""
           # Usually uses jax.vmap over self.solve
           return FamilyData(log10pcs=..., masses=..., radii=..., lambdas=...)

       def get_required_parameters(self) -> list[str]:
           """Return list of additional parameters (e.g., coupling constants)."""
           return ["coupling1", "coupling2", ...]
   ```

2. **Add to JesterTransform factory** in `transforms/transform.py`:
   ```python
   def _create_tov_solver(config: TransformConfig) -> TOVSolverBase:
       if config.tov_solver == "my_new_solver":
           from jesterTOV.tov.my_new import MyNewTOVSolver
           return MyNewTOVSolver(...)
   ```

3. **Update TransformConfig** in `config/schema.py`:
   ```python
   class TransformConfig(BaseModel):
       tov_solver: Literal["gr", "post", "scalar_tensor", "my_new_solver"] = "gr"
       # Add any solver-specific config fields
   ```

4. **Regenerate YAML docs** and **add tests**

### Adding a New Sampler

1. **Create sampler class** in `samplers/` inheriting from `JesterSampler`
2. Implement `sample(prng_key, n_samples, ...) -> SamplerOutput`
3. Add to `SAMPLER_REGISTRY` in `samplers/jester_sampler.py`
4. Add Pydantic config to `config/schema.py`
5. Update `SamplerConfig` discriminated union
6. Regenerate YAML docs and add tests

### Testing Configuration Changes

```bash
# Validate configuration
uv run run_jester_inference config.yaml --validate-only

# Dry run (setup without sampling)
uv run run_jester_inference config.yaml --dry-run
```

## Important Notes

### JAX Configuration

The inference system enables 64-bit precision by default:
```python
jax.config.update("jax_enable_x64", True)
```

For debugging NaN issues, uncomment:
```python
jax.config.update("jax_debug_nans", True)
```

### Type Safety with JAX

**Common type ignore patterns** (required due to JAX tracing limitations):

```python
# vmap batches scalar NamedTuple fields → arrays
masses: Float[Array, "n"] = solutions.M  # type: ignore[assignment]

# Diffrax with throw=False guarantees ys populated
R = sol.ys[0][-1]  # type: ignore[index]

# MetaModel guarantees mu populated (but type system sees Optional)
mu: Float[Array, "n"] = eos_data.mu  # type: ignore[assignment]
# TODO: Consider restructuring Interpolate_EOS_model to make mu non-optional
```

**Anti-pattern:** NEVER use runtime assertions in JAX-traced code (fails during tracing). Use type ignore with explanatory comments instead.

## Result Storage System

**InferenceResult Class** (`result.py`):
- HDF5-based storage format for inference results
- Standardized interface for all sampler types
- Includes posterior samples, metadata, and diagnostics

**Storage Structure:**
```
outdir/{result_id}.h5
├─ posterior/               # Parameter samples + derived quantities
│  ├─ <param_name>         # Each parameter as separate dataset
│  └─ log_prob             # Log probability values
├─ metadata/               # Run configuration and statistics
│  ├─ config               # Original YAML config (string)
│  ├─ sampler_type         # Sampler used
│  ├─ n_samples            # Number of samples
│  └─ run_statistics       # ESS, acceptance rates, etc.
└─ histories/              # Diagnostics (optional)
   ├─ log_prob_history     # Evolution during sampling
   └─ ess_history          # ESS over iterations (for SMC)
```

**Key Methods:**
```python
# Create from sampler output
result = InferenceResult.from_sampler(
    sampler_output=output,
    config=config,
    prior=prior,
)

# Save to HDF5
result.save(outdir / "result.h5")

# Load from HDF5
result = InferenceResult.load(outdir / "result.h5")

# Access data
samples = result.posterior["K_sat"]
log_prob = result.posterior["log_prob"]
metadata = result.metadata
```

**Benefits:**
- Portable: HDF5 is language-agnostic, readable by Python, Julia, R, etc.
- Efficient: Compressed storage, fast I/O for large arrays
- Standardized: Consistent format across all sampler types
- Self-documenting: Includes full config and metadata

## File Naming Conventions

- Configuration: `config.yaml`
- Prior specification: `prior.prior` (Python syntax)
- Example configs: `examples/inference/<sampler_type>/<use_case>/config.yaml`
- Output results: `outdir/{result_id}.h5` (HDF5 format)
  - Legacy: `outdir/results_production.npz`, `outdir/eos_samples.npz` (deprecated)

## Parent Project Context

This module is part of jesterTOV (JESTER). See `../../CLAUDE.md` for:
- Development commands (`uv run pytest`, `uv run pre-commit`)
- Code quality standards (black, ruff, pyright)
- Testing philosophy
- Documentation generation
