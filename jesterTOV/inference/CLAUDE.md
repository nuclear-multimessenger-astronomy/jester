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

**Samplers**: Three backends available
- `type: "flowmc"` - Flow-enhanced MCMC (production ready)
- `type: "smc-rw"` - Sequential Monte Carlo with Random Walk kernel (production ready)
- `type: "smc-nuts"` - Sequential Monte Carlo with NUTS kernel (production ready)
- `type: "blackjax-ns-aw"` - Nested sampling (needs type fixes)

### Inference Documentation
- `docs/inference_index.md` - Navigation hub
- `docs/inference_quickstart.md` - Quick start guide
- `docs/inference.md` - Complete reference
- `docs/inference_yaml_reference.md` - Auto-generated YAML reference

Full details in `jesterTOV/inference/CLAUDE.md`

---

## Running Inference

**ALWAYS use `uv` for Python commands:**

Run inference with config file with venv in root jester directory activated:
```bash
run_jester_inference config.yaml
```

## Architecture

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
│   ├── gw.py            # Gravitational wave events
│   ├── nicer.py         # X-ray timing observations
│   ├── radio.py         # Radio pulsar timing
│   ├── chieft.py        # Chiral EFT constraints
│   ├── rex.py           # PREX/CREX experiments
│   ├── combined.py      # Combined likelihood
│   └── factory.py       # Likelihood creation
├── data/                # Data loading and preprocessing
│   ├── __init__.py      # Data loading functions (NICER, GW posteriors)
│   └── paths.py         # Path management
├── samplers/            # Sampler implementations
│   ├── jester_sampler.py  # Base sampler class
│   └── flowmc.py        # flowMC backend setup
├── base/                # Base classes (copied from Jim v0.2.0)
│   ├── likelihood.py    # LikelihoodBase ABC
│   ├── prior.py         # Prior, CombinePrior, UniformPrior
│   └── transform.py     # NtoMTransform, BijectiveTransform
├── run_inference.py     # Main entry point
└── cli.py               # Command-line interface
```

### Execution Flow

```
config.yaml → Pydantic validation
  ↓
Parse .prior file → CombinePrior object
  ↓
Create transform → JesterTransform.from_config()
  ├─ Instantiate EOS (MetaModel/MetaModelCSE/Spectral)
  └─ Instantiate TOV solver (GR/Post/ScalarTensor)
  ↓
Validate parameters → Check prior contains all required EOS+TOV params
  ↓
Load data (NICER/GW posteriors, construct KDEs)
  ↓
Create likelihoods (factory) → CombinedLikelihood
  ↓
Setup sampler → JesterSampler wrapper (FlowMC/SMC/NS-AW)
  ↓
MCMC/SMC/NS sampling (training + production)
  ↓
Save results → outdir/results_production.npz
  ↓
Generate EOS samples → outdir/eos_samples.npz
```

### EOS/TOV Architecture

**Key Design Principle**: Modular separation of concerns

1. **EOS Classes** (`jesterTOV/eos/`):
   - `MetaModel_EOS_model` - Nuclear empirical parameters (9 NEPs)
   - `MetaModel_with_CSE_EOS_model` - MetaModel + crust-core transition
   - `SpectralDecomposition_EOS_model` - Spectral representation
   - Each implements:
     - `construct_eos(params) -> EOSData` - Build EOS from parameters
     - `get_required_parameters() -> list[str]` - List parameter names

2. **TOV Solvers** (`jesterTOV/tov/`):
   - `GRTOVSolver` - General relativity
   - `PostTOVSolver` - Post-Newtonian corrections
   - `ScalarTensorTOVSolver` - Scalar-tensor gravity
   - Each implements:
     - `construct_family(eos_data, ...) -> FamilyData` - Solve TOV for M-R-Λ family
     - `get_required_parameters() -> list[str]` - List additional parameters (e.g., coupling constants)

3. **JAX Dataclasses** (`jesterTOV/tov/data_classes.py`):
   - `EOSData` - EOS quantities (ns, ps, hs, es, cs2, etc.) - NamedTuple for JAX pytrees
   - `TOVSolution` - Single star solution (M, R, k2)
   - `FamilyData` - M-R-Λ family curves (masses, radii, lambdas)

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

## Configuration System

### YAML Configuration

Configuration files use YAML with Pydantic validation. See `examples/inference/*/config.yaml` for examples.

**Key sections:**
- `seed`: Random seed for reproducibility
- `transform`: EOS transform configuration (MetaModel or MetaModel+CSE)
- `prior`: Path to `.prior` specification file
- `likelihoods`: List of observational constraints to include
- `sampler`: MCMC sampler parameters (chains, loops, learning rate)
- `data_paths`: Override default data file locations

**IMPORTANT**: When modifying `config/schema.py`, regenerate YAML documentation:
```bash
uv run python -m jesterTOV.inference.config.generate_yaml_reference
```

### Prior Specification

Priors are specified in `.prior` files using bilby-style Python syntax:

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

1. Create new file in `likelihoods/` inheriting from `LikelihoodBase`
2. Implement `evaluate(params, data)` method
3. Add likelihood type to `likelihoods/factory.py`
4. Add type to `config/schema.py` LikelihoodConfig
5. Regenerate YAML docs: `uv run python -m jesterTOV.inference.config.generate_yaml_reference`

### Adding a New EOS or TOV Solver

**To add a new EOS**:
1. Create new class in `jesterTOV/eos/` inheriting from `Interpolate_EOS_model`
2. Implement `construct_eos(params) -> EOSData` method
3. Implement `get_required_parameters() -> list[str]` method
4. Add EOS type to `JesterTransform._create_eos()` in `transforms/transform.py`
5. Add type to `config/schema.py` TransformConfig
6. Regenerate YAML docs: `uv run python -m jesterTOV.inference.config.generate_yaml_reference`

**To add a new TOV solver**:
1. Create new class in `jesterTOV/tov/` inheriting from `TOVSolverBase`
2. Implement `construct_family(eos_data, ...) -> FamilyData` method
3. Implement `get_required_parameters() -> list[str]` method
4. Add solver type to `JesterTransform._create_tov_solver()` in `transforms/transform.py`
5. Add type to `config/schema.py` TransformConfig (tov_solver field)
6. Regenerate YAML docs

**No need to create new transform classes** - `JesterTransform` handles all combinations automatically!

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

## File Naming Conventions

- Configuration: `config.yaml`
- Prior specification: `prior.prior` (Python syntax)
- Example configs: `examples/inference/<use_case>/config.yaml`
- Output results: `outdir/results_production.npz`, `outdir/eos_samples.npz`

## Parent Project Context

This module is part of jesterTOV (JESTER). See `../../CLAUDE.md` for:
- Development commands (`uv run pytest`, `uv run pre-commit`)
- Code quality standards (black, ruff, pyright)
- Testing philosophy
- Documentation generation
