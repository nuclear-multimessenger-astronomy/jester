# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

The `jesterTOV/inference/` module provides Bayesian inference for constraining neutron star equation of state (EOS) parameters using multi-messenger observations. It implements a modular, configuration-driven architecture with normalizing flow-enhanced MCMC sampling.

**Status**: Active development - Modular architecture implemented (Phases 1-6 complete)

## Running Inference

**ALWAYS use `uv` for Python commands:**

```bash
# Run inference with config file
uv run run_jester_inference config.yaml

# Alternative module-based command
uv run python -m jesterTOV.inference.run_inference --config config.yaml

# Override output directory
uv run run_jester_inference config.yaml --output-dir ./results/

# Validate config without running
uv run run_jester_inference config.yaml --validate-only

# Dry run (setup without sampling)
uv run run_jester_inference config.yaml --dry-run

# Use example configs
uv run run_jester_inference examples/inference/full_inference/config.yaml
uv run run_jester_inference examples/inference/gw170817_only/config.yaml
uv run run_jester_inference examples/inference/nicer_only/config.yaml
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
├── transforms/          # EOS parameter transforms
│   ├── base.py          # JesterTransformBase ABC
│   ├── metamodel.py     # MetaModel transform (NEP → M-R-Λ)
│   └── metamodel_cse.py # MetaModel + CSE extension
├── likelihoods/         # Observational constraints
│   ├── gw.py            # Gravitational wave events
│   ├── nicer.py         # X-ray timing observations
│   ├── radio.py         # Radio pulsar timing
│   ├── chieft.py        # Chiral EFT constraints
│   ├── rex.py           # PREX/CREX experiments
│   ├── combined.py      # Combined likelihood
│   └── factory.py       # Likelihood creation
├── data/                # Data loading (FIXME: needs implementation)
│   ├── __init__.py      # FIXME: Implement data loading functions
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
Create transform (factory) → JesterTransformBase
  ↓
Load data → FIXME: Need to implement data loading functions
  ↓
Create likelihoods (factory) → CombinedLikelihood
  ↓
Setup flowMC sampler → JesterSampler wrapper
  ↓
MCMC sampling (training + production)
  ↓
Save results → outdir/results_production.npz
  ↓
Generate EOS samples → outdir/eos_samples.npz
```

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
# Nuclear Empirical Parameters
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])

# GW event masses (conditionally included)
mass_1_GW170817 = UniformPrior(1.5, 2.1, parameter_names=["mass_1_GW170817"])

# CSE breaking density (only if nb_CSE > 0)
nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])
```

The parser automatically:
- Includes NEP parameters (`_sat` and `_sym` suffixes)
- Includes `nbreak` only if `nb_CSE > 0` in config
- Adds CSE grid parameters programmatically if `nb_CSE > 0`

## Key Design Principles

### Transform System

Transforms convert between parameter spaces. Two types:

1. **Sample Transforms** (BijectiveTransform):
   - Applied during sampling with Jacobian corrections
   - Must be invertible (1-to-1 mapping)

2. **Likelihood Transforms** (NtoMTransform):
   - Applied before likelihood evaluation
   - Can be N-to-M mapping (e.g., NEP → M-R-Λ curves)
   - No Jacobian corrections

**JESTER uses likelihood transforms**: NEP parameters → Mass-Radius-Tidal deformability curves via TOV solver.

### Base Classes (jimgw Independence)

The `base/` module contains copies of Jim v0.2.0 base classes to remove dependency on jimgw. This gives JESTER full control over interfaces and bug fixes.

**Current dependencies from base/**:
- `LikelihoodBase` - Abstract base class for likelihoods
- `Prior`, `CombinePrior`, `UniformPrior` - Prior system
- `NtoMTransform`, `BijectiveTransform` - Transform interfaces

**Goal**: Keep flowMC as the only external sampler dependency.

### Sampler Architecture

`JesterSampler` is a lightweight wrapper around flowMC that handles:
- Posterior evaluation with transform Jacobians
- Parameter name propagation through transforms
- Converting between named dicts and arrays
- Generic sampling interface

**Critical bug fixes in JesterSampler**:
- Uses `jnp.inf` instead of `jnp.nan` for initialization (avoids NaN propagation)
- Preserves parameter ordering when converting dict → array

## Common Development Tasks

### Adding a New Likelihood

1. Create new file in `likelihoods/` inheriting from `LikelihoodBase`
2. Implement `evaluate(params, data)` method
3. Add likelihood type to `likelihoods/factory.py`
4. Add type to `config/schema.py` LikelihoodConfig
5. Regenerate YAML docs: `uv run python -m jesterTOV.inference.config.generate_yaml_reference`

### Adding a New Transform

1. Create new file in `transforms/` inheriting from `JesterTransformBase`
2. Implement `forward(params)` method
3. Add transform type to `transforms/factory.py`
4. Add type to `config/schema.py` TransformConfig
5. Regenerate YAML docs

### Testing Configuration Changes

```bash
# Validate configuration
uv run run_jester_inference config.yaml --validate-only

# Dry run (setup without sampling)
uv run run_jester_inference config.yaml --dry-run

# Quick test with minimal sampling
uv run run_jester_inference config.yaml  # Edit config: n_chains=2, n_loop_training=1
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

### flowMC Data Argument

Likelihood signature is `likelihood.evaluate(params, data)`. In JESTER, observational data is encapsulated within likelihood objects, so **always pass an empty dict `{}`** as the data argument, not `None`.

### Output Files

Inference produces two output files in `outdir/`:
- `results_production.npz` - All sampled parameters and log probabilities
- `eos_samples.npz` - Subset with full EOS curves (Mass, Radius, Lambda)

### Known Issues

**Prior-only sampling with ZeroLikelihood** (December 2024) - UNRESOLVED:
- Symptoms: NaN log probabilities, 0% acceptance rates
- Root cause: Unknown (under investigation)
- Workaround: Use actual likelihoods (GW, NICER, etc.) for now

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
