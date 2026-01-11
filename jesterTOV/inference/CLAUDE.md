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
- JESTER uses likelihood transforms: NEP → M-R-Λ via TOV solver

**Priors**: Bilby-style Python syntax in `.prior` files
```python
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
```

**Samplers**: Three backends available
- `type: "flowmc"` - Flow-enhanced MCMC (production ready)
- `type: "smc"` - Sequential Monte Carlo (production ready)
  - `kernel_type: "nuts"` or `"random_walk"`
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
Create transform (factory) → JesterTransformBase
  ↓
Load data (NICER/GW posteriors, construct KDEs)
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

# CSE breaking density (for metamodel+CSE transform)
nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])
```

The parser automatically:
- Includes NEP parameters (`_sat` and `_sym` suffixes)
- Adds CSE grid parameters programmatically if `nb_CSE > 0` for metamodel+CSE EOS parametrization

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

### Sampler Architecture

`JesterSampler` is a base class, with subclasses for different sampler algorithms implemented as subclasses of `JesterSampler`.

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
