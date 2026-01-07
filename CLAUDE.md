# CLAUDE.md

This file provides guidance to Claude Code when working with the JESTER repository.

## IMPORTANT GUIDELINES

**Testing Philosophy**: When tests fail, investigate root causes rather than modifying tests to pass. Make notes in CLAUDE.md and fix underlying code issues.

**Documentation Style**: Write clear, concise documentation in full sentences as if by a human researcher. Avoid LLM-like verbosity.

**File Operations**: Use proper tools (Write, Edit, Read) instead of bash heredocs or cat redirection.

---

## Current Status

### Multi-Sampler Architecture

Three sampler backends are now available for Bayesian inference:

1. **FlowMC** (Production Ready) - Normalizing flow-enhanced MCMC
2. **BlackJAX SMC** (Production Ready) - Sequential Monte Carlo with adaptive tempering
   - Gaussian Random Walk kernel with sigma adaptation -- TESTED, THIS IS OUR "DEFAULT" SAMPLER
   - NUTS kernel with Hessian-based mass matrix adaptation -- EXPERIMENTAL, REFRAIN FROM USING NOW
3. **BlackJAX NS-AW** (Needs Testing) - Nested Sampling with Acceptance Walk, mimics bilby setup

**Example Configs Available**:
SMC is production ready and we usually test the following config, which can be executed locally on a laptop without GPU support
```bash
examples/inference/smc_random_walk/chiEFT/config.yaml
```

---

## Project Overview

**JESTER** (**J**ax-based **E**o**S** and **T**ov solv**ER**) is a scientific computing library for neutron star physics using JAX for hardware acceleration and automatic differentiation.

### Core Modules
- `jesterTOV/eos/` - Equation of state models
- `jesterTOV/tov.py` - TOV equation solver
- `jesterTOV/inference/` - Bayesian inference system
- `jesterTOV/utils.py` - Physical constants and unit conversions

### Key Design Principles
- **JAX-first**: Hardware acceleration with automatic differentiation
- **Geometric units**: All physics calculations use geometric units
- **Type safety**: Comprehensive type hints with `jaxtyping` for arrays
- **64-bit precision**: Enabled by default for numerical accuracy

---

## Development Commands

### Always Use `uv`
```bash
# Run Python commands
uv run <command>

# Install dependencies
uv pip install <package>

# Run tests
uv run pytest tests/

# Pre-commit checks
uv run pre-commit run --all-files
```

### Code Quality
```bash
# Format and lint
uv run black .
uv run ruff check --fix .

# Type checking
uv run pyright                 # All files
uv run pyright jesterTOV/      # Specific directory
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_inference/test_config.py

# Run with verbose output
uv run pytest -v tests/
```

## Code Quality Standards

### Documentation
```bash
# Build docs locally
uv pip install -e ".[docs]"
uv run sphinx-build docs docs/_build/html
open docs/_build/html/index.html

# Strict mode (same as CI)
uv run sphinx-build -W --keep-going docs docs/_build/html
```

**Docs URL**: https://nuclear-multimessenger-astronomy.github.io/jester/

---

## Inference System

**Status**: Fully functional modular architecture (config-driven, replaces old argparse interface)

### Running Inference
```bash
# Run inference
run_jester_inference config.yaml
# Validate config only
# (set validate_only: true in config.yaml)

# Dry run without sampling
# (set dry_run: true in config.yaml)
```

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

**YAML Config Auto-Generation**:
```bash
# When modifying config/schema.py, regenerate docs
uv run python -m jesterTOV.inference.config.generate_yaml_reference
```

### Inference Documentation
- `docs/inference_index.md` - Navigation hub
- `docs/inference_quickstart.md` - Quick start guide
- `docs/inference.md` - Complete reference
- `docs/inference_yaml_reference.md` - Auto-generated YAML reference

Full details in `jesterTOV/inference/CLAUDE.md`

---

## Type Hinting Standards

**All new code MUST include comprehensive type hints.**

```python
# Standard library types (Python 3.10+ syntax)
def process_data(values: list[float], threshold: float | None = None) -> dict[str, float]:
    ...

# JAX arrays with jaxtyping
from jaxtyping import Array, Float
def solve_tov(pressure: Float[Array, "n_points"]) -> Float[Array, "n_points"]:
    ...

# Pydantic for configs
from pydantic import BaseModel, Field
class SamplerConfig(BaseModel):
    n_chains: int = Field(gt=0, description="Number of MCMC chains")

# Type aliases for complex types
from typing import TypeAlias
ParameterDict: TypeAlias = dict[str, float]
```

**Type checking**: `uv run pyright jesterTOV/inference/`

---

## Architecture Notes

### Inference Module Structure
```
jesterTOV/inference/
├── config/          # YAML parsing, Pydantic validation
├── priors/          # Prior specification system
├── transforms/      # EOS parameter transforms
├── likelihoods/     # Observational constraints
├── data/            # Data loading and caching
├── samplers/        # FlowMC, SMC, NS-AW implementations
├── base/            # Base classes (copied from Jim v0.2.0)
└── run_inference.py # Main entry point
```

---

## Known Issues & Workarounds

### Open Issues
- **UniformPrior boundaries**: `log_prob()` at exact boundaries causes errors (NaN at xmin, ZeroDivision at xmax)
  - Workaround: Use values strictly inside boundaries
  - Fix: Add numerical guards in LogitTransform
- **TOV solver max_steps**: Some stiff EOS configs hit solver limits
  - May need to increase `max_steps` or adjust EOS parameters
  - Needs further testing to understand when this happens, what the root cause is, and determine best solution

---

## Release Workflow

```bash
# 1. Feature branch for version bump
git checkout -b release/v0.x.x

# 2. Update version in pyproject.toml
# 3. Build and verify
uv build

# 4. Create PR to main
# 5. After merge, tag the commit
git tag v0.x.x
git push origin v0.x.x

# 6. Publish to PyPI
uv publish --token <token>
```
