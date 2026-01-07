# CLAUDE.md

This file provides guidance to Claude Code when working with the JESTER repository.

## Important Guidelines

**Testing Philosophy**: When tests fail, investigate root causes rather than modifying tests to pass. Make notes in CLAUDE.md and fix underlying code issues.

**Documentation Style**: Write clear, concise documentation in full sentences as if by a human researcher. Avoid LLM-like verbosity.

**File Operations**: Use proper tools (Write, Edit, Read) instead of bash heredocs or cat redirection.

---

## Current Status (December 2024)

### Multi-Sampler Architecture ✅ COMPLETE

Three sampler backends are now available for Bayesian inference:

1. **FlowMC** (Production Ready) - Normalizing flow-enhanced MCMC
2. **BlackJAX SMC** (Production Ready) - Sequential Monte Carlo with adaptive tempering
   - NUTS kernel with Hessian-based mass matrix adaptation
   - Gaussian Random Walk kernel with sigma adaptation
   - ✅ All type errors fixed (0 errors)
   - ✅ Example configs created
   - ✅ Dry run validation passed
3. **BlackJAX NS-AW** (Needs Type Checking) - Nested Sampling with Acceptance Walk
   - ⚠️ Has 7 pyright type errors to fix
   - Example configs created
   - Prior-only test run successful

### Unified HDF5 Results Storage ✅ COMPLETE

Single `results.h5` file replaces separate NPZ files:

- ✅ `InferenceResult` class for save/load operations (`jesterTOV/inference/result.py`)
- ✅ Hierarchical HDF5 structure: `/posterior`, `/metadata`, `/histories`
- ✅ Supports all samplers (FlowMC, SMC, NS-AW) with sampler-specific metadata
- ✅ Stores parameters + derived EOS quantities + sampler diagnostics
- ✅ Postprocessing updated to use HDF5 format
- ✅ Type checking passes (0 errors)
- ⚠️ **Needs tests**: See Next Priority Tasks #4

**Example Configs Available**:
```bash
# FlowMC (production ready)
examples/inference/smc-nuts/config.yaml
examples/inference/smc-random-walk/config.yaml

# BlackJAX NS-AW (needs type fixes)
examples/inference/blackjax-ns-aw/prior/config.yaml
examples/inference/blackjax-ns-aw/GW170817/config.yaml
examples/inference/blackjax-ns-aw/NICER_J0030/config.yaml
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

---

## Testing Coverage Assessment (December 2024)

TODO: Rerun to check the actual coverage now.

### Recommendations

**IMMEDIATE ACTIONS** (before next release):
1. **Create `tests/test_inference/test_result.py`** - Test HDF5 save/load for all sampler types
2. **Fix NS-AW type errors** - Run `uv run pyright` and resolve all 7 errors
3. **Document data loading status** - Either implement missing functions or update FIXME

**SHORT-TERM** (next sprint):
4. **Create `tests/test_inference/test_postprocessing.py`** - Basic smoke tests for plotting
5. **Expand SMC tests** - Add tests for evidence calculation, batching behavior
6. **Expand transform tests** - More edge cases for MetaModel and MetaModel+CSE

**LONG-TERM**:
7. **Add integration tests with real data** - Test full workflow with actual GW/NICER data
8. **Performance regression tests** - Track sampling efficiency over time
9. **Documentation tests** - Verify all examples in docs actually run

---

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
uv run run_jester_inference config.yaml

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

### Base Classes (jimgw Independence)
The `base/` module contains copies of Jim v0.2.0 base classes to remove dependency on jimgw:
- `LikelihoodBase` - Abstract likelihood interface
- `Prior`, `CombinePrior`, `UniformPrior` - Prior system
- `NtoMTransform`, `BijectiveTransform` - Transform interfaces

Goal: Keep flowMC as only external sampler dependency.

### Critical Design Notes
- **JAX NaN handling**: Use `jnp.inf` instead of `jnp.nan` for initialization
- **flowMC data argument**: Always pass empty dict `{}`, not `None`
- **Geometric units**: Realistic NS: M~2000m, R~12000m, P~1e-11 m^-2
- **LaTeX in docstrings**: Use raw strings `r"""..."""` to avoid warnings

---

## Known Issues & Workarounds

### Testing Issues (Fixed)
- ✅ LikelihoodConfig validation respects `enabled` field
- ✅ Prior API uses `sample(rng_key, n_samples)` not `sample(u_array)`
- ✅ Transform factory expects Pydantic config, not dict
- ✅ **EOS sample generation array filtering bug** (December 2024, refined January 2026):
  - **Bug**: When generating fewer EOS samples than posterior samples (e.g., 10000 EOS from 800000 posterior),
    arrays had inconsistent lengths causing `ValueError` in cornerplot: "array at index 0 has size 800000 and
    array at index 8 has size 10000"
  - **Root Cause**: `InferenceResult.add_eos_from_transform()` filtered `log_prob` and sampler-specific fields
    but forgot to filter NEP/CSE parameter arrays, leaving them at full posterior size while EOS quantities were subsampled
  - **Fix** (January 2026): Filter ALL arrays (log_prob, sampler fields, AND NEP/CSE parameters) to match
    selected samples; backup full arrays as `*_full`; added validation loop to catch future mismatches
  - **Location**: `jesterTOV/inference/result.py:385-418`
  - **Regression test**: `tests/test_inference/test_integration.py::TestEOSSampleGeneration` (needs update)
- ✅ **SMC parameter ordering bug** (December 2025):
  - **Bug**: `ravel_pytree` uses alphabetical ordering but `add_name` uses `prior.parameter_names` ordering,
    causing scrambled parameters → NaN log_prob
  - **Root Cause**: Inconsistent parameter ordering between flatten and unflatten operations
  - **Fix** (commit a5c863e): Use `self._unflatten_fn(particle_flat)` consistently; added `posterior_from_dict()`
    method to bypass `add_name()`
  - **Location**: `jesterTOV/inference/samplers/blackjax_smc.py:844-852`
  - **Validation**: SMC examples run successfully without NaN values
- ✅ **BlackJAX NS-AW initialization bug** (January 2026):
  - **Bug**: `TypeError: init_fn() got an unexpected keyword argument 'rng_key'` when initializing nested sampler
  - **Root Cause**: The `init_fn()` in `bilby_adaptive_de_sampler_unit_cube` only accepts `particles` parameter,
    but code was passing `rng_key=subkey` argument (initialization doesn't need random key)
  - **Fix** (January 2026): Remove `rng_key=subkey` argument from `nested_sampler.init()` call; added type
    ignore comment for pyright false positive
  - **Location**: `jesterTOV/inference/samplers/blackjax_ns_aw.py:251-253`

### Open Issues
- **UniformPrior boundaries**: `log_prob()` at exact boundaries causes errors (NaN at xmin, ZeroDivision at xmax)
  - Workaround: Use values strictly inside boundaries
  - Fix: Add numerical guards in LogitTransform
- **TOV solver max_steps**: Some stiff EOS configs hit solver limits
  - May need to increase `max_steps` or adjust EOS parameters

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
