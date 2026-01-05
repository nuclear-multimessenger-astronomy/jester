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

### Next Priority Tasks

**CRITICAL** (before next release):

1. **⚠️ TODO: Eliminate redundant TOV solver calls via caching**
   - **Problem**: TOV solver runs TWICE on same samples - once during sampling (for likelihoods),
     once for saving results. This wastes computation time (5-10 seconds per call).
   - **Root cause**: Transform during sampling "forgets" parameters (only returns TOV outputs for likelihood).
     Final result needs BOTH parameters AND TOV outputs, so TOV solver is run again on the same samples.
   - **Goal**: Cache TOV outputs during sampling, reuse them when generating results (zero redundant calls).
   - **Current status**:
     - ✅ **Infrastructure**: Caching infrastructure added to `JesterSampler` base class (sampler-agnostic)
     - ✅ **Fallback**: `InferenceResult.add_eos_from_transform()` checks cache first, falls back to recomputation
     - ❌ **Blocker**: JAX tracing issue - cannot hash/cache inside JAX-compiled functions (e.g., `jax.lax.map`)
     - ⚠️ **All samplers TODO**: Need to implement caching OUTSIDE JAX trace context:
       - **FlowMC**: Cache during production phase (outside JAX compilation)
       - **BlackJAX SMC**: Cache final temperature samples (outside `get_log_prob()`)
       - **BlackJAX NS-AW**: Cache all samples (outside JAX compilation)
   - **Location**:
     - Cache infrastructure: `jesterTOV/inference/samplers/jester_sampler.py:117-438`
     - Cache usage: `jesterTOV/inference/result.py:271-390`
     - Currently DISABLED: `jesterTOV/inference/run_inference.py:491-498`
   - **Status**: Infrastructure complete, but DISABLED due to JAX tracing issues - all samplers need custom implementation
   - **Impact**: Would eliminate ALL redundant TOV solver calls (currently falls back to one recomputation)

2. **⚠️ Add tests for InferenceResult class** (HDF5 results storage) - ✅ COMPLETE (December 2024)
   - Comprehensive tests added in `tests/test_inference/test_result.py` (599 lines)
   - See "Testing Coverage Assessment" section for details

3. **Fix BlackJAX NS-AW type errors** (7 errors in `blackjax_ns_aw.py`)
   - Run: `uv run pyright jesterTOV/inference/samplers/blackjax_ns_aw.py`
   - See type error details in "Testing Coverage Assessment" section

4. **Document data loading status** - `jesterTOV/inference/data/__init__.py`
   - Has FIXME comment about missing DataLoader implementation
   - Either implement missing functions or update documentation

**HIGH PRIORITY**:

4. **Add postprocessing tests** - `tests/test_inference/test_postprocessing.py`
   - 893 lines of visualization code with NO tests
   - Critical for publication-quality figures

5. **Validate SMC with actual sampling runs**
   - Test prior-only sampling (not just dry run)
   - Test with real likelihoods (GW170817, NICER)
   - Test evidence calculation (recent bug fix in commit 7854188)
   - Test batching behavior (commit cbda19f notes "needs further investigating")

**See full testing assessment below for comprehensive analysis.**

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

**Test Status**: 291 tests total, all passing ✅

**Key Testing Insights**:
- Use realistic polytropic EOSs, not linear ones
- MetaModel tested up to ~2 nsat; MetaModel+CSE can go to 6+ nsat
- Stiff EOS: L_sym ≥ 90, Q_sat = 0, K_sym = 0 for realistic NS masses

---

## Testing Coverage Assessment (December 2024)

**CRITICAL TESTING GAPS** - The following areas have NO or INSUFFICIENT test coverage:

### 1. **InferenceResult Class (HDF5 Storage) - ✅ TESTS ADDED (December 2024)**
   - **File**: `jesterTOV/inference/result.py` (516 lines)
   - **Recent Change**: Complete rewrite from NPZ to HDF5 format (commit c1c3f18, Dec 2024)
   - **Test Coverage** (599 lines in `tests/test_inference/test_result.py`):
     - ✅ Save/load roundtrip for all sampler types (FlowMC, SMC, NS-AW)
     - ✅ Scalar vs array dataset handling in sampler_specific data
     - ✅ Metadata serialization/deserialization
     - ✅ Edge cases (empty histories, missing fields, pathlib vs string paths)
     - ✅ Config JSON round-trip
     - ✅ add_derived_eos() method
     - ✅ Summary generation for all sampler types
   - **Additional Coverage**: EOS sample generation filtering (see "Testing Issues (Fixed)" above)
   - **Status**: COMPREHENSIVE - Production infrastructure now well-tested

### 2. **Postprocessing Module - ❌ ZERO TESTS**
   - **File**: `jesterTOV/inference/postprocessing/postprocessing.py` (893 lines)
   - **Risk**: MEDIUM-HIGH - Used for publication-quality figures
   - **Missing Coverage**:
     - Cornerplot generation
     - Mass-radius diagram generation
     - Pressure-density plots
     - HDF5 result loading in postprocessing context
     - Error handling for missing data
     - TeX rendering fallback
   - **Location for tests**: `tests/test_inference/test_postprocessing.py` (DOES NOT EXIST)
   - **Priority**: HIGH - Scientific visualization is critical

### 3. **Data Loading Functions - ⚠️ INCOMPLETE IMPLEMENTATION**
   - **File**: `jesterTOV/inference/data/__init__.py`
   - **Status**: Has FIXME comment - "DataLoader class was removed. Need to implement data loading functionality"
   - **Risk**: MEDIUM - Unclear what functionality exists
   - **Missing Functions** (per FIXME):
     - `load_nicer_kde(psr_name, analysis_group, n_samples)`
     - `load_chieft_bands()`
     - `load_rex_posterior(experiment_name)`
     - `load_gw_nf_model(event_name, model_path)`
   - **Priority**: MEDIUM - Need to clarify implementation status

### 4. **BlackJAX NS-AW Sampler - ⚠️ TYPE ERRORS**
   - **File**: `jesterTOV/inference/samplers/blackjax_ns_aw.py` (503 lines)
   - **Status**: 7 pyright type errors (see below)
   - **Test Coverage**: Basic initialization tests exist, but type safety not verified
   - **Priority**: MEDIUM - Marked as experimental, but should be type-safe

   **Type Errors** (from `uv run pyright jesterTOV/inference/samplers/blackjax_ns_aw.py`):
   ```
   Line 240: Argument missing for parameter "rng_key"
   Line 279: Argument type mismatch (ArrayTree vs ParamDict)
   Line 301: Cannot get len() of ArrayTree
   Line 302: Cannot access attribute "n_likelihood_evals" for NamedTuple
   Line 403: Type conversion issue (2 errors)
   Line 404: Cannot access attribute "std" for MethodType
   ```

### 5. **BlackJAX SMC Sampler - ⚠️ LIMITED TESTING**
   - **File**: `jesterTOV/inference/samplers/blackjax_smc.py` (783 lines)
   - **Test Coverage**: Basic initialization + 2 slow integration tests
   - **Recent Activity**: Multiple bug fixes (commits a5c863e, f65827d, e083bd2)
   - **Missing Coverage**:
     - Evidence calculation (commit 7854188 "Attempt to fix blackjax evidence calculation")
     - Mass matrix building with custom scales
     - Sigma adaptation for random walk kernel
     - Batching behavior (commit cbda19f mentions "needs further investigating")
   - **Priority**: MEDIUM - Experimental, but under active development

### 6. **Test Distribution by Module**
   ```
   Likelihoods:       44 tests ✅ EXCELLENT
   Base classes:      40 tests ✅ EXCELLENT
   Samplers:          33 tests ✅ GOOD (but missing NS-AW/SMC edge cases)
   Config:            29 tests ✅ GOOD
   Priors:            19 tests ✅ ADEQUATE
   Integration:       17 tests ✅ ADEQUATE
   Transforms:        12 tests ⚠️ COULD BE BETTER

   Result class:       0 tests ❌ CRITICAL GAP
   Postprocessing:     0 tests ❌ CRITICAL GAP
   Data loading:       0 tests ❌ CRITICAL GAP
   ```

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
- ✅ **EOS sample generation log_prob filtering bug** (December 2024):
  - **Bug**: When generating fewer EOS samples than posterior samples (e.g., 5000 EOS from 10000 posterior),
    `log_prob` was not filtered to match the randomly selected samples, causing index out of bounds errors
    in postprocessing when trying to color M-R curves by probability
  - **Root Cause**: `generate_eos_samples()` used random selection (`np.random.choice`) for parameters but
    excluded `log_prob` from filtering (line 279), leaving full 10000 `log_prob` values with only 5000 EOS curves
  - **Fix** (commit XXXX): Filter `log_prob` and other sampler-specific fields (`weights`, `ess`, `logL`, `logL_birth`)
    to match selected samples; backup full arrays as `*_full`
  - **Location**: `jesterTOV/inference/run_inference.py:285-298`, `jesterTOV/inference/postprocessing/postprocessing.py:347-352, 458-463`
  - **Regression test**: `tests/test_inference/test_integration.py::TestEOSSampleGeneration`

### Open Issues
- **SMC get_log_prob parameter ordering bug**: `ravel_pytree` uses alphabetical ordering but `add_name` uses `prior.parameter_names` ordering, causing scrambled parameters → NaN log_prob
  - Location: `jesterTOV/inference/samplers/blackjax_smc.py` line 536
  - Fix: Use `self._unflatten_fn(particle)` instead of relying on `self.posterior(particle, {})`
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
