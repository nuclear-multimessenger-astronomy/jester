# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Below, you find the next step we are working on in terms of development. Below that, you find general info on the development process and the repo structure. 

# Important

IMPORTANT: 

When writing tests and something seems off, make a note in CLAUDE.md and start investigating: do NOT change your tests to pass, do not focus on just passing all
tests, but also make sure the code is correct and improved by your testing!

Don't do things like `cat > file.py << 'EOF'`, instead, just make the file and run it with `uv`.

When writing documentation, make it clear and brief to the point, and write full sentences as if by a human researcher and not an LLM, so it is easy to read for humans.

## Next steps

### High Priority: Documentation

**Focus on comprehensive documentation for the JESTER review:**

1. **User Documentation**
   - Tutorial notebooks demonstrating key workflows
   - Quickstart guide for inference with example configs
   - API reference documentation for all modules
   - Physics background and methodology explanations

2. **Developer Documentation**
   - Contribution guidelines
   - Testing strategy and how to add new tests
   - Architecture overview and design decisions
   - Adding new likelihoods, transforms, and priors

3. **Sphinx Documentation Enhancement**
   - Improve existing docstrings where needed (mostly complete for inference module)
   - Add narrative documentation pages
   - Create example gallery
   - Build comprehensive API docs

4. **Inference Module Documentation** (✅ Mostly Complete)
   - Module and class docstrings improved and researcher-friendly
   - Good coverage for transforms, likelihoods, flows
   - Ready for automated documentation generation

### High Priority: Testing Suite for Inference

Develop comprehensive tests for the inference module:
- Unit tests for individual components (priors, transforms, likelihoods)
- Integration tests for end-to-end inference workflows
- Test data loading and preprocessing
- Validate configuration parsing and validation

Location: `jester/tests/test_inference/` (✅ Created)

**Status** (as of 2024-12-25):
- Test infrastructure created (`tests/test_inference/`)
- Config module tests: ✅ **29/29 passing** (all issues resolved)
- Prior module tests: ✅ **19/19 passing** (API documented, boundary issue found)
- Transform module tests: 🚧 **4/12 passing** (investigating TOV solver and API issues)
- Issues discovered and fixed during testing (see below)

#### Issues Found During Testing

**1. LikelihoodConfig validation doesn't respect `enabled` field** ✅ **FIXED**
- **File**: `jesterTOV/inference/config/schema.py:152` (`validate_likelihood_parameters`)
- **Issue**: Parameter validation runs even when `enabled=False`
- **Impact**: Cannot create disabled likelihood configs without providing all required parameters
- **Fix applied**: Added check in validator: `if "enabled" in info.data and not info.data["enabled"]: return v`
- **Result**: `test_disabled_likelihood` now passes
- **Severity**: MEDIUM - Makes configs more verbose but doesn't break functionality

**2. Parser wraps ValidationError in ValueError** ✅ **DOCUMENTED**
- **File**: `jesterTOV/inference/config/parser.py:70`
- **Issue**: `load_config()` catches all exceptions and wraps them in ValueError
- **Impact**: Callers can't distinguish between different error types
- **Decision**: Keep current behavior (ValueError wrapping) for better error messages
- **Fix applied**: Updated test to expect ValueError and added documentation in test docstring
- **Result**: `test_load_config_missing_required_fields_fails` now passes
- **Severity**: LOW - Error message is still informative

**3. Test issue: `test_load_config_with_relative_paths` tries to copy file to itself** ✅ **FIXED**
- **File**: `tests/test_inference/test_config.py:271`
- **Issue**: Test tries to `shutil.copy(sample_prior_file, temp_dir / "test.prior")` but they're the same path
- **Fix applied**: Rewrote test to create prior file with different name ("relative.prior")
- **Result**: Test now passes and properly validates relative path resolution
- **Severity**: TEST BUG - not a code issue

**4. Prior API discovered during testing** (DOCUMENTATION ISSUE) ✅ **DOCUMENTED**
- **File**: `jesterTOV/inference/base/prior.py` (Prior, UniformPrior, CombinePrior)
- **Issue**: Prior `sample()` method signature is `sample(rng_key, n_samples)` not `sample(u_array)`
- **Impact**: Tests need to use correct API with JAX PRNGKey
- **Current behavior**: `sample(rng_key: PRNGKeyArray, n_samples: int) -> dict[str, Array]`
- **Also discovered**: `log_prob()` returns NaN for out-of-bounds instead of -inf
- **Fix applied**: Updated all tests to use correct API
- **Severity**: LOW - API works correctly, just different than initial assumption

**5. UniformPrior boundary values cause errors** (BUG - from Jim/jimgw)
- **File**: `jesterTOV/inference/base/transform.py:320` (LogitTransform)
- **Issue**: Evaluating `log_prob()` at exact boundary values causes errors:
  - `xmin` returns NaN
  - `xmax` raises `ZeroDivisionError` in logit transform: `log(x / (1-x))` when x=1.0
- **Impact**: Cannot evaluate log probability at exact boundaries
- **Root cause**: Logit transform `log(x / (1-x))` is undefined at x=0 and x=1
- **Workaround**: Always use values strictly inside boundaries (e.g., [xmin+ε, xmax-ε])
- **Recommended fix**: Add numerical guards in LogitTransform to return -inf at boundaries
- **Severity**: MEDIUM - Affects sampling at boundaries, but rare in practice
- **Status**: DOCUMENTED - Should be fixed in transform.py with proper boundary handling

**6. Transform factory expects Pydantic config, not dict** (API ISSUE)
- **File**: `jesterTOV/inference/transforms/factory.py:46`
- **Issue**: `create_transform()` expects `TransformConfig` Pydantic object, not plain dict
- **Impact**: Tests need to create Pydantic objects instead of dicts
- **Current behavior**: `config.type` fails with AttributeError when passed a dict
- **Fix needed**: Update tests to use `TransformConfig(**config_dict)` pattern
- **Severity**: LOW - Correct API, just needs proper usage in tests
- **Status**: TO BE FIXED in tests

**7. TOV solver hits max_steps with certain EOS parameters** (PHYSICS ISSUE)
- **File**: `jesterTOV/tov.py:187` (diffeqsolve)
- **Issue**: TOV integration fails with "maximum number of solver steps reached" for some EOS
- **Impact**: Cannot test transforms with all NEP parameter choices
- **Root cause**: Some EOS configurations may be:
  - Too soft (unstable neutron stars)
  - Have numerical issues in MetaModel interpolation
  - Require more solver steps for stiff ODEs
- **Test that failed**: `test_metamodel_forward_realistic_params` with realistic_nep_stiff
- **Recommended investigation**:
  - Check if L_sym=90 is too high without higher-order terms
  - Try softer EOS (L_sym=60)
  - Increase `max_steps` in TOV solver for testing
- **Severity**: MEDIUM - Blocks testing of transforms with realistic physics
- **Status**: NEEDS INVESTIGATION - May be physics issue or numerical issue

**8. CSE parameter count off by one** (TEST BUG)
- **File**: `tests/test_inference/test_transforms.py:205`
- **Issue**: Expected 26 CSE params but got 25
- **Impact**: Test assertion fails
- **Expected**: 8 NEP + 1 nbreak + 8*2 CSE grid + 1 final cs2 = 26
- **Actual**: 25 parameters returned
- **Root cause**: Need to check MetaModelCSETransform.get_parameter_names() implementation
- **Severity**: LOW - Likely just test expectation wrong
- **Status**: TO BE INVESTIGATED 

## Project Overview

JESTER (**J**ax-based **E**o**S** and **T**ov solv**ER**) is a scientific computing library for solving the Tolman-Oppenheimer-Volkoff (TOV) equations for neutron star physics. It provides hardware-accelerated computations via JAX with automatic differentiation capabilities.

## ⚡ IMPORTANT: Use `uv` for All Python Operations

**ALWAYS use `uv` for running Python commands and managing dependencies:**
- Use `uv run <command>` instead of running commands directly
- Use `uv pip install` instead of `pip install`
- Use `uv run pytest` instead of `pytest`
- Use `uv run pre-commit` instead of `pre-commit`

This ensures consistent dependency management and environment handling across all development tasks.

## Development Commands

### Installation and Setup
```bash
# Development install (CPU-only JAX by default)
uv pip install -e .

# Install with GPU support (optional)
uv pip install -e ".[cuda12]"   # For CUDA 12.x
uv pip install -e ".[cuda13]"   # For CUDA 13.x (when available)

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality
```bash
# Run all pre-commit checks
uv run pre-commit run --all-files

# Manual formatting and linting
uv run black .                    # Format code
uv run ruff check --fix .         # Lint and fix issues

# Type checking (install if needed: npm install -g pyright)
uv run pyright                    # Type checking all files
uv run pyright jesterTOV/         # Type check specific directory
```

### Testing
```bash
# Run tests
uv run pytest tests/
```

**TESTING PHILOSOPHY**: Focus on understanding physics and fixing root causes rather than modifying tests to pass. Debug with print statements to understand failures, then fix underlying issues in source code.

**Key Testing Insights**:
- Geometric units require careful handling (realistic NS: M~2000m, R~12000m, P~1e-11 m^-2)
- TOV integration direction: enthalpy decreases center→surface, so dm/dh < 0 is correct
- Use realistic polytropic EOSs, not linear ones, for neutron star tests
- **MetaModel EOS density limits**: MetaModel should only be tested up to ~2 nsat (causality issues at higher densities)
- **MetaModel+CSE**: CSE extension can safely go to 6+ nsat (designed for high-density regions)
- **Stiff EOS parameters**: Use L_sym ≥ 90, Q_sat = 0, K_sym = 0 for realistic neutron star masses

### Test Suite Status
**All 97 tests passing** - No skipped tests!

### Documentation Development
```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Build documentation locally
uv run sphinx-build docs docs/_build/html

# Build with warnings as errors (strict mode, same as CI)
uv run sphinx-build -W --keep-going docs docs/_build/html

# View documentation in browser
open docs/_build/html/index.html  # macOS
xdg-open docs/_build/html/index.html  # Linux

# Clean build artifacts
rm -rf docs/_build
```

**Documentation Structure**:
```
docs/
├── conf.py              # Sphinx configuration (theme, extensions, settings)
├── index.rst            # Main documentation page with table of contents
├── api/                 # Auto-generated API documentation
│   ├── jesterTOV.rst
│   ├── eos.rst
│   ├── tov.rst
│   ├── ptov.rst
│   └── utils.rst
└── _static/             # Static assets (CSS, logos, images)
    ├── style.css        # Custom CSS overrides
    ├── logo_light.svg   # Logo for light mode
    ├── logo_dark.svg    # Logo for dark mode
    └── icon.svg         # Favicon
```

**Theme and Styling**:
- **Theme**: `sphinx-book-theme` (matching flowjax style)
- **Custom CSS**: `docs/_static/style.css` for theme overrides
- **Logo customization**: Update SVG files in `docs/_static/` or modify `conf.py` logo paths
- **Theme options**: Edit `html_theme_options` in `docs/conf.py`

**Adding New Documentation Pages**:
1. Create `.rst` or `.md` file in `docs/` directory
2. Add the file to a `toctree` directive in `index.rst` or parent page
3. Example:
   ```rst
   .. toctree::
      :maxdepth: 2

      new_page
      tutorials/tutorial_1
   ```

**Working with LaTeX in Docstrings**:
- Always use raw strings: `r"""..."""` to avoid escape sequence warnings
- Inline math: `$E = mc^2$` or `\(E = mc^2\)`
- Display math: `$$E = mc^2$$` or `\[E = mc^2\]`
- MathJax is configured in `conf.py` to handle both formats

**Documentation Deployment**:
- **Automatic**: Pushes to `main` branch trigger GitHub Actions workflow (`.github/workflows/docs.yml`)
- **Manual**: Can trigger via GitHub Actions "Run workflow" button
- **URL**: https://nuclear-multimessenger-astronomy.github.io/jester/
- **Build time**: ~2-3 minutes from push to live

**Troubleshooting**:
- **Import errors during build**: Ensure all dependencies are in `pyproject.toml` `[project.optional-dependencies.docs]`
- **Missing modules**: Run `uv pip install -e ".[docs]"` to reinstall
- **Broken links**: Check `toctree` directives reference existing files
- **Theme not loading**: Verify `sphinx-book-theme` is installed and `html_static_path` includes `_static`
- **Logo not showing**: Check SVG file paths in `conf.py` match files in `docs/_static/`

**Documentation Best Practices**:
- Build locally before committing to catch errors early
- Use `sphinx-build -W` (warnings as errors) to match CI behavior
- Keep docstrings concise but complete with type hints
- Add examples to docstrings for complex functions
- Update API docs when adding new modules or public functions

## Architecture

### Core Modules
- **`jesterTOV/eos.py`** - Equation of state models and crust data loading
- **`jesterTOV/tov.py`** - Standard TOV equation solver
- **`jesterTOV/ptov.py`** - Post-TOV solver with extended physics
- **`jesterTOV/utils.py`** - Physical constants and unit conversions
- **`jesterTOV/crust/`** - Neutron star crust models (BPS, DH data files)

### Key Design Principles
- **JAX-first:** All computations designed for hardware acceleration and automatic differentiation
- **Geometric units:** Physics calculations use geometric unit system throughout
- **Type safety:** Strong typing with `jaxtyping` for array shapes and dtypes
- **64-bit precision:** Enabled by default in `__init__.py` for numerical accuracy
- **Type hints:** Comprehensive type annotations for better code maintainability and IDE support

### Type Hinting Standards

**All new code MUST include comprehensive type hints.** Type hints improve code maintainability, enable better IDE support, catch bugs early, and make the codebase more user-friendly.

**Required Type Hints**:
- **Function signatures**: All function parameters and return types
- **Class attributes**: Public attributes should be annotated
- **Module-level variables**: Constants and important module variables
- **Complex data structures**: Dictionaries, lists, and custom types

**Type Hinting Guidelines**:

1. **Use standard library types** (Python 3.10+ syntax):
   ```python
   from typing import Optional, Union, Callable, Any

   # Good: Modern syntax
   def process_data(values: list[float], threshold: float | None = None) -> dict[str, float]:
       ...

   # Avoid: Old-style typing (but acceptable for Python 3.9 compatibility)
   from typing import List, Dict
   def process_data(values: List[float], threshold: Optional[float] = None) -> Dict[str, float]:
       ...
   ```

2. **Use `jaxtyping` for JAX arrays**:
   ```python
   from jaxtyping import Array, Float, Int
   import jax.numpy as jnp

   # Specify array shapes and dtypes
   def solve_tov(
       pressure: Float[Array, "n_points"],
       density: Float[Array, "n_points"]
   ) -> Float[Array, "n_points"]:
       ...
   ```

3. **Use `Pydantic` models for configuration**:
   ```python
   from pydantic import BaseModel, Field

   class SamplerConfig(BaseModel):
       n_chains: int = Field(gt=0, description="Number of MCMC chains")
       learning_rate: float = Field(gt=0.0, le=1.0)
   ```

4. **Document complex types with `TypeAlias`**:
   ```python
   from typing import TypeAlias

   # Define reusable type aliases
   ParameterDict: TypeAlias = dict[str, float]
   EOSFunction: TypeAlias = Callable[[float], float]
   ```

5. **Use `Protocol` for duck typing**:
   ```python
   from typing import Protocol

   class Likelihood(Protocol):
       def evaluate(self, params: dict[str, float], data: dict) -> float:
           ...
   ```

6. **Avoid `Any` when possible**:
   - Use specific types or generics instead of `Any`
   - If `Any` is necessary, add a comment explaining why
   - Consider using `object` for truly unknown types

**Type Checking**:
```bash
# Check types before committing
uv run pyright jesterTOV/inference/

# Fix type errors, don't suppress them unless absolutely necessary
```

**Gradual Adoption**:
- All new files must have complete type hints
- When editing existing files, add type hints to functions you modify
- Large files can be improved incrementally (start with public APIs)

**Benefits**:
- **Better IDE support**: Autocomplete, go-to-definition, refactoring tools
- **Early bug detection**: Catch type mismatches before runtime
- **Living documentation**: Type hints serve as inline documentation
- **Easier onboarding**: New contributors understand interfaces faster
- **Safer refactoring**: Type checker catches broken contracts

### Development Notes
- The codebase is specialized for astrophysics/neutron star modeling
- Pre-commit hooks enforce code quality with black and ruff
- Examples in `examples/` directory demonstrate basic and advanced usage
- Comprehensive test suite with 95 tests covering all major functionality
- **LaTeX in docstrings**: Use raw strings (`r"""..."""`) for math expressions to avoid Pylance warnings

### Python Support
Supports Python 3.10-3.12 with JAX ecosystem dependencies.

### Inference System (Work in Progress)

**Status**: 🚧 Active development on `inference` branch - Modular architecture implementation (Phases 1-6 complete)

**⚠️ IMPORTANT - NO BACKWARDS COMPATIBILITY**: The inference system has been refactored with a complete breaking change. The old argparse-based interface is preserved in `run_inference_old.py` but will be removed. Use the new config-driven system going forward.

The `jesterTOV/inference/` module provides Bayesian inference capabilities for constraining equation of state (EOS) parameters using astrophysical observations.

#### New Modular Structure (Phases 1-6 Complete)

```
jesterTOV/inference/
├── __init__.py
├── config/                      # ✅ Phase 1: Configuration system
│   ├── __init__.py
│   ├── parser.py                # YAML config loading
│   ├── schema.py                # Pydantic validation models
│   └── examples/                # Example config files
│       ├── full_inference.yaml
│       ├── gw170817_only.yaml
│       └── nicer_only.yaml
├── priors/                      # ✅ Phase 2: Prior system (bilby-style format)
│   ├── __init__.py
│   ├── parser.py                # .prior file parsing (executes Python code)
│   ├── library.py               # Common prior definitions
│   └── specifications/          # Prior specification files (Python format)
│       ├── README.md            # Documentation for creating .prior files
│       ├── nep_standard.prior   # K_sat = UniformPrior(...) syntax
│       ├── nep_tight.prior      # Tighter parameter ranges
│       └── cse_8params.prior    # Configuration with CSE
├── transforms/                  # ✅ Phase 3: Transform refactoring
│   ├── __init__.py
│   ├── base.py                  # JesterTransformBase ABC
│   ├── metamodel.py             # MetaModel transform
│   ├── metamodel_cse.py         # MetaModel+CSE transform
│   ├── factory.py               # Transform creation
│   └── auxiliary.py             # Helper functions
├── likelihoods/                 # ✅ Phase 4: Likelihood refactoring
│   ├── __init__.py
│   ├── gw.py                    # GW event likelihoods
│   ├── nicer.py                 # NICER X-ray timing
│   ├── radio.py                 # Radio pulsar timing
│   ├── chieft.py                # ChiEFT constraints
│   ├── rex.py                   # PREX/CREX constraints
│   ├── combined.py              # Combined likelihood
│   └── factory.py               # Likelihood creation
├── data/                        # ✅ Phase 5: Data loading
│   ├── __init__.py
│   ├── loader.py                # Lazy data loading with caching
│   └── paths.py                 # Path management
├── samplers/                    # ✅ Phase 6: Sampler wrappers
│   ├── __init__.py
│   └── jim.py                   # Jim sampler setup
├── run_inference.py             # ✅ Phase 6: New config-driven main script
├── cli.py                       # ✅ Phase 6: Command-line interface
├── run_inference_old.py         # Old argparse version (backup)
├── postprocessing.py            # Analysis utilities (Phase 7 pending)
└── constraints/                 # Old structure (to be removed)
    ├── likelihood.py            # Superseded by likelihoods/
    └── NICER/
        └── get_data.py          # Superseded by data/loader.py
```

#### New Execution Flow (Config-Driven)

```
config.yaml
  ↓
Load config → Validate with Pydantic
  ↓
  ├─→ Parse .prior file → CombinePrior
  ├─→ Create transform via factory → JesterTransformBase
  ├─→ Load data via DataLoader → Lazy loading with caching
  ├─→ Create likelihoods via factory → CombinedLikelihood
  ↓
Setup Jim sampler (jimgw/flowMC)
  ↓
MCMC sampling (training + production)
  ↓
Save results to outdir/
  ↓
Generate EOS samples (TOV solve on selected samples)
```

#### Key Components

**Transforms** (`transforms.py`):
- `MicroToMacroTransform`: NEP parameters → Mass-Radius-Lambda curves
- Two modes: MetaModel only, or MetaModel+CSE (Constant Speed Extension)
- Configurable: density grids, TOV integration, crust models

**Likelihoods** (`constraints/likelihood.py`):
- `GWlikelihood_with_masses`: Gravitational wave events (GW170817)
- `NICERLikelihood`: X-ray timing observations (J0030, J0740)
- `RadioTimingLikelihood`: Pulsar mass measurements
- `ChiEFTLikelihood`: Chiral Effective Field Theory constraints
- `REXLikelihood`: PREX/CREX lead radius experiments
- `CombinedLikelihood`: Sum multiple constraints

**Sampler** (`run_inference.py`):
- Uses `jimgw` (Jim Gravitational Wave library) wrapper around `flowMC`
- Normalizing flow-enhanced MCMC sampling
- Two-stage: training loops + production loops
- Configurable: chains, steps, epochs, learning rate

**Prior Specification** (`priors/`):
- **Format**: Bilby-style Python syntax with variable assignments
- **Example**: `K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])`
- **Conditional parameters**: Use Python if-statements (e.g., `if nb_CSE > 0: nbreak = ...`)
- **CSE grid parameters**: Added automatically by parser based on `nb_CSE` in config
- **Available variables**: `nb_CSE`, `UniformPrior`
- **Documentation**: See `jesterTOV/inference/priors/specifications/README.md`

#### Implementation Status

**✅ Completed (Phases 1-6)**:
- Configuration system with YAML/Pydantic validation
- Prior specification with .prior files (bilby-style Python format)
- Transform base class with MetaModel/MetaModel+CSE implementations
- Modular likelihood system with factory pattern
- Lazy data loading with path management
- New config-driven run_inference.py
- Sampler wrapper for Jim/flowMC

**⏳ Pending (Phase 7)**:
- Postprocessing cleanup (remove hardcoded paths)
- Split postprocessing.py into modular components

**🔮 Future Work**:
- Data downloading utilities
- Normalizing flow training pipeline (train_NF.py)
- Testing suite for inference components
- Tutorial notebooks for config-based inference

**✅ Documentation (Complete)**:
- Comprehensive inference documentation in `docs/` directory
- Auto-generated YAML reference from Pydantic schemas
- Quick start guide, complete reference, and architecture docs
- Documentation maintenance guide for developers


**Dependency Summary**:
```python
# Active dependencies (to be removed):
jimgw.single_event.likelihood.LikelihoodBase  # 8 files
jimgw.prior.Prior                              # 3 files
jimgw.prior.CombinePrior                       # 2 files
jimgw.prior.UniformPrior                       # 2 files (deprecated)
jimgw.transforms.NtoMTransform                 # 4 files
jimgw.transforms.BijectiveTransform            # 1 file

# Removed dependencies:
jimgw.jim.Jim                                  # Replaced by JesterSampler
jimgw.base.LikelihoodBase                      # Deprecated path, use single_event.likelihood
```

**Implementation Plan** (Future):
1. Create `jesterTOV/inference/base/` module
2. Implement standalone base classes:
   - `base/likelihood.py` - LikelihoodBase ABC
   - `base/prior.py` - Prior, CombinePrior ABCs
   - `base/transform.py` - NtoMTransform, BijectiveTransform ABCs
3. Update all imports to use JESTER's base classes
4. Remove jimgw from dependencies (pyproject.toml)
5. Keep flowMC as direct dependency (core MCMC engine)

**Benefits of Removal**:
- Full control over interfaces and bug fixes
- No dependency on external package versions
- Cleaner stack for debugging
- Reduced installation complexity


#### Running Inference

**New config-driven interface** (recommended):
```bash
# Full inference with config file
uv run python -m jesterTOV.inference.run_inference --config config.yaml

# Use example config
uv run python -m jesterTOV.inference.run_inference \
    --config jesterTOV/inference/config/examples/full_inference.yaml

# Override output directory
uv run python -m jesterTOV.inference.run_inference \
    --config config.yaml --output-dir ./my_results/

# Validate config without running
uv run python -m jesterTOV.inference.run_inference \
    --config config.yaml --validate-only

# Dry run (setup without sampling)
uv run python -m jesterTOV.inference.run_inference \
    --config config.yaml --dry-run
```

**Old argparse interface** (deprecated, preserved in `run_inference_old.py`):
```bash
# This interface will be removed - use config files instead
uv run python jesterTOV/inference/run_inference_old.py \
    --sample_GW170817 \
    --sample_J0030 \
    --sample_J0740 \
    --NB_CSE 8 \
    --n_chains 20 \
    --outdir ./results/
```

#### Inference Documentation

**Comprehensive documentation** is available in `docs/`:

- **[docs/inference_index.md](docs/inference_index.md)** - Navigation hub to all inference docs
- **[docs/inference_quickstart.md](docs/inference_quickstart.md)** - 5-minute quick start guide
- **[docs/inference.md](docs/inference.md)** - Complete reference (architecture, config, priors, likelihoods, transforms)
- **[docs/inference_architecture.md](docs/inference_architecture.md)** - Technical architecture details
- **[docs/inference_yaml_reference.md](docs/inference_yaml_reference.md)** - **AUTO-GENERATED** complete YAML options reference
- **[docs/inference_documentation_guide.md](docs/inference_documentation_guide.md)** - How to maintain inference docs

**YAML Configuration Reference**:

The `docs/inference_yaml_reference.md` file is **auto-generated** from Pydantic schemas and serves as the authoritative reference for all YAML configuration options.

**⚠️ IMPORTANT**: When modifying `jesterTOV/inference/config/schema.py`, regenerate the YAML reference:

```bash
uv run python -m jesterTOV.inference.config.generate_yaml_reference
```

This ensures the documentation stays in sync with the actual validation rules. The generator script extracts:
- All field names, types, and defaults
- Required vs. optional fields
- Validation rules
- Complete configuration templates
- Likelihood-specific parameters

**Documentation maintenance**:
- **Auto-generated docs**: YAML reference (regenerate when schema.py changes)
- **Manual docs**: Narrative guides, architecture, examples (update when features change)
- See `docs/inference_documentation_guide.md` for complete maintenance workflow

#### TODOs

**High Priority** (Next Focus):
- [ ] **Documentation development** (user guides, tutorials, API docs)
- [ ] **Testing suite for inference components** (unit + integration tests)

**Future Work**:
- [ ] Implement REX posterior loading in data/loader.py
- [ ] GW190425 data downloading and processing
- [ ] Normalizing flow training pipeline improvements
- [ ] Support for additional constraints (more experiments)
- [ ] Tutorial notebooks for config-based inference
- [ ] Migration guide for users of old argparse interface

**Note**: The modular architecture is now fully functional (Phases 1-7 complete). The old argparse-based interface is preserved in `run_inference_old.py` but should not be used for new work. Use the config-driven system in `run_inference.py` instead.

## Repository Status

### ✅ PyPI Package Release - COMPLETED
**Status**: v0.1.0 published to PyPI - Package available for installation

**IMPORTANT**: Future PyPI releases MUST be done through pull requests, not direct commits to main. This ensures:
- Proper code review before publishing
- Clean git history
- Adherence to branch protection rules
- CI/CD validation before release

**Release Workflow for Future Versions**:
1. Create a feature branch for version bump and changelog
2. Update `version` in pyproject.toml
3. Build with `uv build` to verify
4. Create PR to main branch
5. After PR approval and merge, tag the merge commit: `git tag v0.x.x`
6. Push tag: `git push origin v0.x.x`
7. Publish to PyPI: `uv publish --token <token>`

**v0.1.0 Release Info**:
- Package available: `pip install jesterTOV`
- PyPI page: https://pypi.org/project/jesterTOV/
- Git tag: v0.1.0
- Build system: hatchling with flat package layout

### ✅ Phase 1: Testing Infrastructure - COMPLETED
**Status**: All 97 tests passing with no skipped tests

### ✅ Phase 2: CI/CD & Code Quality - COMPLETED
**Status**: GitHub Actions CI operational with automated testing

### ✅ Phase 3: Documentation System - COMPLETED
**Status**: Sphinx documentation with GitHub Pages deployment operational

**Documentation Details**:
- **Site URL**: https://nuclear-multimessenger-astronomy.github.io/jester/
- **Theme**: sphinx-book-theme (matching flowjax style)
- **Deployment**: Automated via GitHub Actions on push to main
- **Features**: Copy buttons, dark/light mode, responsive design
- **Build Tool**: uv for dependency management

**Local Documentation Build**:
```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Build documentation
uv run sphinx-build docs docs/_build/html

# View locally
open docs/_build/html/index.html
```

**Achievements**:
- ✅ **Documentation site** - GitHub Pages with flowjax-style theme
- ✅ **GitHub Actions CI/CD** - Automated testing and docs deployment
- ✅ **Code quality** - Pre-commit hooks with black, ruff, comprehensive test suite
- ✅ **Type safety** - Reduced pyright errors from 23 to 10 (57% improvement)

### ✅ Type Safety Issues - RESOLVED
**Status**: All major type issues resolved

**Recent Fixes**:
- ✅ Fixed interpax import issue (`from interpax._spline import interp1d`)
- ✅ Resolved JAX array type annotations in eos.py
- ✅ Added guidance for LaTeX docstrings (use raw strings `r"""..."""`)

### Testing Commands
```bash
# Run all tests
uv run pytest

# Run specific categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest -m "not slow"    # Skip slow tests
```