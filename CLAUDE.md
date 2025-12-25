# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Below, you find the next step we are working on in terms of development. Below that, you find general info on the development process and the repo structure. 

## Next steps

### NICER data

Implement ways to download the NICER datasets from Zenodo, open them to get the desired mass and radius posterior samples, and construct KDEs from them to be used in inference. The idea is that we download the dataset from the given URL if it does not exist, then save a file that ONLY has the desired mass and radius samples (there might be more variables from NICER analysis, that are not useful for us, therefore, we save our own file). The idea is to put the source code that does this inside the repo as well, so we can always download the original file again and redo this process for reproducibility and users can follow that approach in case they want to save something extra. 

The data handling should be stored in this dir: `/Users/Woute029/Documents/Code/projects/jester_review/jester/jesterTOV/inference/data`. 

Note: we need to store the information somewhere in a README or docs: for now, do a README in the ./data dir. We can migrate later on. The info should be original link, hotspot models, paper that published this dataset, and so on. 

For now, let's put the code there to download the NICER datasets for the 2 pulsars we are interested in. Note: these are analyzed in two groups, and therefore each PSR has two sets of posterior samples that have to be 'mixed'. Note moreover there are different "hotspot" models so different groups of posterior samples: first make a script that explores the NICER datasets to understand them and put that information in the README as well. 
- NICER PSR J0030+0451: data can be found here: for the "Amsterdam group": https://zenodo.org/records/8239000 but also https://zenodo.org/records/7096789 for older samples, and https://zenodo.org/records/3473466 for the second group (Here, we might have to allow users to download both Amsterdam datasets, and choose which one to use, use last author names to distinguish, and the former is the more recent so use that one, but watch out for different hotspot models.)
- NICER PSR J0740+6620: data can be found here: For Amsterdam: https://zenodo.org/records/10519473 most recent, but also https://zenodo.org/records/6827537 and originally https://zenodo.org/records/5735003. For Maryland: https://zenodo.org/records/4670689

Download the data from zenodo (src code) and use it to save the NICER mass radius posterior samples in a file we have in this repo. 

You can find information on the KDE construction etc and data loading a bit in `/Users/Woute029/Documents/Code/projects/jester_review/jester/jesterTOV/inference/data/old_utils.py`. Note that the KDE construction should use the file we made ourselves, and also, the KDE construction has to be done on the fly when initializing the NICER likelihood in the src code (load samples, make KDE). 

### GW170817, GW190425 data

The same principle as above applies to the two BNS merger events we have so far: GW170817 and GW190425. 

Let us focus on downloading the GW170817 dataset and extracting the following keys from the posterior: `mass_1_source`, `mass_2_source`, `lambda_1`, `lambda_2`. Note that we might also need to save some metadata such as waveform model used, where the data is taken from,... 

Check out the DCC page here in order to get started: https://dcc.ligo.org/LIGO-P1800061/public 

The notebook you see there to extract posterior samples might be easier to read in this format: https://nbviewer.org/urls/dcc.ligo.org/public/0150/P1800061/011/Data%20Release%20Tutorial.ipynb

Check out which posterior samples there are, download the samples and save as npz file for ONLY those variables listed above, then start downloading GW170817 data. 

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

#### jimgw Dependencies (To Be Removed)

**Status**: JESTER inference currently depends on jimgw for base classes and interfaces. Goal is to become fully independent.

**Current Dependencies** (as of 2024-12):

1. **Likelihood Interface**:
   - `from jimgw.single_event.likelihood import LikelihoodBase`
   - Used in: All likelihood files (`likelihoods/*.py`, `constraints/likelihood.py`)
   - Need: Copy/reimplement LikelihoodBase ABC

2. **Prior System**:
   - `from jimgw.prior import Prior`
   - Used in: `priors/simple_priors.py`, `priors/parser.py`, `samplers/jester_sampler.py`
   - `from jimgw.prior import CombinePrior`
   - Used in: `priors/parser.py`, `priors/library.py`
   - `from jimgw.prior import UniformPrior`
   - Used in: `priors/library.py`, `run_inference_old.py`
   - Need: Copy/reimplement Prior, CombinePrior ABCs
   - Note: SimpleUniformPrior already implemented as replacement for UniformPrior

3. **Transform System**:
   - `from jimgw.transforms import NtoMTransform`
   - Used in: `transforms/base.py`, `transforms/auxiliary.py`, `transforms.py`, `samplers/jester_sampler.py`
   - `from jimgw.transforms import BijectiveTransform`
   - Used in: `samplers/jester_sampler.py`
   - Need: Copy/reimplement NtoMTransform, BijectiveTransform ABCs

4. **Sampler (REMOVED)**:
   - ✅ `from jimgw.jim import Jim` - **NO LONGER USED** (replaced by JesterSampler)
   - Old usage: `run_inference_old.py` (deprecated)
   - New: `samplers/jester_sampler.py` - Standalone implementation with bug fixes

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

#### Modular Architecture Benefits

**📋 Full Implementation Plan**: See comprehensive refactoring plan at:
```
/Users/Woute029/.claude/plans/humble-weaving-wave.md
```

This plan includes:
- Complete current state analysis
- 7 implementation phases with code examples
- All file paths to create/modify/remove
- Configuration schemas and prior file formats
- No backwards compatibility (intentional breaking change)

**Key improvements** (now implemented):
- ✅ **Configuration-driven**: YAML config files replace argparse
- ✅ **Prior specification**: `.prior` files parsed and loaded, not hardcoded
- ✅ **Transform hierarchy**: Base class with MetaModel and MetaModel+CSE subclasses
- ✅ **Modular likelihoods**: Factory pattern for likelihood creation
- ✅ **Lazy data loading**: Configurable paths, no import side effects
- ✅ **Clean separation**: config / priors / transforms / likelihoods / data / samplers
- ✅ **Type safety**: Pydantic validation catches configuration errors early
- ✅ **Reproducibility**: Version-controlled config files, seed management

**Structure** (implemented):
```
jesterTOV/inference/
├── config/              # YAML parsing and validation (Pydantic)
├── priors/              # Prior specification and parsing (.prior files)
├── transforms/          # Transform base class and implementations
├── likelihoods/         # Modular likelihood components
├── data/                # Lazy data loading and path management
├── samplers/            # Sampler configuration and wrappers
├── postprocessing/      # Analysis utilities
├── run_inference.py     # New config-driven main script
└── cli.py               # Command-line interface
```

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

**Completed** (Phases 1-6 + Documentation):
- [x] Implement configuration system (YAML/Pydantic)
- [x] Create prior specification format (.prior files)
- [x] Refactor transforms with base class hierarchy
- [x] Modularize likelihood system
- [x] Implement lazy data loading
- [x] Rewrite run_inference.py with new architecture
- [x] Create sampler wrapper (Jim/flowMC)
- [x] Create CLI interface
- [x] **Comprehensive inference documentation** (quick start, reference, architecture)
- [x] **Auto-generated YAML reference** from Pydantic schemas
- [x] **Documentation maintenance guide** for keeping docs in sync

**In Progress** (Phase 7):
- [ ] Clean up postprocessing.py (remove hardcoded paths)
- [ ] Split postprocessing into modular components
- [ ] Remove old constraints/ directory structure

**Future Work**:
- [ ] Implement GW NF model loading in data/loader.py
- [ ] Implement REX posterior loading in data/loader.py
- [ ] Data downloading utilities
- [ ] Normalizing flow training pipeline (`train_NF.py`)
- [ ] Support for additional constraints
- [ ] Testing suite for inference components
- [ ] Tutorial notebooks for config-based inference
- [ ] Migration guide documentation

**Note**: The modular architecture is now functional (Phases 1-6 complete). The old argparse-based interface is preserved in `run_inference_old.py` but should not be used for new work. Use the config-driven system in `run_inference.py` instead.

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