# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JESTER (**J**ax-based **E**o**S** and **T**ov solv**ER**) is a scientific computing library for solving the Tolman-Oppenheimer-Volkoff (TOV) equations for neutron star physics. It provides hardware-accelerated computations via JAX with automatic differentiation capabilities.

## Development Commands

### Installation and Setup
```bash
# Development install
pip install -e .

# Install pre-commit hooks
pre-commit install

# For GPU support (optional)
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Code Quality
```bash
# Run all pre-commit checks
pre-commit run --all-files

# Manual formatting and linting
black .                    # Format code
ruff check --fix .         # Lint and fix issues

# Type checking (install if needed: npm install -g pyright)
pyright                    # Type checking all files
pyright jesterTOV/         # Type check specific directory
```

### Testing
```bash
# Run tests (if available)
pytest tests/
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

### Development Notes
- The codebase is specialized for astrophysics/neutron star modeling
- Pre-commit hooks enforce code quality with black and ruff
- Examples in `examples/` directory demonstrate basic and advanced usage
- Comprehensive test suite with 95 tests covering all major functionality
- **LaTeX in docstrings**: Use raw strings (`r"""..."""`) for math expressions to avoid Pylance warnings

### Python Support
Supports Python 3.10-3.12 with JAX ecosystem dependencies.

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
**Status**: Sphinx documentation with Read the Docs integration operational

**Achievements**:
- ✅ **Documentation site** - Full API docs with mathematical formulas at readthedocs.io
- ✅ **GitHub Actions CI/CD** - Automated testing across Python 3.10, 3.11, 3.12
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
pytest

# Run specific categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```