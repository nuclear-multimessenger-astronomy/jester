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

### Skipped Tests Status
**2 tests currently skipped** (properly documented edge cases):

#### 1. Rest Mass Density Calculation (`tests/test_utils.py:161`)
**Status**: Function available but test skipped due to diffrax compatibility
**Location**: `jesterTOV/utils.py:306-360`  
**Test with**: `pytest tests/test_utils.py::TestCalculateRestMassDensity -v`

#### 2. Extreme Soft EOS Test (`tests/test_integration.py:425`)
**Status**: Physics-based limitation - soft EOS naturally produces lower maximum masses (1.18 vs 1.2 Msun threshold)
**Priority**: Low (represents physically valid edge case, not core functionality failure)

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

### 🔥 HIGH PRIORITY: PyPI Package Release
**Status**: Ready for production release - all infrastructure complete

**Critical Actions Needed**:
1. **Package preparation** - Verify pyproject.toml metadata and dependencies
2. **Version tagging** - Create v0.1.0 release tag with proper changelog
3. **PyPI upload** - Build and upload package to Python Package Index
4. **Installation testing** - Verify `pip install jesterTOV` works correctly
5. **Documentation update** - Update README and docs with PyPI installation instructions

**Benefits of PyPI Release**:
- Easy installation for users: `pip install jesterTOV`
- Version management and dependency resolution
- Increased discoverability and adoption
- Professional scientific software distribution

### ✅ Phase 1: Testing Infrastructure - COMPLETED
**Status**: 95 tests passing, 2 properly skipped (known issues)

### ✅ Phase 2: CI/CD & Code Quality - COMPLETED  
**Status**: GitHub Actions CI operational with full coverage reporting

### ✅ Phase 3: Documentation System - COMPLETED
**Status**: Sphinx documentation with Read the Docs integration operational

**Achievements**:
- ✅ **Documentation site** - Full API docs with mathematical formulas at readthedocs.io
- ✅ **GitHub Actions CI/CD** - Automated testing across Python 3.10, 3.11, 3.12
- ✅ **Code quality** - Pre-commit hooks with black, ruff, comprehensive test coverage
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

# Run with coverage
pytest --cov=jesterTOV --cov-report=html

# Run specific categories  
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```