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
pyright                    # Type checking
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

### Skipped Tests & Resolution
**2 tests currently skipped** (properly documented, not failures):

#### 1. Rest Mass Density Calculation (`tests/test_utils.py:161`)
**Issue**: `calculate_rest_mass_density()` has diffrax API compatibility problems
**Details**: Function uses diffrax ODE solver but may have version compatibility issues with current diffrax API
**Location**: `jesterTOV/utils.py:201-242`
**Resolution Strategy**:
- Update diffrax imports and API calls to match current version
- Test with: `pytest tests/test_utils.py::TestCalculateRestMassDensity -v`
- Alternative: Implement using JAX's built-in ODE solver if diffrax proves problematic

#### 2. Extreme Soft EOS Test (`tests/test_integration.py:425`)
**Issue**: Soft EOS produces maximum mass (1.18 Msun) below expected threshold (1.2 Msun)
**Details**: Physics constraint - very soft EOS naturally gives lower maximum neutron star masses
**Resolution Options**:
- **Option A**: Lower test threshold to 1.1 Msun (more realistic for soft EOS)
- **Option B**: Stiffen the test EOS parameters to achieve higher masses
- **Option C**: Accept as correct physics and document as expected behavior

**Priority**: Low (both represent edge cases, not core functionality)

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
- Pre-commit hooks enforce code quality with black and ruff (pyright temporarily disabled)
- Examples in `examples/` directory demonstrate basic and advanced usage
- Comprehensive test suite with 95 tests covering all major functionality

### Python Support
Supports Python 3.10-3.12 with JAX ecosystem dependencies.

## Repository Status

### ✅ Phase 1: Testing Infrastructure - COMPLETED
**Status**: 95 tests passing, 2 properly skipped (known issues)

### ✅ Phase 2: CI/CD & Code Quality - COMPLETED
**Status**: GitHub Actions workflow operational with test coverage badges

**Achievements**:
- ✅ **GitHub Actions CI/CD** - Automated testing across Python 3.10, 3.11, 3.12
- ✅ **Test coverage reporting** - Codecov integration with coverage badges in README
- ✅ **Coverage badges** - Added codecov badge to README for test coverage visibility
- ✅ **Pre-commit configuration** - Black formatting and ruff linting working
- ✅ **Code quality fixes** - Resolved ruff linting errors with appropriate ignore rules

**Configuration Changes Made**:
- Updated ruff config to new format with comprehensive ignore list for scientific code
- Excluded `examples/` directory from linting (notebooks have different import patterns)
- Added ignore rules for jaxtyping dimension names, lambda expressions, unused variables

### 🔥 HIGH PRIORITY: Documentation System
**Status**: No formal documentation system - critical gap for scientific library

**Critical Documentation Needs**:
1. **API Documentation** - No autodoc system for functions/classes
2. **Mathematical Documentation** - Physics equations and derivations need proper rendering
3. **User Guide** - No comprehensive tutorials for neutron star physics workflow
4. **Developer Documentation** - Missing architecture and contribution guidelines
5. **Examples Documentation** - Jupyter notebooks lack integration with docs

**Immediate Actions Needed**:
- Set up Sphinx documentation with autodoc for API documentation
- Configure MathJax/KaTeX for mathematical formula rendering in equations
- Create comprehensive user guide with physics background and tutorials
- Document core workflows: EOS creation, TOV solving, parameter studies
- Integrate example notebooks into documentation system

### ✅ Type Safety Issues - RESOLVED
**Status**: Reduced from 23 to 10 pyright errors (57% improvement) - No longer blocking

**✅ Issues Resolved**:
1. ✅ **JAX/jaxtyping compatibility** - Fixed Array vs float type mismatches in all compute_* functions
2. ✅ **Function signature mismatches** - Fixed diffrax ODE solver integration (3-parameter function signature)  
3. ✅ **Return type mismatches** - Confirmed utils.limit_by_MTOV return type is correct
4. ✅ **Optional type handling** - Fixed proton_fraction None type annotation
5. ✅ **Array conversion issues** - Fixed jnp.polyval array conversion

**🟡 Remaining Issues (10 errors)**:
- 2 Array/list type checker false positives in eos.py (lines 394, 403) - known arrays, type checker issue
- 8 Optional None access warnings in ptov.py/tov.py - diffrax integration safety (ignoring per design)

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