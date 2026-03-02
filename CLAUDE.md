# CLAUDE.md

This file provides guidance to Claude Code when working with the JESTER repository.

## IMPORTANT GUIDELINES

**Testing Philosophy**: When tests fail, investigate root causes rather than modifying tests to pass.

**Examples**: In the examples, you will sometimes find `submit.sh` files for submitting the tests on a cluster. These might have hardcoded paths etc, but ignore those: these files are just intended as an example and they should not be judged so rigourously as the source code.

**Auto-update CLAUDE.md**: When code changes in infrastructure, make sure CLAUDE.md files in `jester`, and `jester/jesterTOV/inference` are updated to guide future Claude Code sessions.

**Documentation Style**: Write clear, concise documentation in full sentences as if by a human researcher. Avoid LLM-like verbosity.

**Documentation Maintenance**: When making changes to source code (adding/removing/renaming classes, functions, or modules), check if API reference documentation needs updating. Module overview pages in `docs/api/` should list all public classes/functions. See `docs/CLAUDE.md` for detailed documentation guidelines. In case a major refactoring is done, changing the layout of the repo, then we have to check the API references automatic docs building is up to date.

**Math Formatting in Docstrings**: All mathematical expressions in docstrings must use Sphinx/reStructuredText formatting for proper rendering in documentation:
- Use `:math:` role for inline math: `:math:`\Gamma(x)`
- Use `.. math::` directive for display equations
- Always use raw strings (`r"""`) for docstrings containing LaTeX to avoid Python escape sequence warnings
- Follow the style in `jesterTOV/eos/base.py` as the reference example

**File Operations**: Use proper tools (Write, Edit, Read) instead of bash heredocs or cat redirection.

**GitHub Issue Comments**: When posting comments on GitHub issues, always identify as "Claude" or "Claude Code" to make it clear the comment is AI-generated. Never post as if you were the human user. This maintains transparency about AI contributions to the project.

**Backwards compatibility**: There has not been a release yet, so don't worry about breaking changes for now. Focus on code quality, testing, and documentation over supporting legacy APIs!

**Extending JESTER**: When users mention working on adding new components, refer them to the relevant developer guide:
- **Adding a new EOS model**: See `docs/developer_guide/adding_new_eos.md` for complete checklist (EOS class, schema updates, factory registration, tests, documentation)
- **Adding a new TOV solver**: See `docs/developer_guide/adding_new_tov.md` for complete checklist (solver class, schema updates, factory registration, tests, documentation)
- **Adding a new likelihood**: See `docs/developer_guide/adding_new_likelihood.md` for complete checklist (likelihood class, data loading, schema updates, factory registration, tests, documentation)

These guides ensure all integration points are covered (configuration schema, transform factory, parameter validation, tests, and documentation).

---

## Current Status

### Multi-Sampler Architecture

Four sampler backends available for Bayesian inference:

**Production Ready:**
1. **FlowMC** (`type: "flowmc"`) - Normalizing flow-enhanced MCMC
   - Efficient for high-dimensional posteriors
   - Uses learned density model to guide sampling
   - Requires training + production phases

2. **BlackJAX SMC-RW** (`type: "smc-rw"`) - Sequential Monte Carlo with Random Walk kernel
   - **DEFAULT SAMPLER** for testing and lightweight runs
   - Gaussian Random Walk kernel with sigma adaptation
   - Target ESS: 0.9, requires ~10-30 MCMC steps per tempering level
   - Can run locally on laptop without GPU

3. **BlackJAX SMC-NUTS** (`type: "smc-nuts"`) - Sequential Monte Carlo with NUTS kernel
   - Production ready, well-tested
   - NUTS kernel with Hessian-based mass matrix adaptation
   - More efficient than RW for complex posteriors

**Experimental:**
4. **BlackJAX NS-AW** (`type: "blackjax-ns-aw"`) - Nested Sampling with Acceptance Walk
   - For model comparison and evidence estimation
   - Mimics bilby nested sampling setup
   - Needs additional type checking/fixes

**Example Configs Available:**
```bash
# Default lightweight config (runs on laptop)
examples/inference/smc_random_walk/chiEFT/config.yaml

# Other examples organized by sampler type:
examples/inference/flowmc/           # FlowMC examples
examples/inference/smc_random_walk/  # SMC-RW examples
examples/inference/ns/               # Nested sampling examples
examples/inference/spectral/         # Spectral decomposition examples
```

**Sampler Registry** (`jesterTOV/inference/samplers/jester_sampler.py`):
```python
SAMPLER_REGISTRY = {
    "flowmc": FlowMCSampler,
    "smc-rw": BlackJAXSMCRandomWalkSampler,
    "smc-nuts": BlackJAXSMCNUTSSampler,
    "blackjax-ns-aw": BlackJAXNSAWSampler,
}
```

---

## API Migration (January 2025 Refactoring)

**construct_family moved from EOS to TOV solver:**
```python
# OLD: eos_tuple = eos.construct_family(...)
# NEW:
from jesterTOV.tov import GRTOVSolver
eos_data = model.construct_eos(params)  # Returns EOSData NamedTuple
solver = GRTOVSolver()
family_data = solver.construct_family(eos_data, ndat=100, min_nsat=0.75)
# Access: family_data.masses, family_data.radii, family_data.lambdas
```

**E_sat is now required** (was fixed at -16.0):
- Add `E_sat = UniformPrior(-16.1, -15.9, parameter_names=["E_sat"])` to priors
- Applies to `metamodel` and `metamodel_cse` transforms only

**EOSData is NamedTuple** (8 fields):
- Access by name: `eos_data.ns`, `eos_data.ps`, etc.
- Do NOT unpack: ~~`ns, ps, hs, ... = eos_data`~~ (will fail - NamedTuple has 8 fields)
- Fields: `ns`, `ps`, `hs`, `es`, `dloge_dlogps`, `cs2`, `mu` (optional), `extra_constraints` (optional)

**Type ignore patterns for JAX:**

Common patterns required due to JAX tracing limitations:

```python
# 1. vmap batches scalar NamedTuple fields → arrays
masses: Float[Array, "n"] = solutions.M  # type: ignore[assignment]

# 2. Diffrax with throw=False guarantees ys populated (despite Optional type)
R = sol.ys[0][-1]  # type: ignore[index]

# 3. MetaModel guarantees mu populated (but base class has Optional)
mu: Float[Array, "n"] = eos_data.mu  # type: ignore[assignment]
# Note: This pattern suggests Interpolate_EOS_model may need restructuring

# 4. JAX array attribute access (jaxtyping doesn't understand traced attributes)
value = array.item()  # type: ignore[union-attr]
```

**NEVER use runtime assertions in JAX-traced code** - they fail during tracing. Use type ignore with explanatory comments instead.

---

## Project Overview

**JESTER** (**J**ax-based **E**o**S** and **T**ov solv**ER**) is a scientific computing library for neutron star physics using JAX for hardware acceleration and automatic differentiation.

### Core Modules

**jesterTOV/eos/** - Equation of State Models
- Base class: `Interpolate_EOS_model` (abstract base class)
- Available EOS models:
  1. **MetaModel_EOS_model** (`eos/metamodel/base.py`)
     - Nuclear empirical parameter (NEP) based EOS
     - Reference: Margueron et al. (PRD 103, 045803, 2021)
     - Required parameters: 9 NEPs (E_sat, K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym)
     - Combines realistic crust (BPS, DH, DH_fixed, SLy) with core meta-model
  2. **MetaModel_with_CSE_EOS_model** (`eos/metamodel/metamodel_CSE.py`)
     - MetaModel with Crust-core-Saturation Extension
     - Parameters: 9 NEPs + nbreak + nb_CSE grid parameters (typically 4-8)
  3. **SpectralDecomposition_EOS_model** (`eos/spectral/spectral_decomposition.py`)
     - Spectral representation (Lindblom 2010, PRD 82, 103011)
     - Exactly matches LALSuite implementation
     - Parameters: 4 gamma coefficients (gamma_0, gamma_1, gamma_2, gamma_3)
- Each EOS class implements:
  - `construct_eos(params: dict) -> EOSData` - Build EOS from parameters
  - `get_required_parameters() -> list[str]` - List required parameter names
- Returns JAX-compatible `EOSData` NamedTuple with 8 fields

**jesterTOV/tov/** - TOV Equation Solvers
- Base class: `TOVSolverBase` (abstract base class)
- Available solvers:
  1. **GRTOVSolver** (`tov/gr.py`) - General Relativity
     - Standard TOV equations, no additional parameters
  2. **AnisotropyTOVSolver** (`tov/anisotropy.py`) - Beyond-GR modifications
     - Phenomenological sigma terms (Yagi & Yunes 2013)
     - Multiple correction models: Bowers-Liang, Doneva-Yazadjiev, Herrera-Barreto, Post-Newtonian
     - Required parameters: coupling constants (lambda_BL, lambda_DY, etc.)
  3. **ScalarTensorTOVSolver** (`tov/scalar_tensor.py`) - Scalar-tensor gravity
     - Jordan frame implementation (Brown 2023, ApJ 958 125)
     - Required parameters: beta_ST, phi_c, nu_c
- Key methods:
  - `solve(eos_data, pc, **kwargs) -> TOVSolution` - Single star solution
  - `construct_family(eos_data, ndat, min_nsat, **kwargs) -> FamilyData` - M-R-Λ family curves
  - `get_required_parameters() -> list[str]` - List additional parameters
- Uses Diffrax library with Dormand-Prince 5th order integrator (Dopri5)
- Computes Love number k2 for tidal deformability

**jesterTOV/tov/data_classes.py** - JAX Dataclasses
- `EOSData` - EOS quantities (ns, ps, hs, es, cs2, dloge_dlogps, mu, extra_constraints)
- `TOVSolution` - Single star solution (M, R, k2)
- `FamilyData` - M-R-Λ family curves (log10pcs, masses, radii, lambdas)
- All use NamedTuple for automatic JAX pytree compatibility

**jesterTOV/inference/** - Bayesian Inference System
- Modular, configuration-driven architecture with multiple sampler backends
- Single unified `JesterTransform` class coordinates all EOS × TOV combinations
- Automatic parameter validation before sampling (fail-fast with clear errors)
- See `jesterTOV/inference/CLAUDE.md` for detailed architecture
- Structure:
  - `config/` - YAML parsing, Pydantic validation
  - `priors/` - Bilby-style prior specification
  - `transforms/` - Unified JesterTransform for EOS → M-R-Λ
  - `likelihoods/` - Observational constraints (GW, NICER, Radio, ChiEFT, REX, etc.)
  - `flows/` - Normalizing flow utilities for GW likelihoods; includes `bilby_extract.py` for extracting posteriors from bilby HDF5 results
  - `data/` - Data loading and caching
  - `samplers/` - FlowMC, SMC (RW/NUTS), Nested Sampling backends
  - `run_inference.py` - Main orchestration
  - `result.py` - HDF5 result storage

**jesterTOV/utils.py** - Utilities
- Physical constants (c_km, G_km, Msun_km, etc.)
- Unit conversions (geometric ↔ physical units)
- Utility functions

### Key Design Principles
- **JAX-first**: Hardware acceleration with automatic differentiation
  - Avoid Python `if` statements on traced values (use `jnp.where()`)
  - Avoid `float()` casts on traced arrays
  - Use NamedTuple for JAX pytree compatibility (not @dataclass)
  - All physics calculations traced and JIT-compilable
- **Geometric units**: All internal physics calculations use geometric units (c=G=1)
  - Conversions to physical units (M☉, km) only at final output
- **Type safety**: Comprehensive type hints with `jaxtyping` for arrays
  - `Float[Array, "n"]` for shaped JAX arrays
  - Full pyright type checking enabled
- **64-bit precision**: `jax.config.update("jax_enable_x64", True)` by default
- **Modular architecture**: Clean separation of concerns
  - EOS models independent of TOV solvers
  - TOV solvers independent of specific EOS implementations
  - Inference system orchestrates via JesterTransform
  - **No factory methods needed** - JesterTransform handles all combinations
- **ODE Integration**: Diffrax library with adaptive step size (Dopri5 + PIDController)
  - Graceful failure handling with `throw=False`

---

## Development Commands

### Always Use `uv`
```bash
# Run Python commands
uv run <command>

# Install dependencies
uv pip install <package>
```

### CLI Tools

```bash
# Run inference
uv run run_jester_inference config.yaml

# Extract GW posterior samples from a bilby result file (no bilby install needed)
uv run jester_extract_gw_posterior_bilby result.hdf5 --output posterior.npz
```

### Check PR Status

In case we mention we are working on a PR and, e.g., tests fail for it, check it out:

```bash
# View PR status and CI checks
gh pr view <PR_NUMBER> --json statusCheckRollup

# Download CI logs for specific job
gh api repos/nuclear-multimessenger-astronomy/jester/actions/jobs/<JOB_ID>/logs > logs.txt
```

### Code Quality
Run tests, pyright, and pre-commit checks before deciding new changes are ready for review by humans
```bash
# Run tests with verbose output
uv run pytest -v tests/

# Run pyright
uv run pyright

# Run pre-commit
uv run pre-commit run --all-files
```

### Testing
```bash
# Run all tests (excluding slow tests - default for CI)
uv run pytest tests/ -m "not slow"

# Run specific test file
uv run pytest tests/test_inference/test_config.py

# Run E2E tests only
uv run pytest tests/test_inference/test_e2e/ -v

# Run all tests including slow (nightly builds)
uv run pytest tests/
```

**Test Markers**:
- `@pytest.mark.slow` - Slow tests, skipped in regular CI (run in nightly)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end pipeline tests
- `@pytest.mark.blackjax` - Tests requiring BlackJAX samplers

**End-to-End Tests** (`tests/test_inference/test_e2e/`):
- Tests the full inference pipeline: config → sampler.sample() → SamplerOutput
- Uses lightweight hyperparameters for fast execution (<2 min per test)
- Covers all samplers: SMC-RW, FlowMC, BlackJAX NS-AW
- Catches integration bugs that unit tests miss

**CI/CD Configuration**:
- **Regular CI** (`.github/workflows/ci.yml`): Runs on every PR, skips `@slow` tests
- **Nightly CI** (`.github/workflows/nightly.yml`): Runs daily at 2 AM UTC, includes all E2E tests

**CI/CD Test Dependencies**:
- LaTeX packages (texlive-latex-base, texlive-latex-extra, dvipng, cm-super) for matplotlib plotting tests
- IPython (added to `[tests]` dependencies) required by fastprogress (BlackJAX transitive dependency)

## Code Quality Standards

### Documentation
```bash
# Build docs locally
uv pip install -e ".[dev]"
uv run sphinx-build docs docs/_build/html
open docs/_build/html/index.html

# Strict mode (same as CI)
uv run sphinx-build -W --keep-going docs docs/_build/html
```

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
├── config/                      # YAML parsing, Pydantic validation
│   ├── schema.py                # Thin aggregator: InferenceConfig + re-exports
│   ├── schemas/                 # Domain-specific config sub-modules (eos, tov, likelihoods, samplers)
│   ├── parser.py                # YAML loading functions
│   └── generate_yaml_reference.py  # Auto-generate documentation
├── priors/                      # Prior specification system
│   └── parser.py                # Parse .prior files (bilby-style Python)
├── flows/                       # Normalizing flow utilities for GW likelihoods
│   ├── bilby_extract.py         # Extract GW posteriors from bilby HDF5 (+ CLI entry point)
│   ├── config.py                # FlowTrainingConfig Pydantic model
│   ├── train_flow.py            # Flow training entry point
│   └── flow.py                  # Flow model definition
├── transforms/                  # EOS → M-R-Λ transformation
│   ├── transform.py             # JesterTransform (unified for all EOS×TOV)
│   └── __init__.py              # Exports
├── likelihoods/                 # Observational constraints
│   ├── gw.py                    # Gravitational wave (GW170817, GW190425)
│   ├── nicer.py                 # X-ray timing (J0030, J0740, B0437)
│   ├── radio.py                 # Radio pulsar timing
│   ├── chieft.py                # Chiral EFT low-density constraints
│   ├── rex.py                   # PREX/CREX neutron skin experiments
│   ├── combined.py              # CombinedLikelihood wrapper
│   ├── factory.py               # Likelihood creation from config
│   └── constraints.py           # Physical constraints (EOS/TOV/Gamma)
├── data/                        # Data loading and preprocessing
│   ├── __init__.py              # Data loading functions
│   └── paths.py                 # Path management
├── samplers/                    # Sampler implementations
│   ├── jester_sampler.py        # Base JesterSampler + SAMPLER_REGISTRY
│   ├── flowmc.py                # FlowMC backend
│   └── blackjax/                # BlackJAX backends
│       ├── base.py              # BlackjaxSampler base class
│       ├── smc/                 # Sequential Monte Carlo framework
│       │   ├── base.py          # BlackjaxSMCSampler
│       │   ├── random_walk.py   # SMC-RW (production ready)
│       │   └── nuts.py          # SMC-NUTS (production ready)
│       └── nested_sampling/     # Nested sampling
│           └── ns_aw.py         # NS with Acceptance Walk (experimental)
├── base/                        # Base classes (from Jim v0.2.0)
│   ├── likelihood.py            # LikelihoodBase ABC
│   ├── prior.py                 # Prior, CombinePrior, UniformPrior
│   └── transform.py             # NtoMTransform, BijectiveTransform
├── run_inference.py             # Main orchestration script
├── result.py                    # InferenceResult (HDF5 storage)
└── cli.py                       # Command-line interface
```

### Execution Flow
```
config.yaml + prior.prior
    ↓
parse_config() → InferenceConfig (Pydantic validated)
    ↓
parse_prior_file() → CombinePrior object
    ↓
JesterTransform.from_config(config.eos, config.tov)
  ├─ Instantiate EOS (MetaModel/MetaModelCSE/Spectral)
  └─ Instantiate TOV solver (GR/Post/ScalarTensor)
    ↓
Validate parameters
  ├─ Check all required EOS params in prior → raise error if missing
  └─ Check all required TOV params in prior → warn if unused
    ↓
prepare_gw_flows(config, outdir)
  └─ For any GW event with from_bilby_result: extract NPZ + train/cache flow
    ↓
Load data (NICER, GW posteriors, ChiEFT, etc.)
    ↓
create_likelihood() → CombinedLikelihood
    ↓
create_sampler() → Sampler from SAMPLER_REGISTRY
    ↓
sampler.sample(prng_key) → SamplerOutput
  ├─ samples: dict[str, Array]
  ├─ log_prob: Array
  └─ metadata: dict[str, Any]
    ↓
InferenceResult.from_sampler() → HDF5 format
  ├─ posterior (parameters + derived EOS quantities)
  ├─ metadata (config + run statistics)
  └─ histories (diagnostics)
```

---

## Known Issues & Workarounds

### Open Issues
- **UniformPrior boundaries**: `log_prob()` at exact boundaries causes errors
  - NaN at `xmin` due to `log(0)` in LogitTransform
  - ZeroDivisionError at `xmax`
  - **Workaround**: Use values strictly inside boundaries (e.g., add small epsilon)
  - **Fix needed**: Add numerical guards in LogitTransform (e.g., clamp to [epsilon, 1-epsilon])

- **TOV solver max_steps**: Some stiff EOS configurations hit Diffrax solver iteration limits
  - Affects EOS with rapid pressure changes or extreme parameters
  - **Workaround**: Increase `max_steps` in TOV solver config or adjust EOS parameter ranges
  - **Investigation needed**:
    - When exactly does this happen? Which EOS parameter regimes?
    - Is it a numerical stiffness issue or physical instability?
    - Should we use different ODE solver or adaptive tolerances?

- **Spectral EOS gamma bounds**: Some parameter combinations violate causality
  - `extra_constraints` field in EOSData tracks gamma bound violations
  - **Current handling**: ConstraintGammaLikelihood applies penalty
  - **Note**: This is physical, not a bug - spectral parameterization can be non-causal

---

## Release Workflow

When we are ready to make a new release, here are some steps:

```bash
# 1. Feature branch for version bump
git checkout -b release/v0.x.x

# 2. Update version in pyproject.toml and docs/conf.py
# 3. Build and verify
uv build

# 4. Create PR to main
# 5. After merge, tag the commit
git tag v0.x.x
git push origin v0.x.x

# 6. PyPI publishing is NOT possible - jester depends on a specific fork of blackjax
# (https://github.com/handley-lab/blackjax) which cannot be published to PyPI.
# Users install directly from the GitHub repository via git clone.
```
