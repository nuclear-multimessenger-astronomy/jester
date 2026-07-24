# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important developer guidelines

- You do not know everything about samplers. Instead of just doing something that "seems right", please ask more information about samplers and best practices. We can provide src code. *Better to ask for help than to make wrong assumptions and write sloppy code!*
- **blackjax**: For this, the src code is available at `/Users/Woute029/Documents/Code/projects/jester_review/blackjax`: use this to understand how to properly use blackjax samplers and best practices!
- **Postprocessing plot functions**: Every `make_*` plot function call in `generate_all_plots` (in `postprocessing/postprocessing.py`) **must** be wrapped in a `try/except Exception` block that logs the error and continues. This ensures a missing LaTeX installation or any other rendering failure does not abort the entire postprocessing run. The pattern is:
  ```python
  try:
      make_my_plot(...)
  except Exception as e:
      logger.error(f"Failed to create my plot: {e}")
      logger.warning("Continuing with other plots...")
  ```

## Module Overview

The `jesterTOV/inference/` module provides Bayesian inference for constraining neutron star equation of state (EOS) parameters using multi-messenger observations. It implements a modular, configuration-driven architecture with normalizing flow-enhanced MCMC sampling.

### Key Concepts

**Transforms**: Convert parameter spaces
- Sample transforms: Applied during sampling with Jacobian (bijective)
- Likelihood transforms: Applied before likelihood evaluation (N-to-M)
- JESTER uses single unified `JesterTransform` class: NEP → M-R-Λ via EOS + TOV
  - EOS classes know their required parameters
  - TOV solvers know their required parameters
  - Transform coordinates the full pipeline and validates parameters

**Priors**: Bilby-style Python syntax in `.prior` files
```python
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
```

**Samplers**: Five backends available
- `type: "flowmc"` - Flow-enhanced MCMC (production ready)
  - Normalizing flow guidance for efficient sampling
  - Training + production phases
- `type: "smc-rw"` - Sequential Monte Carlo with Random Walk kernel (production ready, **DEFAULT**)
  - Gaussian Random Walk with sigma adaptation
  - Target ESS: 0.9, ~10-30 MCMC steps per tempering level
  - Optional `adaptive_step_size` targets a literal acceptance rate (`target_acceptance_rate`,
    default 0.234) instead of the fixed `random_walk_sigma` throughout; see
    "Adaptive step size (SMC-RW)" below
- `type: "smc-nuts"` - Sequential Monte Carlo with NUTS kernel (production ready)
  - NUTS kernel with Hessian-based mass matrix adaptation
  - More efficient for complex posteriors
- `type: "smc-partial-posteriors-rw"` - SMC on the "path of partial posteriors" (IBIS; data tempering, not lambda tempering)
  - Tempers by number of GW events included, not by inverse-temperature λ; requires at least one `gw` likelihood event
  - The MCMC move in each sub-step is built to target the *old* (pre-substep) mask rather than the new one -- the "resample-move" construction (Gilks & Berzuini 2001), also used by blackjax's own `smc.tempered.build_kernel` and by Chopin's `particles` reference implementation -- which makes the per-substep log-evidence increment exact at any mask-jump size, not just for small jumps; see `_build_jitted_substep_fn`'s docstring in `samplers/blackjax/smc/partial_posteriors.py`. Each event is still ramped in over an adaptive, uncapped ESS-targeting bisection (reusing `smc-rw`'s `target_ess`/`ess_solver`/`dichotomy` machinery), now purely for MCMC-mixing/proposal-adaptation quality rather than evidence correctness
  - Config is split into two levels: the top level (`event_order`, `warm_start_from`, `cadence`) only orchestrates event assimilation; the nested `inner` block (`InnerSMCRandomWalkConfig`: `n_particles`, `n_mcmc_steps`, `target_ess`, `random_walk_sigma`) fully specifies the adaptive SMC-RW loop used to ramp in each event group
  - `cadence` (`int | list[int] | Literal["auto", "automatic"]`, default `1`) groups the not-yet-assimilated events into data-tempering steps of that many events each, instead of one at a time; a list gives explicit group sizes and must sum to exactly the number of new events (checked at sample time in `_compute_event_groups`, after `warm_start_from` is resolved). All events within a group share one mask fraction, bisected jointly — the group's combined log-likelihood is still linear in that shared fraction, so the same ESS-targeting solver applies unmodified
  - `cadence: "auto"` builds groups dynamically instead of a fixed schedule, based on the "full-jump ESS predictor" (`dev/predictors/full_jump_ess_predictor.tex` in the parent development workspace): not-yet-assimilated events are added to a pending queue one at a time, and after each addition `_predict_full_jump_ess` (`partial_posteriors.py`) computes the ESS that jumping the queue's combined mask straight from 0 to 1 would produce against the *current* particle cloud — a read-only importance-weighting diagnostic (`blackjax.smc.ess.log_ess`), never an accepted sampling step. This is exactly the same quantity `ess_solver`/`dichotomy` already evaluate as their first probe at the start of every sub-step bisection, just run standalone before committing to an update. While that ESS stays at or above `auto_ess_threshold`, the event is "not surprising" and stays queued; once it drops below threshold (or the last configured event is reached), the whole queue is assimilated together through the normal, unbiased sub-step bisection — same code path as a fixed-size group. `auto_ess_threshold` (`float | None`, default `None` → falls back to `inner.target_ess`) is only meaningful (and only accepted by config validation) when `cadence == "auto"`
  - Because auto-cadence groups depend on the live particle cloud, they can't be precomputed like fixed-cadence groups; `sample()`'s main loop calls a local `_next_auto_group` closure once per step instead of iterating a precomputed `event_groups` list
  - `plot_diagnostics()` produces the overall across-steps plot plus one per-step sub-step diagnostics plot (mask fraction/ESS/acceptance vs. sub-step) under `outdir/substep_diagnostics/`, plus (auto-cadence only) `outdir/auto_cadence_ess.png` — see below. `metadata["event_groups"]` lists the event names in each processed group. Logging and filenames refer to each group by a short, sequential `batch_<NN>` key (`NN` a 1-indexed counter that increases by exactly 1 for every batch processed, e.g. `batch_01`, `batch_02`, `batch_03`, ... — *not* the absolute index of the group's first event in `event_order`, which used to make `NN` jump by however many events landed in each batch, e.g. `batch_00`, `batch_01`, `batch_04`, `batch_08`, ...) rather than the full `"+"`-joined event-name string — with many events, that string can get arbitrarily long and blow past OS filename limits (`OSError: File name too long`). `metadata["batch_to_sources"]` (`dict[str, list[str]]`, also persisted to the saved `InferenceResult`'s metadata as a JSON string) maps each `batch_<NN>` key to its constituent event names, so the mapping is recoverable without parsing logs. `metadata["n_batches"]` is the cumulative batch count (`n_prev_batches` from a warm-start source, if any, plus the batches processed this run); when `warm_start_from` is set, `_load_warm_start` reads the source result's own `n_batches` back out so this run's numbering continues from there (`batch_<n_prev_batches+1>`, ...) instead of colliding with the source's `batch_01`, `batch_02`, ... filenames if both land under the same `outdir`
  - Auto-cadence only: `metadata["auto_cadence_ess_history"]`/`metadata["auto_cadence_triggered_history"]` (also persisted under `histories/` in the saved `InferenceResult`) record, one entry per not-yet-assimilated event considered by `_next_auto_group`, the full-jump ESS `_predict_full_jump_ess` computed after adding it to the pending queue and whether that check triggered the queue to be assimilated (`ess < auto_ess_threshold`) vs. left it queued. `plot_diagnostics()` turns this into `auto_cadence_ess.png`: one-shot ESS vs. "event added to queue" index, a dashed line at `auto_ess_threshold`, blue dots for deferred (queued) checks and red downward triangles where an update was triggered
  - `warm_start_from` works from *any* sampler's saved result (not just a previous partial-posteriors run): "already covered" GW events are derived from the source run's saved config (`InferenceResult.config_dict["likelihoods"]`) via `_extract_gw_event_order`, and its always-on (non-GW) likelihoods must exactly match this run's (`_canonical_always_on_signature`, checked in `_check_always_on_likelihoods_match`) — `create_sampler` must be given `likelihood_configs=config.likelihoods` for that check to run (otherwise it only warns). The source run may have zero GW events (e.g. radio/ChiEFT-only) — an empty event list is a trivial strict prefix, so its posterior is used purely to seed initial particles and every configured GW event is still assimilated from scratch. The always-on-likelihoods-must-match constraint still applies though: warm-starting from a run whose non-GW likelihood *set* differs from this run's (e.g. adding a new always-on likelihood between stages) is not yet supported and raises `ValueError`.
  - `save_intermediate_results` (`bool`, default `True`) saves a full `InferenceResult` HDF5 (posterior + derived EOS quantities from the TOV solver, via `add_eos_from_transform`) after every data-tempering step, to `outdir/substep_results/results_<n>.h5` where `<n>` is the 1-indexed absolute position of the step's first event within the full `event_order` (stable across warm-started runs). Lets users inspect how the posterior evolves batch-by-batch (e.g. which events are most informative). Implementation: `_save_intermediate_result` temporarily swaps `self._particles_flat`/`self._weights`/`self.final_state`/`self.metadata` to a "so far" snapshot (built by `_build_metadata`, also reused for the real end-of-run metadata) so it can drive `InferenceResult.from_sampler` through the normal `get_sampler_output()` path, then restores them in a `finally` block. Needs the full `InferenceConfig` and outdir, which this sampler doesn't otherwise have — `run_inference.py` wires these in via `sampler.configure_intermediate_saving(...)` right after `create_sampler`, before `sample()` runs.
- `type: "blackjax-ns-aw"` - Nested Sampling with Acceptance Walk (experimental)
  - For model comparison and evidence estimation
  - Needs additional testing/fixes

### Changing Default Values for Likelihood Parameters

When a user asks to change a default value (e.g. `penalty_value`), update **all** of the following:

1. **`likelihoods/<name>.py`** — `__init__` signature default(s) and docstring(s)
2. **`config/schemas/likelihoods.py`** — `Field(default=...)` and any docstring YAML example
3. **`docs/inference/yaml_reference.md`** — update the relevant field entry by hand
4. **`examples/inference/**/config.yaml`** — remove or update the now-redundant explicit value
5. **`tests/test_inference/test_config.py`** — update any assertion on the old default

The factory (`likelihoods/factory.py`) passes `config.<field>` through, so no change needed there.

### GW likelihood batching (`StackedGWLikelihood`)

`create_combined_likelihood` (`likelihoods/factory.py`) builds **one**
`StackedGWLikelihood` (`likelihoods/gw.py`) for a `GWLikelihoodConfig`'s entire
`events` list, not one `GWLikelihood` per event. This replaces a plain Python
`for`/list-comprehension over per-event `GWLikelihood` objects in
`CombinedLikelihood.evaluate`, which JAX fully unrolls at trace time —
compile time then grows superlinearly with the number of events, and once
this sits inside the outer particle `vmap` blackjax already applies, a fully
vectorized event axis multiplies memory by `n_particles * n_events` the same
way a fully vectorized mass-sample axis multiplies it by
`n_particles * N_masses_evaluation` (see `N_masses_batch_size`'s docstring on
`GWLikelihood`, and `dev/FINDINGS.md` — Parts 1 and 4 — in the parent
development workspace for the full benchmarks behind both).

**How it works:** all events must share the same flow architecture
(`flow_type`, `nn_width`, `nn_depth`, ..., dimensionality, standardization
method — everything that determines the pytree *structure* of the trained
weights, checked via `_flow_architecture_signature` and enforced eagerly at
construction with a clear `ValueError` naming the offending event, not a
`jax.tree_util` error buried inside `evaluate()`). Given that, each event's
`flowjax` weights are split via `eqx.partition(flow.flow, eqx.is_array)` and
stacked along a new leading "event" axis with
`jax.tree_util.tree_map(jnp.stack, ...)`; `evaluate()` then runs a single
`jax.lax.map` over that stacked axis (batch size: `event_batch_size`), with
the existing per-event `jax.lax.map` over mass samples (batch size:
`N_masses_batch_size`) nested inside. Numerically this is defined to equal
`sum(GWLikelihood(event).evaluate(params) for event in events)` — a batching
change, not a different likelihood (see
`tests/test_inference/test_likelihoods.py::TestStackedGWLikelihood` and
`TestCombinedLikelihoodFactory::test_create_gw_likelihood_builds_stacked_gw_likelihood`
for the equivalence tests against per-event `GWLikelihood`, the latter using
the real shipped GW170817/GW190425 presets).

**Both `N_masses_batch_size` and `event_batch_size` default to `1`** (a plain
scan over mass samples / events respectively) — this is the safe default for
production SMC/FlowMC runs with many particles and/or many events. Setting
either equal to its total (`N_masses_evaluation`, or the number of events)
degenerates `jax.lax.map` to a plain `jax.vmap` with zero chunking benefit
(see `jax/_src/lax/control_flow/loops.py::map`'s `batch_size` semantics) —
only raise these for faster standalone (non-vmapped) evaluations. The true
concurrent width of flow evaluations during sampling is roughly
`n_particles * event_batch_size * N_masses_batch_size`.

If events genuinely need different flow architectures, they cannot go
through `StackedGWLikelihood` together — no automatic fallback to per-event
`GWLikelihood` exists (this would need to be added, e.g. grouping by
architecture signature, if that scenario becomes common; every flow shipped
with jester currently uses the same architecture, since `train_flow.py`
always trains with the same defaults). `GWLikelihood` itself is unchanged
and still directly usable/importable for a single event or outside the
config system.

### Inference Documentation
- `docs/inference_index.md` - Navigation hub
- `docs/inference_quickstart.md` - Quick start guide
- `docs/inference.md` - Complete reference
- `docs/inference/yaml_reference.md` - Hand-maintained YAML reference

Full details in `jesterTOV/inference/CLAUDE.md`

---

## Architecture

### Testing inference locally

The example in `examples/inference/smc_random_walk/chiEFT` finishes in less than 1 minute: good for testing some changes locally for inference/postprocessing.

### Modular Structure

```
jesterTOV/inference/
├── config/              # YAML parsing and Pydantic validation
│   ├── schema.py        # Thin aggregator: InferenceConfig + re-exports
│   └── schemas/         # Domain-specific config sub-modules
│       ├── eos.py       #   BaseEOSConfig + concrete EOS configs
│       ├── tov.py       #   BaseTOVConfig + GRTOVConfig
│       ├── likelihoods.py #  All likelihood configs (incl. GWEventConfig)
│       └── samplers.py  #   All sampler configs
│   └── parser.py        # YAML loading
├── priors/              # Prior specification system
│   └── parser.py        # Parse .prior files (bilby-style Python format)
├── flows/               # Normalizing flow utilities for GW likelihoods
│   ├── bilby_extract.py # Extract GW posteriors from bilby HDF5 results (+ CLI)
│   ├── config.py        # FlowTrainingConfig Pydantic model
│   ├── train_flow.py    # Flow training entry point
│   ├── flow.py          # Flow model definition
│   └── __init__.py      # Exports Flow, load_model, extract_gw_posterior_from_bilby
├── transforms/          # Unified transform system
│   ├── transform.py     # JesterTransform - single class for all EOS+TOV combinations
│   └── __init__.py      # Exports JesterTransform
├── likelihoods/         # Observational constraints
│   ├── gw.py            # Gravitational wave events (GW170817, GW190425)
│   ├── nicer.py         # X-ray timing (J0030, J0740, B0437)
│   ├── radio.py         # Radio pulsar timing (FIDUCEO/FIDUCEO2)
│   ├── chieft.py        # Chiral EFT low-density constraints
│   ├── rex.py           # PREX/CREX neutron skin experiments
│   ├── constraints.py   # Physical constraints (EOS/TOV/Gamma)
│   ├── combined.py      # CombinedLikelihood wrapper
│   └── factory.py       # Likelihood creation from config
├── data/                # Data loading and preprocessing
│   ├── __init__.py      # Data loading functions (NICER, GW posteriors, ChiEFT)
│   └── paths.py         # Path management and Zenodo caching
├── samplers/            # Sampler implementations
│   ├── jester_sampler.py  # Base JesterSampler + SAMPLER_REGISTRY
│   ├── flowmc.py        # FlowMC backend
│   └── blackjax/        # BlackJAX backends
│       ├── base.py      # BlackjaxSampler base class
│       ├── smc/         # Sequential Monte Carlo framework
│       │   ├── base.py  # BlackjaxSMCSampler (lambda-tempering)
│       │   ├── random_walk.py  # SMC-RW (production ready)
│       │   ├── nuts.py  # SMC-NUTS (production ready)
│       │   └── partial_posteriors.py  # SMC-Partial-Posteriors-RW (data tempering / IBIS)
│       └── nested_sampling/
│           └── ns_aw.py # NS with Acceptance Walk (experimental)
├── base/                # Base classes (copied from Jim v0.2.0)
│   ├── likelihood.py    # LikelihoodBase ABC
│   ├── prior.py         # Prior, CombinePrior, UniformPrior
│   └── transform.py     # NtoMTransform, BijectiveTransform
├── run_inference.py     # Main entry point
└── cli.py               # Command-line interface
```

### Execution Flow

```
config.yaml + prior.prior
    ↓
parse_config() → InferenceConfig (Pydantic validated)
    ↓
parse_prior_file() → ParsedPrior(prior: CombinePrior, fixed_params: dict)
    ↓
JesterTransform.from_config(config.eos, config.tov)
  ├─ Instantiate EOS (MetaModel/MetaModelCSE/Spectral)
  └─ Instantiate TOV solver (GR/Anisotropy)
    ↓
Validate parameters
  ├─ Check all required EOS params in prior → raise error if missing
  └─ Check all required TOV params in prior → warn if unused
    ↓
prepare_gw_flows(config, outdir)   # no-op unless from_bilby_result events exist
  ├─ Extract NPZ from bilby HDF5 (jester_extract_gw_posterior_bilby)
  ├─ Train normalizing flow (FlowTrainingConfig + train_flow_from_config)
  ├─ Hash-based cache: skip training if flow unchanged (flow_config_hash.json)
  └─ Return updated config with resolved nf_model_dir for each event
    ↓
Load data (NICER, GW posteriors, ChiEFT, etc.)
  ├─ Cache downloads from Zenodo
  └─ Construct KDEs for GW posteriors
    ↓
create_likelihood() → CombinedLikelihood
  ├─ Individual likelihoods from factory
  └─ Equal weighting (1/N_likelihoods per likelihood)
    ↓
create_sampler() → Sampler from SAMPLER_REGISTRY
  ├─ FlowMCSampler (flowmc)
  ├─ BlackJAXSMCRandomWalkSampler (smc-rw)
  ├─ BlackJAXSMCNUTSSampler (smc-nuts)
  └─ BlackJAXNSAWSampler (blackjax-ns-aw)
    ↓
sampler.sample(prng_key) → SamplerOutput
  ├─ samples: dict[str, Array]
  ├─ log_prob: Array
  └─ metadata: dict[str, Any] (ESS, weights, acceptance rates, etc.)
    ↓
InferenceResult.from_sampler() → HDF5 format
  ├─ posterior (parameters + derived EOS quantities)
  ├─ metadata (config + run statistics)
  └─ histories (diagnostics: log_prob, ESS, etc.)
    ↓
Save to outdir/{result_id}.h5
```

### EOS/TOV Architecture

**Key Design Principle**: Modular separation of concerns

1. **EOS Classes** (`jesterTOV/eos/`):
   - Base: `Interpolate_EOS_model` (abstract base class)
   - Available models:
     - `MetaModel_EOS_model` - Nuclear empirical parameters (9 NEPs)
       - Reference: Margueron et al. (PRD 103, 045803, 2021)
       - Required: E_sat, K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym
       - Crust options: BPS, DH, SLy
     - `MetaModel_with_CSE_EOS_model` - MetaModel + crust-core-saturation extension
       - Required: 9 NEPs + nbreak + nb_CSE grid parameters (typically 4-8)
     - `SpectralDecomposition_EOS_model` - Spectral representation
       - Reference: Lindblom 2010 (PRD 82, 103011)
       - Required: gamma_0, gamma_1, gamma_2, gamma_3
       - Uses 10-point Gauss-Legendre quadrature
   - Each implements:
     - `construct_eos(params: dict) -> EOSData` - Build EOS from parameters
     - `get_required_parameters() -> list[str]` - List parameter names

2. **TOV Solvers** (`jesterTOV/tov/`):
   - Base: `TOVSolverBase` (abstract base class)
   - Available solvers:
     - `GRTOVSolver` - General Relativity
       - Standard TOV equations, no additional parameters
       - Uses Dopri5 (Dormand-Prince 5th order)
     - `AnisotropyTOVSolver` - Beyond-GR modifications
       - Phenomenological sigma terms (Yagi & Yunes 2013)
       - Models: Bowers-Liang, Doneva-Yazadjiev, Herrera-Barreto, Post-Newtonian
       - Required: coupling constants (lambda_BL, lambda_DY, etc.)
     - `ScalarTensorTOVSolver` - Scalar-tensor gravity
       - Jordan frame (Brown 2023, ApJ 958 125)
       - Required: beta_ST, phi_c, nu_c
   - Each implements:
     - `solve(eos_data, pc, tov_params: dict) -> TOVSolution` - Single star
     - `construct_family(eos_data, ndat, min_nsat, tov_params: dict) -> FamilyData` - M-R-Λ family
     - `get_required_parameters() -> list[str]` - List additional parameters
   - Key features:
     - Uses Diffrax ODE solver with adaptive step size
     - Computes Love number k2 for tidal deformability
     - Parallelized via `jax.vmap()` over central pressures

3. **JAX Dataclasses** (`jesterTOV/tov/data_classes.py`):
   - All use NamedTuple for automatic JAX pytree compatibility
   - `EOSData` - EOS quantities (8 fields)
     - ns, ps, hs, es, dloge_dlogps, cs2, mu (optional), extra_constraints (optional)
   - `TOVSolution` - Single star solution
     - M (mass), R (radius), k2 (Love number)
   - `FamilyData` - M-R-Λ family curves
     - log10pcs, masses (M☉), radii (km), lambdas (dimensionless)

4. **JesterTransform** (`jesterTOV/inference/transforms/transform.py`):
   - Single unified class for all EOS+TOV combinations
   - Created via `JesterTransform.from_config(config)`
   - Coordinates: params → EOS.construct_eos() → TOV.construct_family() → observables
   - Validates: all required params are in prior (raises error if missing)
   - Logs warning: if prior contains unused parameters

**JAX Compatibility Requirements**:
- No Python `if` statements on traced values (use `jnp.where()`)
- No `float()` casts on traced arrays
- Dataclasses must be JAX pytrees (use NamedTuple, not @dataclass)

### Parameter Validation

**Automatic validation at transform setup** (in `run_inference.py`):

After creating `JesterTransform`, the code validates that all required parameters are present in the prior:

```python
transform = JesterTransform.from_config(config.eos, config.tov, ...)
required_params = set(transform.get_parameter_names())
prior_params = set(prior.parameter_names)

# Check for missing parameters
missing_params = required_params - prior_params
if missing_params:
    raise ValueError(
        f"Transform with EOS = {eos_name} and TOV = {tov_name} is missing "
        f"params = {sorted(missing_params)} from the prior file"
    )

# Warn about unused parameters
unused_params = prior_params - required_params
if unused_params:
    logger.warning(f"Prior contains unused parameters: {sorted(unused_params)}")
```

**Benefits**:
- Catch configuration errors before sampling starts (fail-fast)
- Clear error messages identifying which parameters are missing
- Helpful for debugging when switching between EOS types

**Tests**: See `tests/test_inference/test_transform_validation.py` for unit tests

### Sampler Architecture

**Base Class: JesterSampler** (`samplers/jester_sampler.py`)
- Handles parameter transforms (sample + likelihood)
- Manages posterior evaluation with Jacobian corrections
- Provides standardized `SamplerOutput` interface

**Sampler Registry** (`samplers/__init__.py`):
```python
SAMPLER_REGISTRY = {
    "flowmc": FlowMCSampler,
    "blackjax-ns-aw": BlackJAXNSAWSampler,
    "smc-rw": BlackJAXSMCRandomWalkSampler,
    "smc-nuts": BlackJAXSMCNUTSSampler,
    "smc-partial-posteriors-rw": BlackJAXPartialPosteriorsRandomWalkSampler,
    "eos-reweighting": EOSReweightingSampler,
}
```

**BlackJAX Sampler Hierarchy:**
```
JesterSampler (base)
    ├─ FlowMCSampler (flowmc.py)
    └─ BlackjaxSampler (blackjax/base.py) - Shared transform logic
        ├─ BlackjaxSMCSampler (blackjax/smc/base.py) - SMC framework (lambda-tempering)
        │   ├─ BlackJAXSMCRandomWalkSampler (blackjax/smc/random_walk.py)
        │   │   └─ BlackJAXPartialPosteriorsRandomWalkSampler (blackjax/smc/partial_posteriors.py) - data tempering, subclasses RW for kernel reuse
        │   └─ BlackJAXSMCNUTSSampler (blackjax/smc/nuts.py)
        └─ BlackJAXNSAWSampler (blackjax/nested_sampling/ns_aw.py)
```

**SamplerOutput Structure:**
```python
class SamplerOutput:
    samples: dict[str, Array]        # Parameter samples (N_samples × N_params)
    log_prob: Array                  # Log probability (posterior for MCMC, likelihood for NS)
    metadata: dict[str, Any]         # Sampler-specific data
```

**Metadata Contents** (sampler-specific):
- **SMC samplers**: ESS (effective sample size), acceptance rates, weights, tempering schedule
- **FlowMC**: flow training history, MCMC acceptance rates
- **Nested sampling**: evidence (log Z), evidence error, iteration counts

**Key Design Features:**
- Automatic transform application (prior → sampling space)
- Jacobian correction for bijective transforms
- JAX-compatible (JIT compilation, vmap, grad)
- Deterministic sampling via `jax.random.PRNGKey`

### Adaptive step size (SMC-RW)

`SMCRandomWalkSamplerConfig.adaptive_step_size` (default `False`) makes the random-walk proposal
scale adapt toward `target_acceptance_rate` (default 0.234, the Roberts-Rosenthal optimal-scaling
value for random-walk Metropolis) instead of using a fixed `random_walk_sigma` for the whole run.
Use it for high-SNR signals (e.g. ET), where the posterior narrows quickly during annealing and a
fixed sigma otherwise causes the acceptance rate to collapse.

**Why this needed a small custom wrapper, not just a BlackJAX config flag:** BlackJAX already
ships the exact update rule (`blackjax.smc.tuning.from_kernel_info.update_scale_from_acceptance_rate`,
the standard Robbins-Monro/Roberts-Rosenthal scheme), but the generic orchestrator jester uses,
`blackjax.smc.inner_kernel_tuning`, calls `mcmc_parameter_update_fn(key, state, info)` **without**
the previous step's parameter values — so a running scale can't persist across annealing steps
through that API alone. `samplers/blackjax/smc/persistent_inner_kernel_tuning.py` is a ~100-line,
jester-owned drop-in replacement that forwards the previous `parameter_override` into
`mcmc_parameter_update_fn` (new signature: `(key, previous_parameter_override, new_state, info)`),
enabling genuine recursive adaptation. `BlackjaxSMCSampler.sample()` (`smc/base.py`) uses this
wrapper for **all** SMC kernels (RW and NUTS), not just RW.

**Bonus fix as a result:** `smc/nuts.py`'s Hessian/step-size adaptation previously tried to persist
a running step size via a mutated Python closure (`current_step_size = {"value": ...}`) — a no-op
under `jax.lax.while_loop` tracing, so NUTS's dual-averaging step-size adaptation was silently
broken. It now reads `previous_params["step_size"]` instead, which actually persists.

**Implementation notes** (`smc/random_walk.py`):
- `mcmc_step_fn` builds the proposal covariance as `(scale**2) * cov`, where `cov` is the usual
  sigma-scaled empirical covariance (unchanged, recomputed from particles every step) and `scale`
  is a per-particle multiplicative correction (`1.0` = no-op) that only exists when
  `adaptive_step_size=True`.
- `scale` has shape `(n_particles,)` (no leading singleton dim), so BlackJAX's shared-vs-unshared
  parameter dispatch (`blackjax.smc.from_mcmc.unshared_parameters_and_step_fn`: leading dim `1` ⇒
  shared/bound once, anything else ⇒ vmapped per particle) treats it as unshared and vmaps it
  automatically — no manual vmap needed in jester's own code. Because of this, `BlackjaxSMCSampler`
  no longer blanket-wraps `init_params` in `extend_params` itself; each `_setup_mcmc_kernel` is
  responsible for shaping its own returned `init_params` (random_walk.py extends `cov` only; nuts.py
  extends its whole dict, since none of its params vary per particle).
- Before annealing starts, `n_pretune_steps` (default 20) pilot Metropolis steps run on the initial
  prior particles targeting `logprior_fn` (valid since the tempered posterior at λ=0 is the prior),
  self-correcting a poorly-chosen `random_walk_sigma` before real sampling begins. Set to `0` to
  skip pretuning.
- A prototype demonstrating the mechanism (and that the plain BlackJAX API genuinely can't persist
  state this way) lives outside the package at `../../dev_scripts/adaptive_smc_prototype.py`.

## Configuration System

### YAML Configuration

Configuration files use YAML with Pydantic validation. See `examples/inference/*/config.yaml` for examples.

**Key sections:**
- `seed`: Random seed for reproducibility (JAX PRNGKey)
- `transform`: EOS transform configuration
  - `type`: EOS model (metamodel, metamodel_cse, spectral)
  - `nb_CSE`: Number of CSE parameters (only for metamodel_cse)
  - `type`: TOV solver type (gr, anisotropy, scalar_tensor)
  - Grid parameters: ndat, min_nsat, etc.
- `prior`: Path to `.prior` specification file (bilby-style Python)
- `likelihoods`: List of observational constraints (discriminated union)
  - Available types: gw, gw_resampled, nicer, radio, chieft, rex, eos_constraints, tov_constraints, gamma_constraints, zero
  - Each likelihood has `enabled` flag and type-specific parameters
- `sampler`: Sampler configuration (discriminated union by type)
  - FlowMC: n_chains, n_loop_training, n_loop_production, learning_rate, etc.
  - SMC-RW: n_particles, n_mcmc_steps, target_ess, etc.
  - SMC-NUTS: n_particles, n_mcmc_steps, target_ess, etc.
  - NS-AW: n_live_points, max_samples, etc.
- `data_paths`: Override default data file locations (optional)
- `outdir`: Output directory for results (default: "outdir")

**Likelihood Types** (defined in `config/schemas/likelihoods.py`, re-exported from `config/schema.py`):
1. `GWLikelihoodConfig` - Gravitational wave events (pre-sampled)
   - `events`: list of `GWEventConfig` objects — two modes per event:
     - **Pre-trained flow** (default): set `nf_model_dir` to a trained flow directory, or omit to use a built-in preset
     - **From bilby result**: set `from_bilby_result` to a bilby HDF5 path; jester extracts posterior samples and trains a flow automatically via `prepare_gw_flows()` in `run_inference.py`
   - `GWEventConfig` fields: `name` (required), `nf_model_dir`, `from_bilby_result`, `flow_config`, `retrain_flow`
   - `from_bilby_result` and `nf_model_dir` are mutually exclusive; `flow_config`/`retrain_flow` only valid with `from_bilby_result`
2. `GWResampledLikelihoodConfig` - GW with resampling during MCMC
3. `NICERLikelihoodConfig` - X-ray timing
   - sources: list of sources (e.g., ["J0030", "J0740"])
4. `RadioLikelihoodConfig` - Radio pulsar timing
   - database: "FIDUCEO" or "FIDUCEO2"
5. `ChiEFTLikelihoodConfig` - Chiral EFT constraints
   - nb_n: number of density points
6. `REXLikelihoodConfig` - PREX/CREX neutron skin
7. `EOSConstraintsLikelihoodConfig` - EOS physical validity (causality, stability)
8. `TOVConstraintsLikelihoodConfig` - TOV solver success
9. `GammaConstraintsLikelihoodConfig` - Spectral gamma bounds
10. `ZeroLikelihoodConfig` - Prior-only sampling (no data)

**IMPORTANT**: When modifying any file under `config/schemas/`, update `docs/inference/yaml_reference.md` by hand to keep the user documentation in sync.

### Prior Specification

Priors are specified in `.prior` files using bilby-style Python syntax: (note: the following example is specific for `metamodel` or `metamodel_cse`)

```python
# Nuclear Empirical Parameters (required for MetaModel/MetaModelCSE)
E_sat = UniformPrior(-16.1, -15.9, parameter_names=["E_sat"])
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])

# CSE breaking density (for metamodel_cse transform only)
nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])
```

**Important notes**:
- All 9 NEP parameters must be present for MetaModel/MetaModelCSE EOS types
- E_sat is now a free parameter (no longer fixed to -16.0 by default)
- CSE grid parameters (p_0, ..., p_N) are added programmatically if `nb_CSE > 0`
- Parameter validation will raise an error if any required parameter is missing from prior

## Key Design Principles

### Transform System

Transforms convert between parameter spaces. Two types:

1. **Sample Transforms** (BijectiveTransform):
   - Applied during sampling with Jacobian corrections
   - Must be invertible (1-to-1 mapping)
   - Examples: LogitTransform for bounded parameters

2. **Likelihood Transforms** (NtoMTransform):
   - Applied before likelihood evaluation
   - Can be N-to-M mapping (e.g., NEP → M-R-Λ curves)
   - No Jacobian corrections
   - **JesterTransform is the single unified likelihood transform**:
     - Handles all EOS types (metamodel, metamodel_cse, spectral)
     - Handles all TOV solver types (gr, anisotropy, scalar_tensor)
     - Use `JesterTransform.from_config(config)` to instantiate

### Sampler Architecture

`JesterSampler` is a base class, with subclasses for different sampler algorithms implemented as subclasses of `JesterSampler`.

## Common Development Tasks

### Adding a New Likelihood

1. **Create likelihood class** in `likelihoods/` inheriting from `LikelihoodBase`
   ```python
   from jesterTOV.inference.base.likelihood import LikelihoodBase

   class MyNewLikelihood(LikelihoodBase):
       def evaluate(self, params: dict, data: Any) -> float:
           """Compute log probability for parameters."""
           # Your implementation here
           return log_prob
   ```

2. **Add Pydantic config model** to `config/schema.py`:
   ```python
   class MyNewLikelihoodConfig(BaseModel):
       type: Literal["my_new_likelihood"]
       enabled: bool = True
       # Add your likelihood-specific parameters here
       param1: float
       param2: int
   ```

3. **Update LikelihoodConfig discriminated union** in `config/schema.py`:
   ```python
   LikelihoodConfig = Annotated[
       Union[
           # ... existing configs ...
           MyNewLikelihoodConfig,
       ],
       Field(discriminator="type"),
   ]
   ```

4. **Add to factory** in `likelihoods/factory.py`:
   ```python
   elif config.type == "my_new_likelihood":
       from .my_new import MyNewLikelihood
       data = load_my_data()  # If needed
       likelihood = MyNewLikelihood(data)
   ```

5. **Update YAML docs**: Open `docs/inference/yaml_reference.md` and add an entry for your new likelihood type under the appropriate category.

6. **Add tests** in `tests/test_inference/test_likelihoods.py`

### Adding a New EOS Model

**Steps**:
1. **Create EOS class** in `jesterTOV/eos/` inheriting from `Interpolate_EOS_model`
   ```python
   from jesterTOV.eos.base import Interpolate_EOS_model
   from jesterTOV.tov.data_classes import EOSData

   class MyNewEOS(Interpolate_EOS_model):
       def construct_eos(self, params: dict[str, float]) -> EOSData:
           """Build EOS from parameters."""
           # Your implementation here
           return EOSData(ns=..., ps=..., hs=..., es=..., ...)

       def get_required_parameters(self) -> list[str]:
           """Return list of required parameter names."""
           return ["param1", "param2", ...]
   ```

2. **Add to JesterTransform factory** in `transforms/transform.py` with an `isinstance` check:
   ```python
   from jesterTOV.inference.config.schema import ..., MyNewEOSConfig

   def _create_eos(config: BaseEOSConfig, ...) -> Interpolate_EOS_model:
       ...
       elif isinstance(config, MyNewEOSConfig):
           from jesterTOV.eos.my_new import MyNewEOS
           return MyNewEOS(...)
       else:
           raise ValueError(f"Unknown EOS config type: {type(config).__name__}")
   ```

3. **Add config class** to `config/schemas/eos.py` and extend the `EOSConfig` union.
   Inherit from `BaseEOSConfig` (any EOS, has `crust_name`) or `BaseMetamodelEOSConfig`
   (metamodel-based, also has `ndat_metamodel`, `nmax_nsat`, `nmin_MM_nsat`):
   ```python
   class MyNewEOSConfig(BaseEOSConfig):
       type: Literal["my_new_eos"] = "my_new_eos"
       # EOS-specific fields

   EOSConfig = Annotated[
       Union[MetamodelEOSConfig, MetamodelCSEEOSConfig, SpectralEOSConfig, MyNewEOSConfig],
       Discriminator("type"),
   ]
   ```

4. **Update YAML docs** (`docs/inference/yaml_reference.md`) and **add tests**

**No need to create new transform classes** - `JesterTransform` handles all EOS × TOV combinations automatically!

### Adding a New TOV Solver

**Steps**:
1. **Create solver class** in `jesterTOV/tov/` inheriting from `TOVSolverBase`
   ```python
   from jesterTOV.tov.base import TOVSolverBase
   from jesterTOV.tov.data_classes import EOSData, TOVSolution, FamilyData

   class MyNewTOVSolver(TOVSolverBase):
       def solve(self, eos_data: EOSData, pc: float,
                 tov_params: dict[str, float]) -> TOVSolution:
           """Solve TOV for single central pressure."""
           # Your implementation here
           return TOVSolution(M=..., R=..., k2=...)

       def construct_family(self, eos_data: EOSData, ndat: int,
                           min_nsat: float,
                           tov_params: dict[str, float]) -> FamilyData:
           """Build M-R-Λ family curves."""
           # Usually uses jax.vmap over self.solve
           return FamilyData(log10pcs=..., masses=..., radii=..., lambdas=...)

       def get_required_parameters(self) -> list[str]:
           """Return list of additional parameters (e.g., coupling constants)."""
           return ["coupling1", "coupling2", ...]
   ```

2. **Add to JesterTransform factory** in `transforms/transform.py` with an `isinstance` check:
   ```python
   from jesterTOV.inference.config.schema import BaseTOVConfig, GRTOVConfig, MyNewTOVConfig

   def _create_tov_solver(config: BaseTOVConfig) -> TOVSolverBase:
       if isinstance(config, GRTOVConfig):
           return GRTOVSolver()
       elif isinstance(config, MyNewTOVConfig):
           from jesterTOV.tov.my_new import MyNewTOVSolver
           return MyNewTOVSolver(...)
       else:
           raise ValueError(f"Unknown TOV config type: {type(config).__name__}")
   ```

3. **Add config class** to `config/schemas/tov.py`, re-export it from `config/schema.py` and `config/__init__.py`, and extend the `TOVConfig` union:
   ```python
   class MyNewTOVConfig(BaseTOVConfig):
       type: Literal["my_new_solver"] = "my_new_solver"  # type: ignore[override]
       # Solver-specific fields

   TOVConfig = Annotated[
       Union[GRTOVConfig, MyNewTOVConfig],
       Discriminator("type"),
   ]
   ```

4. **Update YAML docs** (`docs/inference/yaml_reference.md`) and **add tests**

### Adding a New Sampler

1. **Create sampler class** in `samplers/` inheriting from `JesterSampler`
2. Implement `sample(prng_key, n_samples, ...) -> SamplerOutput`
3. Add to `SAMPLER_REGISTRY` in `samplers/__init__.py`
4. Add Pydantic config to `config/schema.py`
5. Update `SamplerConfig` discriminated union
6. Update `docs/inference/yaml_reference.md` and add tests

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

### Type Safety with JAX

**Common type ignore patterns** (required due to JAX tracing limitations):

```python
# vmap batches scalar NamedTuple fields → arrays
masses: Float[Array, "n"] = solutions.M  # type: ignore[assignment]

# Diffrax with throw=False guarantees ys populated
R = sol.ys[0][-1]  # type: ignore[index]

# MetaModel guarantees mu populated (but type system sees Optional)
mu: Float[Array, "n"] = eos_data.mu  # type: ignore[assignment]
# TODO: Consider restructuring Interpolate_EOS_model to make mu non-optional
```

**Anti-pattern:** NEVER use runtime assertions in JAX-traced code (fails during tracing). Use type ignore with explanatory comments instead.

## Result Storage System

**InferenceResult Class** (`result.py`):
- HDF5-based storage format for inference results
- Standardized interface for all sampler types
- Includes posterior samples, metadata, and diagnostics

**Storage Structure:**
```
outdir/{result_id}.h5
├─ posterior/               # Parameter samples + derived quantities
│  ├─ <param_name>         # Each parameter as separate dataset
│  └─ log_prob             # Log probability values
├─ metadata/               # Run configuration and statistics
│  ├─ config               # Original YAML config (string)
│  ├─ sampler_type         # Sampler used
│  ├─ n_samples            # Number of samples
│  └─ run_statistics       # ESS, acceptance rates, etc.
└─ histories/              # Diagnostics (optional)
   ├─ log_prob_history     # Evolution during sampling
   └─ ess_history          # ESS over iterations (for SMC)
```

**Key Methods:**
```python
# Create from sampler output
result = InferenceResult.from_sampler(
    sampler_output=output,
    config=config,
    prior=prior,
)

# Save to HDF5
result.save(outdir / "result.h5")

# Load from HDF5
result = InferenceResult.load(outdir / "result.h5")

# Access data
samples = result.posterior["K_sat"]
log_prob = result.posterior["log_prob"]
metadata = result.metadata
```

**Benefits:**
- Portable: HDF5 is language-agnostic, readable by Python, Julia, R, etc.
- Efficient: Compressed storage, fast I/O for large arrays
- Standardized: Consistent format across all sampler types
- Self-documenting: Includes full config and metadata

## File Naming Conventions

- Configuration: `config.yaml`
- Prior specification: `prior.prior` (Python syntax)
- Example configs: `examples/inference/<sampler_type>/<use_case>/config.yaml`
- Output results: `outdir/{result_id}.h5` (HDF5 format)
  - Legacy: `outdir/results_production.npz`, `outdir/eos_samples.npz` (deprecated)

## Parent Project Context

This module is part of jesterTOV (JESTER). See `../../CLAUDE.md` for:
- Development commands (`uv run pytest`, `uv run pre-commit`)
- Code quality standards (black, ruff, pyright)
- Testing philosophy
- Documentation generation
