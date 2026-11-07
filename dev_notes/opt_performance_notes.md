# jester optimisation notes

Branch `opt` of jester (worktree `../jester-opt`, from `origin/main` e676865). Measurements on the Nikhef Quadro GV100
(`wn-lot-001`), venv `/data/gravwav/twouters/projects/64_koehn_jester/modax_bench/.venv` (jax 0.6.2 + CUDA 12, Python 3.12),
code synced to `modax_bench/jester-opt`. Benchmark: `examples/inference/smc_random_walk/GW170817` with GW batch 500 (1000 runs
out of memory on the GV100 with either backend), postprocessing off, no persistent compilation cache.

## Status

| # | Item | Status | Result |
|---|---|---|---|
| 0 | modax TOV backend (`tov.backend: modax`) | done, tests pass | 2.5x sampling, 2.0x wall on the example; same posterior |
| 1 | "Testing likelihood evaluation" eager vmap | done | 41 s -> 29 s, and its compile is reused by item 2 |
| 2 | merge post-sampling log_prob and EOS compiles | done | 29 s -> 2.4 s; example wall 244 -> 200 s |
| 3 | persistent compilation cache | investigated, no code change | blocked by host callbacks (eqx, SMC progress) and by modax's kernel pointer; see log |
| 4 | SMC loop compile 43 s vs likelihood 10.5 s | cause found, needs approval | 4 copies of the likelihood in the step; same fix as 5 |
| 5 | extra likelihood calls per SMC step (12.8 vs 10 moves) | cause found, needs approval | 3 of 13 calls per step are recomputations |
| 6 | rejected TOV steps (~37 % of RHS calls) | investigated, no speed gain | PI gains trade rejections for accepted steps; PI (0.2, 0.6) is a little more accurate at equal cost |
| 7 | EOS table copied per star (4.4 GB at 5000 EOS) | works in an experiment, needs a modax change | TOV 0.332 -> 0.116 s (2.9x), params 4 GB -> 0.04 GB, same accuracy |

Out of scope here because they change results: `ndat_TOV`, `N_masses_evaluation`, SMC settings. float32 for the GW flow exists
on another jester branch and can be ported separately.

## 0. modax backend

- `jesterTOV/tov/gr_modax.py`: same RHS as `_tov_ode` in `s = 1 - h/h0`, one modax Tsit5 launch for all stars; a `custom_vmap`
  rule flattens any vmap over EOS into that launch. Invalid stars get a zero RHS, otherwise a NaN state is rejected up to
  `max_steps` and holds up its warp. Selected with `tov: {type: gr, backend: modax}`.
- Needs a CUDA GPU, no gradients. On wn-lot-001 (glibc 2.34) the numba-enzyme-cuda wheel (manylinux_2_38) does not install;
  Tsit5 only needs its Python layer, so the wheel was unpacked by hand. Rodas5P and modax gradients need Enzyme's binaries.
- TOV alone (5000 EOS x 100 stars): 0.33 s vs 1.6 s (5x); 100 EOS: 0.011 vs 0.30 s (27x).
- Accuracy against diffrax at rtol 1e-10: equal to diffrax (R14, MTOV, Lambda1.4 p50/p99 at the same level).
- Modax amplifies 1e-13 changes of pc to ~1e-4 in Lambda (p50), diffrax ~5e-6; both below their error vs the tight reference.
- Stars are bit-identical whatever the batch around them (tested).

Example, one run each:

| | modax | diffrax |
|---|---|---|
| sampling | 2 min 28 s | 5 min 59 s |
| wall | 244 s | 470 s |
| per SMC step | 9.25 s | 27.7 s |
| logZ | -12.181 | -12.166 |

Posterior medians within 0.06 half-widths (K_sym), others within 0.02; 90 % widths within 0.6 %.

## Where the time goes (modax, 244 s wall)

| stage | s | |
|---|---|---|
| imports | 9 | |
| likelihood setup (flow load, mass pre-sampling) | 11 | |
| "Testing likelihood evaluation", 3 samples | 41 | eager vmap, hundreds of small compiles |
| SMC while compile + step 0 | 61 | XLA compile of the loop 43 s (diffrax 73 s) |
| 9 SMC steps | 83 | |
| final log_prob of the particles | 18 | 14 s compile |
| EOS generation | 10 | 8 s compile, 2 s run |

Per likelihood call on 5000 particles (`profile_parts.py`): transform 0.34 s (EOS construction 0.006 s), GW flow 0.38 s, the
rest < 1 ms; total 0.72 s (diffrax 2.13 s). Standalone compile of transform + likelihood: 10.5 s (diffrax 18 s).

## Log

### Items 1 + 2: one compiled evaluator outside the sampling loop (works)

The 3-sample test ran `jax.vmap(transform.forward)` and `jax.vmap(likelihood.evaluate)` eagerly: every primitive compiles on
its own. Measured in isolation (`item1_test_eval.py`, fresh process each): eager 38.7 s, one jit 15.7 s (modax); diffrax 37.7 s
vs 16.8 s. The log L of the 3 samples differs by up to 0.03 between eager and jit with modax (0.001 with diffrax): ulp-level
differences in the inputs, amplified by the adaptive step sequence. Not a bug, but it shows the size of that noise in log L.

After sampling, SMC recomputed the log posterior of the particles in a `lax.map` (14 s compile) and then
`add_eos_from_transform` solved the TOV equations again in another `lax.map` (8 s compile + 2 s run).

Change (`opt`):
- `JesterSampler.evaluate_in_batches(samples, batch_size)`: one `jit(vmap)` of log prior + likelihood that also returns the
  EOS outputs (`EOS_OUTPUT_KEYS`), called on fixed-size batches (the last one padded), so it compiles once for any N.
- The startup test calls it with `log_prob_batch_size`, so its compile is the one reused later. It now prints the log
  posterior (prior included) instead of the log likelihood.
- `BlackjaxSMCSampler.get_log_prob` uses it when there are no sample transforms (always the case for SMC) and keeps the EOS
  outputs in `sampler.eos_outputs`.
- `InferenceResult.add_eos_from_transform(precomputed=...)` takes these outputs, subset with the same indices as the
  posterior when `n_eos_samples` is smaller; the subset filtering moved into `_filter_to_indices`.
- Tests: `TestBatchedPosteriorEvaluation` (values against `posterior_from_dict`, compiles once for N = 3 and 10),
  `test_final_log_prob_keeps_eos_outputs_of_the_particles` (a small SMC run), `test_precomputed_eos_outputs_are_used_without_a_transform`.

Example (modax, `item12`): wall 244 -> 200 s. Test 29 s (was 41 s), post-sampling 2.4 s (was 29 s). Posterior parameters
and the EOS outputs (M, R, Lambda, n_TOV) are bit-identical to the previous run; the recomputed log_prob differs by up to
0.013 (the noise above).

Open lead: the test compile at batch 1000 takes 29 s where the 3-sample jit took 15.7 s. Possibly the GW `lax.map` inside a
larger vmap, or the prior. Not looked at yet.

### Item 3: persistent compilation cache (does not work as things are; no code change)

Cold and warm run of the example with `JAX_COMPILATION_CACHE_DIR` (modax): 204 s and 194 s, cache 1 MB with 3 small entries.
Neither the evaluator (29 s) nor the SMC loop (43 s) was written. `item3_cache.py` (compile of jit(vmap(transform+likelihood)),
64 samples, two fresh processes per case, `jax_explain_cache_misses`) shows three separate blockers:

1. **Equinox host callbacks.** "Not writing persistent cache entry for 'jit_f' because it uses host callbacks", with diffrax
   and no callback of ours. `eqx.error_if` (used by diffrax `_integrate.py`, `_global_interpolation.py` and flowjax
   `distributions.py`, `bijections/utils.py`, ...) raises through a callback by default. With `EQX_ON_ERROR=nan` the entry is
   written and the second process compiles in 4.3 s instead of 16.9 s (diffrax). Run time per likelihood call is unchanged
   (0.723 s either way; the callbacks sit on branches that are not taken). Cost: errors eqx would raise become NaN silently.
2. **SMC progress printing.** `io_callback(progress_callback, ...)` inside the `while_loop` body makes the whole SMC loop
   uncacheable regardless of 1. Driving the loop from Python (one jitted step per tempering level, printing in Python) would
   make it cacheable and costs nothing at seconds per step. Sampler change, not done.
3. **modax.** The FFI call carries the raw `CUfunction` pointer as an attribute (`_jax_numba_custom_call.ffi_abi_call`,
   `"function": np.int64(launch.function)`), different in every process, so every run writes a new entry and never hits.
   A hit would even be unsafe (a pointer from a dead process). Needs a modax change (look the kernel up by a stable hash at
   launch); worth an upstream issue.

So: with diffrax, `EQX_ON_ERROR=nan` + item 2 + a Python-driven SMC loop would make reruns skip most compilation. With modax
nothing is cacheable until modax changes. Compile still matters only for short runs; for production runs it is ~1 min.

### Items 4 + 5: the SMC step evaluates the likelihood in four places (cause found; fix needs approval)

blackjax (our fork, `smc/tempered.py`, `smc/base.py`, `smc/adaptive_tempered.py`) per tempering step:

| site | calls | needed? |
|---|---|---|
| ESS solver for the next lambda, `jax.vmap(loglikelihood_fn)` | 1 | no: same particles as the reweighting of the previous step |
| `mcmc_init_fn` of each resampled particle (`update_and_take_last`) | 1 | no: their log L is known |
| random-walk moves in the `lax.scan` | 10 | yes |
| reweighting `log_weights_fn` after the moves | 1 | no, if the MCMC state carries log L |

13 calls per step (12.8 measured) and four copies of the likelihood graph in the compiled loop: 4 x 10.5 s = the 43 s compile
of the loop. Carrying (log prior, log L) per particle through the step would give 10 calls per step (-23 %) and one copy
(compile ~12 s). This needs a tempered-SMC step that caches log L, in our blackjax fork or as a jester loop on blackjax's
random-walk kernel. Not started, see the inference CLAUDE.md about sampler changes.

### Item 6: step controller of the modax TOV solve (no speed gain)

`item6_controller.py`: 5000 posterior EOS (500 against diffrax at rtol 1e-10, stable branch above 1 Msun).

| controller | time | accepted / rejected p50 | Lambda err p99 / max |
|---|---|---|---|
| I (0, 1, 0), first 1e-3 (current) | 0.333 s | 42 / 24 | 4.5e-3 / 2.9e-2 |
| I, first 1e-2 | 0.336 s | 42 / 24 | 4.8e-3 / 2.8e-2 |
| PI (0.2, 0.6) | 0.329 s | 49 / 17 | 3.3e-3 / 2.2e-2 |
| PI (0.3, 0.4) | 0.348 s | 56 / 14 | 2.9e-3 / 1.1e-2 |
| PI (0.4, 0.3) | 0.379 s | 65 / 11 | 2.6e-3 / 1.3e-2 |

Fewer rejections come with more accepted steps; attempts per star stay ~66-76. The rejected steps are not the waste they
looked like. PI (0.2, 0.6) is a free accuracy improvement if wanted.

### Item 7: one EOS table per EOS instead of per star (2.9x on TOV; needs modax change)

Each modax thread reads `params[i]`, so every star carries its own copy of the EOS table (1000 doubles), and the 32 threads
of a warp read 32 different copies. Experiment (`item7_shared_table.py`): patched Tsit5 body with `params[i // 100]`, one row
per EOS, h0 moved into the state as a fifth component with zero derivative and zero error weight (other weights x sqrt(5/4)
so the error norm is unchanged). 5000 posterior EOS x 100 stars:

| | time | params | M err p99 | R err p99 | Lambda err p99 / max |
|---|---|---|---|---|---|
| per-star rows | 0.332 s | 4.01 GB | 1.0e-4 | 6.1e-5 | 4.5e-3 / 2.9e-2 |
| per-EOS rows | 0.116 s | 0.04 GB | 9.9e-5 | 6.1e-5 | 4.8e-3 / 2.9e-2 |

Not bit-identical (R, M differ by p50 1e-6, p99 1e-4 between the two: different compiled arithmetic, amplified by the step
sequence), but equally accurate. Expected effect: one likelihood call 0.72 -> ~0.50 s, the GW flow (0.38 s) then dominates.
Implementation options: a modax option for grouped parameters (e.g. `params_group_size` or an index array; upstream PR), or
a jester-side copy of the Tsit5 body. Decision needed.

### State of `opt` (2026-11-07)

Uncommitted in `../jester-opt`: modax backend (`gr_modax.py`, `gr.py` refactor, `tov.backend` config field, transform
factory, `tests/test_tov_modax.py`) and items 1 + 2 (`jester_sampler.py`, `smc/base.py`, `result.py`, `run_inference.py`,
tests in `test_samplers.py` and `test_integration.py`). Pyright clean on the changed files. Fast suite: 662 passed, 3 failed;
the 3 (`TestLogisticDistribution::test_logistic_log_prob`, `TestScaleTransform::test_scale_transform_inverse`,
`TestUniformPrior::test_uniform_prior_log_prob`) fail identically on a clean `origin/main` checkout with the same venv, so they
are not from this branch. `tests/test_utils.py` needs `hypothesis`, not installed locally. The GPU tests of
`test_tov_modax.py` (9) pass on the GV100. Docs (`docs/inference/yaml_reference.md` for `tov.backend`, API pages for
`gr_modax`) and the CLAUDE.md files are not updated yet.

Decisions open: items 4 + 5 (SMC step that caches log L), item 7 (modax grouped parameters), whether to set
`EQX_ON_ERROR=nan` for production runs (item 3), PI (0.2, 0.6) for the modax solve (item 6).

## Round 2 (2026-11-07): latest software, then speed

User direction: hacking in these branches is fine; goal is the latest versions of everything, then as fast as possible.
`EQX_ON_ERROR=nan` (item 3): deferred on request, revisit later (diffrax-only cache gain 16.9 -> 4.3 s per rerun).

### float32 and `dev`

float32 flow evaluation is jester PR #178 ("Bump flowjax and allow float32 evaluation of flows", `use_float32` on the GW
likelihood), merged into `origin/dev`, not into `main`. `dev` also has the stacked GW likelihood (#176, `N_masses_evaluation`
default 500, batch sizes 1), adaptive SMC step size (#177), pruned flowMC/NUTS (#193) and likelihoods (#194), flowjax >= 19.1.1
with retrained flow weights. So `opt` was moved onto `origin/dev` (2d479f1) and the round-1 patch re-applied (3-way, clean;
backup in `opt_on_main.patch` + `opt_on_main_new_files/`). float32 thus comes with `dev`; nothing to port.

### Upstream blackjax (blackjax-devs/blackjax) instead of the ThibeauWouters fork

The fork (`jester` branch, 8758ef11) = upstream as of 2026-02-02 + handley-lab nested sampling (`blackjax/ns/*`,
`mcmc/ss.py`) + one sign fix in `smc/ess.py`. Upstream main (58dcc8bc, 2026-11-03) now has both: `blackjax/ns` and the ESS fix
(PR #915), plus SMC `batch_size` (#929) and several ESS/resampling robustness fixes. Upstream needs Python >= 3.13 and
jax >= 0.9 (jester had `jax<0.8`, Python 3.11-3.13).

Not a big rewrite:
- SMC-RW ran unchanged on upstream blackjax + jax 0.12 + Python 3.13.
- NS-AW: upstream changed the NS kernel convention. `delete_fn(state)` takes no key and returns (dead_idx, target_idx); the
  inner kernel gets one key and the whole state and picks its own start points. Ported in
  `nested_sampling/kernels/acceptance_walk_kernel.py::inner_kernel_wrapper` (uniform choice among survivors above the
  threshold, as the fork's `delete_fn` did). All 6 NS-AW e2e tests pass after that.
- 3 unit tests (`test_logistic_log_prob`, `test_scale_transform_inverse`, `test_uniform_prior_log_prob`) failed with newer
  pytest/jax: `pytest.approx(jnp_scalar)` reads `.flat`, which JAX arrays do not implement. Fixed by `float(...)` on the
  expected value (same check). These also failed on `origin/main` before.
- pyproject: `requires-python >= 3.13`, `jax>=0.9.0`, `blackjax @ git+https://github.com/blackjax-devs/blackjax.git`,
  `.python-version` 3.13; `uv lock --upgrade`: jax 0.12.0.1, blackjax 1.7.2.dev15 (58dcc8bc), flowjax 19.2.1, equinox 0.13.8,
  diffrax 0.7.2, numpy 2.5.4, scipy 1.18.1 (optax stays < 0.2.7, jester's pin).
- Local (macOS CPU): fast suite 655 passed (incl. new tests), e2e 16/16. Pyright clean on all changed files.
- Nikhef: `.venv-latest` (Python 3.13, jax 0.12.0.1 + jax-cuda12-plugin, numba-cuda-mlir 0.6.0 cu12, Enzyme wheel unpacked
  as before), code in `modax_bench/jester-latest`. `.venv-dev` = dev's own pins (jax 0.7.2) for comparison.

### Item 7 implemented: modax `params_group_size` (local modax clone, branch `grouped-params`)

- `modax/tsit5.py`: `solve(..., params_group_size=1)`; trajectory `i` reads `params[i // params_group_size]` (compile-time
  constant in the body, part of the kernel cache key, FFI ABI unchanged). Not differentiable (raises), not under modax's own
  vmap rule. `modax/_jax_common.py::prepare_inputs(params_group_size=...)` checks `rows * group == N`.
- Tests `modax/tests/test_params_groups.py` (GPU): grouped == duplicated rows bit for bit (both shared and local backends),
  exact solution, shape error, not differentiable, default unchanged. Not yet run (GPU busy with benchmarks).
- jester `gr_modax.py`: one row per EOS (`N_HEADER = 0`), state (r, m, H, b, h0) with `ERROR_WEIGHTS = (sqrt(5/4),)*4 + (0,)` so
  the error norm equals the 4-component one; invalid stars get h0 = 0 and the RHS returns zeros before reading the table.
  The vmap rule flattens stars (EOS-major) and rows together. RHS host tests updated and passing; GPU tests pending.

### Items 4 + 5 implemented: cached log L in SMC-RW (`sampler.cache_loglikelihood`, default True)

Each particle is [parameters, log L]. blackjax's tempered SMC gets `loglikelihood_fn = x[n_dim]` (free) and the prior on
`x[:n_dim]`; jester's random-walk move (`random_walk.py::cached_mcmc_step_fn`) uses blackjax's `build_rmh` kernel with a
proposal that moves the parameters, evaluates the true likelihood at the new point and stores it in the last slot, so the
accept/reject is blackjax's own with the same tempered target. Covariance adaptation uses the parameter columns only. The
initial log L comes from the shared evaluator of item 2 (no extra compile). Tests (`TestCachedLogLikelihood`): exact count of
likelihood evaluation events (cached: initial batches + n_mcmc per step; uncached: +3 per step), Gaussian posterior and
analytic logZ in both modes, final log_prob unchanged. GPU timing pending.

### Benchmarks on the latest stack (GW170817 example as on `dev`, postprocessing off, one run each unless stated)

Reference: diffrax float64 (`latest_diffrax_f64`): sampling 354 s, wall 394 s, 30.8 s/step, logZ -12.085.
`compare_runs.py e2e <ref> <runs>` gives the table; shift = median shift in units of the reference 90 % half-width,
width = relative change of the 90 % width, over MTOV, R14, L14, K_sat, L_sym, K_sym, Q_sym, nbreak.

| run | sampling | wall | s/step | logZ (d) | max shift | max width |
|---|---|---|---|---|---|---|
| diffrax f32 | 341 s | 380 s | 29.4 | +0.009 | 0.035 | 2.4 % |
| diffrax f32 + cached log L | 267 s | 304 s | 23.3 | +0.009 | 0.035 | 2.4 % |
| modax f64 (per-star tables) | 93 s | 154 s | 7.16 | +0.001 | 0.034 | 2.1 % |
| modax f32 (per-star tables) | 84 s | 127 s | 6.29 | -0.004 | 0.020 | 2.8 % |
| modax f32 + shared tables | 58 s | 100 s | 3.49 | -0.030 | 0.068 | 2.0 % |
| modax f32 + shared tables + cache | **53 s** | **93 s** | **2.66** | -0.030 | 0.068 | 2.0 % |
| modax f64 + shared tables + cache | 60 s | 103 s | 3.25 | -0.015 | 0.031 | 2.1 % |

- Cache on/off give identical posteriors to all printed digits (same keys, same proposals, same log L values): the cache only
  removes recomputation. Saves 24 % per step with modax (3.49 -> 2.66 s), 21 % with diffrax (29.4 -> 23.3 s).
- Shared tables: 6.29 -> 3.49 s/step.
- The 0.068 shift (L_sym) of the shared-table runs: a different step sequence gives a different MCMC path; seed scatter of this
  example being measured (round 4: 3 more seeds of the optimized setup, 1 more diffrax f64 seed).
- First attempt of shared tables failed: `error_weights=jnp.asarray(...)` became a tracer inside the custom_vmap trace and
  modax converts the weights with NumPy while tracing (`TracerArrayConversionError`). Fixed with `np.asarray`.

### Startup of the optimized run (93 s wall)

imports 4.5 s, likelihood setup 9 s, startup test 18.4 s (of which tracing 10 s: mostly the numba build of the modax kernel,
once per process; XLA 6.2 s), initial log L + setup 6.3 s (incl. 2.3 s compile of a log-prior vmap, now removed by returning
log L from the shared evaluator: `evaluate_in_batches` returns `BatchEvaluation(log_prob, log_likelihood, eos)`), SMC loop
compile 16.2 s + step 0, 9 steps 24 s, post 3.5 s. Compile and setup are now ~2x the sampling for this short example.
Possible later: persistent cache of modax's compiled kernels (~8-10 s/process), smaller SMC loop compile.

### GW likelihood batch size (`gw_batch.py`, 5000 posterior EOS)

`N_masses_batch_size` (dev default 1) changes speed and memory only (|dlogL| <= 2e-15):

| | bs 1 | bs 10 | bs 50 | bs 100 | bs 250 | bs 500 |
|---|---|---|---|---|---|---|
| float64 run | 0.225 s | 0.123 | 0.108 | 0.105 (1.1 GB) | 0.107 | 0.103 (5.5 GB) |
| float32 run | 0.205 s | 0.078 | 0.073 | 0.067 (0.54 GB) | 0.070 | 0.069 (2.4 GB) |

float32 itself changes log L by up to 0.09 (p50 0.012) against float64 (independent of batching); that is dev's float32
accuracy, and likely why float32 runs show logZ offsets of ~0.03.

### Validation across seeds, GW batch 100, and the optimized configuration (rounds 4-5)

| run | sampling | wall | s/step | logZ | max shift | max width |
|---|---|---|---|---|---|---|
| diffrax f64 seed 43 (reference) | 354 s | 394 s | 30.8 | -12.085 | - | - |
| diffrax f64 seed 44 (+cache) | 279 s | 317 s | 23.8 | -12.134 | 0.054 | 2.1 % |
| modax f32 + tables + cache, seed 43 | 53 s | 93 s | 2.66 | -12.115 | 0.068 | 2.0 % |
| idem seed 44 | 55 s | 94 s | 2.68 | -12.099 | 0.048 | 3.8 % |
| idem seed 45 | 55 s | 94 s | 2.66 | -12.066 | 0.027 | 2.6 % |
| idem seed 46 | 53 s | 91 s | 2.66 | -12.095 | 0.046 | 3.9 % |
| **+ GW batch 100 (seed 43)** | **41 s** | **80 s** | **1.80** | -12.115 | 0.068 | 2.0 % |
| modax f64 + tables + cache + GW batch 100 | 45 s | 84 s | 2.19 | -12.100 | 0.031 | 2.1 % |

The seed scatter of the reference itself (seed 43 vs 44) is 0.054 half-widths and 0.049 in logZ; the optimized runs fall
in the same range (logZ -12.094 +- 0.02 over four seeds, vs -12.110 for the two diffrax seeds). GW batch 100 reproduces
batch 1 to all printed digits.

**Optimized configuration**: `examples/inference/smc_random_walk/GW170817_gpu/config.yaml` on `opt` = the GW170817 example +
`tov.backend: modax` (shared EOS tables via modax `params_group_size`), `use_float32: true`, `N_masses_batch_size: 100`,
`cache_loglikelihood: true` (default). Per step 30.8 -> 1.80 s (17x), sampling 354 -> 41 s (8.6x), wall 394 -> 80 s (4.9x);
vs the first `main` diffrax run (470 s wall) 5.9x.

Where the 80 s go now: ~58 s until step 0 (imports 4.5, likelihood setup 9, startup check with modax kernel build ~18,
initial log L, SMC loop compile ~16), 9 steps x 1.8 s, ~3 s after sampling. Per likelihood call: TOV ~0.12 s, GW ~0.07 s.

### State of the branches (2026-11-07, all uncommitted)

- jester `opt` (worktree `../jester-opt`, on `origin/dev` 2d479f1): modax backend with shared tables, items 1+2 (shared
  evaluator, returns log L too), cached log L in SMC-RW, upstream blackjax + NS-AW port, Python 3.13 / jax 0.12 / latest deps
  (`uv.lock` upgraded), 3 test fixes for newer pytest, docs (`yaml_reference.md`, API page `jesterTOV.tov.gr_modax`, strict
  Sphinx build passes; autosummary pages regenerated), CLAUDE.md files, example `GW170817_gpu`. Local: 655 fast + 16 e2e
  pass, pyright clean. GPU (`.venv-latest` on wn-lot-001): `tests/test_tov_modax.py` 10/10.
- modax (`../modax`, branch `grouped-params`, from lawrenceberry/modax ec3e3aa): `params_group_size` in Tsit5 +
  `tests/test_params_groups.py` (GPU: pass). Not pushed anywhere; jester's modax backend needs this branch.

### Next ideas, by expected gain for short runs (long runs are dominated by the 1.8 s steps)

1. Persistent disk cache for modax's compiled kernels (numba build ~8-10 s per process) and a stable kernel id instead of the
   CUfunction pointer in the FFI attributes (would also make JAX's persistent cache usable, item 3).
2. SMC loop compile (~16 s with the likelihood in it once): look at what else is in the loop (tuning, ESS root finder).
3. `EQX_ON_ERROR=nan` (deferred).
4. TOV per call 0.12 s: ndat_TOV / table size choices change results, so only as a user decision.
5. Gradients (smc-nuts, modax sensitivities): discuss before starting, as agreed.

## Round 3: compile caching across runs (works)

User asked to test caching. Done for the optimized GW170817_gpu configuration; three blockers removed.

1. **modax kernels on disk + stable kernel id** (modax branch `kernel-cache`, on top of `grouped-params`):
   - `modax/_kernel_cache.py`: `fingerprint()` hashes a right-hand side's code, closure values, defaults, read globals
     (functions, numba dispatchers, arrays by content, numbers, modules by name), and `kernel_key()` adds modax's own sources,
     the numba-cuda-mlir version and the GPU target. Unknown objects raise `KernelKeyError` -> that kernel is not cached.
     `MODAX_KERNEL_CACHE_DIR` (default `$XDG_CACHE_HOME/modax/kernels`; `off` disables). Atomic writes.
   - `_jax_numba_custom_call.py`: the C++ launcher keeps an id -> CUfunction registry (`modax_register_kernel`); the FFI
     attribute `function` is now that id (60 bits of the key) instead of the raw pointer, so the HLO is identical in every
     process. `compile_kernel(..., cache_key)` loads the cubin from disk or builds and stores it. Without a key the id is the
     pointer, as before (Rodas5P unchanged).
   - `tsit5._make_jax_launch` passes `kernel_key("tsit5", ode_fn, n_vars, gains, spec, shared, params_group_size)`.
   - Tests `tests/test_kernel_cache.py`: fingerprint sees closures, code, array contents, rejects unknown objects (CPU); a
     second process reuses the cached kernel with compilation forbidden and gives identical output; the lowered program is
     byte-identical across processes; `off` works (GPU, pass). Measured: modax build ~6 s per process before, 0 when cached.
   - The 3 failing modax tests (`test_default_first_step_matches_explicit[solve1]`, two in `test_solver_vmap.py`) fail the same
     on the untouched clone: Rodas5P needs Enzyme's vendored clang, which cannot run on glibc 2.34.
2. **SMC loop driven from Python** (jester `smc/base.py`): one `jax.jit(body_fn)` per annealing step, progress printed in
   Python. The `io_callback` inside `lax.while_loop` made the loop uncacheable. Found while testing: the initial carry held
   the log evidence as a Python 0.0 (weak float64) and the step returned a strong float64, so the step compiled twice (20 s +
   18 s). Fixed by building the carry with explicit dtypes; test `test_annealing_step_compiles_once`.
3. **`EQX_ON_ERROR=nan`** (environment, for this test only; the code default is unchanged): needed, otherwise equinox's
   error callbacks in diffrax/flowjax keep every program out of JAX's persistent cache.

Runs (`round7.sh`, fresh caches; `JAX_COMPILATION_CACHE_DIR`, `MODAX_KERNEL_CACHE_DIR`, `CUDA_CACHE_PATH` under
`modax_bench/cachetest2`, `EQX_ON_ERROR=nan`):

| run | wall | sampling | step compiles | vs diffrax f64 (394 s) |
|---|---|---|---|---|
| cold, before the carry fix | 108 s | 65 s | 2 | 3.6x |
| cold | 88 s | 46 s | 1 (20 s) | 4.5x |
| warm | **47 s** | 24 s | 0 | **8.4x** |

Cold and warm results are bit-identical (parameters, EOS outputs, log_prob max diff 0.0). JAX cache 3.4 MB (evaluator,
annealing step, flow sampler), modax cache 2 kernels.

Warm run, where the 47 s go: imports 4.8 s, likelihood setup 10.7 s (not yet looked at: flow loading, mass pre-sampling,
imports of numba/modax?), startup check 4.3 s, to step 0 6.5 s (tracing ~2 s + cache loads), 9 steps 16 s, saving 2.7 s.

Note for Nikhef: the CUDA driver cache defaults to `~/.nv` (was 96 MB in the 1.2 GB home quota) and the modax kernel cache to
`$XDG_CACHE_HOME`: keep both on /data (`CUDA_CACHE_PATH`, `XDG_CACHE_HOME`).

## To test later (curiosity): what the optimizations give on a laptop CPU

Cancelled on 2026-11-07 because other processes were using the laptop's CPU, which would spoil the timing. Run it on an
otherwise idle machine.

- What runs where: `tov.backend: modax` needs an NVIDIA GPU (numba-cuda-mlir kernels, no CPU path), so on a laptop use
  `backend: diffrax`. Everything else is plain JAX and runs on CPU: cached log L (`cache_loglikelihood`), the shared
  evaluator, the Python-driven SMC loop, float32 GW flow, GW `N_masses_batch_size`, upstream blackjax, and compile caching
  (with `EQX_ON_ERROR=nan` for diffrax). The fast test suite already passes on the M2 (CPU), modax tests skipped.
- Test: `examples/inference/smc_random_walk/GW170817_laptop` (1000 particles, 10 MCMC steps) on the `opt` venv, diffrax,
  postprocessing off, same seed, run one after the other:
  - `base`: as dev ships it plus `cache_loglikelihood: false` (float64 flow, GW batch 1);
  - `opt`: `cache_loglikelihood: true`, `use_float32: true`, `N_masses_batch_size: 100`.
  Prepared configs and a runner (`run_both.sh`, uses `run_ts.py` for timestamped logs) are in the session scratchpad
  `.../scratchpad/laptop/`; they are easy to recreate from the two lines above if the scratchpad is gone.
- Compare: sampling time, s/step, wall, logZ, and posteriors with `compare_runs.py`. Optionally a warm rerun of `opt` with
  `JAX_COMPILATION_CACHE_DIR` + `EQX_ON_ERROR=nan` to see the compile-cache gain on CPU.
- Expectation, not measured: the cache should save ~3/13 of the likelihood calls per step; float32 and larger GW batches may
  help less on CPU than on the GPU.

## Round 4: TOV accuracy tests and a faster kernel (2026-11-07)

User direction: larger and low-risk gains only; every edit must keep the solver's accuracy and robustness. Against LAL a
few outliers are expected; the bulk has to match.

### TOV accuracy tests ported from jester-review `04_tov` (`tests/test_tov_lal_crosscheck.py` on `opt`)

- The review compared jester's TOV with LALSuite's (`crosscheck_lal.py`: 108 LAL EOS; `from_jester_to_lal.py`: 200
  metamodel+CSE EOS written as LAL tables with the DH crust), at 100 central pressures 2e33-1e35 Pa, dloge_dlogp clipped to
  [0, 5], M >= 0.75 Msun, relative RMS of R(M), Lambda(M) on a common mass grid. Review results: LAL EOS median R 0.0004 %,
  Lambda 0.24-0.28 %; jester EOS median R 0.07 %, Lambda ~1 %, outliers up to tens of percent.
- Ported for every backend (diffrax always, modax with a GPU): (1) the 12 LAL tables shipped in
  `jesterTOV/tabulated_eos/lalsuite` (they store LAL's own M-R-Lambda, so no lalsuite needed): bulk (median R < 0.005 %,
  median Lambda < 0.3 %), at most 3 outliers beyond 0.02 % / 1 %, none beyond 0.5 % / 5 %, every table gives >= 20 stars;
  (2) slow, needs lalsuite (in the dev extra): 20 jester EOS through LAL: median R < 0.15 %, median Lambda < 2 %, >= 80 %
  within 3.5 %.
- Measured on the 12 tables (diffrax and modax agree): R 0.001-0.01 %, Lambda 0.08-0.4 % for most; outliers MS1 (R 0.39 %,
  Lambda 4.2-4.3 %), MS1B (Lambda ~1 %), SLY230A (Lambda ~1.8 %), the same for both backends.
- Status: 5/5 local (CPU, incl. slow), 9/9 on the GPU (both backends).

### Profiling the modax TOV kernel

Nsight Compute on the node (`/usr/local/cuda-*/bin/ncu`, all 2026.1) no longer supports Volta: "Profiling is not supported
on device 0"; no nvprof. Instead, ablations (`ablate_tov.py`: RHS variants with one cost removed, compared per attempted
star-step, since their step counts differ) and `cuFuncGetAttribute`:

| RHS variant (timing only) | ns per star-step |
|---|---|
| production (while-loop bisection) | 3.6-3.9 |
| no exp/log | 2.8 |
| no search (fixed node) | 1.2 |
| neither | 0.5 |

The search was two thirds of the per-step cost; the search variants also used 224 B local memory per thread (154 regs),
none without it.

### Branchless bracket search (done; bit-identical)

`gr_modax.make_tov_rhs`: the data-dependent while loop is replaced by a fixed sequence of power-of-two candidate steps
(`_search_steps`, 8 for 200 nodes) with an index clamped in bounds and a select instead of a branch (`_bracket_lo` is the
same loop on its own). It returns the same bracket as bisection for every h, so the results are bit-identical.
Tests: `test_bracket_search_matches_the_reference_bisection` (nodes, next-after neighbours, 5000 random, above the table),
`test_rhs_is_bit_identical_to_the_bisection_version` (3000 random states against a frozen copy of the old RHS).

| | TOV, 5000 EOS x 100 stars | per star-step | GW170817 opt run: s/step | sampling (warm) | wall (warm / cold) |
|---|---|---|---|---|---|
| bisection | 0.128 s | 3.74 ns | 1.80 | 24 s | 47 / 88 s |
| branchless | **0.085 s** | 2.47 ns | **1.43** | **20 s** | **43 / 83 s** |

Surface states bit-identical over 500 000 stars; the full run's posterior, EOS outputs and log_prob are bit-identical to the
run before the change (c4 vs c6). All 21 GPU tests pass (modax + LAL cross-check, slow included).
vs diffrax f64 (dev defaults): 21.5x per step, 17.7x sampling, 9.1x wall (warm caches), 4.7x wall (cold).

### Further TOV ideas, not done (ordered by expected gain / risk)

- The search still costs ~1.3 of the 2.47 ns per star-step (no-search variant: 1.16 ns). An exact coarse index (per EOS,
  the node index at uniform log-h bins, stored in the row) would leave one or two candidates: same bracket, so bit-identical.
  More code and a longer row; up to ~1.5x on TOV.
- exp/log ~0.8 ns per star-step: removing them changes the interpolation, so results change. Not low risk.
- Per likelihood call now: TOV ~0.085 s, GW flow (float32, batch 100) ~0.067 s, so the GW flow is about as large as TOV.

## Round 5: batched GW flow (done; end-to-end bit-identical)

Profiling: nsys/ncu do not trace kernels on the GV100 (Volta dropped). `jax.profiler.trace` + `jax.profiler.ProfileData`
does (script `kprof.py` in the session scratchpad / `$B/kprof.py`): per-kernel GPU time without extra packages.

GW likelihood at 5000 particles, float32, batch 100 (68 ms per call): the flowjax MAF under vmap stored the spline
parameters point-major, so the output layer was a transposed GEMM (`volta_sgemm_128x32_tt`, 18 ms), and the spline bin
came from a searchsorted while loop plus a gather (`loop_divide_gather_subtract_fusion`, 16 ms).

`jesterTOV/inference/flows/batched.py::batched_log_prob`: same math as flowjax, for a whole batch, intermediates laid out
`[features, points]` (GEMMs `W @ H`), bin = count of knots below x, values picked by a one-hot sum (exact). A first
point-major version only gave 1.17x (cuBLASLt epilogue kernel 11 ms); the feature-major layout is what helps.
`StackedGWLikelihood(use_batched_flow=None)` uses it whenever `supports_batched_log_prob` accepts the flow (MAF + RQS:
all 16 shipped GW flows), else flowjax.

| | GW call (5000 particles) | s/step | sampling (warm) | wall warm / cold |
|---|---|---|---|---|
| flowjax per point (c6/c7) | 0.068 s | 1.43 | 20 s | 43 / 83 s |
| batched (c8/c9) | **0.040 s** | **1.17** | **18 s** | **40 / 82 s** |

Accuracy: matches flowjax to 1e-13 (float64) on every shipped flow incl. points outside/on the spline interval, random
3-layer flows, float32 to round-off; StackedGW log L identical with and without (batch 1, 7, 20, float32 under jit/vmap).
The GW170817 run is bit-identical to c6 (parameters, EOS outputs, log_prob). vs diffrax f64: 26x per step, 19.7x sampling,
9.8x wall warm. Tests: `tests/test_inference/test_batched_flow.py` (24) + the GW likelihood tests, 51 pass on Nikhef.

Remaining per step (~1.17 s for 10 calls): TOV ~0.085 s and GW ~0.040 s per call, the rest EOS + overhead. In the GW
call the 4 hidden GEMMs (~10 ms) and the bias+ReLU passes (~7 ms) are now the largest parts.

## Round 6: profile after the batched flow, and the coarse TOV index (tried, reverted: no gain)

Profile of one likelihood call, 5000 particles (`prof_all.py`, JAX profiler kernel table), same at prior and posterior:
118 ms per call = TOV kernel 68 ms (58 %), GW flow 40 ms (34 %: hidden GEMMs 9.7, bias+ReLU 6.7, spline ~15,
output layer/copies ~8), EOS 5 ms, cub radix sort 2.6 ms, radio + constraints < 1 ms. 10 calls = 1.18 s = the measured
1.17 s per SMC step, so the SMC machinery itself costs nothing.

Coarse index for the TOV table search: node density on jester tables is very uneven (nodes 0-50 span h = 1e-30..1e-3,
nodes 50-200 span 1e-3..1.4). Bins uniform in log h or h put ~100 nodes in one bin; bins uniform in h^(1/4) work
(NB 128: 99.3 % of bins hold <= 5 nodes). Implemented as header [u0, 1/du, start node per bin] + K branchless steps from
the start + a check + fallback to the full search (exact by construction). Host tests and 1M GPU stars (posterior +
prior tables): bit-identical, fallback 0.00 % of calls. Speed (ns per attempted star-step, interleaved repeats):

| | posterior tables | prior tables |
|---|---|---|
| while-loop bisection (old) | 3.60 | 3.36 |
| branchless full search (current) | **2.62** | **2.27** |
| coarse NB=128 K=3 | 2.80 | 2.43 |
| coarse NB=256 K=2 | 2.65 | 2.33 |
| coarse NB=256 K=3 / NB=128 K=4 | 2.72 / 2.88 | 2.51 / 2.79 |

No variant beats the full branchless search: its 8 steps read a 200-node row that the warp shares (L1), so the extra
header read, two sqrt, int conversions and the check cost as much as the saved steps. The earlier "no-search" ablation
(1.16 ns) overstated the search cost. Reverted; kept a stronger bit-identity test (log-uniform h through the crust,
several EOS) and log-uniform enthalpies in the bracket test.

What is left in the TOV kernel is the Tsit5 arithmetic itself (7 RHS calls per step with exp/log/divisions in double
precision on a GV100). Larger gains would now need fewer steps (looser tolerance: changes results) or float32 in parts
of the RHS (changes results), i.e. accuracy decisions rather than free speed.
