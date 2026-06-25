(inference-workflow)=
# Inference workflow

This page explains what happens under the hood when you run:

```bash
run_jester_inference config.yaml
```

The entry point is `jesterTOV/inference/run_inference.py`. Working through it step by step is the clearest way to understand how all the pieces connect.

## 1. Configuration loading

The YAML file is parsed and validated against Pydantic models, producing an `InferenceConfig` object. Invalid fields (e.g., an unknown sampler type or a misspelled likelihood key) are caught here before any computation begins. Two shortcut flags are evaluated at this stage:

- `validate_only: true` — exits immediately after successful validation.
- `dry_run: true` — continues through all setup steps (prior, transform, likelihood, sampler) but stops before calling `sampler.sample()`. Useful for checking that all components initialise correctly.

See the {ref}`YAML Configuration Reference <yaml-reference>` for a full description of all configuration options.

## 2. Prior setup

The prior specification file (`.prior`) is parsed into a {class}`~jesterTOV.inference.base.prior.CombinePrior` object. Any parameter declared with `Fixed(...)` is extracted into a separate `fixed_params` dictionary and excluded from the sampling space. Fixed parameters are still passed through the transform as constants.

If a likelihood that requires per-sample random keys is enabled (currently `gw_resampled` or `nicer_kde`), a `_random_key` parameter is added to the prior automatically.

For the `metamodel_cse` EOS, the CSE grid parameters (`p_0`, ..., `p_N`) are generated programmatically based on the `nb_CSE` field in the config; they do not need to appear in the `.prior` file. The full expanded prior is printed to the log at startup.

## 3. Transform setup

A {class}`~jesterTOV.inference.transforms.transform.JesterTransform` is constructed from the EOS and TOV configuration blocks. The transform encapsulates the full EOS → TOV pipeline: it takes a parameter dictionary, builds the equation of state, solves the TOV equations, and returns a mass–radius–tidal deformability family.

After construction, the transform validates that every parameter it requires is covered by either the prior or `fixed_params`. A missing parameter raises an error immediately with a clear message listing what is absent. Unused prior parameters generate a warning but do not cause an error.

## 4. GW flow preparation

If any gravitational wave event is configured with `from_bilby_result` or `from_npz_file`, `jester` trains a normalizing flow to approximate the 4D marginal posterior on `(m_1, m_2, \Lambda_1, \Lambda_2)` before sampling starts. The trained weights are cached under `{outdir}/gw_flow_cache/{event_name}/` and reused on subsequent runs unless the flow configuration changes (checked via a SHA-256 hash) or `retrain_flow: true` is set.

For events that already point to a pre-trained flow directory via `nf_model_dir`, this step is skipped. See {doc}`analyze_bns` for full details on GW event configuration.

## 5. Likelihood setup

All enabled likelihoods are instantiated and wrapped in a {class}`~jesterTOV.inference.likelihoods.combined.CombinedLikelihood`. Each component likelihood receives equal weight (1/N). The combined log-likelihood evaluated during sampling is therefore the average of all enabled individual log-likelihoods.

## 6. Sampler setup

The sampler is created from the `sampler` block in the config via the sampler registry. All four backends (FlowMC, SMC-RW, SMC-NUTS, NS-AW) share the same {class}`~jesterTOV.inference.samplers.jester_sampler.JesterSampler` interface. The sampler holds references to the prior, the likelihood, and the transform; it applies the transform internally before likelihood evaluation.

Before sampling begins, `jester` runs a quick sanity check: it draws three prior samples, pushes them through the transform, and evaluates the likelihood. The resulting log-probabilities are printed to the log. A row of `-inf` values at this stage usually indicates a misconfigured likelihood or EOS parameters outside the valid physical range.

## 7. Sampling

`sampler.sample(jax.random.PRNGKey(seed))` is called. What happens internally depends on the sampler type — see the {ref}`yaml-reference` for sampler-specific hyperparameters. For SMC samplers, a `smc_diagnostics.png` plot is generated automatically at the end of sampling.

## 8. EOS quantity generation

After sampling, a subset of posterior samples (controlled by `n_eos_samples`) is passed back through the transform to compute derived EOS quantities: the density grid, pressure, energy density, speed of sound, and the full mass–radius–tidal deformability family. These are added to the result object alongside the raw posterior samples.

This step re-runs the TOV solver in batches (batch size set by `log_prob_batch_size`), so it is the most expensive post-sampling operation. The batch size can be tuned to fit available memory.

## 9. Saving results

Everything is saved to a single HDF5 file at `{outdir}/results.h5`. The file contains the posterior samples, derived EOS quantities, the original config, and sampler diagnostics. Use {class}`~jesterTOV.inference.result.InferenceResult` to load and work with the results.

## 10. Postprocessing

If `postprocessing.enabled: true`, a set of diagnostic plots is generated automatically. The available plots are listed in the `postprocessing` block of the {ref}`yaml-reference`. Each plot is generated independently, so a failure in one (e.g., a missing LaTeX installation affecting corner plots) does not prevent the others from being produced.
