# FAQ and common pitfalls

This page collects answers to frequently asked questions and documents common pitfalls that users encounter when working with JESTER.

---

## Installation and GPU setup

### JAX does not detect my GPU

JAX ships with CPU-only support by default. To enable GPU acceleration, install the CUDA extra when setting up JESTER:

```bash
uv sync --extra cuda12
```

If you already installed JESTER without the CUDA extra, you can upgrade JAX in-place:

```bash
uv pip install -e ".[cuda12]"
```

For more help, check out the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html). 

### Sampling is slow even with CUDA libraries installed

Even with the necessary CUDA libraries installed, JAX might sometimes struggle to detect the presence of the GPU, so that additional debugging steps need to be done. 
When running the inference workflow, the main function of ``run_inference.py`` (which orchestrates the inference pipeline) prints the following jax function call::

```python
    jax.devices()
```

which shows the available devices.
If you are submitting on the cluster, at the top of the log file (or your terminal if you are running locally), verify that something along the following is present in the output:

```bash
    [INFO] jester: JAX devices: [CudaDevice(id=0)]
```

Which shows that indeed the GPU is found and jester is running on it. 
In case the GPU is not found, jax will show an error/warning (but not exit the program), and instead, you will see:

```bash
    [INFO] jester: JAX devices: [CpuDevice(id=0)]
```

### Should I be using gradient-based samplers?

In our experience, we have found that samplers that do not use gradient information, such as the sequential Monte Carlo sampler using random walk MCMC moves, give reliable and robust posteriors, and when ran on high-end GPUs, can finish sampling in a matter of minutes. Gradient-based samplers might introduce additional overhead in the `jit` compilation phase, as well as have more computationally expensive sampling steps, which might only be worth it for a select few EOS and TOV configurations. We recommend starting with the default non-gradient samplers, and only switching to gradient-based samplers if you have a specific reason to do so (e.g., you are using a very high-dimensional parameter space, or you have a very complex posterior geometry that benefits from gradient information), and leave it up to the user to experiment with different samplers and configurations to find the best fit for their specific use case.

---

## Configuration

### Missing required parameter error at startup

JESTER validates that every parameter required by the EOS and TOV solver is present in the prior file before sampling begins. If you see an error like:

```
ValueError: Required EOS parameters missing from prior: ['E_sat']
```

open your `.prior` file and add a prior for the listed parameter. For the MetaModel EOS, `E_sat` is required and a suitable prior is:

```python
E_sat = Uniform(minimum=-16.1, maximum=-15.9, name='E_sat', latex_label='$E_{\\rm sat}$')
```

In case you wish to 'pin' these missing variables to some default values, you can use the `Fixed` prior class, e.g. as follows:

```python
E_sat = Fixed(-16.0, name='E_sat')
```

### The sampler complains about unused TOV parameters

If you include TOV-specific parameters in your prior file (for example, `lambda_BL` for the anisotropy solver) but are running with `type: "gr"` in the TOV config block, JESTER will emit a warning rather than an error. This is intentional: unused parameters are harmless because they are never passed to the solver. You can safely ignore the warning, or remove the unused priors to keep the prior file clean.

---

## JAX-specific pitfalls

### Runtime assertion errors during JIT compilation

Assertions like `assert x > 0` fail during JAX tracing because traced values do not have concrete Python values at compile time. Replace runtime assertions with `jnp.where()` guards or with penalty terms in the likelihood, and add a `# type: ignore` comment with an explanation where needed. See the *Type ignore patterns* section of `CLAUDE.md` for the canonical patterns used in JESTER.

### `float()` or `.item()` raises a `ConcretizationTypeError`

Calling `float(x)` or `x.item()` on a traced JAX array inside a JIT-compiled function forces concrete evaluation, which is not allowed during tracing. Either move the call outside the JIT boundary or restructure the code to avoid needing a concrete scalar.

### EOSData unpacking fails

`EOSData` is a `NamedTuple` with eight fields. Positional unpacking like `ns, ps, hs, ... = eos_data` will raise a `ValueError` because the number of values does not match. Always access fields by name:

```python
eos_data = model.construct_eos(params)
ns  = eos_data.ns
ps  = eos_data.ps
cs2 = eos_data.cs2
```

---

## Numerical issues

### First iteration of SMC produces NaN

This is likely the origin of a NaN somewhere in your likelihood class which messes up the SMC. Note that we have found, in the past, that similar odd behavior can originate from using negative infinity in your likelihoods, e.g., as a penalty value. It might look like the sampler still finishes (i.e., reaches the posterior within a finite number of steps), but often, the posterior is visually problematic. In this case, it is advised to *not* trust this sampling result, and try to find out where the NaN comes from. To debug NaNs during the Bayesian inference workflow, you can turn on JAX's NaN debugging by adding the following flag at the top of your inference config:

```bash
debug_nans: true     # Enable JAX NaN debugging for numerical issues
```

---

## Sphinx documentation build

### A Sphinx warning becomes a build error in CI

CI runs `sphinx-build -W --keep-going`, which promotes every warning to an error. Always verify locally before pushing:

```bash
uv run sphinx-build -W --keep-going docs docs/_build/html
```

Common sources of warnings are unexpected indentation inside docstrings (a `:` followed by a more-indented continuation line), inline bracket-wrapped lists in return descriptions, and missing or malformed cross-references.

### Math does not render in the built docs

Inline math in docstrings must use reStructuredText syntax (`:math:\`expression\``) and docstrings containing LaTeX must be raw strings (`r"""`). Markdown math (`$...$`) is only supported in MyST Markdown files (`.md`), not in Python docstrings. See `jesterTOV/eos/base.py` for reference examples.
