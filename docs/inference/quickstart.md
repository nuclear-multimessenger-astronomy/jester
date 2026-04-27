# Quick Start

**Get started with Bayesian EOS inference in 5 minutes**

This guide explains how to run Bayesian inference with ``jester`` using a simple configuration.
As an example, we will run inference on the {ref}`metamodel + speed-of-sound extension (CSE) <eos-metamodel-cse>` EOS parametrization, using the {ref}`GR TOV solver <tov-gr>`, and sampling the parametrization with {ref}`sequential Monte Carlo <sampler-smc>`.
To constrain the EOS, we will use the {ref}`chiral effective field theory (chiEFT) <likelihood-chieft>` likelihood.
This inference is fast enough to be executed locally on a laptop, making it ideal for testing whether your installation works successfully and for getting familiar with running ``jester``!


## Running your first inference

### ``jester`` inference overview

To run a ``jester`` inference, only two files are required:
1. A ``config.yaml`` file: this specifies all the desired settings and hyperparameters for ``jester``.
2. A ``prior.prior`` file: this specifies the prior distributions on the parameters to be sampled, which includes the EOS parameters and, possibly, additional parameters needed for some TOV solvers.

These two files should be stored in one directory.
The following two sections provide more detail and example files.

### Specify ``config.yaml`` file

The inference in ``jester`` handles everything under the hood and only needs a config file to start up and specify the inference settings.
For this quickstart example, we will use the following ``config.yaml`` file: 

```bash
# ChiEFT-only inference example configuration
# SMC with Random Walk kernel with metamodel+CSE EOS
seed: 44

# Execution options
dry_run: false
validate_only: false

# EOS type and transform settings
transform:
  type: "metamodel_cse"
  ndat_metamodel: 100
  nmax_nsat: 25.0
  nb_CSE: 8
  min_nsat_TOV: 0.75
  ndat_TOV: 100
  nb_masses: 100
  crust_name: "DH"

# Prior specification file
prior:
  specification_file: "prior.prior"

# Likelihood constraints (only chiEFT + EOS constraints enabled)
likelihoods:
  # Physical constraint likelihood - enforces EOS validity
  - type: "constraints_eos"
    enabled: true

  # ChiEFT likelihood - constrains low-density EOS using chiral effective field theory
  - type: "chieft"
    enabled: true
    parameters:
      nb_n: 100

# SMC configuration with Random Walk kernel
sampler:
  type: "smc-rw"
  n_particles: 2000        # Number of particles evolved at each inference step
  n_mcmc_steps: 10         # MCMC steps per tempering level
  target_ess: 0.9          # Target effective sample size (90%)
  random_walk_sigma: 0.1   # Gaussian random walk step size scale
  output_dir: "./outdir/"

# Postprocessing configuration
postprocessing:
  enabled: true
  make_cornerplot: false
  make_massradius: true
  make_pressuredensity: true
```

```{note}
**Further Reading:**
- Full explanation for ``config.yaml`` settings: See {ref}`yaml-reference`
- EOS parametrization explanation: See {ref}`eos-metamodel-cse`
- TOV solver explanation: See {ref}`tov-gr`
- Sampler explanation: See {ref}`sampler-smc`
- ChiEFT likelihood explanation: See {ref}`likelihood-chieft`
``` 

### Specify ``prior.prior`` file

Create the ``prior.prior`` file.
For the metamodel+CSE parametrization, this file must specify the parameters for the metamodel.
The priors on the CSE parameters are generated automatically based on the ``transform`` settings specified in ``config.yaml``.
The ``log`` file that is created during inference specifies the complete prior after prior generation has completed.

Here is an example file with example ranges for the different parameters:

```python
E_sat = UniformPrior(-16.1, -15.9, parameter_names=["E_sat"])
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])
```

### Run the inference

From the directory containing the two files specified above, the inference can be launched with a single command that points to the ``config.yaml`` file (the path to the prior file is specified in the ``config.yaml`` file):
```bash
run_jester_inference config.yaml
```

Note: This command is only recognized after activating the environment where `jester` was installed.

## Result

The ``jester`` logger should have created a large ``log.out`` file that you can review to understand the various steps that ``jester`` has taken.

The results from the ``jester`` inference, such as the EOS posterior samples, are saved to an HDF5 file.
This file can be loaded easily with the {class}`~jesterTOV.inference.result.InferenceResult` class.

For details on loading and analyzing results, see the {class}`~jesterTOV.inference.result.InferenceResult` API reference.

## Postprocessing

``jester`` automatically creates postprocessing diagnostic plots that are saved to the ``outdir``; these can be toggled in the ``config.yaml`` file. Some interesting plots to check out:
- ``smc_diagnostics.png``: Shows the behavior of the SMC sampler during inference.
- ``mass_radius_plot.pdf``: Displays the mass-radius curves corresponding to the posterior EOS samples, color-coded according to the magnitude of their posterior (log-)probability.

For a full walkthrough of how to load the HDF5 result file and analyze the posterior samples, see {doc}`../examples/inference/result`.

At this point, you should have successfully run your first ``jester`` inference. If you encountered any errors, please raise an issue on GitHub explaining the problem so we can fix the source code and/or this documentation.

## Next steps

So, what's next?

1. **Use other EOS constraints**: Try running ``jester`` with additional EOS constraints enabled.
   ```{note}
   Inferences that involve TOV quantities (e.g., mass, radius, tidal deformabilities of neutron stars) are slower during sampling and might require GPU acceleration. Trouble with getting started on GPUs? Refer to this page (TODO: the page does not exist yet so we will have to make that at some point).
   ```
   For inspiration on config files, refer to the inference examples in the jester repository at ``jester/examples/inference/smc_random_walk`` for more examples with the SMC sampler.

   For a complete list of all configuration options, see the {ref}`YAML Configuration Reference <yaml-reference>`.

2. **Try other samplers**: Example config and prior files can be found at:
   - ``jester/examples/inference/flowMC`` for the ``flowMC`` sampler
   - ``jester/examples/inference/blackjax_ns_aw`` for ``blackjax``'s nested sampler

3. **Try other EOS parametrizations**:
   - ``jester/examples/inference/spectral`` shows how to run inference with the spectral expansion using SMC

4. **Try different TOV solvers**: Unfortunately, this is still work in progress, so stay tuned!

---

**Quick Start Version**: 1.0
**Last Updated**: February 2026