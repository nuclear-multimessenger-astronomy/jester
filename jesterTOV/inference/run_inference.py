#!/usr/bin/env python
"""
Modular inference script for jesterTOV
"""

# FIXME: Need to organize this a bit better so that it is more modular and easier to process

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

# FIXME: make a flag that turns this on/off and document it, turn ON by default
# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# FIXME: make a flag that turns this on/off and document it, turn OFF by default
# jax.config.update("jax_debug_nans", True)

from .config.parser import load_config
from .priors.parser import parse_prior_file
from .transforms.factory import create_transform
from .likelihoods.factory import create_combined_likelihood
from .samplers import setup_flowmc_sampler, JesterSampler
# FIXME: DataLoader removed - need to implement data loading functions


def setup_prior(config):
    """
    Setup prior from configuration

    Parameters
    ----------
    config : InferenceConfig
        Configuration object

    Returns
    -------
    CombinePrior
        Combined prior object
    """
    from .base.prior import UniformPrior, CombinePrior

    # Determine conditional parameters
    nb_CSE = (
        config.transform.nb_CSE if config.transform.type == "metamodel_cse" else 0
    )

    # Check if GW or NICER likelihoods are enabled (both need _random_key)
    needs_random_key = False
    for lk in config.likelihoods:
        if lk.enabled and lk.type in ["gw", "nicer"]:
            needs_random_key = True
            break

    # Parse prior file
    prior = parse_prior_file(
        config.prior.specification_file,
        nb_CSE=nb_CSE,
    )

    # Add _random_key prior if GW or NICER likelihoods are enabled
    if needs_random_key:
        print("Adding _random_key prior for likelihood sampling")
        random_key_prior = UniformPrior(
            float(0), float(2**32 - 1), parameter_names=["_random_key"]
        )
        prior = CombinePrior([prior, random_key_prior])

    return prior


def setup_transform(config, keep_names=None):
    """
    Setup transform from configuration

    Parameters
    ----------
    config : InferenceConfig
        Configuration object
    keep_names : list[str], optional
        Parameter names to keep in transformed output

    Returns
    -------
    JesterTransformBase
        Transform instance
    """
    transform = create_transform(config.transform)

    # TODO: Handle name mapping and keep_names properly
    # This will require updates to the transform base class

    return transform


def setup_likelihood(config, transform, data_loader=None):
    """
    Setup combined likelihood from configuration

    Parameters
    ----------
    config : InferenceConfig
        Configuration object
    transform : JesterTransformBase
        Transform instance
    data_loader : None
        DEPRECATED - data loading will be handled differently

    Returns
    -------
    LikelihoodBase
        Combined likelihood instance
    """
    # FIXME: data_loader parameter is deprecated, pass None for now
    return create_combined_likelihood(
        config.likelihoods, data_loader
    )


def run_sampling(flowmc_sampler, seed, outdir):
    """
    Run MCMC sampling and save results

    Parameters
    ----------
    flowmc_sampler : JesterSampler
        JesterSampler instance configured with flowMC backend
    seed : int
        Random seed for sampling
    outdir : str or Path
        Output directory

    Returns
    -------
    dict
        Dictionary containing samples and log probabilities
    """
    print(f"Starting MCMC sampling with seed {seed}...")
    start = time.time()
    flowmc_sampler.sample(jax.random.PRNGKey(seed))
    flowmc_sampler.print_summary()
    end = time.time()
    runtime = end - start

    print(
        f"Sampling complete! Runtime: {int(runtime / 60)} min {int(runtime % 60)} sec"
    )

    ### POSTPROCESSING ###

    # Training (just to count number of samples)
    sampler_state = flowmc_sampler.sampler.get_sampler_state(training=True)
    log_prob_train = sampler_state["log_prob"].flatten()
    nb_samples_training = len(log_prob_train)

    # Production (for saving and plotting)
    sampler_state = flowmc_sampler.sampler.get_sampler_state(training=False)

    # Get the samples as a dictionary
    samples_named = flowmc_sampler.get_samples()
    samples_named_for_saving = {k: np.array(v) for k, v in samples_named.items()}
    samples_named_flat = {k: np.array(v).flatten() for k, v in samples_named.items()}

    # Get the log prob
    log_prob = np.array(sampler_state["log_prob"]).flatten()
    nb_samples_production = len(log_prob)
    total_nb_samples = nb_samples_training + nb_samples_production

    # Save the final results
    print(f"Saving results to {outdir}")
    os.makedirs(outdir, exist_ok=True)

    result_path = os.path.join(outdir, "results_production.npz")
    np.savez(result_path, log_prob=log_prob, **samples_named_for_saving)

    print(f"Number of samples generated in training: {nb_samples_training}")
    print(f"Number of samples generated in production: {nb_samples_production}")
    print(f"Total number of samples: {total_nb_samples}")

    # Save the runtime
    with open(os.path.join(outdir, "runtime.txt"), "w") as f:
        f.write(f"{runtime}\n")
        f.write(f"Training samples: {nb_samples_training}\n")
        f.write(f"Production samples: {nb_samples_production}\n")

    return {
        "samples": samples_named_flat,
        "samples_for_saving": samples_named_for_saving,
        "log_prob": log_prob,
        "runtime": runtime,
    }


def generate_eos_samples(
    config, samples_dict, transform_eos, outdir, n_eos_samples=10_000
):
    """
    Generate EOS curves from sampled parameters

    Parameters
    ----------
    config : InferenceConfig
        Configuration object
    samples_dict : dict
        Dictionary of sampled parameters
    transform_eos : JesterTransformBase
        Transform for generating full EOS quantities
    outdir : str or Path
        Output directory
    n_eos_samples : int, optional
        Number of EOS samples to generate
    """
    samples = samples_dict["samples"]
    log_prob = samples_dict["log_prob"]

    # Cap n_eos_samples at available sample size
    n_available = len(log_prob)
    if n_eos_samples > n_available:
        print(f"Warning: Requested {n_eos_samples} EOS samples but only {n_available} available.")
        print(f"Using all {n_available} samples instead.")
        n_eos_samples = n_available

    print(f"Generating {n_eos_samples} EOS samples...")

    # Randomly select samples
    idx = np.random.choice(np.arange(len(log_prob)), size=n_eos_samples, replace=False)

    chosen_samples = {k: jnp.array(v[idx]) for k, v in samples.items()}

    # Generate EOS curves (with JIT compilation)
    print("JIT compiling and running TOV solver...")
    my_forward = jax.jit(transform_eos.forward)

    # Warm up JIT
    warmup_size = min(100, n_available)
    test_idx = np.random.choice(np.arange(len(log_prob)), size=warmup_size, replace=False)
    test_samples = {k: jnp.array(v[test_idx]) for k, v in samples.items()}
    _ = jax.vmap(my_forward)(test_samples)

    # Run full batch
    TOV_start = time.time()
    transformed_samples = jax.vmap(my_forward)(chosen_samples)
    TOV_end = time.time()
    print(f"TOV solve time: {TOV_end - TOV_start:.2f} s")

    # Combine and save
    chosen_samples.update(transformed_samples)
    selected_log_prob = log_prob[idx]

    eos_path = os.path.join(outdir, "eos_samples.npz")
    np.savez(eos_path, log_prob=selected_log_prob, **chosen_samples)
    print(f"EOS samples saved to {outdir}/eos_samples.npz")


def main(config_path: str):
    """Main inference script

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    """
    # Load configuration
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    outdir = config.sampler.output_dir

    # Print GPU info
    print(f"JAX devices: {jax.devices()}")

    # Validation only
    if config.validate_only:
        print("Configuration valid!")
        return

    # FIXME: DataLoader was removed - need to implement data loading functionality
    # The data_paths from config should be used to configure data loading functions
    # For now, set data_loader to None
    data_loader = None

    # Setup components
    print("Setting up prior...")
    prior = setup_prior(config)
    print(f"Prior parameter names: {prior.parameter_names}")

    print("Setting up transform...")
    transform = setup_transform(config)

    # Create EOS-only transform for postprocessing
    # TODO: This needs proper implementation with keep_names
    transform_eos = setup_transform(config)

    print("Setting up likelihood...")
    likelihood = setup_likelihood(config, transform, data_loader)

    print("Setting up flowMC sampler...")
    flowmc_sampler = setup_flowmc_sampler(
        config.sampler,
        prior,
        likelihood,
        transform,
        seed=config.seed,
    )

    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Transform type: {config.transform.type}")
    print(f"Number of chains: {config.sampler.n_chains}")
    print(f"Training loops: {config.sampler.n_loop_training}")
    print(f"Production loops: {config.sampler.n_loop_production}")
    print(f"Output directory: {outdir}")
    print(f"Random seed: {config.seed}")
    print("=" * 60 + "\n")

    # Dry run option
    if config.dry_run:
        print("Dry run complete!")
        return

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Test likelihood evaluation # FIXME: this should throw an error if a Nan is found
    print("Testing likelihood evaluation...")
    test_samples = prior.sample(jax.random.PRNGKey(0), 3)
    test_samples_transformed = jax.vmap(transform.forward)(test_samples)
    test_log_prob = jax.vmap(likelihood.evaluate)(test_samples_transformed, {})
    print(f"Test log probabilities: {test_log_prob}")

    # Run inference
    results = run_sampling(flowmc_sampler, config.seed, outdir)

    # Generate EOS samples
    n_eos_samples = config.sampler.n_eos_samples
    generate_eos_samples(config, results, transform_eos, outdir, n_eos_samples)

    # Run postprocessing if enabled
    if config.postprocessing.enabled:
        print("\n" + "=" * 60)
        print("Running postprocessing...")
        print("=" * 60)
        from jesterTOV.inference.postprocessing.postprocessing import generate_all_plots

        generate_all_plots(
            outdir=outdir,
            prior_dir=config.postprocessing.prior_dir,
            make_cornerplot_flag=config.postprocessing.make_cornerplot,
            make_massradius_flag=config.postprocessing.make_massradius,
            make_pressuredensity_flag=config.postprocessing.make_pressuredensity,
            make_histograms_flag=config.postprocessing.make_histograms,
            make_contours_flag=config.postprocessing.make_contours,
        )
        print(f"\nPostprocessing complete! Plots saved to {outdir}")

    print(f"\nInference complete! Results saved to {outdir}")


def cli_entry_point():
    """
    Entry point for console script.

    Allows running inference with:
        run_jester_inference config.yaml

    Instead of:
        python -m jesterTOV.inference.run_inference config.yaml
    """
    import sys

    # Check for exactly one argument (the config file path)
    if len(sys.argv) != 2:
        print("Usage: run_jester_inference <config.yaml>")
        print("\nExamples:")
        print("  run_jester_inference config.yaml")
        print("  run_jester_inference examples/inference/full_inference/config.yaml")
        print("\nOptions like dry_run and validate_only should be set in the YAML config file.")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m jesterTOV.inference.run_inference <config.yaml>")
        sys.exit(1)

    main(sys.argv[1])
