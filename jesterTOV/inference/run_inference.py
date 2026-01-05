#!/usr/bin/env python
r"""
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
from .samplers.factory import create_sampler
from .samplers.jester_sampler import JesterSampler
from .result import InferenceResult
from jesterTOV.logging_config import get_logger

# Set up logger
logger = get_logger("jester")


def determine_keep_names(config, prior):
    """
    Determine which parameters need to be preserved in transform output.

    This function checks which likelihoods are enabled and determines which
    prior parameters need to be kept in the transform output for likelihood
    evaluation.

    Parameters
    ----------
    config : InferenceConfig
        Configuration object with likelihood settings
    prior : CombinePrior
        Prior object with parameter names

    Returns
    -------
    list[str] | None
        List of parameter names to keep, or None if no special handling needed

    Raises
    ------
    ValueError
        If a required parameter is missing from the prior
    """
    keep_names = []

    # ChiEFT likelihood requires 'nbreak' parameter
    chieft_enabled = any(lk.enabled and lk.type == "chieft" for lk in config.likelihoods)
    if chieft_enabled:
        if "nbreak" not in prior.parameter_names:
            raise ValueError(
                "ChiEFT likelihood is enabled but 'nbreak' parameter is not in the prior. "
                "Please add 'nbreak' to your prior specification file. "
                f"Current prior parameters: {prior.parameter_names}"
            )
        keep_names.append("nbreak")
        logger.info("ChiEFT likelihood enabled: 'nbreak' parameter will be preserved in transform output")

    # Add future likelihood parameter requirements here
    # Example:
    # some_other_likelihood_enabled = any(lk.enabled and lk.type == "other" for lk in config.likelihoods)
    # if some_other_likelihood_enabled:
    #     if "some_param" not in prior.parameter_names:
    #         raise ValueError("...")
    #     keep_names.append("some_param")

    return keep_names if keep_names else None


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
        logger.info("Adding _random_key prior for likelihood sampling")
        random_key_prior = UniformPrior(
            float(0), float(2**32 - 1), parameter_names=["_random_key"]
        )
        # Flatten the prior structure to avoid nested CombinePrior
        prior = CombinePrior(prior.base_prior + [random_key_prior])

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
    transform = create_transform(config.transform, keep_names=keep_names)

    return transform


def setup_likelihood(config, transform):
    """
    Setup combined likelihood from configuration

    Parameters
    ----------
    config : InferenceConfig
        Configuration object
    transform : JesterTransformBase
        Transform instance

    Returns
    -------
    LikelihoodBase
        Combined likelihood instance
    """
    return create_combined_likelihood(config.likelihoods)


def run_sampling(sampler, seed, config, outdir):
    """
    Run MCMC sampling and create InferenceResult

    Parameters
    ----------
    sampler : JesterSampler
        JesterSampler instance (FlowMC, BlackJAX NS, or BlackJAX SMC)
    seed : int
        Random seed for sampling
    config : InferenceConfig
        Configuration object
    outdir : str or Path
        Output directory

    Returns
    -------
    InferenceResult
        Result object containing samples, metadata, and histories
    """
    logger.info(f"Starting MCMC sampling with seed {seed}...")
    start = time.time()
    sampler.sample(jax.random.PRNGKey(seed))
    sampler.print_summary()
    end = time.time()
    runtime = end - start

    logger.info(
        f"Sampling complete! Runtime: {int(runtime / 60)} min {int(runtime % 60)} sec"
    )

    # Generate diagnostic plots for SMC sampler
    from .samplers.blackjax_smc import BlackJAXSMCSampler
    if isinstance(sampler, BlackJAXSMCSampler):
        logger.info("Generating SMC diagnostic plots...")
        sampler.plot_diagnostics(outdir=outdir, filename="smc_diagnostics.png")

    ### POSTPROCESSING ###

    # Get sample counts (FlowMC has train/production split, others don't)
    nb_samples_training = sampler.get_n_samples(training=True)
    nb_samples_production = sampler.get_n_samples(training=False)
    total_nb_samples = nb_samples_training + nb_samples_production

    logger.info(f"Number of samples generated in training: {nb_samples_training}")
    logger.info(f"Number of samples generated in production: {nb_samples_production}")
    logger.info(f"Total number of samples: {total_nb_samples}")

    # Create InferenceResult from sampler output
    logger.info("Creating InferenceResult from sampler output...")
    result = InferenceResult.from_sampler(
        sampler=sampler,
        config=config,
        runtime=runtime,
        training=False,  # Use production/final samples
    )

    # Save the runtime info to text file for backward compatibility
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "runtime.txt"), "w") as f:
        f.write(f"{runtime}\n")
        f.write(f"Training samples: {nb_samples_training}\n")
        f.write(f"Production samples: {nb_samples_production}\n")

    return result


def generate_eos_samples(
    config, result, transform_eos, outdir, n_eos_samples=10_000
):
    """
    Generate EOS curves from sampled parameters and add to InferenceResult

    Parameters
    ----------
    config : InferenceConfig
        Configuration object
    result : InferenceResult
        Result object with posterior samples
    transform_eos : JesterTransformBase
        Transform for generating full EOS quantities
    outdir : str or Path
        Output directory
    n_eos_samples : int, optional
        Number of EOS samples to generate
    """
    # Get log_prob from result
    log_prob = result.posterior["log_prob"]

    # Cap n_eos_samples at available sample size
    n_available = len(log_prob)
    if n_eos_samples > n_available:
        logger.warning(f"Requested {n_eos_samples} EOS samples but only {n_available} available.")
        logger.warning(f"Using all {n_available} samples instead.")
        n_eos_samples = n_available

    logger.info(f"Generating {n_eos_samples} EOS samples...")

    # Randomly select samples
    idx = np.random.choice(np.arange(len(log_prob)), size=n_eos_samples, replace=False)

    # Filter out metadata fields and derived EOS quantities that aren't transform parameters
    # Only keep fields that are NEP/CSE parameters for the transform
    exclude_keys = {'weights', 'ess', 'logL', 'logL_birth', 'log_prob', '_sampler_specific',
                    'masses_EOS', 'radii_EOS', 'Lambdas_EOS', 'n', 'p', 'e', 'cs2'}
    param_samples = {k: v for k, v in result.posterior.items() if k not in exclude_keys}

    chosen_samples = {k: jnp.array(v[idx]) for k, v in param_samples.items()}

    # CRITICAL: Also filter sampler-specific fields to match the selected samples
    # These fields (log_prob, weights, ess, etc.) must be filtered to match the EOS samples
    # Store the original full log_prob for reference, then update with filtered version
    result.posterior['log_prob_full'] = result.posterior['log_prob'].copy()
    result.posterior['log_prob'] = result.posterior['log_prob'][idx]

    # Filter other sampler-specific fields if present
    sampler_fields_to_filter = ['weights', 'ess', 'logL', 'logL_birth']
    for field in sampler_fields_to_filter:
        if field in result.posterior:
            result.posterior[f'{field}_full'] = result.posterior[field].copy()
            result.posterior[field] = result.posterior[field][idx]

    logger.info(f"Filtered log_prob and sampler fields from {len(log_prob)} to {len(result.posterior['log_prob'])} samples")

    # Generate EOS curves with batched processing
    logger.info("Running TOV solver with batched processing...")
    my_forward = jax.jit(transform_eos.forward)

    # Get batch size from config
    batch_size = config.sampler.log_prob_batch_size
    logger.info(f"Using batch size: {batch_size}")

    # Run with batched processing (JIT compilation happens on first batch)
    TOV_start = time.time()
    transformed_samples = jax.lax.map(my_forward, chosen_samples, batch_size=batch_size)
    TOV_end = time.time()
    logger.info(f"TOV solve time: {TOV_end - TOV_start:.2f} s ({n_eos_samples} samples)")

    # Add derived EOS quantities to result
    result.add_derived_eos(transformed_samples)
    logger.info("Derived EOS quantities added to InferenceResult")


def main(config_path: str):
    """Main inference script

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    outdir = config.sampler.output_dir

    # Print GPU info
    logger.info(f"JAX devices: {jax.devices()}")

    # Validation only
    if config.validate_only:
        logger.info("Configuration valid!")
        return

    # Setup components
    logger.info("Setting up prior...")
    prior = setup_prior(config)

    # Log detailed prior information
    logger.info(f"Prior has {prior.n_dim} dimensions")
    logger.info(f"Prior parameter names: {prior.parameter_names}")

    # Get individual priors - CombinePrior stores them in base_prior attribute
    if hasattr(prior, 'base_prior') and isinstance(prior.base_prior, list):
        individual_priors = prior.base_prior
    else:
        # For single priors, wrap in a list
        individual_priors = [prior]

    # Flatten the list of priors (in case of nested CombinePriors)
    def flatten_priors(priors_list):
        result = []
        for p in priors_list:
            if hasattr(p, 'base_prior') and isinstance(p.base_prior, list):
                result.extend(flatten_priors(p.base_prior))
            else:
                result.append(p)
        return result

    all_priors = flatten_priors(individual_priors)

    # Log each prior with its parameters
    idx = 0
    for param_prior in all_priors:
        for name in param_prior.parameter_names:
            if hasattr(param_prior, 'xmin') and hasattr(param_prior, 'xmax'):
                logger.info(f"  [{idx}] {name}: Uniform({param_prior.xmin}, {param_prior.xmax})")
            else:
                logger.info(f"  [{idx}] {name}: {type(param_prior).__name__}")
            idx += 1

    # Determine which parameters need to be preserved in transform output
    # based on enabled likelihoods (validates required parameters exist in prior)
    keep_names = determine_keep_names(config, prior)

    logger.info("Setting up transform...")
    transform = setup_transform(config, keep_names=keep_names)

    # Log transform details
    logger.info(f"Transform type: {config.transform.type}")
    if config.transform.type == "metamodel_cse":
        logger.info(f"  nb_CSE: {config.transform.nb_CSE}")
    logger.info(f"  ndat_metamodel: {config.transform.ndat_metamodel}")
    logger.info(f"  ndat_TOV: {config.transform.ndat_TOV}")
    logger.info(f"  nmax_nsat: {config.transform.nmax_nsat}")
    if keep_names:
        logger.info(f"  Preserving parameters in output: {keep_names}")

    logger.info("Setting up likelihood...")
    likelihood = setup_likelihood(config, transform)

    # Log detailed likelihood information
    enabled_likelihoods = [lk for lk in config.likelihoods if lk.enabled]
    logger.info(f"Number of enabled likelihoods: {len(enabled_likelihoods)}")
    for lk in enabled_likelihoods:
        logger.info(f"  - {lk.type.upper()}")
        if lk.type == "gw":
            events = lk.parameters.get("events", [])
            logger.info(f"    Events: {[e['name'] for e in events]}")
            logger.info(f"    N masses evaluation: {lk.parameters.get('N_masses_evaluation', 20)}")
        elif lk.type == "nicer":
            pulsars = lk.parameters.get("pulsars", [])
            logger.info(f"    Pulsars: {[p['name'] for p in pulsars]}")
            logger.info(f"    N masses evaluation: {lk.parameters.get('N_masses_evaluation', 100)}")
            logger.info(f"    KDE bandwidth: {lk.parameters.get('kde_bandwidth', 0.02)}")
        elif lk.type == "radio":
            pulsars = lk.parameters.get("pulsars", [])
            logger.info(f"    Pulsars: {[p['name'] for p in pulsars]}")
        elif lk.type == "chieft":
            logger.info(f"    Low bound file: {lk.parameters.get('low_filename', 'default')}")
            logger.info(f"    High bound file: {lk.parameters.get('high_filename', 'default')}")
            logger.info(f"    Integration points: {lk.parameters.get('nb_n', 100)}")
        elif lk.type == "rex":
            logger.info(f"    Experiment: {lk.parameters.get('experiment_name', 'PREX')}")
        elif lk.type == "constraints_eos":
            logger.info(f"    Causality penalty: {lk.parameters.get('penalty_causality', -1e10)}")
            logger.info(f"    Stability penalty: {lk.parameters.get('penalty_stability', -1e5)}")
        elif lk.type == "constraints_tov":
            logger.info(f"    TOV failure penalty: {lk.parameters.get('penalty_tov', -1e10)}")

    logger.info(f"Setting up {config.sampler.type} sampler...")
    sampler = create_sampler(
        config=config.sampler,
        prior=prior,
        likelihood=likelihood,
        likelihood_transforms=[transform],
        seed=config.seed,
    )

    # Log detailed sampler configuration
    logger.info("=" * 60)
    logger.info("Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"Transform: {config.transform.type}")
    logger.info(f"Random seed: {config.seed}")
    logger.info(f"Sampler type: {config.sampler.type}")
    logger.info("Sampler Configuration:")

    # Log sampler-specific config fields
    if config.sampler.type == "flowmc":
        logger.info(f"  Chains: {config.sampler.n_chains}")
        logger.info(f"  Training loops: {config.sampler.n_loop_training}")
        logger.info(f"  Production loops: {config.sampler.n_loop_production}")
        logger.info(f"  Local steps per loop: {config.sampler.n_local_steps}")
        logger.info(f"  Global steps per loop: {config.sampler.n_global_steps}")
        logger.info(f"  Training epochs: {config.sampler.n_epochs}")
        logger.info(f"  Learning rate: {config.sampler.learning_rate}")
        logger.info(f"  Training thinning: {config.sampler.train_thinning}")
        logger.info(f"  Output thinning: {config.sampler.output_thinning}")
    elif config.sampler.type == "nested_sampling":
        logger.info(f"  Live points: {config.sampler.n_live}")
        logger.info(f"  Delete fraction: {config.sampler.n_delete_frac}")
        logger.info(f"  Target MCMC steps: {config.sampler.n_target}")
        logger.info(f"  Termination dlogZ: {config.sampler.termination_dlogz}")
    elif config.sampler.type == "smc":
        logger.info(f"  Particles: {config.sampler.n_particles}")
        logger.info(f"  MCMC steps: {config.sampler.n_mcmc_steps}")
        logger.info(f"  Target ESS: {config.sampler.target_ess}")

    # Log shared sampler config fields
    logger.info(f"  EOS samples to generate: {config.sampler.n_eos_samples}")
    logger.info(f"  Output directory: {outdir}")
    logger.info("=" * 60 + "\n")

    # Dry run option
    if config.dry_run:
        logger.info("Dry run complete!")
        return

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Test likelihood evaluation # FIXME: this should throw an error if a Nan is found
    logger.info("Testing likelihood evaluation...")
    test_samples = prior.sample(jax.random.PRNGKey(0), 3)
    test_samples_transformed = jax.vmap(transform.forward)(test_samples)
    test_log_prob = jax.vmap(likelihood.evaluate)(test_samples_transformed, {})
    logger.info(f"Test log probabilities: {test_log_prob}")

    # TODO: Enable transform caching to avoid redundant TOV solver calls
    # Caching infrastructure exists but has JAX tracing issues (see CLAUDE.md)
    # Need to implement caching outside JAX trace context for all samplers:
    # - FlowMC: Cache during production phase
    # - BlackJAX SMC: Cache final temperature samples
    # - BlackJAX NS-AW: Cache all samples
    # For now, caching is DISABLED and we fall back to recomputation
    # sampler.enable_transform_caching()  # DISABLED

    # Run inference
    result = run_sampling(sampler, config.seed, config, outdir)

    # Generate EOS quantities from cached transforms (if available) or recompute
    # With caching: zero TOV solver calls (uses cached outputs from sampling)
    # Without caching: one TOV solver call (fallback to recomputation)
    result.add_eos_from_transform(
        transform=transform,  # Use the same transform from sampling
        n_eos_samples=config.sampler.n_eos_samples,
        batch_size=config.sampler.log_prob_batch_size,
        sampler=sampler,  # Pass sampler to check for cached transforms
    )

    # Save unified HDF5 file
    result_path = os.path.join(outdir, "results.h5")
    result.save(result_path)
    logger.info(f"Results saved to {result_path}")

    # Run postprocessing if enabled
    if config.postprocessing.enabled:
        logger.info("\n" + "=" * 60)
        logger.info("Running postprocessing...")
        logger.info("=" * 60)
        from jesterTOV.inference.postprocessing.postprocessing import generate_all_plots

        generate_all_plots(
            outdir=outdir,
            prior_dir=config.postprocessing.prior_dir,
            make_cornerplot_flag=config.postprocessing.make_cornerplot,
            make_massradius_flag=config.postprocessing.make_massradius,
            make_pressuredensity_flag=config.postprocessing.make_pressuredensity,
            make_histograms_flag=config.postprocessing.make_histograms,
        )
        logger.info(f"\nPostprocessing complete! Plots saved to {outdir}")

    logger.info(f"\nInference complete! Results saved to {outdir}")


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
        logger.error("Usage: run_jester_inference <config.yaml>")
        logger.info("\nExamples:")
        logger.info("  run_jester_inference config.yaml")
        logger.info("  run_jester_inference examples/inference/full_inference/config.yaml")
        logger.info("\nOptions like dry_run and validate_only should be set in the YAML config file.")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        logger.error("Usage: python -m jesterTOV.inference.run_inference <config.yaml>")
        sys.exit(1)

    main(sys.argv[1])
