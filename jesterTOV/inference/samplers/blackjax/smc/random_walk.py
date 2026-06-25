"""SMC with Gaussian Random Walk Metropolis-Hastings kernel."""

from typing import Callable, cast

import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array

from jesterTOV.inference.base import (
    LikelihoodBase,
    Prior,
    BijectiveTransform,
    NtoMTransform,
)
from jesterTOV.inference.config.schema import SMCRandomWalkSamplerConfig
from jesterTOV.inference.samplers.blackjax.smc.base import BlackjaxSMCSampler
from jesterTOV.logging_config import get_logger

from blackjax.mcmc import random_walk
from blackjax.smc import extend_params
from blackjax.smc.tuning.from_particles import particles_covariance_matrix

logger = get_logger("jester")


class BlackJAXSMCRandomWalkSampler(BlackjaxSMCSampler):
    """SMC with Gaussian Random Walk Metropolis-Hastings kernel.

    This sampler uses a simple random walk proposal with adaptive sigma tuning.
    Recommended for most use cases due to simplicity and robustness.

    The proposal covariance is adapted from current particles at each tempering
    step and scaled by a fixed sigma^2 parameter.

    Parameters
    ----------
    likelihood : LikelihoodBase
        Likelihood object
    prior : Prior
        Prior object
    sample_transforms : list[BijectiveTransform]
        Sample transforms (typically empty for SMC)
    likelihood_transforms : list[NtoMTransform]
        Likelihood transforms
    config : SMCRandomWalkSamplerConfig
        Random walk SMC configuration
    seed : int, optional
        Random seed (default: 0)
    """

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform],
        likelihood_transforms: list[NtoMTransform],
        config: SMCRandomWalkSamplerConfig,
        seed: int = 0,
    ) -> None:
        """Initialize Random Walk SMC sampler."""
        super().__init__(
            likelihood, prior, sample_transforms, likelihood_transforms, config, seed
        )

    def _get_kernel_name(self) -> str:
        """Return kernel name."""
        return "random_walk"

    def _setup_mcmc_kernel(
        self,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        logposterior_fn: Callable,
        initial_particles: Array,
    ) -> tuple[Callable, Callable, dict, Callable]:
        """Setup Random Walk kernel with covariance adaptation.

        The proposal covariance is computed from current particles and scaled by a
        fixed sigma^2 factor. Only the covariance shape is adapted, not the overall scale.

        Parameters
        ----------
        logprior_fn : Callable
            Log prior function (not used for random walk)
        loglikelihood_fn : Callable
            Log likelihood function (not used for random walk)
        logposterior_fn : Callable
            Log posterior function (not used for random walk)
        initial_particles : Array
            Initial particle positions for computing initial covariance

        Returns
        -------
        tuple[Callable, Callable, dict, Callable]
            (mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn)
        """
        # Type narrow config for this subclass
        config = cast(SMCRandomWalkSamplerConfig, self.config)

        logger.info("Using random walk kernel")
        logger.info(f"Fixed sigma scaling: {config.random_walk_sigma}")

        # Setup random walk kernel with additive step
        kernel = random_walk.build_additive_step()

        # Compute initial covariance from initial particles
        init_cov = particles_covariance_matrix(initial_particles)
        # Ensure 2D array (n_dim, n_dim) even for 1D problems
        init_cov = jnp.atleast_2d(init_cov)
        # Scale by fixed sigma^2
        init_cov = init_cov * (config.random_walk_sigma**2)

        init_params = {"cov": init_cov}

        # Define parameter update function with covariance adaptation only
        def mcmc_parameter_update_fn(key, state, info):
            """Adapt proposal covariance based on current particle distribution.

            The covariance matrix is computed from current particles and scaled by
            the fixed sigma^2 parameter. No scale adaptation is performed.
            """
            # Note: state here is TemperedSMCState, particles are at state.particles

            # Compute covariance matrix from current particles
            cov = particles_covariance_matrix(state.particles)
            # Ensure 2D array (n_dim, n_dim) even for 1D problems
            cov = jnp.atleast_2d(cov)

            # Scale covariance by fixed sigma^2
            scaled_cov = cov * (config.random_walk_sigma**2)

            return extend_params({"cov": scaled_cov})  # type: ignore[arg-type]

        # Wrap kernel to match expected signature
        def mcmc_step_fn(rng_key, state, logdensity_fn, **params):
            """Random walk step function with multivariate normal proposal."""
            cov = params.get("cov", init_cov)

            def proposal_distribution(key, position):
                """Multivariate normal proposal using covariance matrix."""
                x, ravel_fn = flatten_util.ravel_pytree(position)
                return ravel_fn(
                    jax.random.multivariate_normal(key, jnp.zeros_like(x), cov)
                )

            return kernel(rng_key, state, logdensity_fn, proposal_distribution)

        # Init function for random walk
        mcmc_init_fn = random_walk.init

        return mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn
