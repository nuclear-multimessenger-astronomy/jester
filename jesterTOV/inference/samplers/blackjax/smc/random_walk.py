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
from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
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

        n_particles = initial_particles.shape[0]

        # Wrap kernel to match expected signature. `scale` is an extra per-particle
        # multiplicative correction on top of the sigma-scaled empirical covariance; it stays
        # at 1.0 (a no-op) unless adaptive_step_size is enabled.
        def mcmc_step_fn(rng_key, state, logdensity_fn, **params):
            """Random walk step function with multivariate normal proposal."""
            cov = params.get("cov", init_cov)
            scale = params.get("scale", 1.0)
            proposal_cov = (scale**2) * cov

            def proposal_distribution(key, position):
                """Multivariate normal proposal using covariance matrix."""
                x, ravel_fn = flatten_util.ravel_pytree(position)
                return ravel_fn(
                    jax.random.multivariate_normal(key, jnp.zeros_like(x), proposal_cov)
                )

            return kernel(rng_key, state, logdensity_fn, proposal_distribution)

        # Init function for random walk
        mcmc_init_fn = random_walk.init

        if not config.adaptive_step_size:
            init_params = extend_params({"cov": init_cov})  # type: ignore[arg-type]

            # Define parameter update function with covariance adaptation only
            def fixed_sigma_update_fn(key, previous_params, state, info):
                """Adapt proposal covariance based on current particle distribution.

                The covariance matrix is computed from current particles and scaled by
                the fixed sigma^2 parameter. No scale adaptation is performed.
                """
                # Note: state here is TemperedSMCState, particles are at state.particles
                cov = particles_covariance_matrix(state.particles)
                cov = jnp.atleast_2d(cov)
                scaled_cov = cov * (config.random_walk_sigma**2)
                return extend_params({"cov": scaled_cov})  # type: ignore[arg-type]

            return mcmc_step_fn, mcmc_init_fn, init_params, fixed_sigma_update_fn  # type: ignore[return-value]

        # --- Adaptive step size path ---
        logger.info(
            f"Adaptive step size enabled: target acceptance rate = "
            f"{config.target_acceptance_rate}"
        )

        if config.n_pretune_steps > 0:
            logger.info(
                f"Pretuning step size with {config.n_pretune_steps} pilot steps on the "
                "prior (lambda=0), so a poorly-chosen random_walk_sigma self-corrects "
                "before annealing starts..."
            )
            pretune_key = jax.random.PRNGKey(self._seed)
            initial_scale = self._pretune_scale(
                pretune_key,
                mcmc_step_fn,
                mcmc_init_fn,
                logprior_fn,
                initial_particles,
                n_particles,
                config.n_pretune_steps,
                config.target_acceptance_rate,
            )
            logger.info(
                f"Pretuning done: mean scale 1.0 -> {float(initial_scale.mean()):.4f}"
            )
        else:
            initial_scale = jnp.ones(n_particles)

        init_params = {
            "cov": extend_params({"cov": init_cov})["cov"],  # type: ignore[arg-type]
            "scale": initial_scale,
        }

        def adaptive_scale_update_fn(key, previous_params, state, info):
            """Adapt proposal covariance shape and per-particle step-size scale.

            The covariance shape/scale-by-sigma is recomputed from current particles exactly
            as in the non-adaptive path. On top of that, a per-particle multiplicative `scale`
            is adapted every annealing step via BlackJAX's `update_scale_from_acceptance_rate`
            (Roberts-Rosenthal / Robbins-Monro scheme), using the previous step's scale
            (threaded through by `persistent_inner_kernel_tuning`) and this step's realized
            acceptance rate, to hone in on `target_acceptance_rate`.
            """
            cov = particles_covariance_matrix(state.particles)
            cov = jnp.atleast_2d(cov)
            scaled_cov = cov * (config.random_walk_sigma**2)

            previous_scale = previous_params["scale"]
            # info.update_info.acceptance_rate has shape (n_particles, n_mcmc_steps): average
            # over the sub-steps taken within this annealing iteration to get one rate per
            # particle before adapting its scale.
            acceptance_rates = info.update_info.acceptance_rate.reshape(
                n_particles, -1
            ).mean(axis=-1)
            new_scale = update_scale_from_acceptance_rate(
                scales=previous_scale,
                acceptance_rates=acceptance_rates,
                target_acceptance_rate=config.target_acceptance_rate,
            )

            return {
                "cov": extend_params({"cov": scaled_cov})["cov"],  # type: ignore[arg-type]
                "scale": new_scale,
            }

        return mcmc_step_fn, mcmc_init_fn, init_params, adaptive_scale_update_fn

    @staticmethod
    def _pretune_scale(
        key: Array,
        mcmc_step_fn: Callable,
        mcmc_init_fn: Callable,
        logprior_fn: Callable,
        initial_particles: Array,
        n_particles: int,
        n_pretune_steps: int,
        target_acceptance_rate: float,
    ) -> Array:
        """Calibrate a per-particle step-size scale via pilot Metropolis steps on the prior.

        Runs `n_pretune_steps` random-walk steps directly on the initial (prior) particles,
        targeting `logprior_fn` (valid since the tempered posterior at lambda=0 is the prior,
        so this needs no likelihood evaluation), adapting `scale` after every pilot step with
        the same Robbins-Monro update used during annealing. This lets a poorly-chosen initial
        `random_walk_sigma` self-correct before real sampling begins.
        """

        def single_particle_step(step_key, position, scale):
            state = mcmc_init_fn(position, logprior_fn)
            new_state, info = mcmc_step_fn(step_key, state, logprior_fn, scale=scale)
            return new_state.position, info.acceptance_rate

        def body_fn(_, carry):
            key, positions, scale = carry
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, n_particles)
            new_positions, acceptance_rates = jax.vmap(single_particle_step)(
                step_keys, positions, scale
            )
            new_scale = update_scale_from_acceptance_rate(
                scales=scale,
                acceptance_rates=acceptance_rates,
                target_acceptance_rate=target_acceptance_rate,
            )
            return key, new_positions, new_scale

        init_scale = jnp.ones(n_particles)
        init_carry = (key, initial_particles, init_scale)
        _, _, final_scale = jax.lax.fori_loop(0, n_pretune_steps, body_fn, init_carry)
        return final_scale
