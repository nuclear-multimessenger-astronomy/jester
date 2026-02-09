"""SMC with NUTS kernel and Hessian-based mass matrix adaptation.

WARNING: This sampler is EXPERIMENTAL. Use with caution and validate results carefully.
"""

from typing import Callable, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array

from jesterTOV.inference.base import (
    LikelihoodBase,
    Prior,
    BijectiveTransform,
    NtoMTransform,
)
from jesterTOV.inference.config.schema import SMCNUTSSamplerConfig
from jesterTOV.inference.samplers.blackjax.smc.base import BlackjaxSMCSampler
from jesterTOV.logging_config import get_logger

from blackjax import nuts
from blackjax.smc import extend_params

logger = get_logger("jester")


class BlackJAXSMCNUTSSampler(BlackjaxSMCSampler):
    """SMC with NUTS kernel and Hessian-based mass matrix adaptation.

    WARNING: This sampler is EXPERIMENTAL. Use with caution and validate results carefully.
    The NUTS kernel with Hessian adaptation has not been thoroughly tested.

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
    config : SMCNUTSSamplerConfig
        NUTS SMC configuration
    seed : int, optional
        Random seed (default: 0)
    """

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform],
        likelihood_transforms: list[NtoMTransform],
        config: SMCNUTSSamplerConfig,
        seed: int = 0,
    ) -> None:
        """Initialize with EXPERIMENTAL warning."""
        super().__init__(
            likelihood, prior, sample_transforms, likelihood_transforms, config, seed
        )
        logger.warning(
            "NUTS kernel is experimental and has not been thoroughly tested yet. "
            "Use with caution and validate results carefully."
        )

    def _get_kernel_name(self) -> str:
        """Return kernel name."""
        return "nuts"

    def _build_mass_matrix(self) -> Array:
        """Create diagonal mass matrix with per-parameter scaling.

        Returns
        -------
        Array
            Diagonal mass matrix (n_dim, n_dim)
        """
        # Type narrow config for this subclass
        config = cast(SMCNUTSSamplerConfig, self.config)

        # Build mass matrix scaling array
        mass_matrix_scale_array = jnp.ones(self.prior.n_dim)

        for param_name, scale in config.mass_matrix_param_scales.items():
            try:
                idx = self.parameter_names.index(param_name)
                mass_matrix_scale_array = mass_matrix_scale_array.at[idx].set(scale)
            except ValueError:
                logger.warning(
                    f"Parameter '{param_name}' not found in parameter list, "
                    f"ignoring mass matrix scale"
                )

        # Mass matrix diagonal = (base * scale)^2
        mass_matrix_diag = (config.mass_matrix_base * mass_matrix_scale_array) ** 2
        return jnp.diag(mass_matrix_diag)

    def _setup_mcmc_kernel(
        self,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        logposterior_fn: Callable,
        initial_particles: Array,
    ) -> tuple[Callable, Callable, dict, Callable]:
        """Setup NUTS kernel with Hessian adaptation.

        Parameters
        ----------
        logprior_fn : Callable
            Log prior function (not used for NUTS)
        loglikelihood_fn : Callable
            Log likelihood function (not used for NUTS)
        logposterior_fn : Callable
            Log posterior function for computing Hessian
        initial_particles : Array
            Initial particle positions (not used for NUTS)

        Returns
        -------
        tuple[Callable, Callable, dict, Callable]
            (mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn)
        """

        # Type narrow config for this subclass
        config = cast(SMCNUTSSamplerConfig, self.config)

        logger.info(f"Initial step size: {config.init_step_size}")
        logger.info(f"Adaptation rate: {config.adaptation_rate}")

        # Hessian for NUTS mass matrix adaptation
        hessian_fn = jax.jit(jax.hessian(logposterior_fn))

        # Build initial mass matrix
        init_inverse_mass_matrix = self._build_mass_matrix()

        # Initial parameters for NUTS
        init_params = {
            "step_size": config.init_step_size,
            "inverse_mass_matrix": init_inverse_mass_matrix,
        }

        # TODO: remove this tracking in case we don't want this for NUTS
        # Track current step size for adaptation
        current_step_size = {"value": config.init_step_size}

        # Define parameter update function for Hessian-based adaptation
        def mcmc_parameter_update_fn(key, state, info):
            """Adapt mass matrix and step size using Hessian at best particle."""
            # Extract log posteriors from NUTS trajectory endpoints
            last_step_info = jax.tree.map(lambda x: x[-1], info.update_info)
            log_posteriors_left = last_step_info.trajectory_leftmost_state.logdensity
            log_posteriors_right = last_step_info.trajectory_rightmost_state.logdensity

            # Take maximum logdensity between endpoints
            log_posteriors = jnp.maximum(log_posteriors_left, log_posteriors_right)

            # Find particle with highest log posterior
            best_idx = jnp.argmax(log_posteriors)

            # Get position from best endpoint
            best_particle = jnp.where(
                log_posteriors_left[best_idx] > log_posteriors_right[best_idx],
                last_step_info.trajectory_leftmost_state.position[best_idx],
                last_step_info.trajectory_rightmost_state.position[best_idx],
            )

            # Compute Hessian at best particle
            hessian = hessian_fn(best_particle)

            # TODO: investigate if this is stable when Lambdas are near zero
            # Eigen decomposition with SoftAbs regularization
            lambdas, V = jnp.linalg.eigh(-hessian)
            soft_lambdas = lambdas / jnp.tanh(5e-3 * lambdas)

            # Reconstruct metric
            G = V @ jnp.diag(soft_lambdas) @ V.T
            adapted_inverse_mass_matrix = jnp.linalg.inv(G)

            # Adapt step size using dual averaging
            mean_acceptance = last_step_info.acceptance_rate.mean()
            log_step_size = jnp.log(current_step_size["value"])
            log_step_size += config.adaptation_rate * (
                mean_acceptance - config.target_acceptance
            )
            adapted_step_size = jnp.exp(log_step_size)
            adapted_step_size = jnp.clip(adapted_step_size, 1e-10, 1e0)

            # Update tracked step size
            current_step_size["value"] = adapted_step_size  # type: ignore[assignment]

            return extend_params(
                {  # type: ignore[arg-type]
                    "step_size": adapted_step_size,
                    "inverse_mass_matrix": adapted_inverse_mass_matrix,
                }
            )

        # Setup NUTS kernel
        mcmc_step_fn = nuts.build_kernel()
        mcmc_init_fn = nuts.init

        return mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn
