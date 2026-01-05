"""BlackJAX Sequential Monte Carlo (SMC) sampler for JESTER inference.

This module provides SMC with adaptive tempering using the BlackJAX library.
Supports two MCMC kernels:
- NUTS (No-U-Turn Sampler) with Hessian-based mass matrix adaptation
- Gaussian Random Walk Metropolis-Hastings
"""

from typing import Any
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax.experimental import io_callback
from jaxtyping import Array, PRNGKeyArray

from ..base import LikelihoodBase, Prior, BijectiveTransform, NtoMTransform
from ..config.schema import SMCSamplerConfig
from .jester_sampler import JesterSampler
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class BlackJAXSMCSampler(JesterSampler):
    """BlackJAX Sequential Monte Carlo with adaptive tempering.

    This sampler implements SMC for Bayesian posterior sampling using adaptive
    tempering (λ: 0 → 1) with configurable MCMC kernels. Works in prior space
    (no unit cube transforms needed).

    Supported kernels:
    - NUTS: No-U-Turn Sampler with Hessian-based mass matrix adaptation
    - Random Walk: Gaussian random walk Metropolis-Hastings

    Parameters
    ----------
    likelihood : LikelihoodBase
        Likelihood object with evaluate(params, data) method
    prior : Prior
        Prior object
    sample_transforms : list[BijectiveTransform]
        Should be empty for SMC (works in prior space)
    likelihood_transforms : list[NtoMTransform]
        N-to-M transforms applied before likelihood evaluation
    config : SMCSamplerConfig
        SMC configuration (includes kernel_type selection)
    seed : int, optional
        Random seed (default: 0)

    Attributes
    ----------
    config : SMCSamplerConfig
        Sampler configuration
    final_state : Any | None
        Final SMC state (after sampling)
    metadata : dict
        Sampling metadata (ESS, time, etc.)
    _unflatten_fn : callable
        Function to convert flat arrays back to parameter dicts
    _particles_flat : Array
        Final particle positions (flat arrays)
    _weights : Array
        Final particle weights

    Notes
    -----
    SMC works best without sample transforms (in prior space). If transforms are
    needed, they should be applied manually before/after SMC.
    """

    config: SMCSamplerConfig
    final_state: Any | None
    metadata: dict
    _unflatten_fn: Any  # Callable[[Array], dict]
    _particles_flat: Array | None
    _weights: Array | None

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform],
        likelihood_transforms: list[NtoMTransform],
        config: SMCSamplerConfig,
        seed: int = 0,
    ) -> None:
        """Initialize BlackJAX SMC sampler."""
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)

        self.config = config
        self.final_state = None
        self.metadata = {}
        self._unflatten_fn = None
        self._particles_flat = None
        self._weights = None
        self._seed = seed

        # Validate that we don't have sample transforms (SMC works in prior space)
        if len(sample_transforms) > 0:
            logger.warning(
                "SMC sampler received sample transforms. SMC typically works best "
                "without sample transforms (in prior space). Proceeding anyway."
            )

        logger.info("Initializing BlackJAX SMC sampler")
        logger.info(f"Kernel type: {config.kernel_type}")

        # Warn about experimental NUTS kernel
        if config.kernel_type == "nuts":
            logger.warning(
                "NUTS kernel is experimental and has not been thoroughly tested yet. "
                "Use with caution and validate results carefully."
            )

        logger.info(f"Configuration: {config.n_particles} particles, "
                    f"{config.n_mcmc_steps} MCMC steps per tempering stage")
        logger.info(f"Target ESS: {config.target_ess}")

    def _build_mass_matrix(self) -> Array:
        """Create diagonal mass matrix with per-parameter scaling.
        TODO: need to check if there are ways to make a better mass matrix?

        Returns
        -------
        Array
            Diagonal mass matrix (n_dim, n_dim)
        """
        # Build mass matrix scaling array
        mass_matrix_scale_array = jnp.ones(self.prior.n_dim)

        for param_name, scale in self.config.mass_matrix_param_scales.items():
            try:
                idx = self.parameter_names.index(param_name)
                mass_matrix_scale_array = mass_matrix_scale_array.at[idx].set(scale)
            except ValueError:
                logger.warning(f"Parameter '{param_name}' not found in parameter list, "
                               f"ignoring mass matrix scale")

        # Mass matrix diagonal = (base * scale)^2
        mass_matrix_diag = (self.config.mass_matrix_base * mass_matrix_scale_array) ** 2
        return jnp.diag(mass_matrix_diag)

    def sample(self, key: PRNGKeyArray, initial_position: Array = jnp.array([])) -> None:
        """Run SMC until λ = 1 (posterior).

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key
        initial_position : Array, optional
            Not used for SMC (samples from prior)
        """
        logger.info("Starting SMC sampling...")
        start_time = time.time()

        # Import BlackJAX SMC
        try:
            from blackjax import inner_kernel_tuning, adaptive_tempered_smc, nuts
            from blackjax.mcmc import random_walk
            from blackjax.smc import extend_params
            from blackjax.smc.resampling import systematic
        except ImportError as e:
            raise ImportError(
                "BlackJAX SMC not found. Install with: pip install blackjax"
            ) from e

        # Sample initial particles from prior
        key, subkey = jax.random.split(key)
        initial_position_dict: dict[str, Array] = self.prior.sample(subkey, self.config.n_particles)

        # Apply sample transforms if any
        for transform in self.sample_transforms:
            initial_position_list = []
            for i in range(self.config.n_particles):
                particle_dict = {
                    name: initial_position_dict[name][i]
                    for name in self.prior.parameter_names
                }
                transformed_dict, _ = transform.transform(particle_dict)
                initial_position_list.append(transformed_dict)
            # Reconstruct dict of arrays
            initial_position_dict = {
                name: jnp.array([p[name] for p in initial_position_list])
                for name in initial_position_list[0].keys()
            }

        # Flatten particles to arrays for BlackJAX SMC
        # NOTE: ravel_pytree uses alphabetical key ordering (deterministic)
        single_sample_dict = tree_map(lambda x: x[0], initial_position_dict)
        _, self._unflatten_fn = ravel_pytree(single_sample_dict)
        self._flatten_fn = lambda x: ravel_pytree(x)[0]

        # Flatten all particles using the flatten function
        initial_position_flat = jax.vmap(self._flatten_fn)(initial_position_dict)

        # Ensure float dtype for NUTS compatibility
        if not jnp.issubdtype(initial_position_flat.dtype, jnp.floating):
            logger.warning(f"Converting initial_position_flat from {initial_position_flat.dtype} to float64")
            initial_position_flat = initial_position_flat.astype(jnp.float64)

        # Helper function to unflatten and apply inverse transforms
        def _unflatten_and_inverse_transform(x_flat: Array, return_jacobian: bool = False):
            """Unflatten particle and apply inverse sample transforms."""
            x_flat = jnp.atleast_1d(x_flat)
            x_dict = self._unflatten_fn(x_flat)

            transform_jacobian = 0.0
            for transform in reversed(self.sample_transforms):
                x_dict, jacobian = transform.inverse(x_dict)
                if return_jacobian:
                    transform_jacobian += jacobian

            return (x_dict, transform_jacobian) if return_jacobian else x_dict

        # Create logprior and loglikelihood functions that work with flat arrays
        # NOTE: BlackJAX adaptive_tempered_smc will vmap these internally for ESS solver
        # So these should work on SINGLE particles, not batches
        def logprior_fn(x_flat: Array) -> float:
            """Log prior for single particle in flattened space."""
            x_dict, transform_jacobian = _unflatten_and_inverse_transform(x_flat, return_jacobian=True)
            return self.prior.log_prob(x_dict) + transform_jacobian

        def loglikelihood_fn(x_flat: Array) -> float:
            """Log likelihood for single particle in flattened space."""
            x_dict = _unflatten_and_inverse_transform(x_flat, return_jacobian=False)

            # Apply likelihood transforms
            for transform in self.likelihood_transforms:
                x_dict = transform.forward(x_dict)

            return self.likelihood.evaluate(x_dict, {})

        # Create posterior for NUTS Hessian initialization (single particle)
        logposterior_fn = lambda x: logprior_fn(x) + loglikelihood_fn(x)

        # Setup kernel-specific parameters and functions
        if self.config.kernel_type == "nuts":
            # TODO: Thoroughly test NUTS kernel with real inference problems to validate
            # correctness of Hessian-based mass matrix adaptation and convergence properties
            logger.info("Initializing SMC with NUTS kernel...")

            # Hessian for NUTS mass matrix adaptation
            hessian_fn = jax.jit(jax.hessian(logposterior_fn))

            # Build initial mass matrix
            init_inverse_mass_matrix = self._build_mass_matrix()

            # Initial parameters for NUTS
            init_params = {
                "step_size": self.config.init_step_size,
                "inverse_mass_matrix": init_inverse_mass_matrix,
            }

            # Track current step size for adaptation
            current_step_size = {"value": self.config.init_step_size}

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

                # Eigen decomposition with SoftAbs regularization
                lambdas, V = jnp.linalg.eigh(-hessian)
                soft_lambdas = lambdas / jnp.tanh(5e-3 * lambdas)

                # Reconstruct metric
                G = V @ jnp.diag(soft_lambdas) @ V.T
                adapted_inverse_mass_matrix = jnp.linalg.inv(G)

                # Adapt step size using dual averaging
                mean_acceptance = last_step_info.acceptance_rate.mean()
                log_step_size = jnp.log(current_step_size["value"])
                log_step_size += self.config.adaptation_rate * (
                    mean_acceptance - self.config.target_acceptance
                )
                adapted_step_size = jnp.exp(log_step_size)
                adapted_step_size = jnp.clip(adapted_step_size, 1e-10, 1e0)

                # Update tracked step size
                current_step_size["value"] = adapted_step_size  # type: ignore[assignment]

                return extend_params(
                    {
                        "step_size": adapted_step_size,
                        "inverse_mass_matrix": adapted_inverse_mass_matrix,
                    }
                )

            # Setup NUTS kernel
            mcmc_step_fn = nuts.build_kernel()
            mcmc_init_fn = nuts.init

        elif self.config.kernel_type == "random_walk":
            logger.info("Initializing SMC with Gaussian Random Walk kernel...")

            # Initial parameters for random walk
            init_params = {}  # Random walk doesn't need parameters

            # Track current sigma for adaptation
            current_sigma = {"value": self.config.random_walk_sigma}

            # Define parameter update function for sigma adaptation
            def mcmc_parameter_update_fn(key, state, info):
                """Adapt step size (sigma) based on acceptance rate."""
                # Extract acceptance rates from last step
                last_step_info = jax.tree.map(lambda x: x[-1], info.update_info)
                mean_acceptance = last_step_info.acceptance_rate.mean()

                # Adapt sigma using dual averaging
                log_sigma = jnp.log(current_sigma["value"])
                log_sigma += self.config.adaptation_rate * (
                    mean_acceptance - self.config.target_acceptance
                )
                adapted_sigma = jnp.exp(log_sigma)
                adapted_sigma = jnp.clip(adapted_sigma, 1e-10, 1e1)

                # Update tracked sigma
                current_sigma["value"] = adapted_sigma  # type: ignore[assignment]

                # Return empty params (random walk doesn't use them in the same way)
                return extend_params({})

            # Setup random walk kernel with additive step
            kernel = random_walk.build_additive_step()

            # Wrap kernel to match expected signature
            def mcmc_step_fn(rng_key, state, logdensity_fn, **params):
                """Random walk step function matching NUTS signature."""
                # Update random step with current sigma
                step_fn = random_walk.normal(jnp.ones(self.prior.n_dim) * current_sigma["value"])
                return kernel(rng_key, state, logdensity_fn, step_fn)

            # Init function for random walk
            mcmc_init_fn = random_walk.init

        else:
            raise ValueError(f"Unknown kernel type: {self.config.kernel_type}")

        # Initialize SMC algorithm with selected kernel
        smc_alg = inner_kernel_tuning(
            smc_algorithm=adaptive_tempered_smc,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params(init_params),
            target_ess=self.config.target_ess,
            num_mcmc_steps=self.config.n_mcmc_steps,
        )

        # Initialize SMC state
        key, subkey = jax.random.split(key)
        state = smc_alg.init(initial_position_flat, subkey)

        # Progress callback for live updates during sampling
        def progress_callback(step: int, lmbda: float, ess: float, acceptance: float) -> None:
            """Print progress update during sampling (called via io_callback)."""
            # Create progress bar
            bar_length = 30
            filled = int(lmbda * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Print update
            logger.info(
                f"Step {step:4d} | λ={lmbda:.6f} | ESS={ess*100:5.1f}% | "
                f"Accept={acceptance*100:5.1f}% | {bar}"
            )

        # Define loop conditions
        def cond_fn(carry):
            state, _, _, _, _, _, _ = carry
            return state.sampler_state.lmbda < 1  # type: ignore[attr-defined]

        def body_fn(carry):
            state, key, step_count, lmbda_history, ess_history, acceptance_history, log_evidence = carry
            key, subkey = jax.random.split(key, 2)
            state, info = smc_alg.step(subkey, state)

            # Accumulate log evidence from log_likelihood_increment TODO: double check the maths here
            log_evidence = log_evidence + info.log_likelihood_increment  # type: ignore[attr-defined]

            # Compute ESS
            weights = state.sampler_state.weights  # type: ignore[attr-defined]
            ess_value = (
                jnp.sum(weights) ** 2 / jnp.sum(weights**2) / self.config.n_particles
            )

            # Extract acceptance rate
            acceptance_rate = info.update_info.acceptance_rate.mean()  # type: ignore[attr-defined]

            # Update histories
            lmbda_history = lmbda_history.at[step_count].set(state.sampler_state.lmbda)  # type: ignore[attr-defined]
            ess_history = ess_history.at[step_count].set(ess_value)
            acceptance_history = acceptance_history.at[step_count].set(acceptance_rate)

            # Print progress update using io_callback
            io_callback(
                progress_callback,
                None,  # No return value
                step_count,
                state.sampler_state.lmbda,  # type: ignore[attr-defined]
                ess_value,
                acceptance_rate
            )

            return (state, key, step_count + 1, lmbda_history, ess_history, acceptance_history, log_evidence)

        # Run SMC with JAX while_loop
        logger.info("=" * 70)
        logger.info("STARTING ADAPTIVE TEMPERING")
        logger.info("=" * 70)
        logger.info(f"Kernel: {self.config.kernel_type.upper()}")
        logger.info(f"Particles: {self.config.n_particles}")
        logger.info(f"MCMC steps per tempering: {self.config.n_mcmc_steps}")
        logger.info(f"Target ESS: {self.config.target_ess * 100:.0f}%")
        logger.info(f"Target acceptance: {self.config.target_acceptance * 100:.0f}%")
        if self.config.kernel_type == "random_walk":
            logger.info(f"Initial sigma: {self.config.random_walk_sigma}")
            logger.info(f"Adaptation rate: {self.config.adaptation_rate}")
        logger.info("Temperature progression: lambda = 0 (prior) -> 1 (posterior)")
        logger.info("Progress updates will be shown after each annealing step")
        logger.info("=" * 70)

        max_steps = 1000
        lmbda_history = jnp.zeros(max_steps)
        ess_history = jnp.zeros(max_steps)
        acceptance_history = jnp.zeros(max_steps)
        log_evidence = 0.0  # Initialize log evidence accumulator

        init_carry = (state, key, 0, lmbda_history, ess_history, acceptance_history, log_evidence)

        logger.info("Running SMC loop (this may take several minutes)...")
        loop_start_time = time.time()

        state, key, steps, lmbda_history, ess_history, acceptance_history, log_evidence = jax.lax.while_loop(
            cond_fn, body_fn, init_carry
        )

        loop_end_time = time.time()
        steps = int(steps)
        end_time = time.time()

        # Extract final particles
        self._particles_flat = state.sampler_state.particles  # type: ignore[attr-defined]
        self._weights = state.sampler_state.weights  # type: ignore[attr-defined]
        self.final_state = state

        # Compute final ESS (weights guaranteed non-None after assignment above)
        assert self._weights is not None
        ess = jnp.sum(self._weights) ** 2 / jnp.sum(self._weights**2)

        # Compute summary statistics
        mean_ess = float(jnp.mean(ess_history[:steps]))
        min_ess = float(jnp.min(ess_history[:steps]))
        mean_acceptance = float(jnp.mean(acceptance_history[:steps]))

        # Compute evidence error estimate
        # Simple estimate: use sqrt(variance / n_steps) as uncertainty
        # Note: This is a rough estimate; proper SMC evidence error requires
        # tracking incremental variances, which BlackJAX doesn't provide directly
        # FIXME: Implement proper evidence error estimation if needed
        log_evidence_err = 0.0  # Placeholder - proper error estimation would require more info

        # Store metadata
        self.metadata = {
            'sampler': 'blackjax_smc',
            'kernel_type': self.config.kernel_type,
            'n_particles': self.config.n_particles,
            'n_mcmc_steps': self.config.n_mcmc_steps,
            'target_ess': self.config.target_ess,
            'annealing_steps': steps,
            'final_ess': float(ess),
            'final_ess_percent': float(ess / self.config.n_particles * 100),
            'mean_ess': mean_ess,
            'min_ess': min_ess,
            'mean_acceptance': mean_acceptance,
            'logZ': float(log_evidence),
            'logZ_err': float(log_evidence_err),
            'sampling_time_seconds': end_time - start_time,
            'loop_time_seconds': loop_end_time - loop_start_time,
            'lmbda_history': lmbda_history[:steps].tolist(),
            'ess_history': ess_history[:steps].tolist(),
            'acceptance_history': acceptance_history[:steps].tolist(),
        }

        # Display progress summary table only
        logger.info("")
        self._print_progress_summary(steps, lmbda_history, ess_history, acceptance_history)

    def _print_progress_summary(
        self,
        steps: int,
        lmbda_history: Array,
        ess_history: Array,
        acceptance_history: Array
    ) -> None:
        """Print a progress summary table showing annealing progression.

        Parameters
        ----------
        steps : int
            Number of annealing steps completed
        lmbda_history : Array
            Temperature (lambda) values at each step
        ess_history : Array
            ESS values at each step
        acceptance_history : Array
            Acceptance rates at each step
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("ANNEALING PROGRESS SUMMARY")
        logger.info("=" * 70)

        # Determine how many rows to show
        # Show first, last, and evenly spaced intermediate steps
        max_rows = 15
        if steps <= max_rows:
            # Show all steps
            indices = list(range(steps))
        else:
            # Show first, last, and evenly spaced intermediate
            n_intermediate = max_rows - 2
            step_size = (steps - 2) / n_intermediate
            indices = [0]  # First step
            indices.extend([int(1 + i * step_size) for i in range(n_intermediate)])
            indices.append(steps - 1)  # Last step

        # Print table header
        logger.info(f"{'Step':<8} {'Lambda':<12} {'ESS (%)':<12} {'Accept (%)':<12} {'Progress':<20}")
        logger.info("-" * 70)

        # Print table rows
        for idx in indices:
            step_num = idx
            lmbda = float(lmbda_history[idx])
            ess_pct = float(ess_history[idx]) * 100
            acc_pct = float(acceptance_history[idx]) * 100

            # Create progress bar (20 chars)
            bar_length = 20
            filled = int(lmbda * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)

            logger.info(f"{step_num:<8} {lmbda:<12.6f} {ess_pct:<12.1f} {acc_pct:<12.1f} {bar}")

        logger.info("-" * 70)
        logger.info(f"Total steps: {steps}")
        logger.info(f"Temperature range: λ = 0.000000 (prior) → {float(lmbda_history[steps-1]):.6f} (posterior)")
        logger.info("=" * 70)
        logger.info("")

    def print_summary(self, transform: bool = True) -> None:
        """Print summary of SMC run.

        Parameters
        ----------
        transform : bool, optional
            Not used for SMC

        Notes
        -----
        Summary information is already displayed during and after sampling,
        so this method does nothing to avoid redundancy.
        """
        # Summary already displayed during sampling - no need to repeat
        pass

    def plot_diagnostics(self, outdir: str | Path = ".", filename: str = "smc_diagnostics.png") -> None:
        """Generate diagnostic plots for SMC sampling run.

        Creates a 3-panel figure showing:
        - Temperature (lambda) progression from 0 to 1
        - Effective Sample Size (ESS) evolution
        - Acceptance rate evolution

        Parameters
        ----------
        outdir : str or Path, optional
            Output directory for saving the plot (default: current directory)
        filename : str, optional
            Filename for the diagnostic plot (default: "smc_diagnostics.png")

        Notes
        -----
        This method requires matplotlib to be installed. It should be called after
        sampling is complete (after calling `sample()`).
        """
        if self.final_state is None:
            logger.warning("No samples yet - run sample() first")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not found. Install with: pip install matplotlib")
            return

        # Extract histories from metadata
        lmbda_history = self.metadata['lmbda_history']
        ess_history = self.metadata['ess_history']
        acceptance_history = self.metadata['acceptance_history']
        n_steps = self.metadata['annealing_steps']

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        fig.suptitle(f"SMC Diagnostics ({self.config.kernel_type.upper()} kernel)",
                     fontsize=14, fontweight='bold')

        # Plot 1: Lambda (temperature) progression
        axes[0].plot(range(n_steps), lmbda_history, 'b-o', linewidth=2)
        axes[0].set_ylabel(r'Temperature $\lambda$', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        axes[0].axhline(y=1, color='black', linestyle='--', alpha=0.3, linewidth=1)

        # Plot 2: ESS evolution
        ess_percent = [ess * 100 for ess in ess_history]
        axes[1].plot(range(n_steps), ess_percent, 'g-o', linewidth=2)
        axes[1].axhline(y=self.config.target_ess * 100, color='black', linestyle='--',
                       alpha=0.5, linewidth=1.5, label=f'Target ({self.config.target_ess*100:.0f}%)')
        axes[1].set_ylabel('ESS (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best', fontsize=10)
        axes[1].set_ylim(0, 105)

        # Plot 3: Acceptance rate evolution
        acceptance_percent = [acc * 100 for acc in acceptance_history]
        axes[2].plot(range(n_steps), acceptance_percent, 'orange', linestyle='-', marker='o', linewidth=2)
        axes[2].axhline(y=self.config.target_acceptance * 100, color='black', linestyle='--',
                       alpha=0.5, linewidth=1.5, label=f'Target ({self.config.target_acceptance*100:.0f}%)')
        axes[2].set_ylabel('Acceptance Rate (%)', fontsize=12)
        axes[2].set_xlabel('Annealing Step', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='best', fontsize=10)
        axes[2].set_ylim(0, 105)

        plt.tight_layout()

        # Save figure
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        output_path = outdir_path / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved diagnostic plot to {output_path}")
        plt.close(fig)

    def get_samples(self, training: bool = False) -> dict:
        """Return final particle positions.

        Parameters
        ----------
        training : bool, optional
            Not used for SMC (no train/production split)

        Returns
        -------
        dict
            Dictionary with:
            - Parameter samples (transformed back to prior space)
            - 'weights': particle weights
            - 'ess': effective sample size
        """
        if self.final_state is None or self._particles_flat is None or self._weights is None:
            raise RuntimeError("No samples available - run sample() first")

        # Transform particles back to structured format (guaranteed non-None after check above)
        assert self._particles_flat is not None
        assert self._weights is not None
        particles_dict = jax.vmap(self._unflatten_fn)(self._particles_flat)

        # Apply inverse sample transforms if any
        for transform in reversed(self.sample_transforms):
            particles_list = []
            n_particles = len(self._particles_flat)
            for i in range(n_particles):
                particle_dict = {name: particles_dict[name][i] for name in particles_dict.keys()}
                transformed_dict, _ = transform.inverse(particle_dict)
                particles_list.append(transformed_dict)
            # Reconstruct dict of arrays
            particles_dict = {
                name: jnp.array([p[name] for p in particles_list])
                for name in particles_list[0].keys()
            }

        # Add weights and ESS to output
        particles_dict['weights'] = self._weights
        particles_dict['ess'] = self.metadata['final_ess']

        return particles_dict

    def get_log_prob(self, training: bool = False) -> Array:
        """Get log posterior probabilities from SMC.

        Parameters
        ----------
        training : bool, optional
            Not used for SMC (no train/production split)

        Returns
        -------
        Array
            Log posterior probability values (1D array)
            Note: At λ=1 (final tempering), these are true posterior values.
            Weights are approximately uniform at convergence.
        """
        if self.final_state is None or self._particles_flat is None:
            raise RuntimeError("No samples available - run sample() first")

        # For SMC at λ=1, we have log posterior values
        # Compute from particles using batched processing
        # logger.info(f"Computing log probabilities from {len(self._particles_flat)} particles using batched processing...")
        # logger.info(f"Batch size: {self.config.log_prob_batch_size}")

        # CRITICAL: Must use _unflatten_fn because ravel_pytree uses alphabetical ordering,
        # which differs from self.parameter_names ordering used by add_name()
        assert self._particles_flat is not None

        def compute_log_prob(particle_flat):
            # Convert from flat array (alphabetical order) to dict using _unflatten_fn
            x_dict = self._unflatten_fn(particle_flat)
            # Use base class method to compute posterior from dict
            return self.posterior_from_dict(x_dict, {})

        # Use batched processing for efficiency
        log_probs = jax.lax.map(compute_log_prob, self._particles_flat, batch_size=self.config.log_prob_batch_size)
        logger.info(f"Computed {len(log_probs)} log probability values")

        return log_probs

    def get_n_samples(self, training: bool = False) -> int:
        """Get number of particles from SMC.

        Parameters
        ----------
        training : bool, optional
            If True, returns 0 (SMC has no training phase).
            If False, returns number of particles.

        Returns
        -------
        int
            Number of particles (0 if training=True)
        """
        # SMC has no training phase - return 0 if training requested
        if training:
            return 0

        if self._particles_flat is None:
            return 0

        return len(self._particles_flat)
