"""Base class for BlackJAX Sequential Monte Carlo (SMC) samplers.

This module provides BlackjaxSMCSampler, which implements shared SMC functionality
(adaptive tempering, particle management, result handling) and delegates
only the kernel-specific parts to subclasses.
"""

from abc import abstractmethod
from typing import Any, Callable, cast
import time
from pathlib import Path
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random
from jax import flatten_util
from jax.tree_util import tree_map
from jax.experimental import io_callback
from jaxtyping import Array, PRNGKeyArray

from jesterTOV.inference.base import (
    LikelihoodBase,
    Prior,
    BijectiveTransform,
    NtoMTransform,
)
from jesterTOV.inference.config.schema import (
    SMCRandomWalkSamplerConfig,
    SMCNUTSSamplerConfig,
)
from jesterTOV.inference.samplers.jester_sampler import SamplerOutput
from jesterTOV.inference.samplers.blackjax.base import BlackjaxSampler
from jesterTOV.logging_config import get_logger

from blackjax import inner_kernel_tuning, adaptive_tempered_smc
from blackjax.smc import extend_params
from blackjax.smc.base import SMCInfo
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.smc.resampling import systematic
from blackjax.smc.tempered import TemperedSMCState

logger = get_logger("jester")


class BlackjaxSMCSampler(BlackjaxSampler):
    """Base class for BlackJAX Sequential Monte Carlo with adaptive tempering.

    This abstract base class implements all the shared SMC functionality
    (adaptive tempering, particle management, result handling) and delegates
    only the kernel-specific parts to subclasses.

    Key differences from parent BlackjaxSampler:
    - Adds flattening/unflattening utilities (SMC requires flat arrays)
    - Wraps dict-based functions from parent for flat array API
    - Implements full SMC sampling loop

    Subclasses must implement:
    - _setup_mcmc_kernel(): Return (mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn)
    - _get_kernel_name(): Return string name for logging/plotting

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
    config : SMCRandomWalkSamplerConfig | SMCNUTSSamplerConfig
        SMC configuration
    seed : int, optional
        Random seed (default: 0)

    Attributes
    ----------
    config : SMCRandomWalkSamplerConfig | SMCNUTSSamplerConfig
        Sampler configuration
    final_state : Any | None
        Final SMC state (after sampling)
    metadata : dict
        Sampling metadata (ESS, time, etc.)
    _unflatten_fn : callable
        Function to convert flat arrays back to parameter dicts
    _flatten_fn : callable
        Function to convert parameter dicts to flat arrays
    _particles_flat : Array
        Final particle positions (flat arrays)
    _weights : Array
        Final particle weights
    """

    config: SMCRandomWalkSamplerConfig | SMCNUTSSamplerConfig
    final_state: Any | None
    metadata: dict
    _unflatten_fn: Any  # Callable[[Array], dict]
    _flatten_fn: Any  # Callable[[dict], Array]
    _particles_flat: Array | None
    _weights: Array | None

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform],
        likelihood_transforms: list[NtoMTransform],
        config: SMCRandomWalkSamplerConfig | SMCNUTSSamplerConfig,
        seed: int = 0,
    ) -> None:
        """Initialize BlackJAX SMC sampler."""
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)

        self.config = config
        self.final_state = None
        self.metadata = {}
        self._unflatten_fn = None
        self._flatten_fn = None
        self._particles_flat = None
        self._weights = None
        self._seed = seed

        # Validate that we don't have sample transforms (SMC works in prior space)
        if len(sample_transforms) > 0:
            logger.warning(
                "SMC sampler received sample transforms. SMC typically works best "
                "without sample transforms (in prior space). Proceeding anyway."
            )

        logger.info(
            f"Initializing BlackJAX SMC sampler with {self._get_kernel_name()} kernel"
        )
        logger.info(
            f"Configuration: {config.n_particles} particles, "
            f"{config.n_mcmc_steps} MCMC steps per tempering stage"
        )
        logger.info(f"Target ESS: {config.target_ess}")

    def _create_flatten_unflatten_utilities(
        self, initial_position_dict: dict[str, Array]
    ) -> None:
        """Create flatten/unflatten functions for SMC's flat array API.

        Parameters
        ----------
        initial_position_dict : dict[str, Array]
            Dictionary of initial particle positions (each value is array of shape (n_particles,))
        """
        # Extract single sample to determine structure
        single_sample_dict = tree_map(lambda x: x[0], initial_position_dict)

        # Create unflatten function using ravel_pytree (alphabetical ordering)
        _, self._unflatten_fn = flatten_util.ravel_pytree(single_sample_dict)

        # Create flatten function
        self._flatten_fn = lambda x: flatten_util.ravel_pytree(x)[0]

    def _wrap_dict_fn_for_flat_arrays(
        self, dict_fn: Callable[[dict], float]
    ) -> Callable[[Array], float]:
        """Wrap a dict-based function to work with flat arrays.

        This is the bridge between BlackjaxSampler's dict functions
        and SMC's flat array API.

        Parameters
        ----------
        dict_fn : Callable[[dict], float]
            Function that takes parameter dict and returns float

        Returns
        -------
        Callable[[Array], float]
            Function that takes flat array and returns float

        Examples
        --------
        >>> logprior_dict = self._create_logprior_fn_from_dict()
        >>> logprior_flat = self._wrap_dict_fn_for_flat_arrays(logprior_dict)
        >>> # Now logprior_flat can be passed to BlackJAX SMC
        """

        def flat_fn(x_flat: Array) -> float:
            """Convert flat array to dict, evaluate function."""
            x_flat = jnp.atleast_1d(x_flat)
            x_dict = self._unflatten_fn(x_flat)
            return dict_fn(x_dict)

        return flat_fn

    @abstractmethod
    def _setup_mcmc_kernel(
        self,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        logposterior_fn: Callable,
        initial_particles: Array,
    ) -> tuple[Callable, Callable, dict, Callable]:
        """Setup kernel-specific components.

        Parameters
        ----------
        logprior_fn : Callable
            Log prior function for single particle (flat array)
        loglikelihood_fn : Callable
            Log likelihood function for single particle (flat array)
        logposterior_fn : Callable
            Log posterior function for single particle (flat array, for NUTS Hessian)
        initial_particles : Array
            Initial particle positions (flat arrays, shape: (n_particles, n_dim))

        Returns
        -------
        tuple[Callable, Callable, dict, Callable]
            - mcmc_step_fn: MCMC step function
            - mcmc_init_fn: MCMC initialization function
            - init_params: Initial parameter dict for the kernel
            - mcmc_parameter_update_fn: Function to adapt parameters
        """
        pass

    @abstractmethod
    def _get_kernel_name(self) -> str:
        """Return the kernel name for logging/plotting."""
        pass

    def sample(self, key: PRNGKeyArray) -> None:
        """Run SMC until λ = 1 (posterior).

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key

        Notes
        -----
        Initial particles are sampled from the prior internally.
        """
        logger.info(f"Starting SMC sampling with {self._get_kernel_name()} kernel...")
        start_time = time.time()

        # Sample initial particles from prior
        key, subkey = jax.random.split(key)
        initial_position_dict: dict[str, Array] = self.prior.sample(
            subkey, self.config.n_particles
        )

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

        # Create flatten/unflatten utilities
        self._create_flatten_unflatten_utilities(initial_position_dict)

        # Flatten all particles using the flatten function
        initial_position_flat = jax.vmap(self._flatten_fn)(initial_position_dict)

        # Ensure float dtype for compatibility
        if not jnp.issubdtype(initial_position_flat.dtype, jnp.floating):
            logger.warning(
                f"Converting initial_position_flat from {initial_position_flat.dtype} to float64"
            )
            initial_position_flat = initial_position_flat.astype(jnp.float64)

        # Create dict-based functions from parent class
        logprior_dict = self._create_logprior_fn_from_dict()
        loglikelihood_dict = self._create_loglikelihood_fn_from_dict()

        # Wrap for flat arrays (SMC requirement)
        logprior_fn = self._wrap_dict_fn_for_flat_arrays(logprior_dict)
        loglikelihood_fn = self._wrap_dict_fn_for_flat_arrays(loglikelihood_dict)

        # Create posterior for kernel setup (e.g., NUTS Hessian)
        logposterior_fn = lambda x: logprior_fn(x) + loglikelihood_fn(x)

        # Setup kernel-specific components
        mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn = (
            self._setup_mcmc_kernel(
                logprior_fn, loglikelihood_fn, logposterior_fn, initial_position_flat
            )
        )

        # Initialize SMC algorithm with kernel
        smc_alg = inner_kernel_tuning(
            smc_algorithm=adaptive_tempered_smc,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params(init_params),  # type: ignore[arg-type]
            target_ess=self.config.target_ess,
            num_mcmc_steps=self.config.n_mcmc_steps,
        )

        # Initialize SMC state
        key, subkey = jax.random.split(key)
        state = smc_alg.init(initial_position_flat, subkey)

        # Progress callback for live updates during sampling
        def progress_callback(
            step: int, tempering_param: float, ess: float, acceptance: float
        ) -> None:
            """Print progress update during sampling (called via io_callback)."""
            # Create progress bar
            bar_length = 30
            filled = int(tempering_param * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Print update
            logger.info(
                f"Step {step:4d} | λ={tempering_param:.6f} | ESS={ess*100:5.1f}% | "
                f"Accept={acceptance*100:5.1f}% | {bar}"
            )

        # Define loop conditions with proper type hints
        # Carry is: (StateWithParameterOverride, key, step_count, tempering_param_history, ess_history, acceptance_history, log_evidence)
        def cond_fn(
            carry: tuple[
                StateWithParameterOverride,
                PRNGKeyArray,
                int,
                Array,
                Array,
                Array,
                float,
            ]
        ) -> bool:
            state, _, _, _, _, _, _ = carry
            # Cast to proper type for type checker (runtime type is correct)
            sampler_state = cast(TemperedSMCState, state.sampler_state)
            # Type checker sees this as potentially returning Array, but at runtime
            # tempering_param is a scalar float, so comparison returns bool
            return sampler_state.tempering_param < 1  # type: ignore[return-value]

        def body_fn(
            carry: tuple[
                StateWithParameterOverride,
                PRNGKeyArray,
                int,
                Array,
                Array,
                Array,
                float,
            ]
        ):

            (
                state,
                key,
                step_count,
                tempering_param_history,
                ess_history,
                acceptance_history,
                log_evidence,
            ) = carry
            key, subkey = jax.random.split(key, 2)
            state, info = smc_alg.step(subkey, state)
            # Cast to proper types for type checker (runtime types are correct)
            state = cast(StateWithParameterOverride, state)
            info = cast(SMCInfo, info)
            sampler_state = cast(TemperedSMCState, state.sampler_state)

            # Accumulate log evidence from log_likelihood_increment
            log_evidence = log_evidence + info.log_likelihood_increment

            # Compute ESS
            weights = sampler_state.weights
            ess_value = (
                jnp.sum(weights) ** 2 / jnp.sum(weights**2) / self.config.n_particles
            )

            # Extract acceptance rate
            # Note: update_info is kernel-specific NamedTuple, not fully typed in blackjax
            acceptance_rate = info.update_info.acceptance_rate.mean()  # type: ignore[attr-defined]

            # Update histories
            tempering_param_history = tempering_param_history.at[step_count].set(
                sampler_state.tempering_param
            )
            ess_history = ess_history.at[step_count].set(ess_value)
            acceptance_history = acceptance_history.at[step_count].set(acceptance_rate)

            # Print progress update using io_callback
            io_callback(
                progress_callback,
                None,  # No return value
                step_count,
                sampler_state.tempering_param,
                ess_value,
                acceptance_rate,
            )

            return (
                state,
                key,
                step_count + 1,
                tempering_param_history,
                ess_history,
                acceptance_history,
                log_evidence,
            )

        # Run SMC with JAX while_loop
        logger.info("=" * 70)
        logger.info("STARTING ADAPTIVE TEMPERING")
        logger.info("=" * 70)
        logger.info(f"Kernel: {self._get_kernel_name().upper()}")
        logger.info(f"Particles: {self.config.n_particles}")
        logger.info(f"MCMC steps per tempering: {self.config.n_mcmc_steps}")
        logger.info(f"Target ESS: {self.config.target_ess * 100:.0f}%")
        logger.info("Temperature progression: lambda = 0 (prior) -> 1 (posterior)")
        logger.info("Progress updates will be shown after each annealing step")
        logger.info("=" * 70)

        max_steps = 1000
        tempering_param_history = jnp.zeros(max_steps)
        ess_history = jnp.zeros(max_steps)
        acceptance_history = jnp.zeros(max_steps)
        log_evidence = 0.0  # Initialize log evidence accumulator

        init_carry = (
            state,
            key,
            0,
            tempering_param_history,
            ess_history,
            acceptance_history,
            log_evidence,
        )

        logger.info("Running SMC loop (this may take several minutes)...")
        loop_start_time = time.time()

        (
            state,
            key,
            steps,
            tempering_param_history,
            ess_history,
            acceptance_history,
            log_evidence,
        ) = jax.lax.while_loop(
            cond_fn, body_fn, init_carry  # type: ignore[arg-type]
        )

        loop_end_time = time.time()
        steps = int(steps)
        end_time = time.time()

        # Extract final particles
        # Cast to proper type for type checker (runtime type is correct)
        final_sampler_state = cast(TemperedSMCState, state.sampler_state)
        self._particles_flat = cast(Array, final_sampler_state.particles)
        self._weights = final_sampler_state.weights
        self.final_state = state

        # Compute final ESS (weights guaranteed non-None after assignment above)
        assert self._weights is not None
        ess = jnp.sum(self._weights) ** 2 / jnp.sum(self._weights**2)

        # Compute summary statistics
        mean_ess = float(jnp.mean(ess_history[:steps]))
        min_ess = float(jnp.min(ess_history[:steps]))
        mean_acceptance = float(jnp.mean(acceptance_history[:steps]))

        # FIXME: Need to implement a way to compute evidence error estimate
        log_evidence_err = 0.0  # Placeholder for now

        # Store metadata (kernel name will be set by subclass)
        self.metadata = {
            "sampler": f"blackjax_smc_{self._get_kernel_name()}",
            "kernel_type": self._get_kernel_name(),
            "n_particles": self.config.n_particles,
            "n_mcmc_steps": self.config.n_mcmc_steps,
            "target_ess": self.config.target_ess,
            "annealing_steps": steps,
            "final_ess": float(ess),
            "final_ess_percent": float(ess / self.config.n_particles * 100),
            "mean_ess": mean_ess,
            "min_ess": min_ess,
            "mean_acceptance": mean_acceptance,
            "logZ": float(log_evidence),
            "logZ_err": float(log_evidence_err),
            "sampling_time_seconds": end_time - start_time,
            "loop_time_seconds": loop_end_time - loop_start_time,
            "tempering_param_history": tempering_param_history[:steps].tolist(),
            "ess_history": ess_history[:steps].tolist(),
            "acceptance_history": acceptance_history[:steps].tolist(),
        }

    def plot_diagnostics(
        self, outdir: str | Path = ".", filename: str = "smc_diagnostics.png"
    ) -> None:
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

        # Extract histories from metadata
        tempering_param_history = self.metadata["tempering_param_history"]
        ess_history = self.metadata["ess_history"]
        acceptance_history = self.metadata["acceptance_history"]
        n_steps = self.metadata["annealing_steps"]

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        kernel_name = self._get_kernel_name().upper()
        fig.suptitle(
            f"SMC Diagnostics ({kernel_name} kernel)",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: Lambda (temperature) progression
        axes[0].plot(range(n_steps), tempering_param_history, "b-o", linewidth=2)
        axes[0].set_ylabel(r"Inverse temperature $\lambda$", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        axes[0].axhline(y=1, color="black", linestyle="--", alpha=0.3, linewidth=1)

        # Plot 2: ESS evolution
        ess_percent = [ess * 100 for ess in ess_history]
        axes[1].plot(range(n_steps), ess_percent, "g-o", linewidth=2)
        axes[1].axhline(
            y=self.config.target_ess * 100,
            color="black",
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label=f"Target ({self.config.target_ess*100:.0f}%)",
        )
        axes[1].set_ylabel("ESS (%)", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best", fontsize=10)
        axes[1].set_ylim(0, 105)

        # Plot 3: Acceptance rate evolution
        acceptance_percent = [acc * 100 for acc in acceptance_history]
        axes[2].plot(
            range(n_steps),
            acceptance_percent,
            "orange",
            linestyle="-",
            marker="o",
            linewidth=2,
        )
        axes[2].set_ylabel("Acceptance Rate (%)", fontsize=12)
        axes[2].set_xlabel("Annealing Step", fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 105)

        plt.tight_layout()

        # Save figure
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        output_path = outdir_path / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved diagnostic plot to {output_path}")
        plt.close(fig)

    def get_samples(self) -> dict:
        """Return final particle positions.

        Returns
        -------
        dict
            Dictionary with:
            - Parameter samples (transformed back to prior space)
            - 'weights': particle weights
            - 'ess': effective sample size
        """
        if (
            self.final_state is None
            or self._particles_flat is None
            or self._weights is None
        ):
            raise RuntimeError("No samples available - run sample() first")

        # Transform particles back to structured format
        assert self._particles_flat is not None
        assert self._weights is not None
        particles_dict = jax.vmap(self._unflatten_fn)(self._particles_flat)

        # Apply inverse sample transforms if any
        for transform in reversed(self.sample_transforms):
            particles_list = []
            n_particles = len(self._particles_flat)
            for i in range(n_particles):
                particle_dict = {
                    name: particles_dict[name][i] for name in particles_dict.keys()
                }
                transformed_dict, _ = transform.inverse(particle_dict)
                particles_list.append(transformed_dict)
            # Reconstruct dict of arrays
            particles_dict = {
                name: jnp.array([p[name] for p in particles_list])
                for name in particles_list[0].keys()
            }

        # Add weights and ESS to output
        particles_dict["weights"] = self._weights
        particles_dict["ess"] = self.metadata["final_ess"]

        return particles_dict

    def get_log_prob(self) -> Array:
        """Get log posterior probabilities from SMC.

        Returns
        -------
        Array
            Log posterior probability values (1D array)
            Note: At λ=1 (final tempering), these are true posterior values.
        """
        if self.final_state is None or self._particles_flat is None:
            raise RuntimeError("No samples available - run sample() first")

        assert self._particles_flat is not None

        def compute_log_prob(particle_flat):
            # Convert from flat array (alphabetical order) to dict using _unflatten_fn
            x_dict = self._unflatten_fn(particle_flat)
            # Use base class method to compute posterior from dict
            return self.posterior_from_dict(x_dict, {})

        # Use batched processing for efficiency
        log_probs = jax.lax.map(
            compute_log_prob,
            self._particles_flat,
            batch_size=self.config.log_prob_batch_size,
        )
        logger.info(f"Computed {len(log_probs)} log probability values")

        return log_probs

    def get_n_samples(self) -> int:
        """Get number of particles from SMC.

        Returns
        -------
        int
            Number of particles
        """
        if self._particles_flat is None:
            return 0

        return len(self._particles_flat)

    def get_sampler_output(self) -> SamplerOutput:
        """Get standardized sampler output.

        Returns
        -------
        SamplerOutput
            - samples: Parameter samples (dict of arrays, no weights/ess)
            - log_prob: Log posterior at λ=1 (final tempering)
            - metadata: {"weights": Array, "ess": float}

        Raises
        ------
        RuntimeError
            If sampling has not been run yet.
        """
        if self._particles_flat is None:
            raise RuntimeError("No samples available. Run sample() first.")

        # Get current samples dict (includes weights, ess)
        all_data = self.get_samples()

        # Separate parameters from metadata
        samples: dict[str, Array] = {}
        metadata: dict[str, Any] = {}

        metadata_keys = {"weights", "ess"}
        for key, value in all_data.items():
            if key in metadata_keys:
                metadata[key] = value
            else:
                samples[key] = value

        # Get log probabilities
        log_prob = self.get_log_prob()

        return SamplerOutput(
            samples=samples,
            log_prob=log_prob,
            metadata=metadata,
        )
