"""BlackJAX Nested Sampling sampler for JESTER inference.

This module provides nested sampling using the BlackJAX library (handley-lab fork)
with acceptance walk kernel for efficient exploration of the parameter space.

# FIXME: this is still being tested, use with care!
"""

from typing import Any
import time

import jax
import jax.numpy as jnp
import jax.random
from jax.experimental import io_callback
from jaxtyping import Array, PRNGKeyArray

from ..base import LikelihoodBase, Prior, BijectiveTransform, NtoMTransform
from ..config.schema import BlackJAXNSAWConfig
from .jester_sampler import JesterSampler, SamplerOutput
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class BlackJAXNSAWSampler(JesterSampler):
    """BlackJAX Nested Sampling with acceptance walk kernel.

    This sampler implements nested sampling for Bayesian evidence calculation
    and posterior sampling. It uses unit cube transforms (all parameters mapped
    to [0, 1]) and the acceptance walk kernel for MCMC proposals.

    Parameters
    ----------
    likelihood : LikelihoodBase
        Likelihood object with evaluate(params, data) method
    prior : Prior
        Prior object (must be CombinePrior of UniformPrior)
    sample_transforms : list[BijectiveTransform]
        Unit cube transforms (created by transform_factory)
    likelihood_transforms : list[NtoMTransform]
        N-to-M transforms applied before likelihood evaluation
    config : BlackJAXNSAWConfig
        Nested sampling configuration
    seed : int, optional
        Random seed (default: 0)

    Attributes
    ----------
    config : BlackJAXNSAWConfig
        Sampler configuration
    final_state : Any | None
        Final nested sampling state (after sampling)
    metadata : dict
        Sampling metadata (evidence, time, etc.)
    _logprior_fn : callable
        Pre-compiled log prior function (unit cube → prior space)
    _loglikelihood_fn : callable
        Pre-compiled log likelihood function (unit cube → likelihood)

    Notes
    -----
    Requires BoundToBound [0,1] transforms for all parameters (created automatically
    by transform_factory for nested sampling).
    """

    config: BlackJAXNSAWConfig
    final_state: Any | None
    metadata: dict
    _logprior_fn: Any  # Compiled JAX function
    _loglikelihood_fn: Any  # Compiled JAX function
    _unit_cube_stepper: Any  # Unit cube stepper function
    _filtered_samples_cache: dict | None  # Cache for filtered samples from anesthetic

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform],
        likelihood_transforms: list[NtoMTransform],
        config: BlackJAXNSAWConfig,
        seed: int = 0,
    ) -> None:
        """Initialize BlackJAX nested sampling sampler."""
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)

        self.config = config
        self.final_state = None
        self.metadata = {}
        self._seed = seed
        self._filtered_samples_cache = None

        # Nested sampling requires unit cube transforms
        # If not provided, create them automatically
        if len(sample_transforms) == 0:
            logger.info(
                "No sample transforms provided - creating unit cube transforms for NS-AW"
            )
            from .transform_factory import create_sample_transforms

            sample_transforms = create_sample_transforms(config, prior)
            # Update the sample_transforms in the parent class
            self.sample_transforms = sample_transforms
            # Recompute parameter names after adding transforms
            for transform in self.sample_transforms:
                self.parameter_names = transform.propagate_name(self.parameter_names)

        logger.info("Initializing BlackJAX Nested Sampling (Acceptance Walk) sampler")
        logger.info(
            f"Configuration: {config.n_live} live points, "
            f"delete fraction {config.n_delete_frac}"
        )
        logger.info(f"Termination: dlogZ < {config.termination_dlogz}")

        # Pre-compile log prior and log likelihood functions
        self._create_logprior_fn()
        self._create_loglikelihood_fn()

        # Create unit cube stepper function (wraps at [0, 1] boundaries)
        self._create_unit_cube_stepper()

        # Import BlackJAX nested sampling (lazy import to avoid dependency issues)
        try:
            from blackjax.ns.utils import finalise

            self._finalise = finalise
        except ImportError as e:
            raise ImportError(
                "BlackJAX nested sampling not found. Install with: "
                "pip install git+https://github.com/handley-lab/blackjax@nested_sampling"
            ) from e

        # Note: The actual nested sampler is created in sample() method
        # since it requires the acceptance walk kernel implementation

    def _create_logprior_fn(self) -> None:
        """Pre-compile log prior function in unit cube space.

        This function:
        1. Applies inverse sample transforms (unit cube → prior space)
        2. Evaluates prior log probability
        3. Adds Jacobian corrections from transforms
        """

        def logprior_fn(params_dict: dict[str, float]) -> float:
            """Evaluate log prior in unit cube space."""
            transform_jacobian = 0.0
            named_params = params_dict.copy()

            # Apply inverse transforms (unit cube → prior)
            for transform in reversed(self.sample_transforms):
                named_params, jacobian = transform.inverse(named_params)
                transform_jacobian += jacobian

            # Evaluate prior + Jacobian
            return self.prior.log_prob(named_params) + transform_jacobian

        # JIT compile for performance
        self._logprior_fn = jax.jit(logprior_fn)

    def _create_loglikelihood_fn(self) -> None:
        """Pre-compile log likelihood function in unit cube space.

        This function:
        1. Applies inverse sample transforms (unit cube → prior space)
        2. Applies forward likelihood transforms (prior → likelihood params)
        3. Evaluates likelihood
        """

        def loglikelihood_fn(params_dict: dict[str, float]) -> float:
            """Evaluate log likelihood in unit cube space."""
            named_params = params_dict.copy()

            # Apply inverse sample transforms (unit cube → prior)
            for transform in reversed(self.sample_transforms):
                named_params, _ = transform.inverse(named_params)

            # Apply likelihood transforms (prior → likelihood params)
            for transform in self.likelihood_transforms:
                named_params = transform.forward(named_params)

            # Evaluate likelihood
            return self.likelihood.evaluate(named_params)

        # JIT compile for performance
        self._loglikelihood_fn = jax.jit(loglikelihood_fn)

    def _create_unit_cube_stepper(self) -> None:
        """Create stepper function that wraps parameters at [0, 1] boundaries.

        For JESTER, all parameters are bounded but not periodic, so we use
        modulo wrapping for all parameters to keep them in [0, 1].
        """

        def unit_cube_stepper(
            position: dict, direction: dict, step_size: float
        ) -> dict:
            """Step in unit cube with periodic boundary wrapping."""
            proposed = jax.tree.map(
                lambda pos, d: pos + step_size * d,
                position,
                direction,
            )
            # Wrap all parameters to [0, 1] using modulo
            return jax.tree.map(lambda prop: jnp.mod(prop, 1.0), proposed)

        self._unit_cube_stepper = unit_cube_stepper

    def sample(
        self, key: PRNGKeyArray, initial_position: Array = jnp.array([])
    ) -> None:
        """Run nested sampling until termination criterion.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key
        initial_position : Array, optional
            Not used for nested sampling (samples from prior in unit cube)
        """
        logger.info("Starting nested sampling...")
        start_time = time.time()

        # Import acceptance walk sampler from kernels
        from .kernels import bilby_adaptive_de_sampler_unit_cube

        # Configure sampler
        n_delete = int(self.config.n_live * self.config.n_delete_frac)

        logger.info(f"Sampling {self.config.n_live} live points, batch size {n_delete}")

        # Sample initial positions from prior
        key, subkey = jax.random.split(key)
        initial_particles = self.prior.sample(subkey, self.config.n_live)

        # Transform to unit cube
        for transform in self.sample_transforms:
            initial_particles = jax.vmap(transform.forward)(initial_particles)

        # Initialize nested sampler with acceptance walk kernel
        nested_sampler = bilby_adaptive_de_sampler_unit_cube(
            logprior_fn=self._logprior_fn,
            loglikelihood_fn=self._loglikelihood_fn,
            nlive=self.config.n_live,
            n_target=self.config.n_target,
            max_mcmc=self.config.max_mcmc,
            num_delete=n_delete,
            stepper_fn=self._unit_cube_stepper,
            max_proposals=self.config.max_proposals,
        )

        # Initialize sampler state
        # Note: init_fn only takes particles, pyright incorrectly expects rng_key
        state = nested_sampler.init(initial_particles)  # type: ignore[call-arg]

        def terminate(state):
            """Termination condition: stop when remaining evidence is small."""
            dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
            return jnp.isfinite(dlogz) and dlogz < self.config.termination_dlogz

        # JIT compile step function for performance
        step_fn = jax.jit(nested_sampler.step)

        # Progress callback for live updates during sampling
        def progress_callback(iteration: int, logZ: float, dlogZ: float) -> None:
            """Print progress update during nested sampling (called via io_callback)."""
            # Format logZ and dlogZ with appropriate precision
            logZ_str = f"{logZ:+10.2f}" if jnp.isfinite(logZ) else "      -inf"
            dlogZ_str = f"{dlogZ:8.4f}" if jnp.isfinite(dlogZ) else "     inf"

            # Print update
            logger.info(
                f"Iteration {iteration:4d} | logZ={logZ_str} | dlogZ={dlogZ_str}"
            )

        # Run nested sampling loop
        logger.info("=" * 70)
        logger.info("STARTING NESTED SAMPLING")
        logger.info("=" * 70)
        logger.info(f"Live points: {self.config.n_live}")
        logger.info(
            f"Delete fraction: {self.config.n_delete_frac} ({n_delete} points per iteration)"
        )
        logger.info(f"Termination: dlogZ < {self.config.termination_dlogz}")
        logger.info(f"Max MCMC steps: {self.config.max_mcmc}")
        logger.info("Progress updates will be shown after each iteration")
        logger.info("=" * 70)

        dead = []
        n_iterations = 0

        while not terminate(state):
            key, subkey = jax.random.split(key)
            state, dead_info = step_fn(subkey, state)
            dead.append(dead_info)
            n_iterations += 1

            # Compute current evidence and termination criterion
            current_logZ = float(state.logZ)  # type: ignore[attr-defined]
            current_dlogZ = float(jnp.logaddexp(0, state.logZ_live - state.logZ))  # type: ignore[attr-defined]

            # Print progress update using io_callback
            io_callback(
                progress_callback,
                None,  # No return value
                n_iterations,
                current_logZ,
                current_dlogZ,
            )

        # Store evidence from state before finalization
        # (NSState has logZ, but NSInfo from finalise does not)
        logZ = float(state.logZ)  # type: ignore[attr-defined]
        # Estimate uncertainty from remaining evidence in live points
        logZ_err = float(jnp.logaddexp(0, state.logZ_live - state.logZ))  # type: ignore[attr-defined]

        # Finalize nested sampling results
        logger.info("Finalizing nested sampling results...")
        final_info = self._finalise(state, dead)  # type: ignore[arg-type]

        # Transform particles back to prior space
        logger.info("Transforming samples back to prior space...")
        physical_particles = final_info.particles
        for transform in reversed(self.sample_transforms):
            # Type note: vmap preserves PyTree structure; physical_particles remains ArrayTree
            physical_particles = jax.vmap(transform.backward)(physical_particles)  # type: ignore[arg-type]

        # Store final info with physical parameters
        # Note: final_info is NSInfo (not NSState), so we store it with replaced particles
        self.final_state = final_info._replace(particles=physical_particles)

        # Store metadata
        end_time = time.time()
        sampling_time = end_time - start_time

        # Get number of samples from pytree (particles is a dict, not array)
        particles_leaves = jax.tree_util.tree_leaves(final_info.particles)
        n_samples = int(particles_leaves[0].shape[0]) if particles_leaves else 0

        self.metadata = {
            "sampler": "blackjax_ns_aw",
            "n_live": self.config.n_live,
            "n_delete": n_delete,
            "n_delete_frac": self.config.n_delete_frac,
            "n_target": self.config.n_target,
            "max_mcmc": self.config.max_mcmc,
            "max_proposals": self.config.max_proposals,
            "termination_dlogz": self.config.termination_dlogz,
            "sampling_time_seconds": sampling_time,
            "sampling_time_minutes": sampling_time / 60,
            "n_iterations": n_iterations,
            "n_samples": n_samples,
            "n_likelihood_evaluations": int(jnp.sum(final_info.inner_kernel_info.n_likelihood_evals)),  # type: ignore[attr-defined]
            "logZ": logZ,
            "logZ_err": logZ_err,
        }

        logger.info("=" * 70)
        logger.info("NESTED SAMPLING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total iterations: {n_iterations}")
        logger.info(f"Dead points generated: {len(dead) * n_delete}")
        logger.info(f"Final evidence: log(Z) = {logZ:.2f} ± {logZ_err:.2f}")
        logger.info(
            f"Final dlogZ: {logZ_err:.4f} (termination criterion: {self.config.termination_dlogz})"
        )
        logger.info(
            f"Sampling time: {(sampling_time)//60:.0f} minutes {(sampling_time)%60:.1f} seconds"
        )
        logger.info(f"Likelihood evaluations: {int(jnp.sum(final_info.inner_kernel_info.n_likelihood_evals))}")  # type: ignore[attr-defined]
        logger.info("=" * 70)

    def print_summary(self, transform: bool = True) -> None:
        """Print summary of nested sampling run.

        Parameters
        ----------
        transform : bool, optional
            Not used for nested sampling (always returns physical parameters)
        """
        logger.info("=" * 70)
        logger.info("NESTED SAMPLING SUMMARY")
        logger.info("=" * 70)

        if self.final_state is None:
            logger.warning("No samples yet - run sample() first")
            return

        # Print evidence
        if "logZ" in self.metadata:
            logger.info(
                f"log(Z) = {self.metadata['logZ']:.2f} ± {self.metadata['logZ_err']:.2f}"
            )

        # Print sampling info
        logger.info(f"Live points: {self.config.n_live}")
        logger.info(
            f"Sampling time: {self.metadata.get('sampling_time_seconds', 0):.1f}s"
        )

        if "n_samples" in self.metadata:
            logger.info(f"Posterior samples: {self.metadata['n_samples']}")

    def get_samples(self) -> dict:
        """Return unweighted posterior samples from nested sampling.

        This method computes importance weights using anesthetic, then resamples
        to produce approximately ESS (effective sample size) unweighted posterior
        samples. This ensures downstream analysis (plotting, postprocessing) treats
        all samples as equally weighted, which is the expected behavior.

        Returns
        -------
        dict
            Dictionary with:
            - Parameter samples (resampled, unweighted)
            - 'logL': log likelihood values (resampled)
            - 'logL_birth': birth log likelihoods (resampled)

        Notes
        -----
        The original weighted samples are cached in _filtered_samples_cache for
        advanced users who need access to the full weighted set.
        """
        if self.final_state is None:
            raise RuntimeError("No samples available - run sample() first")

        # Handle birth likelihoods (replace NaN with -inf)
        logL_birth = self.final_state.loglikelihood_birth.copy()
        logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

        # Compute importance weights using anesthetic (if available)
        # Note: anesthetic may drop invalid samples (logL <= logL_birth)
        try:
            from anesthetic.samples import NestedSamples
            import warnings

            # Note: self.final_state is NSInfo, which has particles in physical (prior) space
            # Suppress the logL <= logL_birth warning (it's handled internally by anesthetic)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "out of .* samples have logL <= logL_birth"
                )
                ns_samples = NestedSamples(
                    self.final_state.particles,
                    logL=self.final_state.loglikelihood,
                    logL_birth=logL_birth,
                    logzero=jnp.nan,
                    dtype=jnp.float64,
                )

            # Get posterior weights (anesthetic computes from logL and logL_birth)
            # NOTE: anesthetic may drop invalid samples, so we need to use the filtered data
            weights = ns_samples.get_weights()

            # Extract filtered samples from anesthetic (it's a DataFrame)
            # Get all columns except metadata columns
            param_cols = [
                col
                for col in ns_samples.columns
                if col not in ["logL", "logL_birth", "weights"]
            ]
            samples = {col: jnp.array(ns_samples[col].values) for col in param_cols}

            # Add metadata
            samples["weights"] = jnp.array(weights)
            samples["logL"] = jnp.array(ns_samples["logL"].values)
            samples["logL_birth"] = jnp.array(ns_samples["logL_birth"].values)

            # Cache filtered samples BEFORE resampling (for get_log_prob() to use if needed)
            self._filtered_samples_cache = samples.copy()

            # Resample weighted samples to produce unweighted posterior samples
            # This is critical for downstream analysis that assumes equal weights
            logger.info(
                "Resampling weighted NS samples to produce unweighted posterior..."
            )

            # Compute effective sample size
            weights_array = samples["weights"]
            ess = jnp.sum(weights_array) ** 2 / jnp.sum(weights_array**2)
            logger.info(
                f"Effective sample size: {ess:.1f} / {len(weights_array)} raw samples"
            )

            # Number of samples to draw: use ESS as target
            # Round to nearest integer, minimum 100 samples
            n_resample = max(100, int(jnp.round(ess)))
            logger.info(f"Resampling to {n_resample} unweighted posterior samples...")

            # Normalize weights for sampling
            normalized_weights = weights_array / jnp.sum(weights_array)

            # Resample with replacement using weighted sampling
            key = jax.random.PRNGKey(
                self._seed + 1000
            )  # Offset seed for reproducibility
            indices = jax.random.choice(
                key,
                len(weights_array),
                shape=(n_resample,),
                replace=True,
                p=normalized_weights,
            )

            # Create resampled samples dict
            resampled_samples = {}
            for key_name, value in samples.items():
                if key_name == "weights":
                    # All samples now have equal weight
                    continue
                elif key_name in ["logL", "logL_birth"]:
                    # Keep metadata
                    resampled_samples[key_name] = value[indices]
                else:
                    # Resample parameter arrays
                    resampled_samples[key_name] = value[indices]

            # Replace samples with resampled version
            samples = resampled_samples
            logger.info(
                f"Resampling complete: {len(samples[list(samples.keys())[0]])} unweighted samples"
            )

            # Update cache to match resampled data (for get_log_prob() consistency)
            self._filtered_samples_cache = samples.copy()

            # Store evidence from anesthetic computation (more accurate than our estimate)
            try:
                # Note: logZ() returns the evidence value; std() is accessed on the samples
                logZ_result = ns_samples.logZ()
                logZ_anesthetic = float(logZ_result)  # type: ignore[arg-type]
                # Get standard deviation from the logZ samples
                # Note: anesthetic stores logZ values in the samples dataframe
                logZ_err_anesthetic = float(ns_samples.logZ().std())  # type: ignore[union-attr]
                # Only set if both succeed
                self.metadata["logZ_anesthetic"] = logZ_anesthetic
                self.metadata["logZ_err_anesthetic"] = logZ_err_anesthetic
            except Exception as e:
                logger.warning(f"Could not compute anesthetic evidence: {e}")

        except ImportError:
            logger.warning(
                "anesthetic not available - using all samples without resampling"
            )
            # Use all samples without filtering or resampling
            samples = dict(self.final_state.particles)
            samples["logL"] = self.final_state.loglikelihood
            samples["logL_birth"] = logL_birth
            self._filtered_samples_cache = samples
            logger.warning(
                "Without anesthetic, cannot compute proper weights. "
                "All samples treated equally (may include low-weight samples)."
            )
        except Exception as e:
            logger.warning(
                f"anesthetic weight computation failed: {e} - using all samples without resampling"
            )
            # Use all samples without filtering or resampling
            samples = dict(self.final_state.particles)
            samples["logL"] = self.final_state.loglikelihood
            samples["logL_birth"] = logL_birth
            self._filtered_samples_cache = samples
            logger.warning(
                "Without proper weights, all samples treated equally (may include low-weight samples)."
            )

        return samples

    def get_log_prob(self) -> Array:
        """Get log likelihoods from nested sampling.

        Returns
        -------
        Array
            Log likelihood values (1D array)
            Note: For NS, this is log likelihood, not log posterior.
            Use weights separately for posterior inference.

        Notes
        -----
        This method returns filtered log likelihoods (matching get_samples()).
        If anesthetic has dropped invalid samples, the length will be less than
        the raw NSInfo.loglikelihood array.
        """
        if self.final_state is None:
            raise RuntimeError("No samples available - run sample() first")

        # Use cached filtered samples if available (from get_samples())
        # This ensures get_log_prob() and get_samples() have consistent lengths
        if self._filtered_samples_cache is not None:
            return self._filtered_samples_cache["logL"]

        # Fallback: return all log likelihoods (unfiltered)
        # This shouldn't happen in normal usage since get_samples() is called first
        logger.warning(
            "get_log_prob() called before get_samples() - returning unfiltered logL"
        )
        return self.final_state.loglikelihood

    def get_n_samples(self) -> int:
        """Get number of posterior samples from nested sampling.

        Returns
        -------
        int
            Number of posterior samples

        Notes
        -----
        This method returns the number of filtered samples (matching get_samples()).
        If anesthetic has dropped invalid samples, the count will be less than
        the raw NSInfo particle count.
        """
        if self.final_state is None:
            return 0

        # Use cached filtered samples if available (from get_samples())
        # This ensures get_n_samples() matches get_samples() and get_log_prob()
        if self._filtered_samples_cache is not None:
            # Get length from any parameter array
            first_param = next(iter(self._filtered_samples_cache.keys()))
            return len(self._filtered_samples_cache[first_param])

        # Fallback: return all particles (unfiltered)
        return len(self.final_state.particles)

    def get_sampler_output(self) -> SamplerOutput:
        """
        Get standardized sampler output.

        Returns
        -------
        SamplerOutput
            - samples: Unweighted parameter samples (dict of arrays, no metadata fields)
            - log_prob: Log likelihood (NOT log posterior - NS works in likelihood space)
            - metadata: {"logL": Array, "logL_birth": Array}

        Raises
        ------
        RuntimeError
            If sampling has not been run yet.

        Notes
        -----
        Samples are resampled using importance weights to produce unweighted posterior samples.
        This ensures downstream analysis treats all samples equally.
        log_prob contains log likelihood, not log posterior (standard for NS).
        """
        if self.final_state is None:
            raise RuntimeError("No samples available. Run sample() first.")

        # Get current samples dict (includes weights, logL, logL_birth)
        all_data = self.get_samples()

        # Separate parameters from metadata
        samples: dict[str, Array] = {}
        metadata: dict[str, Any] = {}

        metadata_keys = {"weights", "logL", "logL_birth"}
        for key, value in all_data.items():
            if key in metadata_keys:
                metadata[key] = value
            else:
                samples[key] = value

        # Get log probabilities (log likelihood for NS-AW)
        log_prob = self.get_log_prob()

        return SamplerOutput(
            samples=samples,
            log_prob=log_prob,
            metadata=metadata,
        )
