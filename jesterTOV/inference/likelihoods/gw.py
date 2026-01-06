r"""Gravitational wave event likelihood implementations"""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.flows.flow import Flow
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class GWLikelihoodResampled(LikelihoodBase):
    """
    Gravitational wave likelihood for a single GW event using normalizing flow posteriors

    This likelihood evaluates the GW posterior by:
    1. Sampling masses (m1, m2) from the trained normalizing flow
    2. Interpolating tidal deformabilities (Λ1, Λ2) from the EOS
    3. Evaluating the NF log probability on (m1, m2, Λ1, Λ2)

    Parameters
    ----------
    event_name : str
        Name of the GW event (e.g., "GW170817")
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float, optional
        Penalty value for samples where masses exceed Mtov (default: -99999.0)
    N_masses_evaluation : int, optional
        Number of mass samples per likelihood evaluation (default: 20)
    N_masses_batch_size : int, optional
        Batch size for processing mass samples (default: 10)

    Attributes
    ----------
    event_name : str
        Name of the GW event
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float
        Penalty value for samples where masses exceed Mtov
    N_masses_evaluation : int
        Number of mass samples per likelihood evaluation
    N_masses_batch_size : int
        Batch size for processing mass samples
    flow : Flow
        Normalizing flow model for this GW event
    """

    event_name: str
    model_dir: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    flow: Flow

    def __init__(
        self,
        event_name: str,
        model_dir: str,
        penalty_value: float = -99999.0,
        N_masses_evaluation: int = 20,
        N_masses_batch_size: int = 10,
    ) -> None:
        super().__init__()
        self.event_name = event_name
        self.model_dir = model_dir
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size

        # Load Flow model for this event
        logger.info(f"Loading NF model for {event_name} from {model_dir}")
        self.flow = Flow.from_directory(model_dir)
        logger.info(f"Loaded NF model for {event_name}")

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - '_random_key': Random seed for mass sampling (cast to int64)
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS

        Returns
        -------
        Float
            Log likelihood value for this GW event
        """
        # Extract parameters
        sampled_key = params["_random_key"].astype("int64")
        key = jax.random.key(sampled_key)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        # Sample all N_masses_evaluation samples from NF in one go
        all_nf_samples: Float[Array, "n_samples 2"] = self.flow.sample(
            key, (self.N_masses_evaluation,)
        )

        def process_sample(sample: Float[Array, " 2"]) -> Float:
            """
            Process a single NF sample

            Note: jax.lax.map with batch_size still applies the function to individual
            elements, not batches. The batch_size parameter is for compilation optimization.

            Parameters
            ----------
            sample : Float[Array, " 2"]
                Single sample with [m1, m2]

            Returns
            -------
            Float
                Log probability including penalties for this sample
            """
            m1 = sample[0]
            m2 = sample[1]

            # Interpolate lambdas
            lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
            lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)

            # Evaluate log_prob on single sample
            ml_sample = jnp.array([m1, m2, lambda_1, lambda_2])
            logpdf = self.flow.log_prob(ml_sample)

            # Penalties for masses exceeding Mtov
            penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
            penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)

            # Return log prob + penalties for this sample
            return logpdf + penalty_m1 + penalty_m2

        # Use jax.lax.map with batching for memory-efficient processing
        # batch_size helps with compilation memory, not runtime batching
        all_logprobs = jax.lax.map(
            process_sample, all_nf_samples, batch_size=self.N_masses_batch_size
        )

        # Average over all samples for this event
        log_likelihood = jnp.mean(all_logprobs)

        return log_likelihood


class GWLikelihood(LikelihoodBase):
    """
    Gravitational wave likelihood using pre-sampled masses for deterministic evaluation

    This likelihood improves upon GWLikelihoodResampled by pre-sampling mass pairs once at
    initialization, eliminating the need for the _random_key parameter and providing
    deterministic likelihood evaluations critical for sampler convergence.

    Key improvements over GWLikelihoodResampled:
    1. Deterministic: Same EOS parameters → same likelihood value
    2. No _random_key hack: Uses fixed seed at initialization
    3. Scalable: Can use N=10,000+ samples efficiently on GPU
    4. Fair comparison: All EOS evaluated at identical mass points
    5. Better convergence: Smooth likelihood surface for MCMC/SMC

    The likelihood works by:
    1. Pre-sampling (m1, m2) pairs from the trained flow at initialization
    2. For each EOS evaluation:
       a. Interpolate Λ1, Λ2 from the candidate EOS at the fixed mass points
       b. Evaluate flow log_prob on (m1, m2, Λ1_EOS, Λ2_EOS)
       c. Apply penalties for masses exceeding Mtov
       d. Average over all pre-sampled mass pairs

    Parameters
    ----------
    event_name : str
        Name of the GW event (e.g., "GW170817")
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float, optional
        Penalty value for samples where masses exceed Mtov (default: -99999.0)
    N_masses_evaluation : int, optional
        Number of mass samples to pre-sample (default: 2000)
        Large values recommended - GPU parallelization makes this cheap!
    N_masses_batch_size : int, optional
        Batch size for jax.lax.map processing (default: 1000)
    seed : int, optional
        Random seed for mass pre-sampling (default: 42)
        Fixed seed ensures reproducibility across runs

    Attributes
    ----------
    event_name : str
        Name of the GW event
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float
        Penalty value for samples where masses exceed Mtov
    N_masses_evaluation : int
        Number of pre-sampled mass pairs
    N_masses_batch_size : int
        Batch size for processing
    seed : int
        Random seed used for pre-sampling
    flow : Flow
        Normalizing flow model for this GW event
    fixed_mass_samples : Float[Array, "n_samples 2"]
        Pre-sampled (m1, m2) pairs from the flow, shape [N, 2]

    Notes
    -----
    This class does NOT require _random_key in the parameter dictionary,
    unlike GWLikelihoodResampled. The seed is only used once at initialization.

    GPU parallelization via jax.lax.map means N=10,000 samples costs nearly
    the same as N=20, so use large N for near-integration accuracy.

    Examples
    --------
    Configure in YAML:

    >>> likelihoods:
    >>>   - type: "gw"
    >>>     enabled: true
    >>>     parameters:
    >>>       events:
    >>>         - name: "GW170817"
    >>>       N_masses_evaluation: 2000  # Default value
    >>>       N_masses_batch_size: 1000
    >>>       seed: 42
    """

    event_name: str
    model_dir: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    seed: int
    flow: Flow
    fixed_mass_samples: Float[Array, "n_samples 2"]

    def __init__(
        self,
        event_name: str,
        model_dir: str,
        penalty_value: float = -99999.0,
        N_masses_evaluation: int = 2000,
        N_masses_batch_size: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.event_name = event_name
        self.model_dir = model_dir
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size
        self.seed = seed

        # Load Flow model for this event
        logger.info(f"Loading NF model for {event_name} from {model_dir}")
        self.flow = Flow.from_directory(model_dir)
        logger.info(f"Loaded NF model for {event_name}")

        # Pre-sample masses ONCE at initialization
        logger.info(
            f"Pre-sampling {N_masses_evaluation} mass pairs with seed={seed} for {event_name}"
        )
        key = jax.random.key(seed)
        samples = self.flow.sample(key, (N_masses_evaluation,))
        # Extract only (m1, m2), discard Lambda values from flow
        self.fixed_mass_samples = samples[:, :2]  # Shape: [N, 2]
        logger.info(
            f"Pre-sampled mass range: m1=[{jnp.min(self.fixed_mass_samples[:, 0]):.3f}, "
            f"{jnp.max(self.fixed_mass_samples[:, 0]):.3f}] Msun, "
            f"m2=[{jnp.min(self.fixed_mass_samples[:, 1]):.3f}, "
            f"{jnp.max(self.fixed_mass_samples[:, 1]):.3f}] Msun"
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS

            Note: Does NOT require '_random_key' (unlike GWLikelihood)

        Returns
        -------
        Float
            Log likelihood value for this GW event
        """
        # Extract EOS parameters (no _random_key needed!)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        def process_sample(sample: Float[Array, " 2"]) -> Float:
            """
            Process a single pre-sampled mass pair

            Note: jax.lax.map with batch_size applies function to individual
            elements. The batch_size parameter is for compilation optimization.

            Parameters
            ----------
            sample : Float[Array, " 2"]
                Pre-sampled mass pair [m1, m2]

            Returns
            -------
            Float
                Log probability including penalties for this sample
            """
            m1 = sample[0]
            m2 = sample[1]

            # Interpolate lambdas from candidate EOS
            lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
            lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)

            # Evaluate log_prob on single sample
            ml_sample = jnp.array([m1, m2, lambda_1, lambda_2])
            logpdf = self.flow.log_prob(ml_sample)

            # Penalties for masses exceeding Mtov
            penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
            penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)

            # Return log prob + penalties for this sample
            return logpdf + penalty_m1 + penalty_m2

        # Use jax.lax.map with batching for memory-efficient processing
        # Process all pre-sampled mass pairs
        all_logprobs = jax.lax.map(
            process_sample, self.fixed_mass_samples, batch_size=self.N_masses_batch_size
        )

        # Average over all pre-sampled mass pairs
        log_likelihood = jnp.mean(all_logprobs)

        return log_likelihood
