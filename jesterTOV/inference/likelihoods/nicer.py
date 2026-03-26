r"""
NICER X-ray timing likelihood implementations

This module provides two implementations:
1. NICERLikelihood - Flow-based (NEW DEFAULT, more efficient)
2. NICERKDELikelihood - KDE-based (legacy, for backward compatibility)
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.logging_config import get_logger
import jesterTOV.utils as utils

logger = get_logger("jester")


class NICERLikelihood(LikelihoodBase):
    """
    NICER likelihood using normalizing flows (NEW DEFAULT).

    This is the recommended NICER likelihood implementation that uses
    pre-trained normalizing flows on M-R posteriors for efficient and
    deterministic likelihood evaluation.

    For the legacy KDE-based version, see NICERKDELikelihood.

    The likelihood loads pre-trained flow models for Amsterdam and Maryland
    groups and evaluates the likelihood by:
    1. Pre-sampling masses ONCE at initialization (deterministic with seed)
    2. During evaluation: interpolating radius from the EOS for pre-sampled masses
    3. Evaluating the flow log probability at (mass, radius)
    4. Averaging over all samples

    Parameters
    ----------
    psr_name : str
        Pulsar name (e.g., "J0030", "J0740")
    amsterdam_model_dir : str | None
        Path to directory containing Amsterdam flow model
        (flow_weights.eqx, metadata.json, flow_kwargs.json).
        If None, uses preset model path.
    maryland_model_dir : str | None
        Path to directory containing Maryland flow model.
        If None, uses preset model path.
    penalty_value : float, optional
        Penalty value for samples where mass exceeds Mtov (default: -99999.0)
    N_masses_evaluation : int, optional
        Number of mass samples per likelihood evaluation (default: 20)
    N_masses_batch_size : int, optional
        Batch size for processing mass samples (default: 10)
    seed : int, optional
        Random seed for pre-sampling masses (default: 42)

    Attributes
    ----------
    psr_name : str
        Pulsar name
    penalty_value : float
        Penalty value for samples where mass exceeds Mtov
    N_masses_evaluation : int
        Number of mass samples per likelihood evaluation
    N_masses_batch_size : int
        Batch size for processing mass samples
    seed : int
        Random seed for deterministic pre-sampling
    amsterdam_flow : Flow
        Normalizing flow for Amsterdam M-R posterior
    maryland_flow : Flow
        Normalizing flow for Maryland M-R posterior
    amsterdam_fixed_mass_samples : Float[Array, "n_samples"]
        Pre-sampled mass values from Amsterdam flow (fixed at initialization)
    maryland_fixed_mass_samples : Float[Array, "n_samples"]
        Pre-sampled mass values from Maryland flow (fixed at initialization)
    """

    psr_name: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    seed: int

    def __init__(
        self,
        psr_name: str,
        amsterdam_model_dir: str | None = None,
        maryland_model_dir: str | None = None,
        penalty_value: float = -99999.0,
        N_masses_evaluation: int = 20,
        N_masses_batch_size: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.psr_name = psr_name
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size
        self.seed = seed

        # Import Flow here to avoid circular imports
        from jesterTOV.inference.flows.flow import Flow

        # Validate that both model directories are provided
        if amsterdam_model_dir is None or maryland_model_dir is None:
            raise ValueError(
                f"Both amsterdam_model_dir and maryland_model_dir must be provided for {psr_name}. "
                "Preset model paths are not yet implemented. "
                "Please provide explicit paths to trained flow models "
                "(see TODO_FLOW_TRAINING.md Phase 3)."
            )

        # Use provided model paths
        logger.info(f"Using Amsterdam model directory: {amsterdam_model_dir}")
        logger.info(f"Using Maryland model directory: {maryland_model_dir}")

        # Load flow models
        logger.info(f"Loading Amsterdam flow for {psr_name} from {amsterdam_model_dir}")
        self.amsterdam_flow = Flow.from_directory(amsterdam_model_dir)

        logger.info(f"Loading Maryland flow for {psr_name} from {maryland_model_dir}")
        self.maryland_flow = Flow.from_directory(maryland_model_dir)

        logger.info(f"Loaded normalizing flows for {psr_name}")

        # Pre-sample masses ONCE at initialization (deterministic with seed)
        logger.info(
            f"Pre-sampling {N_masses_evaluation} masses with seed={seed} for {psr_name}"
        )
        key = jax.random.key(seed)
        key_amsterdam, key_maryland = jax.random.split(key)

        # Sample (mass, radius) from flows
        amsterdam_samples = self.amsterdam_flow.sample(
            key_amsterdam, (N_masses_evaluation,)
        )
        maryland_samples = self.maryland_flow.sample(
            key_maryland, (N_masses_evaluation,)
        )

        # Extract only masses (first column), discard radius values
        self.amsterdam_fixed_mass_samples = amsterdam_samples[:, 0]  # Shape: (N,)
        self.maryland_fixed_mass_samples = maryland_samples[:, 0]  # Shape: (N,)

        logger.info(
            f"Pre-sampled Amsterdam mass range: "
            f"[{jnp.min(self.amsterdam_fixed_mass_samples):.3f}, "
            f"{jnp.max(self.amsterdam_fixed_mass_samples):.3f}] Msun"
        )
        logger.info(
            f"Pre-sampled Maryland mass range: "
            f"[{jnp.min(self.maryland_fixed_mass_samples):.3f}, "
            f"{jnp.max(self.maryland_fixed_mass_samples):.3f}] Msun"
        )

    def _get_preset_model_path(self, psr_name: str, group: str) -> str:
        """
        Get preset model path for a pulsar and analysis group.

        Parameters
        ----------
        psr_name : str
            Pulsar name (e.g., "J0030", "J0740")
        group : str
            Analysis group ("amsterdam" or "maryland")

        Returns
        -------
        str
            Path to preset model directory

        Raises
        ------
        ValueError
            If no preset exists for this pulsar/group combination
        """
        # TODO: Define preset paths once NICER flow models are trained
        # For now, this is a placeholder that will be updated in Phase 3

        # Example preset structure (to be implemented):
        # base_dir = Path(__file__).parent.parent / "flows" / "models" / "nicer_maf"
        # model_dir = base_dir / psr_name / f"{psr_name}_{group}_NICER_model"

        raise NotImplementedError(
            f"Preset model paths for {psr_name} {group} not yet implemented. "
            "Please provide explicit model_dir paths or train NICER flows first "
            "(see TODO_FLOW_TRAINING.md Phase 3)."
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters.
        Handles two segments in case of phase transition.
        Segments defined by detecting sudden jump in data gap along pc.
        Uses pre-sampled masses from initialization (deterministic evaluation).

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'radii_EOS': Array of neutron star radii from EOS

        Returns
        -------
        Float
            Log likelihood value for this NICER observation
        """
        # Extract parameters
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        radii_EOS: Float[Array, " n_points"] = params["radii_EOS"]
        mtov: Float = jnp.max(masses_EOS)
        
        # mass-radius split wrt phase transition jump
        split_idx = utils.get_MR_split_index(masses_EOS, radii_EOS)
        
        # divide by mask
        idx = jnp.arange(masses_EOS.shape[0])
        mask1 = idx < split_idx
        mask2 = idx >= split_idx

        # Segment 1
        m_eos_1 = jnp.where(mask1, masses_EOS, jnp.inf)
        r_eos_1 = jnp.where(mask1, radii_EOS, 0.0)
        sort_1 = jnp.argsort(m_eos_1)
        m_eos_1, r_eos_1 = m_eos_1[sort_1], r_eos_1[sort_1]
        
        # Boundary 1: Find valid min and max mass for this segment
        seg1_min = m_eos_1[0]
        seg1_max = jnp.max(jnp.where(m_eos_1 == jnp.inf, -jnp.inf, m_eos_1))

        # Segment 2
        m_eos_2 = jnp.where(mask2, masses_EOS, jnp.inf)
        r_eos_2 = jnp.where(mask2, radii_EOS, 0.0)
        sort_2 = jnp.argsort(m_eos_2)
        m_eos_2, r_eos_2 = m_eos_2[sort_2], r_eos_2[sort_2]

        # Boundary 2: Find valid min and max mass for this segment
        seg2_min = m_eos_2[0]
        seg2_max = jnp.max(jnp.where(m_eos_2 == jnp.inf, -jnp.inf, m_eos_2))

        def process_sample_amsterdam(mass: Float, m_eos: Array, r_eos: Array, max_m_tov: Float, seg_min: Float, seg_max: Float) -> Float:
            radius = jnp.interp(mass, m_eos, r_eos)
            mr_point = jnp.array([[mass, radius]])
            logpdf = self.amsterdam_flow.log_prob(mr_point)
            
            # Do not extrapolate: zero probability mask for extrapolated points, so no need for weighing
            in_segment = (mass >= seg_min) & (mass <= seg_max)
            logpdf = jnp.where(in_segment, logpdf, -jnp.inf)
            
            penalty = jnp.where(mass > max_m_tov, self.penalty_value, 0.0)
            return logpdf + penalty

        def process_sample_maryland(mass: Float, m_eos: Array, r_eos: Array, max_m_tov: Float, seg_min: Float, seg_max: Float) -> Float:
            radius = jnp.interp(mass, m_eos, r_eos)
            mr_point = jnp.array([[mass, radius]])
            logpdf = self.maryland_flow.log_prob(mr_point)
            
            # Zero probability mask for extrapolated points, so no need for weighing
            in_segment = (mass >= seg_min) & (mass <= seg_max)
            logpdf = jnp.where(in_segment, logpdf, -jnp.inf)
            
            penalty = jnp.where(mass > max_m_tov, self.penalty_value, 0.0)
            return logpdf + penalty

        amsterdam_logprobs_1 = jax.lax.map(
            lambda m: process_sample_amsterdam(m, m_eos_1, r_eos_1, mtov, seg1_min, seg1_max),
            self.amsterdam_fixed_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        maryland_logprobs_1 = jax.lax.map(
            lambda m: process_sample_maryland(m, m_eos_1, r_eos_1, mtov, seg1_min, seg1_max),
            self.maryland_fixed_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        amsterdam_logprobs_2 = jax.lax.map(
            lambda m: process_sample_amsterdam(m, m_eos_2, r_eos_2, mtov, seg2_min, seg2_max),
            self.amsterdam_fixed_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        maryland_logprobs_2 = jax.lax.map(
            lambda m: process_sample_maryland(m, m_eos_2, r_eos_2, mtov, seg2_min, seg2_max),
            self.maryland_fixed_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        # Average over all samples for each group
        # In this case, even if data is accidentally splitted somewhere, log_likelihood will be exactly the same.
        N_amsterdam = amsterdam_logprobs_1.shape[0]
        N_maryland = maryland_logprobs_1.shape[0]

        logL_amsterdam_1 = logsumexp(amsterdam_logprobs_1) - jnp.log(N_amsterdam) 
        logL_maryland_1 = logsumexp(maryland_logprobs_1) - jnp.log(N_maryland) 
        
        logL_amsterdam_2 = logsumexp(amsterdam_logprobs_2) - jnp.log(N_amsterdam) 
        logL_maryland_2 = logsumexp(maryland_logprobs_2) - jnp.log(N_maryland) 

        log_likelihood = logsumexp(
            jnp.array([logL_amsterdam_1, logL_maryland_1, logL_amsterdam_2, logL_maryland_2])
        ) - jnp.log(2.0)

        return log_likelihood


class NICERKDELikelihood(LikelihoodBase):
    """
    NICER likelihood using KDE (Kernel Density Estimation) approach.

    This is the original NICER likelihood implementation that uses KDE
    on M-R posterior samples. For the flow-based version, see NICERLikelihood.

    TODO: Generalize to e.g. only one group, weights between different hotspot models,...

    This likelihood loads posterior samples from Amsterdam and Maryland groups,
    constructs KDEs, and evaluates the likelihood by:
    1. Sampling masses from the NICER posterior samples
    2. Interpolating radius from the EOS for those masses
    3. Evaluating the KDE log probability at (mass, radius)
    4. Averaging over all samples

    Parameters
    ----------
    psr_name : str
        Pulsar name (e.g., "J0030", "J0740")
    amsterdam_samples_file : str
        Path to npz file with Amsterdam group posterior samples
        Expected to contain 'mass' (Msun) and 'radius' (km) arrays
    maryland_samples_file : str
        Path to npz file with Maryland group posterior samples
        Expected to contain 'mass' (Msun) and 'radius' (km) arrays
    penalty_value : float, optional
        Penalty value for samples where mass exceeds Mtov (default: -99999.0)
    N_masses_evaluation : int, optional
        Number of mass samples per likelihood evaluation (default: 20)
    N_masses_batch_size : int, optional
        Batch size for processing mass samples (default: 10)

    Attributes
    ----------
    psr_name : str
        Pulsar name
    penalty_value : float
        Penalty value for samples where mass exceeds Mtov
    N_masses_evaluation : int
        Number of mass samples per likelihood evaluation
    N_masses_batch_size : int
        Batch size for processing mass samples
    amsterdam_masses : Float[Array, " n_amsterdam"]
        Mass samples from Amsterdam group
    maryland_masses : Float[Array, " n_maryland"]
        Mass samples from Maryland group
    amsterdam_posterior : gaussian_kde
        KDE of Amsterdam (mass, radius) posterior
    maryland_posterior : gaussian_kde
        KDE of Maryland (mass, radius) posterior
    """

    psr_name: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    amsterdam_masses: Float[Array, " n_amsterdam"]
    maryland_masses: Float[Array, " n_maryland"]
    amsterdam_posterior: gaussian_kde
    maryland_posterior: gaussian_kde

    def __init__(
        self,
        psr_name: str,
        amsterdam_samples_file: str,
        maryland_samples_file: str,
        penalty_value: float = -99999.0,
        N_masses_evaluation: int = 20,
        N_masses_batch_size: int = 10,
    ) -> None:
        super().__init__()
        self.psr_name = psr_name
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size

        # Load samples from npz files
        logger.info(
            f"Loading Amsterdam samples for {psr_name} from {amsterdam_samples_file}"
        )
        amsterdam_data = np.load(amsterdam_samples_file, allow_pickle=True)

        logger.info(
            f"Loading Maryland samples for {psr_name} from {maryland_samples_file}"
        )
        maryland_data = np.load(maryland_samples_file, allow_pickle=True)

        # Extract mass and radius samples
        # File format: mass (Msun), radius (km)
        amsterdam_mass = amsterdam_data["mass"]
        amsterdam_radius = amsterdam_data["radius"]
        maryland_mass = maryland_data["mass"]
        maryland_radius = maryland_data["radius"]

        # Store mass samples as JAX arrays for random sampling
        self.amsterdam_masses = jnp.array(amsterdam_mass)
        self.maryland_masses = jnp.array(maryland_mass)

        # Stack into [mass, radius] arrays for KDE
        # Convert to JAX arrays for JAX KDE
        amsterdam_mr = jnp.vstack([amsterdam_mass, amsterdam_radius])
        maryland_mr = jnp.vstack([maryland_mass, maryland_radius])

        # Construct KDEs using JAX implementation
        logger.info(f"Constructing JAX KDEs for {psr_name}")
        self.amsterdam_posterior = gaussian_kde(amsterdam_mr)
        self.maryland_posterior = gaussian_kde(maryland_mr)
        logger.info(f"Loaded JAX KDEs for {psr_name}")

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters.
        Handles two segments in case of phase transition.
        Segments defined by detecting sudden jump in data gap along pc.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - '_random_key': Random seed for mass sampling (cast to int64)
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'radii_EOS': Array of neutron star radii from EOS

        Returns
        -------
        Float
            Log likelihood value for this NICER observation
        """
        # Extract parameters
        sampled_key = params["_random_key"].astype("int64")
        key = jax.random.key(sampled_key)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        radii_EOS: Float[Array, " n_points"] = params["radii_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        # mass-radius split wrt phase transition jump
        split_idx = utils.get_MR_split_index(masses_EOS, radii_EOS)
        
        # divide by mask
        idx = jnp.arange(masses_EOS.shape[0])
        mask1 = idx < split_idx
        mask2 = idx >= split_idx

        # Segment 1
        m_eos_1 = jnp.where(mask1, masses_EOS, jnp.inf)
        r_eos_1 = jnp.where(mask1, radii_EOS, 0.0)
        sort_1 = jnp.argsort(m_eos_1)
        m_eos_1, r_eos_1 = m_eos_1[sort_1], r_eos_1[sort_1]
        
        # Boundary 1: Find valid min and max mass for this segment
        seg1_min = m_eos_1[0]
        seg1_max = jnp.max(jnp.where(m_eos_1 == jnp.inf, -jnp.inf, m_eos_1))

        # Segment 2
        m_eos_2 = jnp.where(mask2, masses_EOS, jnp.inf)
        r_eos_2 = jnp.where(mask2, radii_EOS, 0.0)
        sort_2 = jnp.argsort(m_eos_2)
        m_eos_2, r_eos_2 = m_eos_2[sort_2], r_eos_2[sort_2]

        # Boundary 2: Find valid min and max mass for this segment
        seg2_min = m_eos_2[0]
        seg2_max = jnp.max(jnp.where(m_eos_2 == jnp.inf, -jnp.inf, m_eos_2))

        # Split key for Amsterdam and Maryland sampling
        key_amsterdam, key_maryland = jax.random.split(key)

        # Sample masses from the NICER posterior samples
        # Each group gets half of N_masses_evaluation samples
        n_samples_per_group: int = self.N_masses_evaluation // 2

        # Sample indices and get mass samples
        amsterdam_indices = jax.random.choice(
            key_amsterdam,
            len(self.amsterdam_masses),
            shape=(n_samples_per_group,),
            replace=True,
        )
        maryland_indices = jax.random.choice(
            key_maryland,
            len(self.maryland_masses),
            shape=(n_samples_per_group,),
            replace=True,
        )

        amsterdam_mass_samples: Float[Array, " n_amsterdam_samples"] = (
            self.amsterdam_masses[amsterdam_indices]
        )
        maryland_mass_samples: Float[Array, " n_maryland_samples"] = (
            self.maryland_masses[maryland_indices]
        )

        def process_sample_amsterdam(mass: Float, m_eos: Array, r_eos: Array, max_m_tov: Float, seg_min: Float, seg_max: Float) -> Float:
            """
            Process a single Amsterdam mass sample

            Parameters
            ----------
            mass : Float
                Sampled mass value
            m_eos : Array
                Segment mass array
            r_eos : Array
                Segment radius array
            max_m_tov : Float
                Maximum mass of the EOS
            seg_min : Float
                Minimum mass of the segment
            seg_max : Float
                Maximum mass of the segment

            Returns
            -------
            Float
                Log probability from Amsterdam KDE including penalty
            """
            # Interpolate radius from EOS
            radius = jnp.interp(mass, m_eos, r_eos)

            # Evaluate Amsterdam KDE at (mass, radius)
            mr_point = jnp.array([[mass], [radius]])  # Shape: (2, 1)
            logpdf = self.amsterdam_posterior.logpdf(mr_point)

            # Do not extrapolate: zero probability mask for extrapolated points
            in_segment = (mass >= seg_min) & (mass <= seg_max)
            logpdf = jnp.where(in_segment, logpdf, -jnp.inf)

            # Penalty for mass exceeding Mtov
            penalty = jnp.where(mass > max_m_tov, self.penalty_value, 0.0)

            return logpdf + penalty

        def process_sample_maryland(mass: Float, m_eos: Array, r_eos: Array, max_m_tov: Float, seg_min: Float, seg_max: Float) -> Float:
            """
            Process a single Maryland mass sample

            Parameters
            ----------
            mass : Float
                Sampled mass value
            m_eos : Array
                Segment mass array
            r_eos : Array
                Segment radius array
            max_m_tov : Float
                Maximum mass of the EOS
            seg_min : Float
                Minimum mass of the segment
            seg_max : Float
                Maximum mass of the segment

            Returns
            -------
            Float
                Log probability from Maryland KDE including penalty
            """
            # Interpolate radius from EOS
            radius = jnp.interp(mass, m_eos, r_eos)

            # Evaluate Maryland KDE at (mass, radius)
            mr_point = jnp.array([[mass], [radius]])  # Shape: (2, 1)
            logpdf = self.maryland_posterior.logpdf(mr_point)

            # Zero probability mask for extrapolated points
            in_segment = (mass >= seg_min) & (mass <= seg_max)
            logpdf = jnp.where(in_segment, logpdf, -jnp.inf)

            # Penalty for mass exceeding Mtov
            penalty = jnp.where(mass > max_m_tov, self.penalty_value, 0.0)

            return logpdf + penalty

        # Use jax.lax.map with batching for memory-efficient processing
        amsterdam_logprobs_1 = jax.lax.map(
            lambda m: process_sample_amsterdam(m, m_eos_1, r_eos_1, mtov, seg1_min, seg1_max),
            amsterdam_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        maryland_logprobs_1 = jax.lax.map(
            lambda m: process_sample_maryland(m, m_eos_1, r_eos_1, mtov, seg1_min, seg1_max),
            maryland_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        amsterdam_logprobs_2 = jax.lax.map(
            lambda m: process_sample_amsterdam(m, m_eos_2, r_eos_2, mtov, seg2_min, seg2_max),
            amsterdam_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        maryland_logprobs_2 = jax.lax.map(
            lambda m: process_sample_maryland(m, m_eos_2, r_eos_2, mtov, seg2_min, seg2_max),
            maryland_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        # Average over all samples for each group (log-mean = logsumexp - log(N))
        N_amsterdam = amsterdam_logprobs_1.shape[0]
        N_maryland = maryland_logprobs_1.shape[0]
        
        logL_amsterdam_1 = logsumexp(amsterdam_logprobs_1) - jnp.log(N_amsterdam)
        logL_maryland_1 = logsumexp(maryland_logprobs_1) - jnp.log(N_maryland)
        
        logL_amsterdam_2 = logsumexp(amsterdam_logprobs_2) - jnp.log(N_amsterdam)
        logL_maryland_2 = logsumexp(maryland_logprobs_2) - jnp.log(N_maryland)

        # Combine all likelihoods
        log_likelihood = logsumexp(
            jnp.array([logL_amsterdam_1, logL_maryland_1, logL_amsterdam_2, logL_maryland_2])
        ) - jnp.log(2.0)

        return log_likelihood
