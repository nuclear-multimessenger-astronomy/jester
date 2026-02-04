r"""
NICER X-ray timing likelihood implementations
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class NICERLikelihoodResampled(LikelihoodBase):
    """
    NICER likelihood using mass sampling and EOS interpolation (legacy version)

    This is the legacy version that resamples masses during each likelihood evaluation.
    For production use, prefer NICERLikelihood which pre-samples masses at initialization
    for deterministic, reproducible evaluations.

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
        Evaluate log likelihood for given EOS parameters

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

        def process_sample_amsterdam(mass: Float) -> Float:
            """
            Process a single Amsterdam mass sample

            Parameters
            ----------
            mass : Float
                Sampled mass value

            Returns
            -------
            Float
                Log probability from Amsterdam KDE including penalty
            """
            # Interpolate radius from EOS
            radius = jnp.interp(mass, masses_EOS, radii_EOS)

            # Evaluate Amsterdam KDE at (mass, radius)
            mr_point = jnp.array([[mass], [radius]])  # Shape: (2, 1)
            logpdf = self.amsterdam_posterior.logpdf(mr_point)

            # Penalty for mass exceeding Mtov
            penalty = jnp.where(mass > mtov, self.penalty_value, 0.0)

            return logpdf + penalty

        def process_sample_maryland(mass: Float) -> Float:
            """
            Process a single Maryland mass sample

            Parameters
            ----------
            mass : Float
                Sampled mass value

            Returns
            -------
            Float
                Log probability from Maryland KDE including penalty
            """
            # Interpolate radius from EOS
            radius = jnp.interp(mass, masses_EOS, radii_EOS)

            # Evaluate Maryland KDE at (mass, radius)
            mr_point = jnp.array([[mass], [radius]])  # Shape: (2, 1)
            logpdf = self.maryland_posterior.logpdf(mr_point)

            # Penalty for mass exceeding Mtov
            penalty = jnp.where(mass > mtov, self.penalty_value, 0.0)

            return logpdf + penalty

        # Use jax.lax.map with batching for memory-efficient processing
        amsterdam_logprobs = jax.lax.map(
            process_sample_amsterdam,
            amsterdam_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        maryland_logprobs = jax.lax.map(
            process_sample_maryland,
            maryland_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        # Average over all samples for each group
        logL_amsterdam = logsumexp(amsterdam_logprobs)
        logL_maryland = logsumexp(maryland_logprobs)

        # Average the two groups (equal weights)
        log_likelihood = logsumexp(jnp.array([logL_amsterdam, logL_maryland]))

        return log_likelihood


class NICERLikelihood(LikelihoodBase):
    """
    NICER likelihood using pre-sampled masses for deterministic evaluation

    This likelihood improves upon NICERLikelihoodResampled by pre-sampling mass values once at
    initialization, eliminating the need for the _random_key parameter and providing
    deterministic likelihood evaluations critical for sampler convergence.

    Key improvements over NICERLikelihoodResampled:
    1. Deterministic: Same EOS parameters → same likelihood value
    2. No _random_key hack: Uses fixed seed at initialization
    3. Scalable: Can use N=10,000+ samples efficiently on GPU
    4. Fair comparison: All EOS evaluated at identical mass points
    5. Better convergence: Smooth likelihood surface for MCMC/SMC

    The likelihood works by:
    1. Pre-sampling mass values from Amsterdam and Maryland posterior samples at initialization
    2. For each EOS evaluation: interpolate radius from the candidate EOS at the fixed mass points,
       evaluate KDE log_prob on (mass, radius_EOS), apply penalties for masses exceeding Mtov,
       and average over all pre-sampled mass values

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
        Number of mass samples to pre-sample (default: 2000)
        Large values recommended - GPU parallelization makes this cheap!
    N_masses_batch_size : int, optional
        Batch size for jax.lax.map processing (default: 1000)
    seed : int, optional
        Random seed for mass pre-sampling (default: 42)
        Fixed seed ensures reproducibility across runs

    Attributes
    ----------
    psr_name : str
        Pulsar name
    penalty_value : float
        Penalty value for samples where mass exceeds Mtov
    N_masses_evaluation : int
        Number of pre-sampled mass values
    N_masses_batch_size : int
        Batch size for processing
    seed : int
        Random seed used for pre-sampling
    amsterdam_posterior : gaussian_kde
        KDE of Amsterdam (mass, radius) posterior
    maryland_posterior : gaussian_kde
        KDE of Maryland (mass, radius) posterior
    fixed_amsterdam_masses : Float[Array, "n_samples"]
        Pre-sampled mass values from Amsterdam posterior
    fixed_maryland_masses : Float[Array, "n_samples"]
        Pre-sampled mass values from Maryland posterior

    Notes
    -----
    This class does NOT require _random_key in the parameter dictionary,
    unlike NICERLikelihoodResampled. The seed is only used once at initialization.

    GPU parallelization via jax.lax.map means N=10,000 samples costs nearly
    the same as N=20, so use large N for near-integration accuracy.

    Examples
    --------
    Configure in YAML::

        likelihoods:
          - type: "nicer"
            enabled: true
            parameters:
              pulsars:
                - name: "J0030"
              N_masses_evaluation: 2000  # Default value
              N_masses_batch_size: 1000
              seed: 42
    """

    psr_name: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    seed: int
    amsterdam_posterior: gaussian_kde
    maryland_posterior: gaussian_kde
    fixed_amsterdam_masses: Float[Array, " n_samples"]
    fixed_maryland_masses: Float[Array, " n_samples"]

    def __init__(
        self,
        psr_name: str,
        amsterdam_samples_file: str,
        maryland_samples_file: str,
        penalty_value: float = -99999.0,
        N_masses_evaluation: int = 2000,
        N_masses_batch_size: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.psr_name = psr_name
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size
        self.seed = seed

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

        # Stack into [mass, radius] arrays for KDE
        # Convert to JAX arrays for JAX KDE
        amsterdam_mr = jnp.vstack([amsterdam_mass, amsterdam_radius])
        maryland_mr = jnp.vstack([maryland_mass, maryland_radius])

        # Construct KDEs using JAX implementation
        logger.info(f"Constructing JAX KDEs for {psr_name}")
        self.amsterdam_posterior = gaussian_kde(amsterdam_mr)
        self.maryland_posterior = gaussian_kde(maryland_mr)
        logger.info(f"Loaded JAX KDEs for {psr_name}")

        # Pre-sample masses ONCE at initialization
        logger.info(
            f"Pre-sampling {N_masses_evaluation} mass values with seed={seed} for {psr_name}"
        )
        key = jax.random.key(seed)
        key_amsterdam, key_maryland = jax.random.split(key)

        # Each group gets half of N_masses_evaluation samples
        n_samples_per_group = N_masses_evaluation // 2

        # Sample indices from posterior samples
        amsterdam_indices = jax.random.choice(
            key_amsterdam,
            len(amsterdam_mass),
            shape=(n_samples_per_group,),
            replace=True,
        )
        maryland_indices = jax.random.choice(
            key_maryland,
            len(maryland_mass),
            shape=(n_samples_per_group,),
            replace=True,
        )

        # Store pre-sampled masses as JAX arrays
        amsterdam_mass_jax = jnp.array(amsterdam_mass)
        maryland_mass_jax = jnp.array(maryland_mass)

        self.fixed_amsterdam_masses = amsterdam_mass_jax[amsterdam_indices]
        self.fixed_maryland_masses = maryland_mass_jax[maryland_indices]

        logger.info(
            f"Pre-sampled mass range for Amsterdam: "
            f"[{jnp.min(self.fixed_amsterdam_masses):.3f}, "
            f"{jnp.max(self.fixed_amsterdam_masses):.3f}] Msun"
        )
        logger.info(
            f"Pre-sampled mass range for Maryland: "
            f"[{jnp.min(self.fixed_maryland_masses):.3f}, "
            f"{jnp.max(self.fixed_maryland_masses):.3f}] Msun"
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'radii_EOS': Array of neutron star radii from EOS

            Note: Does NOT require '_random_key' (unlike NICERLikelihoodResampled)

        Returns
        -------
        Float
            Log likelihood value for this NICER observation
        """
        # Extract EOS parameters (no _random_key needed!)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        radii_EOS: Float[Array, " n_points"] = params["radii_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        def process_sample_amsterdam(mass: Float) -> Float:
            """
            Process a single pre-sampled Amsterdam mass value

            Note: jax.lax.map with batch_size applies function to individual
            elements. The batch_size parameter is for compilation optimization.

            Parameters
            ----------
            mass : Float
                Pre-sampled mass value

            Returns
            -------
            Float
                Log probability from Amsterdam KDE including penalty
            """
            # Interpolate radius from candidate EOS
            radius = jnp.interp(mass, masses_EOS, radii_EOS)

            # Evaluate Amsterdam KDE at (mass, radius)
            mr_point = jnp.array([[mass], [radius]])  # Shape: (2, 1)
            logpdf = self.amsterdam_posterior.logpdf(mr_point)

            # Penalty for mass exceeding Mtov
            penalty = jnp.where(mass > mtov, self.penalty_value, 0.0)

            return logpdf + penalty

        def process_sample_maryland(mass: Float) -> Float:
            """
            Process a single pre-sampled Maryland mass value

            Note: jax.lax.map with batch_size applies function to individual
            elements. The batch_size parameter is for compilation optimization.

            Parameters
            ----------
            mass : Float
                Pre-sampled mass value

            Returns
            -------
            Float
                Log probability from Maryland KDE including penalty
            """
            # Interpolate radius from candidate EOS
            radius = jnp.interp(mass, masses_EOS, radii_EOS)

            # Evaluate Maryland KDE at (mass, radius)
            mr_point = jnp.array([[mass], [radius]])  # Shape: (2, 1)
            logpdf = self.maryland_posterior.logpdf(mr_point)

            # Penalty for mass exceeding Mtov
            penalty = jnp.where(mass > mtov, self.penalty_value, 0.0)

            return logpdf + penalty

        # Use jax.lax.map with batching for memory-efficient processing
        # Process all pre-sampled mass values
        amsterdam_logprobs = jax.lax.map(
            process_sample_amsterdam,
            self.fixed_amsterdam_masses,
            batch_size=self.N_masses_batch_size,
        )

        maryland_logprobs = jax.lax.map(
            process_sample_maryland,
            self.fixed_maryland_masses,
            batch_size=self.N_masses_batch_size,
        )

        # Take logsumexp over all pre-sampled mass values for each group
        logL_amsterdam = logsumexp(amsterdam_logprobs)
        logL_maryland = logsumexp(maryland_logprobs)

        # Average the two groups (equal weights) using logsumexp
        log_likelihood = logsumexp(jnp.array([logL_amsterdam, logL_maryland]))

        return log_likelihood


# TODO: will implement or remove later
# class NICERLikelihood_with_masses(LikelihoodBase):
#     """
#     NICER likelihood with mass as a sampled parameter (no marginalization)

#     This likelihood loads posterior samples from Amsterdam and Maryland groups,
#     constructs KDEs, and evaluates the likelihood at a specific mass value.

#     Parameters
#     ----------
#     psr_name : str
#         Pulsar name (e.g., "J0030", "J0740")
#     amsterdam_samples_file : str
#         Path to npz file with Amsterdam group posterior samples
#         Expected to contain 'mass' (Msun) and 'radius' (km) arrays
#     maryland_samples_file : str
#         Path to npz file with Maryland group posterior samples
#         Expected to contain 'mass' (Msun) and 'radius' (km) arrays
#     """

#     def __init__(
#         self,
#         psr_name: str,
#         amsterdam_samples_file: str,
#         maryland_samples_file: str,
#     ):
#         super().__init__()
#         self.psr_name = psr_name

#         # Load samples from npz files
#         print(f"Loading Amsterdam samples for {psr_name} from {amsterdam_samples_file}")
#         amsterdam_data = np.load(amsterdam_samples_file, allow_pickle=True)

#         print(f"Loading Maryland samples for {psr_name} from {maryland_samples_file}")
#         maryland_data = np.load(maryland_samples_file, allow_pickle=True)

#         # Extract mass and radius samples
#         amsterdam_mass = amsterdam_data['mass']
#         amsterdam_radius = amsterdam_data['radius']
#         maryland_mass = maryland_data['mass']
#         maryland_radius = maryland_data['radius']

#         # Stack into [mass, radius] arrays for KDE
#         # Convert to JAX arrays for JAX KDE
#         amsterdam_mr = jnp.vstack([amsterdam_mass, amsterdam_radius])
#         maryland_mr = jnp.vstack([maryland_mass, maryland_radius])

#         # Construct KDEs using JAX implementation
#         print(f"Constructing JAX KDEs for {psr_name} (with masses)")
#         self.amsterdam_posterior = gaussian_kde(amsterdam_mr)
#         self.maryland_posterior = gaussian_kde(maryland_mr)
#         print(f"Loaded JAX KDEs for {psr_name} (with masses)")

#     def evaluate(self, params: dict[str, Float], data: dict) -> Float:
#         """
#         Evaluate log likelihood for given EOS parameters and sampled mass

#         Parameters
#         ----------
#         params : dict[str, Float]
#             Must contain:
#             - 'masses_EOS': Array of neutron star masses from EOS
#             - 'radii_EOS': Array of neutron star radii from EOS
#             - f'mass_{self.psr_name}': Sampled mass for this pulsar
#         data : dict
#             Not used (data encapsulated in likelihood object)

#         Returns
#         -------
#         float
#             Log likelihood value for this NICER observation
#         """
#         masses_EOS, radii_EOS = params["masses_EOS"], params["radii_EOS"]
#         mass = params[f"mass_{self.psr_name}"]
#         radius = jnp.interp(mass, masses_EOS, radii_EOS, left=0, right=0)

#         # Evaluate KDE at specific (mass, radius) point
#         mr_grid = jnp.vstack([mass, radius])
#         logL_maryland = self.maryland_posterior.logpdf(mr_grid)
#         logL_amsterdam = self.amsterdam_posterior.logpdf(mr_grid)

#         # Average the two groups (equal weights) using logsumexp
#         logL_array = jnp.array([logL_maryland, logL_amsterdam])
#         log_likelihood = logsumexp(logL_array) - jnp.log(2)

#         return log_likelihood
