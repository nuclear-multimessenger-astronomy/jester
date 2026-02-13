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
from typing import Optional

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.logging_config import get_logger

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
    amsterdam_model_dir : Optional[str]
        Path to directory containing Amsterdam flow model
        (flow_weights.eqx, metadata.json, flow_kwargs.json).
        If None, uses preset model path.
    maryland_model_dir : Optional[str]
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
        amsterdam_model_dir: Optional[str] = None,
        maryland_model_dir: Optional[str] = None,
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

        # Determine model paths (use presets if not provided)
        if amsterdam_model_dir is None:
            amsterdam_model_dir = self._get_preset_model_path(psr_name, "amsterdam")
            logger.warning(
                f"No amsterdam_model_dir provided for {psr_name}, "
                f"using preset: {amsterdam_model_dir}"
            )

        if maryland_model_dir is None:
            maryland_model_dir = self._get_preset_model_path(psr_name, "maryland")
            logger.warning(
                f"No maryland_model_dir provided for {psr_name}, "
                f"using preset: {maryland_model_dir}"
            )

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
        self.amsterdam_fixed_mass_samples = amsterdam_samples[:, :1]  # Shape: (N, 1)
        self.maryland_fixed_mass_samples = maryland_samples[:, :1]  # Shape: (N, 1)

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

        def process_sample_amsterdam(mass: Float[Array, " 1"]) -> Float:
            """
            Process a single mass sample

            Parameters
            ----------
            mass : Float[Array, " 1"]
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
            logpdf = self.amsterdam_flow.log_prob(mr_point)

            # Penalty for mass exceeding Mtov
            penalty = jnp.where(mass > mtov, self.penalty_value, 0.0)

            return logpdf + penalty

        def process_sample_maryland(mass: Float[Array, " 1"]) -> Float:
            """
            Process a single Maryland mass sample

            Parameters
            ----------
            mass : Float[Array, " 1"]
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
            logpdf = self.maryland_flow.log_prob(mr_point)

            # Penalty for mass exceeding Mtov
            penalty = jnp.where(mass > mtov, self.penalty_value, 0.0)

            return logpdf + penalty

        # Use jax.lax.map with batching for memory-efficient processing
        amsterdam_logprobs = jax.lax.map(
            process_sample_amsterdam,
            self.amsterdam_fixed_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        maryland_logprobs = jax.lax.map(
            process_sample_maryland,
            self.maryland_fixed_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        # Average over all samples for each group
        logL_amsterdam = logsumexp(amsterdam_logprobs)
        logL_maryland = logsumexp(maryland_logprobs)

        # Average the two groups (equal weights)
        log_likelihood = logsumexp(jnp.array([logL_amsterdam, logL_maryland]))

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
