"""
NICER X-ray timing likelihood implementations
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
from jaxtyping import Float

from jesterTOV.inference.base.likelihood import LikelihoodBase


class NICERLikelihood(LikelihoodBase):
    """
    NICER likelihood using mass sampling and EOS interpolation
    
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
    N_masses_evaluation : int, optional
        Number of mass samples per likelihood evaluation (default: 20)
    N_masses_batch_size : int, optional
        Batch size for processing mass samples (default: 10)
    """

    def __init__(
        self,
        psr_name: str,
        amsterdam_samples_file: str,
        maryland_samples_file: str,
        N_masses_evaluation: int = 20,
        N_masses_batch_size: int = 10,
    ):
        super().__init__()
        self.psr_name = psr_name
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size

        # Load samples from npz files
        print(f"Loading Amsterdam samples for {psr_name} from {amsterdam_samples_file}")
        amsterdam_data = np.load(amsterdam_samples_file, allow_pickle=True)

        print(f"Loading Maryland samples for {psr_name} from {maryland_samples_file}")
        maryland_data = np.load(maryland_samples_file, allow_pickle=True)

        # Extract mass and radius samples
        # File format: mass (Msun), radius (km)
        amsterdam_mass = amsterdam_data['mass']
        amsterdam_radius = amsterdam_data['radius']
        maryland_mass = maryland_data['mass']
        maryland_radius = maryland_data['radius']

        # Store mass samples as JAX arrays for random sampling
        self.amsterdam_masses = jnp.array(amsterdam_mass)
        self.maryland_masses = jnp.array(maryland_mass)

        # Stack into [mass, radius] arrays for KDE
        # Convert to JAX arrays for JAX KDE
        amsterdam_mr = jnp.vstack([amsterdam_mass, amsterdam_radius])
        maryland_mr = jnp.vstack([maryland_mass, maryland_radius])

        # Construct KDEs using JAX implementation
        print(f"Constructing JAX KDEs for {psr_name}")
        self.amsterdam_posterior = gaussian_kde(amsterdam_mr)
        self.maryland_posterior = gaussian_kde(maryland_mr)
        print(f"Loaded JAX KDEs for {psr_name}")

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, Float]
            Must contain:
            - '_random_key': Random seed for mass sampling (cast to int64)
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'radii_EOS': Array of neutron star radii from EOS
        data : dict
            Not used (data encapsulated in likelihood object)

        Returns
        -------
        float
            Log likelihood value for this NICER observation
        """
        # Extract parameters
        sampled_key = params["_random_key"].astype("int64")
        key = jax.random.key(sampled_key)
        masses_EOS = params["masses_EOS"]
        radii_EOS = params["radii_EOS"]

        # Split key for Amsterdam and Maryland sampling
        key_amsterdam, key_maryland = jax.random.split(key)

        # Sample masses from the NICER posterior samples
        # Each group gets half of N_masses_evaluation samples
        n_samples_per_group = self.N_masses_evaluation // 2

        # Sample indices and get mass samples
        amsterdam_indices = jax.random.choice(
            key_amsterdam,
            len(self.amsterdam_masses),
            shape=(n_samples_per_group,),
            replace=True
        )
        maryland_indices = jax.random.choice(
            key_maryland,
            len(self.maryland_masses),
            shape=(n_samples_per_group,),
            replace=True
        )

        amsterdam_mass_samples = self.amsterdam_masses[amsterdam_indices]
        maryland_mass_samples = self.maryland_masses[maryland_indices]

        def process_sample_amsterdam(mass):
            """
            Process a single Amsterdam mass sample

            Parameters
            ----------
            mass : float
                Sampled mass value

            Returns
            -------
            float
                Log probability from Amsterdam KDE
            """
            # Interpolate radius from EOS
            radius = jnp.interp(mass, masses_EOS, radii_EOS)

            # Evaluate Amsterdam KDE at (mass, radius)
            mr_point = jnp.array([[mass], [radius]])  # Shape: (2, 1)
            logpdf = self.amsterdam_posterior.logpdf(mr_point)

            return logpdf

        def process_sample_maryland(mass):
            """
            Process a single Maryland mass sample

            Parameters
            ----------
            mass : float
                Sampled mass value

            Returns
            -------
            float
                Log probability from Maryland KDE
            """
            # Interpolate radius from EOS
            radius = jnp.interp(mass, masses_EOS, radii_EOS)

            # Evaluate Maryland KDE at (mass, radius)
            mr_point = jnp.array([[mass], [radius]])  # Shape: (2, 1)
            logpdf = self.maryland_posterior.logpdf(mr_point)

            return logpdf

        # Use jax.lax.map with batching for memory-efficient processing
        amsterdam_logprobs = jax.lax.map(
            process_sample_amsterdam,
            amsterdam_mass_samples,
            batch_size=self.N_masses_batch_size
        )

        maryland_logprobs = jax.lax.map(
            process_sample_maryland,
            maryland_mass_samples,
            batch_size=self.N_masses_batch_size
        )

        # Average over all samples for each group
        logL_amsterdam = jnp.mean(amsterdam_logprobs)
        logL_maryland = jnp.mean(maryland_logprobs)

        # Average the two groups (equal weights)
        log_likelihood = (logL_amsterdam + logL_maryland) / 2.0

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
