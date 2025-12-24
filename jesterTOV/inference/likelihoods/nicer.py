"""
NICER X-ray timing likelihood implementations

TODO: Generalize to e.g. only one group, weights between different hotspot models,...
"""

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import gaussian_kde
from jaxtyping import Float

from jesterTOV.inference.base import LikelihoodBase


class NICERLikelihood(LikelihoodBase):
    """
    NICER likelihood marginalizing over mass using M-R interpolation

    This likelihood loads posterior samples from Amsterdam and Maryland groups,
    constructs KDEs, and marginalizes over mass to compute the likelihood.

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
        Number of mass points for marginalization grid (default: 100)
        Mass range is automatically determined from EOS (1.0 to max(masses_EOS))
    N_masses_batch_size : int, optional
        Batch size for processing mass grid points (default: 20)
    """

    def __init__(
        self,
        psr_name: str,
        amsterdam_samples_file: str,
        maryland_samples_file: str,
        N_masses_evaluation: int = 100,
        N_masses_batch_size: int = 20,
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
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'radii_EOS': Array of neutron star radii from EOS
        data : dict
            Not used (data encapsulated in likelihood object)

        Returns
        -------
        float
            Log likelihood value for this NICER observation
        """
        import jax

        masses_EOS, radii_EOS = params["masses_EOS"], params["radii_EOS"]

        # Create mass grid and interpolate radii from EOS
        m = jnp.linspace(1.0, jnp.max(masses_EOS), self.N_masses_evaluation)
        r = jnp.interp(m, masses_EOS, radii_EOS)

        # Stack into (N, 2) array for batch processing
        mr_points = jnp.stack([m, r], axis=1)

        def process_point(mr_point):
            """
            Process a single (mass, radius) point

            Note: jax.lax.map with batch_size still applies the function to individual
            elements, not batches. The batch_size parameter is for compilation optimization.

            Parameters
            ----------
            mr_point : array, shape (2,)
                Single point with [mass, radius]

            Returns
            -------
            array, shape (2,)
                [log_prob_maryland, log_prob_amsterdam]
            """
            # Reshape to (2, 1) for KDE evaluation
            mr_grid = mr_point.reshape(2, 1)

            # Evaluate KDE at this point
            logy_maryland = self.maryland_posterior.logpdf(mr_grid)
            logy_amsterdam = self.amsterdam_posterior.logpdf(mr_grid)

            return jnp.array([logy_maryland, logy_amsterdam])

        # Use jax.lax.map with batching for memory-efficient processing
        # batch_size helps with compilation memory, not runtime batching
        all_logprobs = jax.lax.map(
            process_point,
            mr_points,
            batch_size=self.N_masses_batch_size
        )

        # Extract Maryland and Amsterdam log probabilities
        logy_maryland = all_logprobs[:, 0]
        logy_amsterdam = all_logprobs[:, 1]

        # Marginalize over mass using logsumexp
        logL_maryland = logsumexp(logy_maryland) - jnp.log(len(logy_maryland))
        logL_amsterdam = logsumexp(logy_amsterdam) - jnp.log(len(logy_amsterdam))

        # Average the two groups (equal weights) using logsumexp for numerical stability
        # log(0.5 * (L_maryland + L_amsterdam)) = logsumexp([logL_maryland, logL_amsterdam]) - log(2)
        logL_array = jnp.array([logL_maryland, logL_amsterdam])
        log_likelihood = logsumexp(logL_array) - jnp.log(2)

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
