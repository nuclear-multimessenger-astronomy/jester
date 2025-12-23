"""NICER X-ray timing likelihood implementations"""

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Float
from scipy.stats import gaussian_kde

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
    """

    def __init__(
        self,
        psr_name: str,
        amsterdam_samples_file: str,
        maryland_samples_file: str,
        N_masses_evaluation: int = 100,
    ):
        super().__init__()
        self.psr_name = psr_name
        self.N_masses_evaluation = N_masses_evaluation
        
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
        amsterdam_mr = np.vstack([amsterdam_mass, amsterdam_radius])
        maryland_mr = np.vstack([maryland_mass, maryland_radius])

        # Construct KDEs
        print(f"Constructing KDEs for {psr_name}")
        self.amsterdam_posterior = gaussian_kde(amsterdam_mr)
        self.maryland_posterior = gaussian_kde(maryland_mr)
        print(f"Loaded KDEs for {psr_name}")

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
        masses_EOS, radii_EOS = params["masses_EOS"], params["radii_EOS"]

        # Create mass grid and interpolate radii from EOS
        m = jnp.linspace(1.0, jnp.max(masses_EOS), self.N_masses_evaluation)
        r = jnp.interp(m, masses_EOS, radii_EOS)

        # Evaluate KDE log probability on M-R grid
        mr_grid = jnp.vstack([m, r])
        logy_maryland = self.maryland_posterior.logpdf(mr_grid)
        logy_amsterdam = self.amsterdam_posterior.logpdf(mr_grid)

        # Marginalize over mass using logsumexp
        logL_maryland = logsumexp(logy_maryland) - jnp.log(len(logy_maryland))
        logL_amsterdam = logsumexp(logy_amsterdam) - jnp.log(len(logy_amsterdam))

        # Average the two groups (equal weights)
        L_maryland = jnp.exp(logL_maryland)
        L_amsterdam = jnp.exp(logL_amsterdam)
        L = 0.5 * (L_maryland + L_amsterdam)
        log_likelihood = jnp.log(L)

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
#         amsterdam_mr = np.vstack([amsterdam_mass, amsterdam_radius])
#         maryland_mr = np.vstack([maryland_mass, maryland_radius])

#         # Construct KDEs
#         print(f"Constructing KDEs for {psr_name} (with masses)")
#         self.amsterdam_posterior = gaussian_kde(amsterdam_mr)
#         self.maryland_posterior = gaussian_kde(maryland_mr)
#         print(f"Loaded KDEs for {psr_name} (with masses)")

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
