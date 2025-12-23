"""Gravitational wave event likelihood implementations"""

import os
import jax
import jax.numpy as jnp
import numpy as np

from jesterTOV.inference.base import LikelihoodBase
from jesterTOV.inference.flows.train_flow import Flow

class GWlikelihood_with_masses(LikelihoodBase):
    """
    Gravitational wave likelihood using normalizing flow posteriors

    This likelihood evaluates the GW posterior by:
    1. Sampling masses (m1, m2) from the trained normalizing flow
    2. Interpolating tidal deformabilities (Λ1, Λ2) from the EOS
    3. Evaluating the NF log probability on (m1, m2, Λ1, Λ2)
    4. Importance sampling to remove the prior used during NF training

    Parameters
    ----------
    eos : str
        Equation of state name
    ifo_network : str
        Interferometer network (e.g., "HLV")
    suffix : str
        Additional identifier suffix
    id : str
        Event identifier (e.g., "GW170817")
    very_negative_value : float, optional
        Penalty value for invalid samples (default: -99999.0)
    N_samples_masses : int, optional
        Number of samples for determining mass range (default: 2000)
    N_masses_evaluation : int, optional
        Number of mass samples per likelihood evaluation (default: 1)
    nf_prior : Transformed, optional
        Prior distribution used during NF training (for importance sampling)
    hdi_prob : float, optional
        Probability level for credible intervals (default: 0.90)
    """

    def __init__(
        self,
        eos: str,
        ifo_network: str,
        suffix: str,
        id: str,
        very_negative_value: float = -99999.0,
        N_samples_masses: int = 2_000,
        N_masses_evaluation: int = 1,
        nf_prior = None,  # TODO: Add type hint when flowjax is imported
        hdi_prob: float = 0.90,
    ):
        super().__init__()
        self.eos = eos
        self.ifo_network = ifo_network
        self.suffix = suffix
        self.id = id
        if len(self.suffix) > 0:
            self.name = f"{self.eos}_{self.ifo_network}_{self.id}_{self.suffix}"
            saved_location = f"models/{self.eos}/{self.ifo_network}/{self.id}_{self.suffix}"
        else:
            self.name = f"{self.eos}_{self.ifo_network}_{self.id}"
            saved_location = f"models/{self.eos}/{self.ifo_network}/{self.id}"
        self.very_negative_value = very_negative_value
        self.N_masses_evaluation = N_masses_evaluation
        self.nf_prior = nf_prior

        # FIXME: NF_PATH should be configurable or passed via data loading system
        NF_PATH = os.environ.get("NF_PATH", "./models")

        # Construct model directory path
        model_dir = os.path.join(NF_PATH, saved_location)

        print(f"Loading the trained NF model from: {model_dir}")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"NF model directory not found: {model_dir}\n"
                f"Expected structure: {model_dir}/flow_weights.eqx, flow_kwargs.json, metadata.json"
            )

        # Load the normalizing flow using the Flow class
        # This handles all the architecture reconstruction and weight loading
        flow_wrapper = Flow.from_directory(model_dir)
        self.NS_posterior = flow_wrapper.flow  # Extract the underlying flowjax model
        self.flow_metadata = flow_wrapper.metadata
        self.flow_kwargs = flow_wrapper.flow_kwargs
        
        print(f"Loaded the NF for run {self.eos}_{self.id}")
        seed = np.random.randint(0, 100000)
        key = jax.random.key(seed)
        key, subkey = jax.random.split(key)
        
        # Generate some samples from the NS posterior to know the mass range
        nf_samples = self.NS_posterior.sample(subkey, (N_samples_masses,))
        
        # Use it to get the range of m1 and m2
        m1 = nf_samples[:, 0]
        m2 = nf_samples[:, 1]

        # Use credible interval based on percentiles
        lower_percentile = (1 - hdi_prob) / 2 * 100
        upper_percentile = (1 - (1 - hdi_prob) / 2) * 100
        self.m1_min, self.m1_max = np.percentile(np.array(m1), [lower_percentile, upper_percentile])
        self.m2_min, self.m2_max = np.percentile(np.array(m2), [lower_percentile, upper_percentile])
        
        print(f"The range of m1 for {self.eos}_{self.id} is: {self.m1_min:.4f} to {self.m1_max:.4f}")
        print(f"The range of m2 for {self.eos}_{self.id} is: {self.m2_min:.4f} to {self.m2_max:.4f}")
        

    def evaluate(self, params: dict[str, float], data: dict) -> float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, float]
            Must contain 'masses_EOS' and 'Lambdas_EOS' arrays from transform,
            plus 'key' for random sampling
        data : dict
            Not used (data encapsulated in likelihood object)

        Returns
        -------
        float
            Log likelihood value
        """
    
        # Generate some samples from the NS posterior to know the mass range
        sampled_key = params["key"].astype("int64")
        key = jax.random.key(sampled_key)
        masses_EOS, Lambdas_EOS = params['masses_EOS'], params['Lambdas_EOS']
        mtov = jnp.max(masses_EOS)
        
        ### Old method -- using large array of samples at once. This works fine but is restricted in memory quite easily
        nf_samples = self.NS_posterior.sample(key, (self.N_masses_evaluation,))
        # Use the NF to sample the masses, then we discard Lambdas and instead infer them from the sampled EOS
        m1 = nf_samples[:, 0].flatten()
        m2 = nf_samples[:, 1].flatten()
        
        penalty_mass1_mtov = jnp.where(m1 > mtov, self.very_negative_value, 0.0).at[0].get()
        penalty_mass2_mtov = jnp.where(m2 > mtov, self.very_negative_value, 0.0).at[0].get()
        
        # Lambdas: interpolate to get the values
        lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right = 1.0)
        lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right = 1.0)
        
        # Make a 4D array of the m1, m2, and lambda values and evalaute NF log prob on it
        ml_grid = jnp.array([m1, m2, lambda_1, lambda_2]).T
        
        logpdf_NS = self.NS_posterior.log_prob(ml_grid)
        logpdf_NS = jnp.mean(logpdf_NS)
        
        # Evaluate the log prior so we can subtract it (importance sampling strategy)
        logpdf_prior = self.nf_prior.log_prob(ml_grid)
        logpdf_prior = jnp.mean(logpdf_prior)
        
        log_likelihood = logpdf_NS - logpdf_prior + penalty_mass1_mtov + penalty_mass2_mtov
        
        return log_likelihood