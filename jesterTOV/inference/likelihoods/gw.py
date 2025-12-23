"""Gravitational wave event likelihood implementations"""

import jax
import jax.numpy as jnp
import numpy as np

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.flows.train_flow import Flow


class GWLikelihood(LikelihoodBase):
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
    """

    def __init__(
        self,
        event_name: str,
        model_dir: str,
        penalty_value: float = -99999.0,
        N_masses_evaluation: int = 20,
        N_masses_batch_size: int = 10,
    ):
        super().__init__()
        self.event_name = event_name
        self.model_dir = model_dir
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size

        # Load Flow model for this event
        print(f"Loading NF model for {event_name} from {model_dir}")
        self.flow = Flow.from_directory(model_dir)
        print(f"Loaded NF model for {event_name}")


    def evaluate(self, params: dict[str, float], data: dict) -> float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, float]
            Must contain:
            - '_random_key': Random seed for mass sampling (cast to int64)
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS
        data : dict
            Not used (data encapsulated in likelihood object)

        Returns
        -------
        float
            Log likelihood value for this GW event
        """
        # Extract parameters
        sampled_key = params["_random_key"].astype("int64")
        key = jax.random.key(sampled_key)
        masses_EOS = params["masses_EOS"]
        Lambdas_EOS = params["Lambdas_EOS"]
        mtov = jnp.max(masses_EOS)

        # Split into batches
        n_full_batches = self.N_masses_evaluation // self.N_masses_batch_size
        remainder = self.N_masses_evaluation % self.N_masses_batch_size

        def process_batch(batch_key):
            """Process one batch of mass samples"""
            # Sample masses from flow (returns numpy array)
            nf_samples = self.flow.sample(batch_key, (self.N_masses_batch_size,))
            nf_samples = jnp.array(nf_samples)  # Convert to JAX array
            m1 = nf_samples[:, 0]
            m2 = nf_samples[:, 1]

            # Interpolate lambdas from EOS
            lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
            lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)

            # Evaluate log_prob on (m1, m2, lambda_1, lambda_2)
            ml_grid = jnp.stack([m1, m2, lambda_1, lambda_2], axis=-1)
            logpdf = self.flow.log_prob(np.array(ml_grid))
            logpdf = jnp.array(logpdf)  # Convert back to JAX array

            # Penalties for masses exceeding Mtov
            penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
            penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)

            # Return sum of log probabilities and penalties for this batch
            return jnp.sum(logpdf) + jnp.sum(penalty_m1) + jnp.sum(penalty_m2)

        # Process full batches using jax.lax.map for memory efficiency
        total_sum = 0.0
        if n_full_batches > 0:
            batch_keys = jax.random.split(key, n_full_batches)
            batch_sums = jax.lax.map(process_batch, batch_keys)
            total_sum = jnp.sum(batch_sums)

        # Handle remainder batch (if N_masses_evaluation not divisible by batch size)
        if remainder > 0:
            key, remainder_key = jax.random.split(key)
            nf_samples = self.flow.sample(remainder_key, (remainder,))
            nf_samples = jnp.array(nf_samples)
            m1 = nf_samples[:, 0]
            m2 = nf_samples[:, 1]

            lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
            lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)

            ml_grid = jnp.stack([m1, m2, lambda_1, lambda_2], axis=-1)
            logpdf = self.flow.log_prob(np.array(ml_grid))
            logpdf = jnp.array(logpdf)

            penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
            penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)

            total_sum += jnp.sum(logpdf) + jnp.sum(penalty_m1) + jnp.sum(penalty_m2)

        # Average over all samples for this event
        log_likelihood = total_sum / self.N_masses_evaluation

        return log_likelihood