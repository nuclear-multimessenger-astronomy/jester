r"""Gravitational wave event likelihood implementations"""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.flows.flow import Flow
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


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


    def evaluate(self, params: dict[str, Float | Array], data: dict[str, Any]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - '_random_key': Random seed for mass sampling (cast to int64)
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS
        data : dict[str, Any]
            Not used (data encapsulated in likelihood object)

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
        all_nf_samples: Float[Array, "n_samples 2"] = self.flow.sample(key, (self.N_masses_evaluation,))

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
            process_sample,
            all_nf_samples,
            batch_size=self.N_masses_batch_size
        )

        # Average over all samples for this event
        log_likelihood = jnp.mean(all_logprobs)

        return log_likelihood