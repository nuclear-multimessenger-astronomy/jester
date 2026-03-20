r"""Multimessenger event likelihood implementations"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.flows.flow import Flow
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class MultiMessengerLikelihood(LikelihoodBase):
    """
    Likelihood for a Gravitational wave event where there is also an independent posterior for the ejecta masses
    from an electromagnetic counterpart.

    The likelihood works by:

    1. Passing population parameters to a population model and generating (m1, m2) pairs.
    2. Interpolate Λ1, Λ2 from the candidate EOS at
       the generated mass points, evaluate GW flow log_prob on (m1, m2, Λ1_EOS, Λ2_EOS),
       apply penalties for masses exceeding Mtov, and average over all (m1, m2) pairs.
    3. Determine and log10_mej_dyn and log10_mej_wind from the EOS, m1, m2 using some fitting formulae,
       and evaluate EM flow log_prob one (log10_mej_dyn, log10_mej_wind).
    4. Add the log_probs and average over all generated (m1, m2) pairs.

    Parameters
    ----------
    event_name : str
        Name of the GW event (e.g., "GW170817")
    dir_gw : str
        Path to directory containing the trained normalizing flow for the GW posterior
    dir_em : str
        Path to directory containing the trained normalizing flow for the EM posterior
    penalty_value : float, optional
        Penalty value for samples where masses exceed Mtov (default: 0.0, i.e. no penalty)
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
    N_masses_batch_size : int
        Batch size for processing
    seed : int
        Random seed used for pre-sampling
    flow_gw : Flow
        Normalizing flow model for the GW event
    flow_em : Flow
        Normalizing flow model for the EM event

    Notes
    -----

    GPU parallelization via jax.lax.map means N=10,000 samples costs nearly
    the same as N=20, so use large N for near-integration accuracy.

    """

    event_name: str
    model_dir: str
    penalty_value: float
    N_masses_batch_size: int
    seed: int
    flow: Flow
    fixed_mass_samples: Float[Array, "n_samples 2"]

    def __init__(
        self,
        event_name: str,
        dir_gw: str,
        dir_em: str,
        zeta: float,
        penalty_value: float = 0.0,
        N_masses_batch_size: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.event_name = event_name
        self.dir_gw = dir_gw
        self.dir_em = dir_em
        self.penalty_value = penalty_value
        self.N_masses_batch_size = N_masses_batch_size
        self.seed = seed

        self.zeta = zeta

        # Load Flow model for this event
        logger.info(f"Loading NF for GW posterior {event_name} from {dir_gw}.")
        self.flow_gw = Flow.from_directory(dir_gw)
        logger.info(f"Loading NF for EM posterior {event_name} from {dir_em}.")
        self.flow_em = Flow.from_directory(dir_em)
        logger.info(f"Loaded NFs for {event_name}.")


    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'radii_EOS': Array of neutron star radii from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS
            - 'masses_1_pop': Array of heavier NS masses sampled from the population
            - 'masses_2_pop': Array of ligher NS sampled from the population
            - 'log10_mej_dyn': Array of dynamical ejecta from the NS masses
            - 'log10_mdisk': Array of disk masses from the NS masses
        Returns
        -------
        Float
            Log likelihood value for this GW event
        """
        # Extract EOS parameters (no _random_key needed!)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        log10_mej_wind = jnp.log10(self.zeta) +  params["log10_mdisk"]
        mm_samples = jnp.array([params["masses_1_pop"],
                                params["masses_2_pop"],
                                params["log10_mej_dyn"],
                                log10_mej_wind]).T
        


        def process_sample(sample: Float[Array, " 4"]) -> Float:
            """
            Process a single pre-sampled mass pair

            Note: jax.lax.map with batch_size applies function to individual
            elements. The batch_size parameter is for compilation optimization.

            Parameters
            ----------
            sample : Float[Array, " 4"]
                Pre-sampled mass pair [m1, m2]

            Returns
            -------
            Float
                Log probability including penalties for this sample
            """
            m1 = sample[0]
            m2 = sample[1]
            log10_mej_dyn = sample[2]
            log10_mej_wind = sample[3]


            # Interpolate lambdas from candidate EOS
            lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
            lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)

            # Evaluate GW log_prob on single sample
            gw_sample = jnp.array([m1, m2, lambda_1, lambda_2])
            logpdf_gw = self.flow_gw.log_prob(gw_sample)

            # Penalties for masses exceeding Mtov
            penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
            penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)
            
            # Evaluate log
            em_sample = jnp.array([log10_mej_dyn, log10_mej_wind])
            logpdf_em = self.flow_em.log_prob(em_sample)

            # Return log prob + penalties for this sample
            return logpdf_gw + penalty_m1 + penalty_m2 + logpdf_em
        
        # Use jax.lax.map with batching for memory-efficient processing
        # Process all pre-sampled mass pairs
        all_logprobs = jax.lax.map(
            process_sample, mm_samples, batch_size=self.N_masses_batch_size
        )

        # Take logsumexp over all pre-sampled mass pairs
        log_likelihood = logsumexp(all_logprobs) - jnp.log(mm_samples.shape[0])

        return log_likelihood