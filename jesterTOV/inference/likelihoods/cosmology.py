r"""Cosmolgy likelihood implementations"""

from typing import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.flows.flow import Flow, ConditionalFlow
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

class CosmoMultiMessengerLikelihood(LikelihoodBase):

    def __init__(
        self,
        event_name: str,
        dir_gw: str,
        dir_em: str,
        N_eval: int,
        redshift_mean: float,
        redshift_sigma: float,
        logprior_gw: Callable | None = None,
        logprior_em: Callable | None = None,
        penalty_value: float = 0.0,
        N_masses_batch_size: int = 1000,
        key: jax.random.PRNGKey = jax.random.key(0),
    ) -> None:
        

        super().__init__()
        self.event_name = event_name
        self.dir_gw = dir_gw
        self.dir_em = dir_em
        self.penalty_value = penalty_value
        self.N_masses_batch_size = N_masses_batch_size

        # Load GW posterior flow for this event
        logger.info(f"Loading NF for GW posterior {event_name} from {dir_gw}.")
        self.flow_gw = Flow.from_directory(dir_gw)
        logger.info(f"Loading NF for EM posterior {event_name} from {dir_em}.")
        self.flow_em = Flow.from_directory(dir_em)
        logger.info(f"Loaded NFs for GW and EM posterior.")

        # Set up prior subtraction 
        if logprior_gw is None:
            logprior_gw = lambda x: 0
        self.logprior_gw = jax.jit(logprior_gw)

        if logprior_em is None:
            logprior_em = lambda x: 0
        self.logprior_em = jax.jit(logprior_em)

        # setup inclination array
        self.cos_theta_jn_arr, key = self.setup_inclination_array(key, N_eval)

        # setup redshift array
        key, subkey = jax.random.split(key)
        self.redshift_arr = jax.random.normal(subkey, shape=(N_eval,)) *redshift_sigma + redshift_mean

    def setup_inclination_array(self, key, N_eval: int):
        """
        This methods provides the integration array for the cos_inclination parameter
        over which the posteriors are later marginalized.
        """
        
        key, subkey = jax.random.split(key)
        samples = self.flow_gw.sample(subkey, shape=(2000,))
        cos_theta_jn_max = samples[:, 5].max()
        cos_theta_jn_min = samples[:, 5].min()

        cos_theta_jn_max = jnp.minimum(1, cos_theta_jn_max+0.01)
        cos_theta_jn_min = jnp.maximum(-1, cos_theta_jn_min-0.01)

        key, subkey = jax.random.split(key)
        cos_theta_jn_arr = jax.random.uniform(
            subkey, 
            shape=(N_eval,),
            minval=cos_theta_jn_min, 
            maxval=cos_theta_jn_max
        )

        return cos_theta_jn_arr, key


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
            - 'dL_fn_redshift_arr': x-array of redshift for the luminosity distance function
            - 'dL_fn_distance_arr': y-array of distance for the luminosity distance function
        Returns
        -------
        Float
            Log likelihood value for this GW event
        """
        # Extract EOS parameters (no _random_key needed!)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)
        luminosity_distance_arr = jnp.interp(self.redshift_arr, params["dL_fn_redshift_arr"], params["dL_fn_distance_arr"])

        samples = jnp.array([
            params["masses_1_pop"],
            params["masses_2_pop"],
            params["log10_mej_dyn"],
            params["log10_mdisk"],
            self.redshift_arr,
            luminosity_distance_arr,
            self.cos_theta_jn_arr,
        ]).T
        

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
            log10_mdisk = sample[3]
            redshift = sample[4]
            luminosity_distance = sample[5]
            cos_theta_jn = sample[6]

            # Interpolate lambdas from candidate EOS
            lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
            lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)

            # Evaluate GW log_posterior on single sample
            gw_sample = jnp.array([m1, m2, lambda_1, lambda_2, luminosity_distance])
            logpdf_gw = self.flow_gw.log_prob(gw_sample)

            # subtract the prior
            gw_sample = jnp.array([m1, m2, lambda_1, lambda_2, luminosity_distance])
            logprior_gwvalue = self.logprior_gw(gw_sample)
            logpdf_gw -= logprior_gwvalue
            
            # Penalties for masses exceeding Mtov
            penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
            penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)
            
            # Evaluate log
            em_sample = jnp.array([log10_mej_dyn, luminosity_distance])
            logpdf_em = self.flow_em.log_prob(em_sample)

            #subtract the prior
            logprior_emvalue = self.logprior_em(em_sample)
            logpdf_em -= logprior_emvalue

            # Return log prob + penalties for this sample
            return logpdf_gw + penalty_m1 + penalty_m2 + logpdf_em
        
        # Use jax.lax.map with batching for memory-efficient processing
        # Process all pre-sampled mass pairs
        all_logprobs = jax.lax.map(
            process_sample, samples, batch_size=self.N_masses_batch_size
        )

        # Take logsumexp over all pre-sampled mass pairs
        log_likelihood = logsumexp(all_logprobs) - jnp.log(samples.shape[0])

        return log_likelihood