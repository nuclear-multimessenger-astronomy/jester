r"""Cosmolgy likelihood implementations"""

from typing import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.flows.flow import Flow, ConditionalFlow
from jesterTOV.utils import solar_mass_in_meter, lambda1_lambda2_to_lambda_tilde
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

def dynamic_mass_fitting_prompt_collapse(
        mass_1,
        mass_2,
        lambda_1,
        lambda_2,
        a=1.25e-4,
        b=9.82e-1,
        c=-2.44,
):
    """
    See https://arxiv.org/pdf/2411.02342, Eq. (9)
    """
    q = mass_2 / mass_1
    lambda_tilde = lambda1_lambda2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)

    mdyn = a*lambda_tilde*(q**(-1) -b) * jnp.exp(c/q) # this is always positive
    mdyn = jnp.maximum(1e-5, mdyn)

    return mdyn
    
def dynamic_mass_fitting(
    mass_1,
    mass_2,
    compactness_1,
    compactness_2,
    a=-9.3335,
    b=114.17,
    c=-337.56,
    n=1.5465,
):
    """
    See https://arxiv.org/pdf/2002.07728.pdf
    """
    mdyn = mass_1 * (
        a / compactness_1 + b * jnp.power(mass_2 / mass_1, n) + c * compactness_1
    )
    mdyn += mass_2 * (
        a / compactness_2 + b * jnp.power(mass_1 / mass_2, n) + c * compactness_2
    )
    mdyn *= 1e-3
    mdyn = jnp.maximum(1e-5, mdyn)
    return mdyn

def parameter_conversion(mass_1: Array, mass_2: Array, params: dict[str, Array]):

    # Extract EOS parameters 
    masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
    radii_EOS: Float[Array, " n_point"] = params["radii_EOS"]
    Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
    mtov: Float = jnp.max(masses_EOS)
    mtov: Float = jnp.max(masses_EOS)
    m_coll = params.get("k_coll", 1.3) * mtov

    # tidal deformability and radii
    lambda_1 = jnp.interp(mass_1, masses_EOS, Lambdas_EOS)
    lambda_2 = jnp.interp(mass_2, masses_EOS, Lambdas_EOS)
    radii_1 = jnp.interp(mass_1, masses_EOS, radii_EOS)
    radii_2 = jnp.interp(mass_2, masses_EOS, radii_EOS)
    compactness_1 = mass_1 / radii_1 * solar_mass_in_meter * 1e-3
    compactness_2 = mass_2 / radii_2 * solar_mass_in_meter * 1e-3

    # ejecta masses
    prompt_collapse = m_coll < (mass_1 + mass_2)
    mej_dyn = jnp.where(
        prompt_collapse,
        dynamic_mass_fitting_prompt_collapse(mass_1, mass_2, lambda_1, lambda_2), 
        dynamic_mass_fitting(mass_1, mass_2, compactness_1, compactness_2)
    )
    log10_mej_dyn = jnp.log10(mej_dyn)

    return lambda_1, lambda_2, log10_mej_dyn

class CosmoMultiMessengerLikelihood(LikelihoodBase):

    def __init__(
        self,
        event_name: str,
        dir_gw: str,
        dir_gw_cond: str,
        dir_em: str,
        population_logpdf: Callable,
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
        self.cflow_gw = ConditionalFlow.from_directory(dir_gw_cond)
        logger.info(f"Loading NF for EM posterior {event_name} from {dir_em}.")
        self.flow_em = Flow.from_directory(dir_em)
        logger.info(f"Loaded NFs for GW and EM posterior.")

        self.N_eval =  N_eval
        self.population_logpdf = jax.jit(population_logpdf)

        # Set up prior subtraction 
        if logprior_gw is None:
            logprior_gw = lambda x: 0
        self.logprior_gw = jax.jit(logprior_gw)

        if logprior_em is None:
            logprior_em = lambda x: 0
        self.logprior_em = jax.jit(logprior_em)

        # setup redshift array
        key, subkey = jax.random.split(key)
        self.redshift_arr = jax.random.normal(subkey, shape=(N_eval,)) * redshift_sigma + redshift_mean

        self.key = key


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
            - 'dL_fn_redshift_arr': x-array of redshift for the luminosity distance function
            - 'dL_fn_distance_arr': y-array of distance for the luminosity distance function
            - population parameters
        Returns
        -------
        Float
            Log likelihood value for this GW event
        """

        # Sample masses and inclinations from the posterior
        mass1_det, mass2_det, cos_theta_jn = self.flow_gw.sample(self.key, (self.N_eval,)).T

        # calculate population probability
        mass_1, mass_2 = mass1_det / (1 + self.redshift_arr), mass_2 = mass2_det / (1 + self.redshift_arr)
        all_logprobs = self.population_logpdf(mass_1, mass_2, params)

        # convert to EM and GW posterior parameters
        lambda_1, lambda_2, log10_mej_dyn = parameter_conversion(mass_1, mass_2, params)

        # get luminosity distance
        luminosity_distance_arr = jnp.interp(self.redshift_arr, params["dL_fn_redshift_arr"], params["dL_fn_distance_arr"])

        samples = jnp.array([
            mass1_det,
            mass2_det,
            lambda_1,
            lambda_2,
            log10_mej_dyn,
            luminosity_distance_arr,
            cos_theta_jn
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
            m1_det= sample[0]
            m2_det = sample[1]
            lambda_1 = sample[2]
            lambda_2 = sample[3]
            log10_mej_dyn = sample[4]
            luminosity_distance = sample[5]
            cos_theta_jn = sample[6]

            # Evaluate GW log_posterior on single sample
            cond_sample = jnp.array([m1_det, m2_det, cos_theta_jn])
            eval_sample = jnp.array([lambda_1, lambda_2, luminosity_distance])
            logpdf_gw = self.cflow_gw.log_prob(eval_sample, cond_sample)

            # subtract the prior
            gw_sample = jnp.array([m1_det, m2_det, lambda_1, lambda_2, luminosity_distance, cos_theta_jn])
            logprior_gwvalue = self.logprior_gw(gw_sample)
            logpdf_gw -= logprior_gwvalue
            
            # Evaluate log
            em_sample = jnp.array([log10_mej_dyn, luminosity_distance, cos_theta_jn])
            logpdf_em = self.flow_em.log_prob(em_sample)

            #subtract the prior
            logprior_emvalue = self.logprior_em(em_sample)
            logpdf_em -= logprior_emvalue

            # Return log prob + penalties for this sample
            return logpdf_gw + logpdf_em
        
        # Use jax.lax.map with batching for memory-efficient processing
        # Process all pre-sampled mass pairs
        all_logprobs += jax.lax.map(
            process_sample, samples, batch_size=self.N_masses_batch_size
        )

        # Take logsumexp over all pre-sampled mass pairs
        log_likelihood = logsumexp(all_logprobs) - jnp.log(all_logprobs.shape[0])

        return log_likelihood