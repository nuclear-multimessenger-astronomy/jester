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

def convert_to_log10_mej_dyn(
        m_coll: Float,
        mass_1: Array,
        compactness_1: Array,
        lambda_1: Array,
        mass_2: Array,
        compactness_2: Array,
        lambda_2: Array
    ):

    # ejecta masses
    prompt_collapse = m_coll < (mass_1 + mass_2)
    mej_dyn = jnp.where(
        prompt_collapse,
        dynamic_mass_fitting_prompt_collapse(mass_1, mass_2, lambda_1, lambda_2), 
        dynamic_mass_fitting(mass_1, mass_2, compactness_1, compactness_2)
    )
    log10_mej_dyn = jnp.log10(mej_dyn)

    return log10_mej_dyn

class CosmoMultiMessengerLikelihood(LikelihoodBase):
    """
    Multi-messenger likelihood for a single BNS merger.

    This class evaluates the likelihood for a joint inference of the EOS, 
    mass distribution, and cosmology from an EM-bright BNS merger. 
    The basic mechanism is as follows:
    
    1. Sample (m1, m2, cos_theta_jn) from the GW posterior.
    2. Sample the redshift from the host galaxy redshift measurement.
    3. Interpolate tidal deformabilities (Λ1, Λ2) from the EOS.
    4. Convert the redshift to luminosity distance by using the sampled cosmology.
    5. Evaluate the conditional probabilities p(Λ1, Λ2, dL | m1, m2, cos_theta_jn, GW)
       through a conditional normalizing flow.
    6. Evaluate the mass distribution function on (m1, m2)
    7. Optional: If asked for, one can additional factor in a light curve posterior
       by evaluating the flow p(log10_mej_dyn, dL, cos_theta_jn | EM),
       similar to the MultiMessengerLikelihood. 
    8. Multiply and sum everything together to get the result.
    

    Parameters
    ----------
    event_name : str
        Name of the GW event (e.g., "GW170817")
    posterior_gw : str
        Path to an *.npz file containing the GW posterior samples. 
        Has to include entries for mass_1, mass_2, cos_theta_jn.
    conditional_flow_gw: str
        Path to a conditional normalizing flow to evaluate p(Λ1, Λ2, dL | m1, m2, cos_theta_jn, GW)
    mass_model_logpdf: Callable
        Binary mass function that takes as input jnp Arrays for mass_1, mass_2, and then a dict 
        with the mass model's parameters. Must be jittable.
    redshift_mean: float
        Measured redshift of the host galaxy.
    redshift_sigma: float
        Uncertainty of the redshift measurement.
    flow_em: str
        Path to a normalizing flow to evaluate p(log10_mej_dyn, dL, cos_theta_jn | EM). 
        If not set, will ignore the EM contribution to the likelihood.
    use_em: bool, optional
        Whether to use the EM contribution to the likelihood. If False,
        flow_em argument will be ignored. (default: True)
    logprior_gw: Callable | None:
        Prior used to obtain the GW posterior. 
        Should be a function that accepts a dict with the GW parameters as keys.
        If None, uniform priors will be assumed.
    logprior_em: Callable | None:
        Prior used to obtain the light curve posterior. 
        Should be a function that accepts a dict with the EM parameters as keys.
        If None, uniform priors will be assumed.
    penalty_value : float, optional
        Penalty value for samples where masses exceed Mtov (default: 0.0, i.e. no penalty)
    N_masses_evaluation : int, optional
        Number of samples to take from the GW posterior. 
        Will never be larger than the number of samples in posterior_gw. (default: 2000)
    N_masses_batch_size : int, optional
        Batch size for processing mass samples (default: 1000)
    key: jax.random.PRNGKey, optional
        Random key used to initialize the samples. (default: jax.random.key(0))

    """


    def __init__(
        self,
        event_name: str,
        posterior_gw: str,
        conditional_flow_gw: str,
        mass_model_logpdf: Callable,
        redshift_mean: float,
        redshift_sigma: float,
        flow_em: str | None = None,
        use_em: bool = True,
        logprior_gw: Callable | None = None,
        logprior_em: Callable | None = None,
        penalty_value: float = 0.0,
        N_masses_evaluation: int = 2000,
        N_masses_batch_size: int = 1000,
        key: jax.random.PRNGKey = jax.random.key(0),
    ) -> None:
        

        super().__init__()
        self.event_name = event_name

        self.N_masses_evaluation = N_masses_evaluation
        self.penalty_value = penalty_value
        self.N_masses_batch_size = N_masses_batch_size

        # setup redshift array
        key, subkey = jax.random.split(key)
        normal_arr = jax.random.normal(
            subkey, 
            shape=(self.N_masses_evaluation,)
        ) 
        self.redshift_arr = normal_arr * redshift_sigma + redshift_mean
        self.redshift_arr = jnp.clip(self.redshift_arr, 0, 100)

        # set up GW posterior
        key, subkey = jax.random.split(key)
        self.setup_gw_likelihood(
            posterior_gw,
            conditional_flow_gw,
            logprior_gw,
            subkey
        )

        # set up mass function
        self.mass_distribution_logpdf = jax.jit(mass_model_logpdf)

        # set up EM posterior if wanted
        self.use_em = use_em
        if self.use_em:
            self.setup_em_likelihood(
                flow_em,
                logprior_em,
            )


    def setup_gw_likelihood(
            self,
            posterior_gw: str,
            conditional_flow_gw: str,
            logprior_gw: Callable | None,
            key: jax.random.PRNGKey
        ) -> None:

        # set up integration arrays
        posterior_gw = jnp.load(posterior_gw)
        Nsamp = posterior_gw["mass_1"].shape[0]

        if Nsamp < self.N_masses_evaluation:
            logger.warning(f"For event {self.event_name}, N_masses_evaluation is larger than available GW posterior samples.",
                           "Setting N_masses_evaluation from {self.N_masses_evaluation} to {Nsamp}.")
            self.N_masses_evaluation = Nsamp
            self.redshift_arr = self.redshift_arr[:self.N_masses_evaluation]

        mask = jnp.zeros(Nsamp, dtype=bool).at[
            jax.random.choice(key, Nsamp, (self.N_masses_evaluation,), replace=False)
        ].set(True)

        self.mass_1_det = posterior_gw["mass_1"][mask]
        self.mass_2_det = posterior_gw["mass_2"][mask]
        self.mass_1_source = self.mass_1_det / (1 + self.redshift_arr)
        self.mass_2_source = self.mass_2_det / (1 + self.redshift_arr)
        self.cos_theta_jn = posterior_gw["cos_theta_jn"][mask]

        # Load GW conditional flow for this event 
        self.cflow_gw = ConditionalFlow.from_directory(conditional_flow_gw)


        # Set up prior subtraction 
        if logprior_gw is None:
            logprior_gw = lambda x: 0
        self.logprior_gw = jax.jit(logprior_gw)

        logger.info(f"Loaded GW likelihood for {self.event_name} from {posterior_gw} and {conditional_flow_gw}.")


    def setup_em_likelihood(
            self,
            flow_em: str,
            logprior_em: Callable | None,
    ) -> None:
        
        # Load EM flow for this event
        self.flow_em = Flow.from_directory(flow_em)

        # Set up prior subtraction
        if logprior_em is None:
            logprior_em = lambda x: 0
        self.logprior_em = jax.jit(logprior_em)

        logger.info(f"Loaded EM likelihood for {self.event_name} from {flow_em}.")


    def process_gw_likelihood(
            self,
            samples: dict[str, Array]
    ) -> Array:
        
        def process_single(sample):
            
            # Evaluate GW log_posterior on single sample
            cond_sample = jnp.array([sample["mass_1_source"], sample["mass_2_source"], sample["cos_theta_jn"]])
            eval_sample = jnp.array([sample["lambda_1"], sample["lambda_2"], sample["luminosity_distance"]])
            logpdf_gw = self.cflow_gw.log_prob(eval_sample, cond_sample)

            # subtract the prior
            logprior_gwvalue = self.logprior_gw(sample)
            logpdf_gw -= logprior_gwvalue

            return logpdf_gw
        
        log_probs_gw = jax.lax.map(
            process_single,
            samples,
            batch_size=self.N_masses_batch_size
        )

        return log_probs_gw

    def process_em_likelihood(
            self,
            samples: dict[str, Array]
        ) -> Array:


        def process_single(sample):
            # Evaluate log
            array_sample = jnp.array([sample["log10_mej_dyn"], sample["luminosity_distance"], sample["cos_theta_jn"]])
            logpdf_em = self.flow_em.log_prob(array_sample)

            #subtract the prior
            logprior_emvalue = self.logprior_em(sample)
            logpdf_em -= logprior_emvalue

            return logpdf_em
        
        log_probs_em = jax.lax.map(
            process_single,
            samples,
            batch_size=self.N_masses_batch_size
        )

        return log_probs_em

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS, mass distribution, and cosmology parameters

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
            Log likelihood value for this multi-messenger event
        """

        mtov = params["masses_EOS"].max()

        mass_distribution_logprobs = self.mass_distribution_logpdf(
            self.mass_1_source, 
            self.mass_2_source, 
            params
        )

        # get tidal deformabilities
        lambda_1 = jnp.interp(self.mass_1_source, params["masses_EOS"], params["Lambdas_EOS"])
        lambda_2 = jnp.interp(self.mass_2_source, params["masses_EOS"], params["Lambdas_EOS"])
        
        # get luminosity distance
        luminosity_distance = jnp.interp(self.redshift_arr, params["dL_fn_redshift_arr"], params["dL_fn_distance_arr"])

        # collect all in one dict
        samples = dict(
            mass_1_source=self.mass_1_source,
            mass_2_source=self.mass_2_source,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            luminosity_distance=luminosity_distance,
            cos_theta_jn=self.cos_theta_jn
        )

        
        # evaluate GW log probs
        logprobs_gw = self.process_gw_likelihood(samples)

        log_probs = mass_distribution_logprobs + logprobs_gw

        # evaluate EM log probs if wanted
        if self.use_em:
            m_coll = params["k_coll"] * mtov
            radii_1 = jnp.interp(self.mass_1_source, params["masses_EOS"], params["radii_EOS"])
            radii_2 = jnp.interp(self.mass_2_source, params["masses_EOS"], params["radii_EOS"])
            compactness_1 = self.mass_1_source / radii_1 * solar_mass_in_meter * 1e-3
            compactness_2 = self.mass_2_source / radii_2 * solar_mass_in_meter * 1e-3
            log10_mej_dyn = convert_to_log10_mej_dyn(
                m_coll,
                self.mass_1_source, compactness_1, lambda_1,
                self.mass_2_source, compactness_2, lambda_2
                )
            samples["log10_mej_dyn"] = log10_mej_dyn

            logprobs_em = self.process_em_likelihood(samples)
            log_probs += logprobs_em
    
        # Take logsumexp over all samples
        log_likelihood = logsumexp(log_probs) - jnp.log(log_probs.shape[0])

        return log_likelihood
