r"""Multimessenger event likelihood implementations"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp

from jesterTOV.utils import solar_mass_in_meter, lambda1_lambda2_to_lambda_tilde
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
    N_masses_evaluation : int, optional
        Number of mass samples to pre-sample (default: 2000)
        Large values recommended - GPU parallelization makes this cheap!
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
    N_masses_evaluation : int
        Number of pre-sampled mass pairs
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
    N_masses_evaluation: int
    N_masses_batch_size: int
    seed: int
    flow: Flow
    fixed_mass_samples: Float[Array, "n_samples 2"]

    def __init__(
        self,
        event_name: str,
        dir_gw: str,
        dir_em: str,
        population,
        pop_random_key: jax.random.PRNGKey,
        zeta: float,
        penalty_value: float = 0.0,
        N_masses_evaluation: int = 2000,
        N_masses_batch_size: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.event_name = event_name
        self.dir_gw = dir_gw
        self.dir_em = dir_em
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size
        self.seed = seed

        self.population = population
        self.pop_random_key = pop_random_key

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
            - **population parameters: The population parameters for the population model that are sampled over.
            - **fitting parameters: The parameters for the fitting relation that are sampled over.

        Returns
        -------
        Float
            Log likelihood value for this GW event
        """
        # Extract EOS parameters (no _random_key needed!)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        radii_EOS: Float[Array, " n_points"] = params["radii_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        mass_samples = self.population(self.pop_random_key, params, self.N_masses_evaluation)
        ejecta_samples = self.multimessenger_conversion(*mass_samples, params)

        mm_samples = jnp.vstack([mass_samples, ejecta_samples])

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
            return logpdf_gw + penalty_m1 + penalty_m2
        
        # Use jax.lax.map with batching for memory-efficient processing
        # Process all pre-sampled mass pairs

        all_logprobs = jax.lax.map(
            process_sample, mm_samples, batch_size=self.N_masses_batch_size
        )

        # Take logsumexp over all pre-sampled mass pairs
        log_likelihood = logsumexp(all_logprobs) - jnp.log(self.N_masses_evaluation)

        return log_likelihood
    
    def multimessenger_conversion(self, 
                          mass_1: Float,
                          mass_2: Float,
                          params: dict[str, Array]):
        
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        radii_EOS: Float[Array, " n_points"] = params["radii_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)
        r16 = jnp.interp(1.6, masses_EOS, radii_EOS)
        m_coll = params.get("k_coll", 1.3) * mtov
        prompt_collapse = m_coll < (mass_1 + mass_2)

        lambda_1 = jnp.interp(mass_1, masses_EOS, Lambdas_EOS)
        lambda_2 = jnp.interp(mass_2, masses_EOS, Lambdas_EOS)
        radii_1 = jnp.interp(mass_1, masses_EOS, radii_EOS)
        radii_2 = jnp.interp(mass_2, masses_EOS, radii_EOS)
        compactness_1 = mass_1 / radii_1 * solar_mass_in_meter * 1e-3
        compactness_2 = mass_2 / radii_2 * solar_mass_in_meter * 1e-3


        mej_dyn = jnp.where(prompt_collapse, 
                                  self.dynamic_mass_fitting_prompt_collapse(mass_1, mass_2, lambda_1, lambda_2), 
                                  self.dynamic_mass_fitting(mass_1, mass_2, compactness_1, compactness_2)
                                  )
        log10_mej_dyn = jnp.log10(mej_dyn)
       

        log10_mdisk = jnp.where(prompt_collapse,
                                self.log10_disk_mass_fitting_prompt_collapse(mass_1, mass_2, lambda_1, lambda_2),
                                self.log10_disk_mass_fitting(mass_1+mass_2, mass_1/mass_2, mtov, r16))
        
        #zeta = jnp.where(prompt_collapse,
        #                 jax.random.truncated_normal(jax.random.key(5810), lower=-2., upper=2., shape=(self.N_masses_evaluation, )) * 0.1 + 0.2,
        #                 jax.random.truncated_normal(jax.random.key(4187), lower=-2.5, upper=1.5, shape=(self.N_masses_evaluation,)) * 0.2 + 0.5)
        
        log10_mej_wind = jnp.log10(self.zeta) + log10_mdisk

        
        return jnp.stack([log10_mej_dyn, log10_mej_wind])

    def dynamic_mass_fitting_prompt_collapse(
            self,
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

        return mdyn
    
    def dynamic_mass_fitting(
        self,
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

        mdyn = jnp.maximum(0.0, mdyn)

        return mdyn
    

    def log10_disk_mass_fitting(
        self,
        total_mass,
        mass_ratio,
        MTOV,
        R16,
        a0=-1.725,
        delta_a=-2.337,
        b0=-0.564,
        delta_b=-0.437,
        c=0.958,
        d=0.057,
        beta=5.879,
        q_trans=0.886,
    ):
        """
        See https://arxiv.org/pdf/2205.08513 Eq. (22)
        The coefficients a0, delta_a etc. have been updated since then,
        the ones here are the correct ones.
        The threshold mass is from https://arxiv.org/pdf/1908.05442.pdf.
        """
        k = -3.606 * MTOV / R16 + 2.38
        threshold_mass = k * MTOV

        xi = 0.5 * jnp.tanh(beta * (mass_ratio - q_trans))

        a = a0 + delta_a * xi
        b = b0 + delta_b * xi

        log10_mdisk = a * (1 + b * jnp.tanh((c - total_mass / threshold_mass) / d))
        log10_mdisk = jnp.maximum(-3.0, log10_mdisk)

        return log10_mdisk
    


    def log10_disk_mass_fitting_prompt_collapse(
            self,
            mass_1,
            mass_2,
            lambda_1,
            lambda_2,
            a=7.70,
            b=-13.4,
            c=8.16e-3):
        """
        See https://arxiv.org/pdf/2411.02342, Eq. (11)
        Typo for b, b=-13.4 confirmed through author correspondence
        """
        q = mass_2 / mass_1
        lambda_tilde = lambda1_lambda2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
        log10_mdisk = a + b * q + c * lambda_tilde * q**2

        log10_mdisk = jnp.minimum(log10_mdisk, -1)

        return log10_mdisk