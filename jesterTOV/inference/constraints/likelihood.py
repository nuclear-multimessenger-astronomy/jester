"""
Likelihood functions for various observational constraints used in the inference.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float
from jax.scipy.stats import gaussian_kde

from jimgw.single_event.likelihood import LikelihoodBase
from jesterTOV.inference.transforms import MicroToMacroTransform
      
class NICERLikelihood(LikelihoodBase):
    
    def __init__(self,
                 psr_name: str,
                 transform: MicroToMacroTransform = None,
                 m_min: float = 1.0,
                 m_max: float = 2.5,
                 nb_masses: int = 100,
                 use_NF: bool = False,
                 nn_depth: int = 5,
                 nn_block_dim: int = 8,
                 ):
        
        self.psr_name = psr_name
        self.transform = transform
        self.counter = 0
        self.nb_masses = nb_masses
        self.m_min = m_min
        self.m_max = m_max
        self.nb_masses = nb_masses
        self.masses = jnp.linspace(m_min, m_max, nb_masses)
        self.dm = self.masses[1] - self.masses[0]
        self.use_NF = use_NF
        
        # Load the data
        if use_NF:
            # Define the PyTree structure for deserialization
            like_flow = block_neural_autoregressive_flow(
                key=jax.random.PRNGKey(0),
                base_dist=Normal(jnp.zeros(2)),
                nn_depth=nn_depth,
                nn_block_dim=nn_block_dim
            )
            
            # Locate the file
            nf_file_amsterdam = f"NF/NF_model_{psr_name}_amsterdam.eqx"
            nf_file_maryland = f"NF/NF_model_{psr_name}_maryland.eqx"
            
            loaded_model_amsterdam: Transformed = eqx.tree_deserialise_leaves(nf_file_amsterdam, like=like_flow)
            loaded_model_maryland: Transformed = eqx.tree_deserialise_leaves(nf_file_maryland, like=like_flow)
            
            self.amsterdam_posterior = loaded_model_amsterdam
            self.maryland_posterior = loaded_model_maryland
        else:
            self.amsterdam_posterior = kde_dict[psr_name]["amsterdam"]
            self.maryland_posterior = kde_dict[psr_name]["maryland"]
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        masses_EOS, radii_EOS = params["masses_EOS"], params["radii_EOS"]
        
        m = jnp.linspace(1.0, jnp.max(masses_EOS), self.nb_masses)
        r = jnp.interp(m, masses_EOS, radii_EOS)
        
        mr_grid = jnp.vstack([m, r])
        if self.use_NF:
            logy_maryland = self.maryland_posterior.log_prob(mr_grid.T)
            logy_amsterdam = self.amsterdam_posterior.log_prob(mr_grid.T)
        else:
            logy_maryland = self.maryland_posterior.logpdf(mr_grid)
            logy_amsterdam = self.amsterdam_posterior.logpdf(mr_grid)
        
        logL_maryland = logsumexp(logy_maryland) - jnp.log(len(logy_maryland))
        logL_amsterdam = logsumexp(logy_amsterdam) - jnp.log(len(logy_amsterdam))
        
        L_maryland = jnp.exp(logL_maryland)
        L_amsterdam = jnp.exp(logL_amsterdam)
        L = 1/2 * (L_maryland + L_amsterdam)
        log_likelihood = jnp.log(L)
        
        return log_likelihood
    

class NICERLikelihood_with_masses(LikelihoodBase):
    
    def __init__(self,
                 psr_name: str,
                 transform: MicroToMacroTransform = None):
        
        self.psr_name = psr_name
        self.transform = transform
        
        self.amsterdam_posterior = kde_dict[psr_name]["amsterdam"]
        self.maryland_posterior = kde_dict[psr_name]["maryland"]
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        masses_EOS, radii_EOS = params["masses_EOS"], params["radii_EOS"]
        mass = params[f"mass_{self.psr_name}"]
        radius = jnp.interp(mass, masses_EOS, radii_EOS, left=0, right=0)
        
        mr_grid = jnp.vstack([mass, radius])
        logL_maryland = self.maryland_posterior.logpdf(mr_grid)
        logL_amsterdam = self.amsterdam_posterior.logpdf(mr_grid)
        
        logL_array = jnp.array([logL_maryland, logL_amsterdam])
        log_likelihood = logsumexp(logL_array) - jnp.log(2)
        
        return log_likelihood
    
# FIXME: this should be changed to the one with the marginalized mass sampling thing idea
class GWlikelihood_with_masses(LikelihoodBase):

    def __init__(self,
                 GW_name: str,
                 nf_model_filename: str,
                 transform: MicroToMacroTransform = None,
                 very_negative_value: float = -9999999.0,
                 nn_depth: int = 5,
                 nn_block_dim: int = 8,
                 ):
        
        self.GW_name = GW_name
        self.nf_model_filename = nf_model_filename
        self.transform = transform
        self.counter = 0
        self.very_negative_value = very_negative_value
        
        # Define the PyTree structure for deserialization
        like_flow = block_neural_autoregressive_flow(
            key=jax.random.PRNGKey(0),
            base_dist=Normal(jnp.zeros(4)),
            nn_depth=nn_depth,
            nn_block_dim=nn_block_dim
        )
        
        # Load the normalizing flow
        loaded_model: Transformed = eqx.tree_deserialise_leaves(nf_model_filename, like=like_flow)
        self.NS_posterior = loaded_model
        

    def evaluate(self, params: dict[str, float], data: dict) -> float:
        
        m1, m2 = params[f"mass_1_{self.GW_name}"], params[f"mass_2_{self.GW_name}"]
        penalty_masses = jnp.where(m1 < m2, self.very_negative_value, 0.0)
        
        masses_EOS, Lambdas_EOS = params['masses_EOS'], params['Lambdas_EOS']
        mtov = jnp.max(masses_EOS)
        
        penalty_mass1_mtov = jnp.where(m1 > mtov, self.very_negative_value, 0.0)
        penalty_mass2_mtov = jnp.where(m2 > mtov, self.very_negative_value, 0.0)

        # Lambdas: interpolate to get the values
        lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right = 1.0)
        lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right = 1.0)

        # Make a 4D array of the m1, m2, and lambda values and evalaute NF log prob on it
        ml_grid = jnp.array([m1, m2, lambda_1, lambda_2])
        logpdf_NS = self.NS_posterior.log_prob(ml_grid)
        
        log_likelihood = logpdf_NS + penalty_masses + penalty_mass1_mtov + penalty_mass2_mtov
        
        return log_likelihood

    
class REXLikelihood(LikelihoodBase):
    
    def __init__(self,
                 experiment_name: str,
                 # likelihood calculation kwargs
                 nb_masses: int = 100):
        
        assert experiment_name in ["PREX", "CREX"], "Only PREX and CREX are supported as experiment name arguments."
        self.experiment_name = experiment_name
        self.counter = 0
        self.nb_masses = nb_masses
        
        # Load the data
        self.posterior: gaussian_kde = kde_dict[experiment_name]
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        log_likelihood_array = self.posterior.logpdf(jnp.array([params["E_sym"], params["L_sym"]]))
        log_likelihood = log_likelihood_array.at[0].get()
        return log_likelihood
    
class RadioTimingLikelihood(LikelihoodBase):
    
    def __init__(self,
                 psr_name: str,
                 mean: float, 
                 std: float,
                 nb_masses: int = 100,
                 transform: MicroToMacroTransform = None):
        
        self.psr_name = psr_name
        self.transform = transform
        self.nb_masses = nb_masses
        
        self.mean = mean
        self.std = std
        
    # def evaluate(self, params: dict[str, Float], data: dict) -> Float:
    #     masses_EOS = params["masses_EOS"]
    #     mtov = jnp.max(masses_EOS)
    #     log_likelihood = jax.scipy.stats.norm.logcdf(
    #         mtov, loc=self.mean, scale=self.std
    #     )
    #     log_likelihood -= jnp.log(mtov)

    #     return log_likelihood
    
    # TODO: this is the old method that was bugged -- remove later on
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # Log likelihood is a Gaussian with give mean and std, evalaute it on the masses:
        masses_EOS = params["masses_EOS"]
        mtov = jnp.max(masses_EOS)
        m = jnp.linspace(1.0, mtov, self.nb_masses)
        
        log_likelihood_array = -0.5 * (m - self.mean)**2 / self.std**2
        # Do integration with discrete sum
        log_likelihood = logsumexp(log_likelihood_array) - jnp.log(len(log_likelihood_array))
        log_likelihood -= mtov
        
        return log_likelihood
    

    
class ChiEFTLikelihood(LikelihoodBase):
    
    def __init__(self,
                 transform: MicroToMacroTransform = None,
                 nb_n: int = 100):
        
        self.transform = transform
        
        # Load the chi EFT data
        low_filename = "/projects/prjs1678/paper_jose/src/paper_jose/inference/data/chiEFT/low.dat"
        f = np.loadtxt(low_filename)
        n_low = jnp.array(f[:, 0]) / 0.16 # convert to nsat
        p_low = jnp.array(f[:, 1])
        # NOTE: this is not a spline but it is the best I can do -- does this matter? Need to check later on
        EFT_low = lambda x: jnp.interp(x, n_low, p_low)
        
        high_filename = "/projects/prjs1678/paper_jose/src/paper_jose/inference/data/chiEFT/high.dat"
        f = np.loadtxt(high_filename)
        n_high = jnp.array(f[:, 0]) / 0.16 # convert to nsat
        p_high = jnp.array(f[:, 1])
        
        EFT_high = lambda x: jnp.interp(x, n_high, p_high)
        
        self.n_low = n_low
        self.p_low = p_low
        self.EFT_low = EFT_low
        
        self.n_high = n_high
        self.p_high = p_high
        self.EFT_high = EFT_high
        
        self.nb_n = nb_n
        
        # TODO: remove once debugged
        # print(f"Init of chiEFT likelihood")
        # print("self.n_low range")
        # print(jnp.min(self.n_low))
        # print(jnp.max(self.n_low))
        
        # print("self.n_high")
        # print(jnp.min(self.n_high))
        # print(jnp.max(self.n_high))
        
        # print("self.p_low range")
        # print(jnp.min(self.p_low))
        # print(jnp.max(self.p_low))
        
        # print("self.p_high")
        # print(jnp.min(self.p_high))
        # print(jnp.max(self.p_high))
        
        
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # Get relevant parameters
        n, p = params["n"], params["p"]
        nbreak = params["nbreak"]
        
        # Convert to nsat for convenience
        nbreak = nbreak / 0.16
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        
        # TODO: remove once debugged
        # print("nbreak:")
        # print(nbreak)
        
        # print("n_range:")
        # print(jnp.min(n))
        # print(jnp.max(n))
        
        # print("p_range:")
        # print(jnp.min(p))
        # print(jnp.max(p))
        
        prefactor = 1 / (nbreak - 0.75 * 0.16)
        
        # Lower limit is at 0.12 fm-3
        this_n_array = jnp.linspace(0.75, nbreak, self.nb_n)
        dn = this_n_array.at[1].get() - this_n_array.at[0].get()
        low_p = self.EFT_low(this_n_array)
        high_p = self.EFT_high(this_n_array)
        
        # Evaluate the sampled p(n) at the given n
        sample_p = jnp.interp(this_n_array, n, p)
        
        # Compute f
        def f(sample_p, low_p, high_p):
            beta = 6/(high_p-low_p)
            return_value = (
                -beta * (sample_p - high_p) * jnp.heaviside(sample_p - high_p, 0) +
                -beta * (low_p - sample_p) * jnp.heaviside(low_p - sample_p, 0) +
                1 * jnp.heaviside(sample_p - low_p, 0) * jnp.heaviside(high_p - sample_p, 0) # FIXME: 0 or 1? Hauke has 1 but then low_p with the log
            )
            
            return return_value
            
        f_array = f(sample_p, low_p, high_p) # Well actually already log f
        
        log_likelihood = prefactor * jnp.sum(f_array) * dn
        
        return log_likelihood
        
    
class CombinedLikelihood(LikelihoodBase):
    
    def __init__(self,
                 likelihoods_list: list[LikelihoodBase],
                 transform: MicroToMacroTransform = None):
        
        # TODO: remove transform input?
        
        super().__init__()
        self.likelihoods_list = likelihoods_list
        self.transform = transform
        self.counter = 0
        
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        all_log_likelihoods = jnp.array([likelihood.evaluate(params, data) for likelihood in self.likelihoods_list])
        return jnp.sum(all_log_likelihoods)
    
class ZeroLikelihood(LikelihoodBase):
    def __init__(self,
                 transform: MicroToMacroTransform = None):
        
        # TODO: remove transform input?
        
        super().__init__()
        self.transform = transform
        self.counter = 0
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        return 0.0