import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float
from jax.scipy.stats import gaussian_kde
import pandas as pd
import copy

from jimgw.base import LikelihoodBase
from jimgw.transforms import NtoMTransform
from jimgw.prior import UniformPrior, CombinePrior

import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.distributions import Normal, Transformed

from jesterTOV.eos import MetaModel_with_CSE_EOS_model, MetaModel_EOS_model, construct_family
import jesterTOV.utils as jose_utils

#################
### CONSTANTS ###
#################

NEP_CONSTANTS_DICT = {
    # This is a set of MM parameters that gives a decent initial guess for Hauke's Set A maximum likelihood EOS
    "E_sym": 33.431808,
    "L_sym": 77.178344,
    "K_sym": -129.761344,
    "Q_sym": 0.0,
    "Z_sym": 0.0,
    
    "E_sat": -16.0,
    "K_sat": 285.527411,
    "Q_sat": 0.0,
    "Z_sat": 0.0,
    
    "nbreak": 0.153406,
    
    # FIXME: this has been changed now because of uniform [0, 1] sampling!
    # "n_CSE_0": 3 * 0.16,
    # "n_CSE_1": 4 * 0.16,
    # "n_CSE_2": 5 * 0.16,
    # "n_CSE_3": 6 * 0.16,
    # "n_CSE_4": 7 * 0.16,
    # "n_CSE_5": 8 * 0.16,
    # "n_CSE_6": 9 * 0.16,
    # "n_CSE_7": 10 * 0.16,
    
    "cs2_CSE_0": 0.5,
    "cs2_CSE_1": 0.7,
    "cs2_CSE_2": 0.5,
    "cs2_CSE_3": 0.4,
    "cs2_CSE_4": 0.8,
    "cs2_CSE_5": 0.6,
    "cs2_CSE_6": 0.9,
    "cs2_CSE_7": 0.8,
    
    # This is the final entry
    "cs2_CSE_8": 0.9,
}

def merge_dicts(dict1: dict, dict2: dict):
    """
    Merges 2 dicts, but if the key is already in dict1, it will not be overwritten by dict2.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary. Do not use its values if keys are in dict1
    """
    
    result = {}
    for key, value in dict1.items():
        result[key] = value
        
    for key, value in dict2.items():
        if key not in result.keys():
            result[key] = value
            
    return result

############
### DATA ###
############

PSR_PATHS_DICT = {"J0030": {"maryland": "/projects/prjs1678/paper_jose/src/paper_jose/inference/data/J0030/J0030_RM_maryland.txt",
                            "amsterdam": "/projects/prjs1678/paper_jose/src/paper_jose/inference/data/J0030/ST_PST__M_R.txt"},
                  "J0740": {"maryland": "/projects/prjs1678/paper_jose/src/paper_jose/inference/data/J0740/J0740_NICERXMM_full_mr.txt",
                            "amsterdam": "/projects/prjs1678/paper_jose/src/paper_jose/inference/data/J0740/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat"}}
SUPPORTED_PSR_NAMES = list(PSR_PATHS_DICT.keys()) # we do not include the most recent PSR for now


empty = {"maryland": {}, "amsterdam": {}}
data_samples_dict: dict[str, dict[str, pd.Series]] = {"J0030": copy.deepcopy(empty), "J0740": copy.deepcopy(empty)}
kde_dict: dict[str, dict[str, gaussian_kde]] = {"J0030": copy.deepcopy(empty), "J0740": copy.deepcopy(empty)}

### NICER pulsars

N_samples_KDE = 10_000
N_samples_plot = 10_000
for psr in ["J0030", "J0740"]:
    for group in ["amsterdam", "maryland"]:
        
        # Get the paths
        path = PSR_PATHS_DICT[psr][group]
        if group == "maryland":
            samples = pd.read_csv(path, sep=" ", names=["R", "M", "weight"] , skiprows = 6)
        else:
            if psr == "J0030":
                samples = pd.read_csv(path, sep=" ", names=["weight", "M", "R"])
            else:
                samples = pd.read_csv(path, sep=" ", names=["M", "R"])
                samples["weight"] = np.ones_like(samples["M"])
        
        if pd.isna(samples["weight"]).any():
            print("Warning: weights not properly specified, assuming constant weights instead.")
            samples["weight"] = np.ones_like(samples["weight"])
            
        # Get as samples and as KDE
        m, r, w = samples["M"].values, samples["R"].values, samples["weight"].values
        
        # Generate N_samples samples for the KDE:
        idx = np.random.choice(len(samples), size = N_samples_KDE)
        m, r, w = m[idx], r[idx], w[idx]
        
        # Generate the KDEs
        data_2d = jnp.array([m, r])
        posterior = gaussian_kde(data_2d, weights = w)

        # Append data samples and KDE for later on
        data_samples_dict[psr][group] = samples
        kde_dict[psr][group] = posterior
            

try:        
    prex_posterior = gaussian_kde(np.loadtxt("/projects/prjs1678/paper_jose/src/paper_jose/inference/data/PREX/PREX_samples.txt", skiprows = 1).T)
    crex_posterior = gaussian_kde(np.loadtxt("/projects/prjs1678/paper_jose/src/paper_jose/inference/data/CREX/CREX_samples.txt", skiprows = 1).T)

    kde_dict["PREX"] = prex_posterior
    kde_dict["CREX"] = crex_posterior
except Exception as e:
    print("Could not load PREX/CREX data, likely because the paths are not correct. Please check.")

##################
### TRANSFORMS ###
##################

class MicroToMacroTransform(NtoMTransform):
    
    def __init__(self,
                 name_mapping: tuple[list[str], list[str]],
                 keep_names: list[str] = None,
                 # metamodel kwargs:
                 ndat_metamodel: int = 100,
                 # CSE kwargs
                 nmax_nsat: float = 25,
                 nb_CSE: int = 8,
                 # TOV kwargs
                 min_nsat_TOV: float = 0.75,
                 ndat_TOV: int = 100,
                 ndat_CSE: int = 100,
                 nb_masses: int = 100,
                 fixed_params: dict[str, float] = None,
                 # NEW
                 crust_name: str = "DH",
                ):
    
        # By default, keep all names
        if keep_names is None:
            keep_names = name_mapping[0]
        super().__init__(name_mapping, keep_names=keep_names)
    
        # Save as attributes
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nmax = nmax_nsat * 0.16
        self.nb_CSE = nb_CSE
        self.min_nsat_TOV = min_nsat_TOV
        self.ndat_TOV = ndat_TOV
        self.ndat_CSE = ndat_CSE
        self.nb_masses = nb_masses
        
        if crust_name not in ["DH", "BPS", "DH_fixed"]:
            raise ValueError(f"crust_name must be either 'DH', 'DH_fixed' or 'BPS', got {crust_name} instead.")
        
        print(f"Crust name: {crust_name}")
        
        # Create the EOS object -- there are several choices for the parametrizations
        if nb_CSE > 0:
            eos = MetaModel_with_CSE_EOS_model(nmax_nsat=self.nmax_nsat,
                                               ndat_metamodel=self.ndat_metamodel,
                                               ndat_CSE=self.ndat_CSE,
                                               crust_name=crust_name
                    )
            self.transform_func = self.transform_func_MM_CSE
        else:
            print(f"WARNING: This is a metamodel run with no CSE parameters!")
            eos = MetaModel_EOS_model(nmax_nsat = self.nmax_nsat,
                                      ndat = self.ndat_metamodel)
        
            self.transform_func = self.transform_func_MM
        
        self.eos = eos
        
        # Remove those NEPs from the fixed values that we sample over
        if fixed_params is None:
            fixed_params = copy.deepcopy(NEP_CONSTANTS_DICT)
        
        self.fixed_params = fixed_params 
        for name in self.name_mapping[0]:
            if name in list(self.fixed_params.keys()):
                self.fixed_params.pop(name)
                
        print("Fixed params loaded inside the MicroToMacroTransform:")
        for key, value in self.fixed_params.items():
            print(f"    {key}: {value}")
            
        # Construct a lambda function for solving the TOV equations, fix the given parameters
        self.construct_family_lambda = lambda x: construct_family(x, ndat = self.ndat_TOV, min_nsat = self.min_nsat_TOV)
        
    def transform_func_MM(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP)
        
        # Limit cs2 so that it is causal
        idx = jnp.argmax(cs2 >= 1.0)
        final_n = ns.at[idx].get()
        first_n = ns.at[0].get()
        
        ns_interp = jnp.linspace(first_n, final_n, len(ns))
        ps_interp = jnp.interp(ns_interp, ns, ps)
        hs_interp = jnp.interp(ns_interp, ns, hs)
        es_interp = jnp.interp(ns_interp, ns, es)
        dloge_dlogps_interp = jnp.interp(ns_interp, ns, dloge_dlogps)
        cs2_interp = jnp.interp(ns_interp, ns, cs2)
        
        # Solve the TOV equations
        eos_tuple = (ns_interp, ps_interp, hs_interp, es_interp, dloge_dlogps_interp)
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {"logpc_EOS": logpc_EOS, "masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
                       "n": ns_interp, "p": ps_interp, "h": hs_interp, "e": es_interp, "dloge_dlogp": dloge_dlogps_interp, "cs2": cs2_interp}

        return return_dict

    def transform_func_MM_CSE(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        
        # Separate the MM and CSE parameters
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        NEP["nbreak"] = params["nbreak"]
        
        ngrids_u = jnp.array([params[f"n_CSE_{i}_u"] for i in range(self.nb_CSE)])
        ngrids_u = jnp.sort(ngrids_u)
        cs2grids = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE)])
        
        # From the "quantiles", i.e. the values between 0 and 1, convert between nbreak and nmax
        width = (self.nmax - params["nbreak"])
        ngrids = params["nbreak"] + ngrids_u * width
        
        # Append the final cs2 value, which is fixed at nmax 
        ngrids = jnp.append(ngrids, jnp.array([self.nmax]))
        cs2grids = jnp.append(cs2grids, jnp.array([params[f"cs2_CSE_{self.nb_CSE}"]]))
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP, ngrids, cs2grids)
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        
        # Solve the TOV equations
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {"logpc_EOS": logpc_EOS, "masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
                       "n": ns, "p": ps, "h": hs, "e": es, "dloge_dlogp": dloge_dlogps, "cs2": cs2}
        
        return return_dict
    
    def transform_func_MM_NN(self, params: dict[str, Float]) -> dict[str, Float]:
        
        # NOTE: I am trying to figure out how to do it but params must be NN params I guess
        # Separate the MM and CSE parameters
        params.update(self.fixed_params)
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        NEP["nbreak"] = params["nbreak"]
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP, params["nn_state"])
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        
        # Solve the TOV equations
        p_c_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {"masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS, "p_c_EOS": p_c_EOS,
                    "n": ns, "p": ps, "h": hs, "e": es, "dloge_dlogp": dloge_dlogps, "cs2": cs2}
        
        return return_dict
    
def detector_frame_M_c_q_to_source_frame_m_1_m_2(params: dict) -> dict:
    
    M_c, q, d_L = params['M_c'], params['q'], params['d_L']
    H0 = params.get('H0', 67.4) # (km/s) / Mpc
    c = params.get('c', 299_792.4580) # km / s
    
    # Calculate source frame chirp mass
    z = d_L * H0 * 1e3 / c
    M_c_source = M_c / (1.0 + z)

    # Get source frame mass_1 and mass_2
    M_source = M_c_source * (1.0 + q) ** 1.2 / q**0.6
    m_1_source = M_source / (1.0 + q)
    m_2_source = M_source * q / (1.0 + q)

    return {'m_1': m_1_source, 'm_2': m_2_source}

class ChirpMassMassRatioToSourceComponentMasses(NtoMTransform):
        
    def __init__(
        self,
    ):
        name_mapping = (["M_c", "q", "d_L"], ["m_1", "m_2"])
        super().__init__(name_mapping=name_mapping, keep_names = "all")
        
        self.transform_func = detector_frame_M_c_q_to_source_frame_m_1_m_2
        
class ChirpMassMassRatioToLambdas(NtoMTransform):
    
    def __init__(
        self,
        name_mapping,
    ):
        super().__init__(name_mapping=name_mapping, keep_names = "all")
        
        self.mass_transform = ChirpMassMassRatioToSourceComponentMasses()
        
    def transform_func(self, params: dict[str, Float]) -> dict[str, Float]:
        
        masses_EOS = params["masses_EOS"]
        Lambdas_EOS = params["Lambdas_EOS"]
        
        # Get masses
        m_params = self.mass_transform.forward(params)
        m_1, m_2 = m_params["m_1"], m_params["m_2"]
        
        # Interpolate to get Lambdas
        lambda_1_interp = jnp.interp(m_1, masses_EOS, Lambdas_EOS, right = -1.0)
        lambda_2_interp = jnp.interp(m_2, masses_EOS, Lambdas_EOS, right = -1.0)
        
        return {"lambda_1": lambda_1_interp, "lambda_2": lambda_2_interp}
        
        
###################
### LIKELIHOODS ###
###################

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
   
#############
### PRIOR ###
#############

NMAX_NSAT = 25
NMAX = NMAX_NSAT * 0.16
# N = 100
NB_CSE = 8

### NEP priors
K_sat_prior = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat_prior = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])

E_sym_prior = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym_prior = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
Q_sym_prior = UniformPrior(-800.0, 800.0, parameter_names=["Q_sym"])
Z_sym_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sym"])

prior_list = [
    E_sym_prior,
    L_sym_prior, 
    K_sym_prior,
    Q_sym_prior,
    Z_sym_prior,

    K_sat_prior,
    Q_sat_prior,
    Z_sat_prior,
]

### CSE priors
nbreak_prior = UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"])
prior_list.append(nbreak_prior)
for i in range(NB_CSE):
    prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"n_CSE_{i}_u"]))
    prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))

# Final point to end
prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))
prior = CombinePrior(prior_list)
sampled_param_names = prior.parameter_names
name_mapping = (sampled_param_names, ["logpc_EOS", "masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])