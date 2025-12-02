import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float

import copy
from jimgw.transforms import NtoMTransform
from jesterTOV.eos import MetaModel_with_CSE_EOS_model, MetaModel_EOS_model, construct_family

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
        super().__init__(name_mapping) # TODO: , keep_names = keep_names
    
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
        
        # TODO: check if needed or not
        self.fixed_params = {}
        
        # # Remove those NEPs from the fixed values that we sample over
        # if fixed_params is None:
        #     fixed_params = copy.deepcopy(NEP_CONSTANTS_DICT)
        
        # self.fixed_params = fixed_params 
        # for name in self.name_mapping[0]:
        #     if name in list(self.fixed_params.keys()):
        #         self.fixed_params.pop(name)
                
        # print("Fixed params loaded inside the MicroToMacroTransform:")
        # for key, value in self.fixed_params.items():
        #     print(f"    {key}: {value}")
            
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
        super().__init__(name_mapping=name_mapping) # TODO: , keep_names = "all"
        
        self.transform_func = detector_frame_M_c_q_to_source_frame_m_1_m_2
        
class ChirpMassMassRatioToLambdas(NtoMTransform):
    
    def __init__(
        self,
        name_mapping,
    ):
        super().__init__(name_mapping=name_mapping) # TODO: , keep_names = "all"
        
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