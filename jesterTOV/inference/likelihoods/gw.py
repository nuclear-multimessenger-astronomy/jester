# FIXME: this needs to be updated to the rest of the codebase, this is just a copy of an old file

import numpy as np
import jax
import jax.numpy as jnp
import os
import json
import arviz

from jimgw.base import LikelihoodBase
from jimgw.prior import Prior
from jimgw.transforms import NtoMTransform

import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.distributions import Normal, Transformed

from joseTOV.eos import MetaModel_with_CSE_EOS_model, Crust_with_CSE_EOS_model, construct_family
import joseTOV.utils as jose_utils

class GWlikelihood_with_masses(LikelihoodBase):

    def __init__(self,
                 eos: str,
                 ifo_network: str,
                 suffix: str,
                 id: str,
                 transform: MicroToMacroTransform = None,
                 very_negative_value: float = -99999.0,
                 N_samples_masses: int = 2_000,
                 N_masses_evaluation: int = 1,
                 nf_prior: Transformed = None,
                 hdi_prob: float = 0.90):
        
        self.eos = eos
        self.ifo_network = ifo_network
        self.suffix = suffix
        self.id = id
        if len(self.suffix) > 0:
            self.name = f"{self.eos}_{self.ifo_network}_{self.id}_{self.suffix}"
            saved_location = f"models/{self.eos}/{self.ifo_network}/{self.id}_{self.suffix}"
        else:
            self.name = f"{self.eos}_{self.ifo_network}_{self.id}"
            saved_location = f"models/{self.eos}/{self.ifo_network}/{self.id}"
        self.transform = transform
        self.very_negative_value = very_negative_value
        self.N_masses_evaluation = N_masses_evaluation
        self.nf_prior = nf_prior
        
        # Locate the file
        print(f"Loading the trained NF model from: {saved_location}")
        nf_file = os.path.join(NF_PATH, saved_location + ".eqx")
        nf_kwargs_file = os.path.join(NF_PATH, saved_location + "_kwargs.json")
        
        if not os.path.exists(nf_file):
            print(f"Tried looking for the NF architecture at path {nf_file}, but it doesn't exist!")
        
        # Load the kwargs used to train the NF to define the PyTree structure
        with open(nf_kwargs_file, "r") as f:
            nf_kwargs = json.load(f)
            
        like_flow = make_flow(jax.random.PRNGKey(0), nf_kwargs["nn_depth"], nf_kwargs["nn_block_dim"])
        
        # Load the normalizing flow
        loaded_model: Transformed = eqx.tree_deserialise_leaves(nf_file, like=like_flow)
        self.NS_posterior = loaded_model
        
        print(f"Loaded the NF for run {self.eos}_{self.id}")
        seed = np.random.randint(0, 100000)
        key = jax.random.key(seed)
        key, subkey = jax.random.split(key)
        
        # Generate some samples from the NS posterior to know the mass range
        nf_samples = self.NS_posterior.sample(subkey, (N_samples_masses,))
        
        # Use it to get the range of m1 and m2
        m1 = nf_samples[:, 0]
        m2 = nf_samples[:, 1]
        
        # Instead, we use the 99% credible interval:
        self.m1_min, self.m1_max = arviz.hdi(np.array(m1), hdi_prob=hdi_prob)
        self.m2_min, self.m2_max = arviz.hdi(np.array(m2), hdi_prob=hdi_prob)
        
        print(f"The range of m1 for {self.eos}_{self.id} is: {self.m1_min:.4f} to {self.m1_max:.4f}")
        print(f"The range of m2 for {self.eos}_{self.id} is: {self.m2_min:.4f} to {self.m2_max:.4f}")
        

    def evaluate(self, params: dict[str, float], data: dict) -> float:
    
        # Generate some samples from the NS posterior to know the mass range
        sampled_key = params["key"].astype("int64")
        key = jax.random.key(sampled_key)
        masses_EOS, Lambdas_EOS = params['masses_EOS'], params['Lambdas_EOS']
        mtov = jnp.max(masses_EOS)
        
        ### Old method -- using large array of samples at once. This works fine but is restricted in memory quite easily
        nf_samples = self.NS_posterior.sample(key, (self.N_masses_evaluation,))
        # Use the NF to sample the masses, then we discard Lambdas and instead infer them from the sampled EOS
        m1 = nf_samples[:, 0].flatten()
        m2 = nf_samples[:, 1].flatten()
        
        penalty_mass1_mtov = jnp.where(m1 > mtov, self.very_negative_value, 0.0).at[0].get()
        penalty_mass2_mtov = jnp.where(m2 > mtov, self.very_negative_value, 0.0).at[0].get()
        
        # Lambdas: interpolate to get the values
        lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right = 1.0)
        lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right = 1.0)
        
        # Make a 4D array of the m1, m2, and lambda values and evalaute NF log prob on it
        ml_grid = jnp.array([m1, m2, lambda_1, lambda_2]).T
        
        logpdf_NS = self.NS_posterior.log_prob(ml_grid)
        logpdf_NS = jnp.mean(logpdf_NS)
        
        # Evaluate the log prior so we can subtract it (importance sampling strategy)
        logpdf_prior = self.nf_prior.log_prob(ml_grid)
        logpdf_prior = jnp.mean(logpdf_prior)
        
        log_likelihood = logpdf_NS - logpdf_prior + penalty_mass1_mtov + penalty_mass2_mtov
        
        return log_likelihood