import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
import copy

PSR_PATHS_DICT = {"J0030": {"maryland": "/projects/prjs1678/paper_jose/src/paper_jose/inference/data/J0030/J0030_RM_maryland.txt",
                            "amsterdam": "/projects/prjs1678/paper_jose/src/paper_jose/inference/data/J0030/ST_PST__M_R.txt"},
                  "J0740": {"maryland": "/projects/prjs1678/paper_jose/src/paper_jose/inference/data/J0740/J0740_NICERXMM_full_mr.txt",
                            "amsterdam": "/projects/prjs1678/paper_jose/src/paper_jose/inference/data/J0740/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat"}}
SUPPORTED_PSR_NAMES = list(PSR_PATHS_DICT.keys()) # we do not include the most recent PSR for now

# TODO: move!

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
            