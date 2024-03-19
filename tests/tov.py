import os

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from joseTOV import eos

def test_tov():
    working_dir = os.path.dirname(__file__)
    data_dir = os.path.join(working_dir, "data")

    # load the eos files
    micro_eos = jnp.load(
        f'{data_dir}/micro_eos.npz',
    )
    # create the eos object
    eos_input = eos.Interpolate_EOS_model(
        micro_eos['n'],
        micro_eos['p'],
        micro_eos['e']
    )
    # solve the TOV equation for M-R-Lambda curve
    logpcs, ms, rs, Lambdas = eos.construct_family(eos_input)
