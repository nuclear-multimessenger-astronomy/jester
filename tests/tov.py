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
        f"{data_dir}/micro_eos.npz",
    )
    # create the eos object
    eos_input = eos.Interpolate_EOS_model(
        micro_eos["n"], micro_eos["p"], micro_eos["e"]
    )
    eos_tuple = (
        eos_input.n,
        eos_input.p,
        eos_input.h,
        eos_input.e,
        eos_input.dloge_dlogp,
    )
    # solve the TOV equation for M-R-Lambda curve
    logpcs, ms, rs, Lambdas = eos.construct_family(eos_tuple)
