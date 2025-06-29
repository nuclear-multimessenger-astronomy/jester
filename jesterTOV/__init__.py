from jax import config

config.update("jax_enable_x64", True)

# Import main modules
from . import eos, tov, ptov, stt_tov, utils
