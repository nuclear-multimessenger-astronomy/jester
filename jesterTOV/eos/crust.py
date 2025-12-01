r"""Neutron star crust models and data loading."""

import os
import jax.numpy as jnp
from jaxtyping import Array

# Get the path to the crust directory (relative to this module's parent)
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
CRUST_DIR = f"{DEFAULT_DIR}/crust"


def load_crust(name: str) -> tuple[Array, Array, Array]:
    r"""
    Load neutron star crust equation of state data from the default directory.

    This function loads pre-computed crust EOS data from tabulated files, supporting
    both built-in crust models (BPS, DH) and custom files.

    Args:
        name (str): Name of the crust model to load (e.g., 'BPS', 'DH') or a filename
                   with .npz extension for custom files.

    Returns:
        tuple[Array, Array, Array]: A tuple containing:
            - Number densities :math:`n` [:math:`\mathrm{fm}^{-3}`]
            - Pressures :math:`p` [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]
            - Energy densities :math:`\varepsilon` [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]

    Raises:
        ValueError: If the specified crust model is not found in the default directory.

    Note:
        Available built-in crust models can be found in the crust/ subdirectory.
    """

    # Get the available crust names
    available_crust_names = [
        f.split(".")[0] for f in os.listdir(CRUST_DIR) if f.endswith(".npz")
    ]

    # If a name is given, but it is not a filename, load the crust from the jose directory
    if not name.endswith(".npz"):
        if name in available_crust_names:
            name = os.path.join(CRUST_DIR, f"{name}.npz")
        else:
            raise ValueError(
                f"Crust {name} not found in {CRUST_DIR}. Available crusts are {available_crust_names}"
            )

    # Once the correct file is identified, load it
    crust = jnp.load(name)
    n, p, e = crust["n"], crust["p"], crust["e"]
    return n, p, e
