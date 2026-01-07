r"""Equation of state models and utilities for neutron star structure calculations."""

# Base classes
from jesterTOV.eos.base import Interpolate_EOS_model

# Crust models
from jesterTOV.eos.crust import load_crust, CRUST_DIR

# Meta-model parametrizations
from jesterTOV.eos.metamodel import (
    MetaModel_EOS_model,
    MetaModel_with_CSE_EOS_model,
    MetaModel_with_peakCSE_EOS_model,
)

# Neutron star family construction utilities
from jesterTOV.eos.families import (
    locate_lowest_non_causal_point,
    construct_family,
    construct_family_nonGR,
    construct_family_ST,
    construct_family_ST_sol,
)

__all__ = [
    # Base
    "Interpolate_EOS_model",
    # Crust
    "load_crust",
    "CRUST_DIR",
    # Meta-model
    "MetaModel_EOS_model",
    "MetaModel_with_CSE_EOS_model",
    "MetaModel_with_peakCSE_EOS_model",
    # Families
    "locate_lowest_non_causal_point",
    "construct_family",
    "construct_family_nonGR",
    "construct_family_ST",
    "construct_family_ST_sol",
]
