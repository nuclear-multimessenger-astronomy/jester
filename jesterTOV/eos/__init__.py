r"""Equation of state models and utilities for neutron star structure calculations."""

# Base classes
from jesterTOV.eos.base import Interpolate_EOS_model

# Crust models
from jesterTOV.eos.crust import Crust, CRUST_DIR

# Meta-model parametrizations
from jesterTOV.eos.metamodel import (
    MetaModel_EOS_model,
    MetaModel_with_CSE_EOS_model,
    MetaModel_with_peakCSE_EOS_model,
)

# Spectral decomposition parametrization
from jesterTOV.eos.spectral import SpectralDecomposition_EOS_model

__all__ = [
    # Base
    "Interpolate_EOS_model",
    # Crust
    "Crust",
    "CRUST_DIR",
    # Meta-model
    "MetaModel_EOS_model",
    "MetaModel_with_CSE_EOS_model",
    "MetaModel_with_peakCSE_EOS_model",
    # Spectral
    "SpectralDecomposition_EOS_model",
]
