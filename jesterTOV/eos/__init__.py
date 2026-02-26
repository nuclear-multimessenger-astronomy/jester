r"""Equation of state models and utilities for neutron star structure calculations.

Submodules
----------
- base: Base classes for EOS models
- crust: Crust equation of state models
- metamodel: Meta-model parametrizations (MetaModel, MetaModel+CSE, MetaModel+peakCSE)
- spectral: Spectral decomposition parametrization
- piecewise_polytrope: Piecewise polytrope parametrization (Read et al. 2009)

Import classes from their specific submodules:

    from jesterTOV.eos.base import Interpolate_EOS_model
    from jesterTOV.eos.crust import Crust, CRUST_DIR
    from jesterTOV.eos.metamodel import MetaModel_EOS_model, MetaModel_with_CSE_EOS_model
    from jesterTOV.eos.spectral import SpectralDecomposition_EOS_model
    from jesterTOV.eos.piecewise_polytrope import PiecewisePolytrope_EOS_model
"""
