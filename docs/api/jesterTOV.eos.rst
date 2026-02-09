``jesterTOV.eos`` module
========================

.. currentmodule:: jesterTOV.eos

.. automodule:: jesterTOV.eos

The equation of state (EOS) module provides classes for modeling neutron star matter equations of state using various parametrizations.

Mathematical Background
-----------------------

The metamodel equation of state is based on the formulation from Margueron et al. (2021), which provides a flexible framework for modeling nuclear matter at various densities.

The total energy density is given by:

.. math::

   \varepsilon(n, \delta) = \varepsilon_{\text{kinetic}}(n, \delta) + \varepsilon_{\text{potential}}(n, \delta)

where :math:`n` is the baryon number density and :math:`\delta = 1 - 2Y_p` is the isospin asymmetry parameter.

Submodules
----------

.. toctree::
    :maxdepth: 1

    jesterTOV.eos.metamodel
    jesterTOV.eos.spectral

Base Classes
------------

.. autosummary::
   :toctree: _autosummary

   Interpolate_EOS_model

Crust Models
------------

.. autosummary::
   :toctree: _autosummary

   Crust
   CRUST_DIR
