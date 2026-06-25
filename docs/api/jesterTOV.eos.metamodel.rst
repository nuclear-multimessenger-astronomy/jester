``jesterTOV.eos.metamodel`` module
===================================

.. currentmodule:: jesterTOV.eos.metamodel

Meta-model equation of state implementations based on nuclear empirical parameters.

Mathematical Background
-----------------------

The metamodel equation of state is based on the formulation from Margueron et al. (2021), which provides a flexible framework for modeling nuclear matter at various densities.

The total energy density is given by:

.. math::

   \varepsilon(n, \delta) = \varepsilon_{\text{kinetic}}(n, \delta) + \varepsilon_{\text{potential}}(n, \delta)

where :math:`n` is the baryon number density and :math:`\delta = 1 - 2Y_p` is the isospin asymmetry parameter.

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   MetaModel_EOS_model
   MetaModel_with_CSE_EOS_model
   MetaModel_with_peakCSE_EOS_model
