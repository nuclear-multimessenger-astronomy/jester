.. _overview-eos:

Equation of State (EOS) Models
================================

JESTER provides multiple parametrizations for the neutron star equation of state, each with different physical assumptions and use cases.

Available Parametrizations
---------------------------

**Metamodel**
   Taylor expansion of energy density around saturation density.

   :doc:`eos/metamodel`

**Metamodel + CSE**
   Metamodel with speed-of-sound extrapolation to high densities.

   :doc:`eos/metamodel_cse`

**Spectral Decomposition**
   4-parameter spectral expansion of the EOS.

   :doc:`eos/spectral`

.. toctree::
   :hidden:

   eos/metamodel
   eos/metamodel_cse
   eos/spectral
