.. _overview-eos:

EOS models
================================

JESTER provides multiple parametrizations for the neutron star equation of state, each with different physical assumptions and use cases.

Metamodel
---------

Taylor expansion of energy density around saturation density.

:doc:`eos/metamodel`

Metamodel + CSE
---------------

Metamodel with speed-of-sound extrapolation to high densities.

:doc:`eos/metamodel_cse`

Metamodel + peakCSE
-------------------

Metamodel with a Gaussian-peaked speed-of-sound extrapolation to high densities.

:doc:`eos/metamodel_peakcse`

Spectral Decomposition
----------------------

4-parameter spectral expansion of the EOS.

:doc:`eos/spectral`

.. toctree::
   :hidden:

   eos/metamodel
   eos/metamodel_cse
   eos/metamodel_peakcse
   eos/spectral
