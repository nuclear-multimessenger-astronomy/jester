.. _overview-likelihoods:

Likelihood constraints
======================

JESTER supports multiple observational constraints from multi-messenger astronomy to constrain the neutron star equation of state. These likelihoods can be combined in Bayesian inference.

Base class
-----------

Information about how the base likelihood class is implemented. Useful for those wishing to know how to extend ``jester`` and add new likelihoods.

:doc:`likelihoods/base_class`

Chiral effective field theory (ChiEFT)
-----------------------------------------

Chiral effective field theory constraints on low-density nuclear matter.

:doc:`likelihoods/chieft`

Radio Timing
------------

High-precision pulsar mass measurements.

:doc:`likelihoods/radio`

NICER
-----

X-ray timing constraints on mass and radius (PSR J0030+0451, PSR J0740+6620).

:doc:`likelihoods/nicer`

Gravitational Waves
-------------------

Binary neutron star merger observations (GW170817, etc.).

:doc:`likelihoods/gw`

.. toctree::
   :hidden:

   likelihoods/base_class
   likelihoods/chieft
   likelihoods/radio
   likelihoods/nicer
   likelihoods/gw
