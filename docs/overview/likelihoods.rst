.. _overview-likelihoods:

Likelihood Constraints
======================

JESTER supports multiple observational constraints from multi-messenger astronomy to constrain the neutron star equation of state. These likelihoods can be combined in Bayesian inference.

Available Constraints
---------------------

**Gravitational Waves**
   Binary neutron star merger observations (GW170817, etc.).

   :doc:`likelihoods/gw`

**NICER**
   X-ray timing constraints on mass and radius (PSR J0030+0451, PSR J0740+6620).

   :doc:`likelihoods/nicer`

**Nuclear Experiments (ChiEFT)**
   Chiral effective field theory constraints on low-density nuclear matter.

   :doc:`likelihoods/chieft`

**Radio Timing**
   High-precision pulsar mass measurements.

   :doc:`likelihoods/radio`

.. toctree::
   :hidden:

   likelihoods/gw
   likelihoods/nicer
   likelihoods/chieft
   likelihoods/radio
