:orphan:

This section contains Jupyter notebook examples demonstrating JESTER's core functionality for equation of state modeling and TOV solving.

Basic Examples
--------------

**Constructing an EOS and solving the TOV equations**
   Basic tutorial showing how to construct an equation of state and solve the Tolman-Oppenheimer-Volkoff equations for neutron star structure.

   :doc:`eos_tov/eos_tov`

**Automatic differentiating through TOV solvers**
   Being written in ``jax``, ``jester`` supports automatic differentiation through the TOV solvers, allowing for gradient-based optimization routines. This example demonstrates how to compute gradients of neutron star properties with respect to EOS parameters.

   :doc:`eos_tov/automatic_differentiation`

**Prior predictive checks for EOS models**
   Draws samples from the prior distributions of EOS parametrizations available in JESTER (MetaModel, MetaModel + CSE, MetaModel + peakCSE, Spectral reparametrized), solves the TOV equations for each, and plots the resulting ensemble of neutron star properties as a sanity check before inference. After this, users should be ready to dive into the full Bayesian inference workflow: get started here: :doc:`../inference/quickstart`.

   :doc:`eos_tov/prior_predictive`

.. toctree::
   :hidden:

   eos_tov/eos_tov
   eos_tov/automatic_differentiation
   eos_tov/prior_predictive
