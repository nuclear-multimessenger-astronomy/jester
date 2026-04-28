This section contains Jupyter notebook examples demonstrating JESTER's core functionality for equation of state modeling and TOV solving.

Basic Examples
--------------

**Constructing an EOS and solving the TOV equations**
   Basic tutorial showing how to construct an equation of state and solve the Tolman-Oppenheimer-Volkoff equations for neutron star structure.

   :doc:`eos_tov/eos_tov`

**Automatic differentiating through TOV solvers**
   Being written in ``jax``, ``jester`` supports automatic differentiation through the TOV solvers, allowing for gradient-based optimization routines. This example demonstrates how to compute gradients of neutron star properties with respect to EOS parameters.

   :doc:`eos_tov/automatic_differentiation`

.. toctree::
   :hidden:

   eos_tov/eos_tov
   eos_tov/automatic_differentiation