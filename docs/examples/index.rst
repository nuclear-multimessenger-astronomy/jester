This section contains Jupyter notebook examples demonstrating JESTER's core functionality for equation of state modeling and TOV solving.

Basic Examples
--------------

**Constructing an EOS and solving the TOV equations**
   Basic tutorial showing how to construct an equation of state and solve the Tolman-Oppenheimer-Volkoff equations for neutron star structure.

   :doc:`eos_tov/eos_tov`

**Scalar-tensor theory TOV solver**
   Exploration of modified gravity theories using JESTER's scalar-tensor TOV solver to study deviations from general relativity.

   :doc:`eos_tov/eos_STtov`

**Automatic differentiation for TOV equations**
   Demonstrates how JAX's automatic differentiation enables efficient computation of derivatives and gradients for TOV solutions.

   :doc:`eos_tov/automatic_differentiation`

**Automatic differentiation with MetaModel EOS**
   Example demonstrating the use of automatic differentiation with the MetaModel equation of state parametrization.

   :doc:`eos_tov/automatic_differentiation_metamodel`

.. toctree::
   :hidden:

   eos_tov/eos_tov
   eos_tov/eos_STtov
   eos_tov/automatic_differentiation
   eos_tov/automatic_differentiation_metamodel
