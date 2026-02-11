TOV Solvers
===========

JESTER provides TOV (Tolman-Oppenheimer-Volkoff) equation solvers to compute neutron star structure from a given equation of state. All solvers are JAX-accelerated and support automatic differentiation.

Available Solvers
-----------------

**General Relativity**
   Standard TOV solver in general relativity.

   :doc:`tov/gr`

**Modified Gravity**
   Scalar-tensor theories and alternative gravity frameworks.

   :doc:`tov/scalar_tensor`

**Pressure Anisotropy**
   Post-TOV solver including anisotropic effects.

   :doc:`tov/anisotropy`

.. toctree::
   :hidden:

   tov/gr
   tov/scalar_tensor
   tov/anisotropy
