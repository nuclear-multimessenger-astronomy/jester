``jesterTOV.tov`` module
========================

.. currentmodule:: jesterTOV.tov

.. automodule:: jesterTOV.tov

The TOV (Tolman-Oppenheimer-Volkoff) module provides solvers for the structure equations of neutron stars in various theories of gravity.

Mathematical Background
-----------------------

The TOV equations describe hydrostatic equilibrium in spherically symmetric, non-rotating neutron stars. In general relativity, the equations are:

.. math::

   \frac{dP}{dr} = -\frac{(\varepsilon + P)(m + 4\pi r^3 P)}{r(r - 2m)}

.. math::

   \frac{dm}{dr} = 4\pi r^2 \varepsilon

where :math:`P` is the pressure, :math:`\varepsilon` is the energy density, :math:`m` is the enclosed mass, and :math:`r` is the radial coordinate.

Data Classes
------------

.. autosummary::
   :toctree: _autosummary

   EOSData
   TOVSolution
   FamilyData

Solver Classes
--------------

.. autosummary::
   :toctree: _autosummary

   TOVSolverBase
   GRTOVSolver
   PostTOVSolver
   ScalarTensorTOVSolver
