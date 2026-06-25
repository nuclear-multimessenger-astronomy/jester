``jesterTOV.tov.gr`` module
============================

.. currentmodule:: jesterTOV.tov.gr

General Relativity TOV solver implementation.

Mathematical Background
-----------------------

The TOV equations in general relativity describe hydrostatic equilibrium in spherically symmetric neutron stars:

.. math::

   \frac{dP}{dr} = -\frac{(\varepsilon + P)(m + 4\pi r^3 P)}{r(r - 2m)}

.. math::

   \frac{dm}{dr} = 4\pi r^2 \varepsilon

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   GRTOVSolver
