jesterTOV.tov module
====================

The TOV (Tolman-Oppenheimer-Volkoff) module solves the stellar structure equations for neutron stars.

.. automodule:: jesterTOV.tov.gr
   :members:
   :undoc-members:
   :show-inheritance:

Mathematical Background
-----------------------

The TOV equations describe the structure of a spherically symmetric, static star in general relativity:

.. math::

   \frac{dP}{dr} = -\frac{G(\varepsilon + P)(M + 4\pi r^3 P)}{r(r - 2GM)}

.. math::

   \frac{dM}{dr} = 4\pi r^2 \varepsilon

where:
- :math:`P(r)` is the pressure at radius :math:`r`
- :math:`\varepsilon(r)` is the energy density at radius :math:`r`
- :math:`M(r)` is the mass enclosed within radius :math:`r`
- :math:`G` is the gravitational constant (set to 1 in geometric units)