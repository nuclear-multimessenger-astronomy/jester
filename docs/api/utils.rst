jesterTOV.utils module
======================

Utility functions for unit conversions, physical constants, and numerical methods.

.. automodule:: jesterTOV.utils
   :members:
   :undoc-members:
   :show-inheritance:

Physical Constants
------------------

The module provides physical constants in geometric units where :math:`G = c = 1`:

- Nuclear saturation density: :math:`n_{\text{sat}} = 0.16 \text{ fm}^{-3}`
- Solar mass: :math:`M_{\odot} = 1.477 \text{ km}`
- Conversion factors between CGS and geometric units

Numerical Methods
-----------------

Key numerical utilities include:

- **Cubic spline interpolation**: For smooth interpolation between data points
- **Cumulative trapezoidal integration**: For integrating differential equations
- **Unit conversions**: Between different unit systems used in neutron star physics