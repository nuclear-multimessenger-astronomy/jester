jesterTOV.ptov module
=====================

The post-TOV (pTOV) module extends the standard TOV equations to include tidal deformability calculations.

.. automodule:: jesterTOV.tov.post
   :members:
   :undoc-members:
   :show-inheritance:

Mathematical Background
-----------------------

The post-TOV equations include additional perturbative equations for calculating the tidal Love number :math:`k_2`:

.. math::

   \frac{dH}{dr} = \beta(r) H + \alpha(r) b

.. math::

   \frac{db}{dr} = H + \gamma(r) b

where :math:`H` and :math:`b` are auxiliary functions related to the tidal deformation, and :math:`\alpha`, :math:`\beta`, :math:`\gamma` are coefficients that depend on the background stellar structure.

The tidal Love number is then:

.. math::

   k_2 = \frac{8C^5}{5}(1-2C)^2[2 + 2C(y_R - 1) - y_R] \times \{2C[6-3y_R + 3C(5y_R-8)] + 4C^3[13-11y_R + C(3y_R-2) + 2C^2(1+y_R)] + 3(1-2C)^2[2-y_R + 2C(y_R-1)]\ln(1-2C)\}^{-1}

where :math:`C = GM/Rc^2` is the compactness and :math:`y_R` is related to the tidal response at the surface.