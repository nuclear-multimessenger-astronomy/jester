.. _eos-piecewise-polytrope:

Piecewise Polytrope
===================

The piecewise polytrope is a 4-parameter EOS parametrization introduced by
Read et al. (2009). It represents the pressure as a piecewise power law
:math:`p = K_i \rho^{\Gamma_i}` in each density interval.

Free parameters
---------------

- :math:`\log_{10} p_1^\mathrm{SI}` — :math:`\log_{10}` of the pressure in Pa at the reference density :math:`\rho_1 = 10^{17.7}` kg/m³. Typical range: 33.5 to 34.5.
- :math:`\Gamma_1, \Gamma_2, \Gamma_3` — adiabatic indices for three successive high-density core pieces. Physical values are in roughly :math:`[1.2, 4.5]`.

The low-density crust is fixed to the analytic SLy4 4-piece fit from LALSuite
and is not a free parameter.

Example prior file
------------------

.. code-block:: python

   logp1_si = UniformPrior(33.5, 34.5, parameter_names=["logp1_si"])
   gamma1 = UniformPrior(1.2, 4.5, parameter_names=["gamma1"])
   gamma2 = UniformPrior(1.2, 4.5, parameter_names=["gamma2"])
   gamma3 = UniformPrior(1.2, 4.5, parameter_names=["gamma3"])

Example config snippet
----------------------

.. code-block:: yaml

   eos:
     type: piecewise_polytrope
     n_points: 500

API reference
-------------

:class:`jesterTOV.eos.piecewise_polytrope.PiecewisePolytrope_EOS_model`

References
----------

Read et al., Physical Review D 79, 124032 (2009).
