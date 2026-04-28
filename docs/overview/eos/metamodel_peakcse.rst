.. _eos-metamodel-peakcse:

Metamodel + peakCSE
===================

This page describes the Metamodel + peakCSE EOS parametrization and its implementation in ``jester``.
It is closely related to the :ref:`eos-metamodel-cse` model but uses a physically motivated functional
form for the high-density speed of sound rather than a free piecewise interpolation.

Physical motivation
-------------------

The plain CSE extension places no constraints on the shape of :math:`c_s^2(n)` above the break density
and requires many parameters when a fine density grid is needed.
The peakCSE model of :cite:`Greif:2018njt` imposes more structure by combining two physically
motivated components: a logistic function that drives the speed of sound smoothly toward the
conformal limit :math:`c_s^2 = 1/3` expected from perturbative QCD at asymptotically high densities,
and a Gaussian peak that captures a possible phase transition — where matter briefly stiffens before
settling toward the conformal limit.

Speed-of-sound parametrization
--------------------------------

While this model was introduced in :cite:`Greif:2018njt`, here we use the implementation and notation from :cite:`Pang:2025fes`.
Above the break density :math:`n_\mathrm{break}`, the squared speed of sound is parametrized as follows:

.. math::
   :label: cs2_peakcse

   c_s^2 = \xi + \frac{\tfrac{1}{3} - \xi}{1 + e^{-l_\mathrm{sig}(n - n_\mathrm{sig})}}
   + c_{s,\mathrm{peak}}^2 \exp\!\left[-\frac{1}{2}\left(\frac{n - n_\mathrm{peak}}{\sigma_\mathrm{peak}}\right)^2\right] ,

where :math:`\xi` is a constant offset chosen at construction time to ensure continuity of
:math:`c_s^2` at :math:`n_\mathrm{break}`.
The logistic term drives :math:`c_s^2 \to 1/3` at large densities regardless of the other parameters,
while the Gaussian term is a transient feature that can represent a rapid stiffening associated with a
phase transition.

The full EOS above :math:`n_\mathrm{break}` is then obtained by integrating the same thermodynamic
relations as in the CSE model (see :ref:`eos-metamodel-cse` for similar equations and more information):

.. math::

   \mu(n) &= \mu_\mathrm{break}\,\exp\!\left[\int_{n_\mathrm{break}}^n \frac{c_s^2(n')}{n'}\,dn'\right], \\
   p(n) &= p_\mathrm{break} + \int_{n_\mathrm{break}}^n c_s^2(n')\,\mu(n')\,dn', \\
   \varepsilon(n) &= \varepsilon_\mathrm{break} + \int_{n_\mathrm{break}}^n \mu(n')\,dn'.

The example below shows the pressure and speed of sound for a representative EOS
with the peakCSE extension starting at :math:`n_\mathrm{break} = 2\,n_\mathrm{sat}`.

.. plot:: overview/eos/metamodel_peakcse_plot.py

   Pressure (top) and squared speed of sound (bottom) versus baryon number density
   for a representative Metamodel + peakCSE EOS.
   The grey, orange, and green shaded bands indicate the crust, the metamodel core,
   and the peakCSE extension region, respectively.
   The dashed grey curve in the lower panel is the logistic baseline (the Gaussian
   term removed), showing how :math:`c_s^2` settles smoothly toward the conformal
   limit :math:`c_s^2 = 1/3` (dotted red line).
   Annotations mark the Gaussian peak amplitude :math:`c_{s,\mathrm{peak}}^2`,
   its central density :math:`n_\mathrm{peak}`, and the one-sigma half-width
   :math:`\sigma_\mathrm{peak}`.

Parameters
----------

The peakCSE model adds 5 parameters on top of the 9 NEPs and the :math:`n_{\rm{break}}` density:

- ``gaussian_peak`` (:math:`c_{s,\mathrm{peak}}^2`) — amplitude of the Gaussian peak in :math:`c_s^2/c^2`.
- ``gaussian_mu`` (:math:`n_\mathrm{peak}`) — central density of the peak in :math:`\mathrm{fm}^{-3}`.
- ``gaussian_sigma`` (:math:`\sigma_\mathrm{peak}`) — width of the peak in :math:`\mathrm{fm}^{-3}`.
- ``logit_growth_rate`` (:math:`l_\mathrm{sig}`) — rate at which the logistic term rises.
- ``logit_midpoint`` (:math:`n_\mathrm{sig}`) — density at which the logistic term reaches half its asymptotic value.

In total the model has :math:`9 + 1 + 5 = 15` free parameters.

Usage example
-------------

**Configuration file:**

.. code-block:: yaml

   eos:
     type: metamodel_peakcse
     ndat_metamodel: 100
     nmax_nsat: 12.0
     crust_name: DH

**Prior file** (excerpt):

.. code-block:: python

   nbreak         = UniformPrior(0.16, 0.48, parameter_names=["nbreak"])
   gaussian_peak  = UniformPrior(0.0, 1.0,   parameter_names=["gaussian_peak"])
   gaussian_mu    = UniformPrior(0.3, 1.5,   parameter_names=["gaussian_mu"])
   gaussian_sigma = UniformPrior(0.05, 0.5,  parameter_names=["gaussian_sigma"])
   logit_growth_rate = UniformPrior(1.0, 20.0, parameter_names=["logit_growth_rate"])
   logit_midpoint    = UniformPrior(0.5, 2.0,  parameter_names=["logit_midpoint"])

Further resources
-----------------

* API reference: :class:`jesterTOV.eos.metamodel.MetaModel_with_peakCSE_EOS_model`
* CSE background: :ref:`eos-metamodel-cse`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
