.. _tov-anisotropy:

Pressure anisotropy
=====================================

This page describes the beyond-GR TOV solver with phenomenological pressure anisotropy and its
implementation in ``jester``.

Physical motivation
-------------------

In the standard GR treatment, neutron star matter is assumed to be a perfect fluid with isotropic
pressure.
Several physical scenarios — superfluid vortex structures, strong magnetic fields, and exotic
phase transitions — can produce a local anisotropy between the radial and tangential pressures.
Rather than committing to a specific microphysical model, the ``AnisotropyTOVSolver`` parametrizes
the effect through phenomenological correction terms added directly to the stellar structure equations. 
This allows Bayesian inference to jointly constrain the EOS and potential deviations from the
isotropic GR result.

Modified structure equations
-----------------------------

There are several models for pressure anisotropy in neutron stars, and the ``AnisotropyTOVSolver`` includes three of them, which can be turned on or off independently.
An overview of the models can be found in :cite:`Rahmansyah:2020gar`, Eqs. (12) - (14) (note: in Eq. 12, the power of epsilon should be 2). 

The radial pressure gradient in the standard TOV equation is modified by an anisotropy correction
:math:`\sigma(r)`:

.. math::
   :label: anisotropy_tov_dpdr

   \frac{dp_r}{dr} = -\frac{m\varepsilon\left(1 + \frac{p_r}{\varepsilon}\right)\!\left(1 + \frac{4\pi r^3 p_r}{m}\right)}{r^2\!\left(1 - \frac{2m}{r}\right)} - \frac{2\sigma}{r} \, .

When :math:`\sigma = 0` the equation reduces to the ordinary TOV result.

We have the following models implemented for the ``jester`` anisotropy paper :cite:`Pang:2025fes`:

* Bowers-Liang model (:math:`\sigma_\mathrm{BL}`) (ref: :cite:`Bowers:1974tgi`)
* Horvat et al (:math:`\sigma_\mathrm{DY}`) (ref: :cite:`Horvat:2010xf`)
* Cosenza model (:math:`\sigma_\mathrm{HB}`) (ref: :cite:`Cosenza:1981myi`)

The correction :math:`\sigma` can be decomposed into contributions from these three independently
parametrized phenomenological models so that :math:`\sigma = \sigma_\mathrm{BL} + \sigma_\mathrm{DY} + \sigma_\mathrm{HB}`.
The parametrizations have the following form:

.. math::
   :label: sigma_BL

   \sigma_\mathrm{BL} = -\frac{\lambda_\mathrm{BL}\,\varepsilon^2 r^2}{3}
   \left(1 + \frac{3p}{\varepsilon}\right)\!\left(1 + \frac{p}{\varepsilon}\right)
   \frac{1}{1 - 2m/r}

.. math::
   :label: sigma_DY

   \sigma_\mathrm{DY} = \lambda_\mathrm{DY}\,\frac{2m}{r}\,p

.. math::
   :label: sigma_HB

   \sigma_\mathrm{HB} = -\!\left(\frac{1}{\lambda_\mathrm{HB}} - 1\right)\frac{r}{2}\frac{dp}{dr}

Setting :math:`\lambda_\mathrm{BL} = 0`, :math:`\lambda_\mathrm{DY} = 0`, and
:math:`\lambda_\mathrm{HB} = 1` recovers the standard GR equations with pressure isotropy.

The tidal deformability equations are also modified consistently by including the derivative of
:math:`\sigma` with respect to pressure inside :meth:`~jesterTOV.tov.anisotropy.AnisotropyTOVSolver.solve`.

Usage
-----

**Configuration file:**

.. code-block:: yaml

   tov:
     type: anisotropy
     min_nsat_TOV: 0.75
     ndat_TOV: 100
     nb_masses: 100

**Prior file** (excerpt):

Note that these priors have to be defined on top of your priors for the EOS parameters, which are not shown here. 

.. code-block:: python

   # Only sample lambda_DY, while fixing the other two couplings to their GR values
   # This is what was done in Pang:2025fes
   lambda_BL = Fixed(0.0,  parameter_names=["lambda_BL"])
   lambda_DY = UniformPrior(-0.5, 0.5, parameter_names=["lambda_DY"])
   lambda_HB = Fixed(1.0,  parameter_names=["lambda_HB"])

Any of the three coupling constants can be fixed to its GR value or treated as a free parameter
depending on the analysis goal.
Using ``Fixed`` for the couplings you do not want to constrain is more efficient than placing a
broad prior on them.
Complete working examples are in ``examples/inference/anisotropy/``.

Further resources
-----------------

* API reference: :class:`jesterTOV.tov.anisotropy.AnisotropyTOVSolver`
* Standard GR TOV: :ref:`tov-gr`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
