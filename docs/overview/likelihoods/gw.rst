.. _likelihood-gw:

Gravitational Wave Constraints
===============================

.. note::

   This page is a placeholder. The detailed content below is to be written soon.

Binary neutron star (BNS) mergers observed with gravitational waves provide direct
constraints on tidal deformability. During the inspiral phase, each neutron star is
tidally deformed by its companion's gravitational field, and this deformation leaves
an imprint on the gravitational waveform. The tidal deformability
:math:`\Lambda = \frac{2}{3} k_2 \left(\frac{R}{M}\right)^5`
depends sensitively on the equation of state through both the radius :math:`R` and the
Love number :math:`k_2`.

JESTER currently supports two confirmed BNS events from the LIGO–Virgo catalogue:

* **GW170817** — the first BNS merger, detected on 17 August 2017 (Abbott et al. 2017).
  Multiple posterior datasets are available covering both GWTC-1 and updated analyses,
  with low-spin and high-spin prior choices.
* **GW190425** — the second BNS event, detected on 25 April 2019 (Abbott et al. 2020).
  Several waveform models are supported (IMRPhenomPNRT, IMRPhenomDNRT, TaylorF2).

For each event, JESTER trains a normalizing flow on the GW posterior samples.
The flow learns the joint density over component source-frame masses and tidal
deformabilities :math:`(m_1, m_2, \Lambda_1, \Lambda_2)`, and
:class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood` evaluates the EOS
likelihood by querying this flow. The corner plots below show the original
posterior samples (blue) and samples drawn from the trained flow (red), illustrating
how faithfully the flow captures the posterior.

----

GW170817
--------

.. plot:: overview/likelihoods/gw_corner_gw170817.py

   Corner plot for GW170817 (GWTC-1, low-spin prior). Blue shows the original
   posterior samples; red shows samples drawn from the trained normalizing flow.
   The diagonal panels display marginal KDE curves; the off-diagonal panels show
   68% (darker) and 90% (lighter) filled credible-interval contours.

GW190425
--------

.. plot:: overview/likelihoods/gw_corner_gw190425.py

   Corner plot for GW190425 (IMRPhenomPNRT, low-spin prior). Blue shows the
   original posterior samples; red shows samples drawn from the trained
   normalizing flow.

----

[Placeholder: The sections below are still to be written.]

This page will cover:

* Tidal deformability measurement methodology and waveform models
* Details of each supported event and posterior dataset
* How posterior samples are stored and loaded (``data/gw170817/``, ``data/gw190425/``)
* The normalizing-flow likelihood implementation (:class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood`)
* The resampling variant (:class:`~jesterTOV.inference.likelihoods.gw.GWLikelihoodResampled`)
* Configuration in YAML files
* Usage examples
* References
