.. _likelihood-nicer:

NICER Constraints
=================

.. note::

   This page is a placeholder. The detailed content below is to be written soon.

NICER (Neutron star Interior Composition Explorer) is a NASA X-ray telescope on the
International Space Station that measures X-ray pulse profiles from millisecond pulsars.
By modelling the anisotropic X-ray emission from hot spots on the neutron star surface,
NICER constrains the stellar mass and radius simultaneously, providing direct input for
equation-of-state inference.

JESTER currently supports four pulsars observed by NICER:

* **PSR J0030+0451** — multiple hotspot models from Amsterdam (Riley et al. 2019) and
  Maryland (Miller et al. 2019) groups; NICER-only data.
* **PSR J0437-4715** — Amsterdam CST+PDT analysis (Choudhury et al. 2024); NICER-only data.
* **PSR J0614-3329** — Amsterdam ST+PDT analysis (Dittmann et al. 2025); NICER-only data.
* **PSR J0740+6620** — Amsterdam gamma analysis (Salmi et al. 2024) and Maryland
  analyses (Miller et al. 2021); NICER+XMM-Newton data.

The figure below shows the mass–radius posteriors for one representative analysis
per pulsar, with filled contours at the 68% and 90% credible intervals.

.. plot:: overview/likelihoods/nicer_mr_plot.py

   Mass–radius posteriors from NICER for the four supported pulsars.
   Filled contours show the 68% (darker) and 90% (lighter) credible intervals
   from kernel density estimation of the published posterior samples.
   One representative analysis is plotted per pulsar.

----

[Placeholder: The sections below are to still to be written.]

This page will cover:

* X-ray pulse profile modelling and the mass–radius inference method
* Details of each supported pulsar and analysis group
* How posterior samples are stored and loaded (``data/NICER/``)
* The normalizing-flow likelihood implementation (:class:`~jesterTOV.inference.likelihoods.nicer.NICERLikelihood`)
* The legacy KDE-based implementation (:class:`~jesterTOV.inference.likelihoods.nicer.NICERKDELikelihood`)
* Configuration in YAML files
* Usage examples
* References
