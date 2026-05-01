.. _likelihood-gw:

Gravitational wave constraints from binary neutron star mergers
=================================================================

.. note::

   Running some of the scripts mentioned here require an installation of ``bilby`` to convert masses to source frame. This is an optional dependency that is not listed in the JESTER repo, so you may need to install it manually with ``uv pip install bilby`` or ``uv sync --extra dev`` in your environment.

Binary neutron star (BNS) mergers observed with gravitational waves provide direct constraints on tidal deformability. During the inspiral phase, each neutron star is tidally deformed by its companion's gravitational field, and this deformation leaves an imprint on the gravitational waveform. The tidal deformability :math:`\Lambda = \frac{2}{3} k_2 \left(\frac{R}{M}\right)^5` depends sensitively on the equation of state through both the radius :math:`R` and the Love number :math:`k_2`.

JESTER currently supports two confirmed BNS events from the LIGO-Virgo catalogue:

* **GW170817** :cite:`LIGOScientific:2017vwq`, :cite:`LIGOScientific:2018hze`, GWOSC page: `<https://gwosc.org/eventapi/html/GWTC-1-confident/GW170817/v3/>`_
* **GW190425** :cite:`LIGOScientific:2020aai`, GWOSC page: `<https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW190425/v3/>`_

For both, several posteriors are available from the LIGO-Virgo-KAGRA (LVK) collaboration and independent analyses, using different waveform models and spin priors.
Posterior samples for both events are stored as compressed NumPy archives (``.npz``) under ``jesterTOV/inference/data/``. Each file contains the four parameters required by the likelihood: ``mass_1_source``, ``mass_2_source``, ``lambda_1``, and ``lambda_2``, all in source frame (solar masses for masses, dimensionless for tidal deformabilities), together with a ``metadata`` dictionary recording the data source, waveform model, and sample count.
The processed files live in ``jesterTOV/inference/data/gw170817/`` and ``jesterTOV/inference/data/gw190425/`` for GW170817 and GW190425, respectively.

For each dataset, JESTER has a trained normalizing flow on the GW posterior samples. The flow learns the joint density over component source-frame masses and tidal deformabilities :math:`(m_1, m_2, \Lambda_1, \Lambda_2)`, and :class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood` evaluates the EOS likelihood by querying this flow. The corner plots below show the original posterior samples (blue) and samples drawn from the trained flow (red), illustrating how faithfully the flow captures the posterior.

Below, we provide more information on the available datasets.
The datasets can be downloaded automatically using the scripts ``jesterTOV/inference/data/gw170817/download_gw170817.py``, and ``jesterTOV/inference/data/gw190425/download_gw190425.py``, respectively.
These scripts download the raw files from the LIGO DCC, convert detector-frame masses to source frame, and save the four-parameter excerpts as ``.npz`` archives.
Note that the original files are gitignored, and only the processed ``.npz`` files are tracked in the repository, so you will need to run the download scripts to get the raw, complete datasets.

----

GW170817
--------

**LVK analyses** 

The table below shows the sets of posterior samples released by the LVK collaboration exist, taken from the following sources:

* P1800370 (GWTC-1 PE release) — `<https://dcc.ligo.org/LIGO-P1800370/public>`_
* P1800061 (*Properties of the binary neutron star merger GW170817*, :cite:`LIGOScientific:2018hze`) — `<https://dcc.ligo.org/LIGO-P1800061/public>`_

.. list-table::
   :header-rows: 1
   :widths: 40 20 15 30 10

   * - File
     - Release
     - Waveform
     - Spin prior
     - Samples
   * - ``gw170817_low_spin_posterior.npz``
     - P1800061
     - IMRPhenomPv2_NRTidal
     - below 0.05
     - 3,952
   * - ``gw170817_high_spin_posterior.npz``
     - P1800061
     - IMRPhenomPv2_NRTidal
     - below 0.89
     - 9,117
   * - ``gw170817_gwtc1_lowspin_posterior.npz``
     - P1800370 (GWTC-1)
     - IMRPhenomPv2_NRTidal
     - below 0.05
     - 8,078
   * - ``gw170817_gwtc1_highspin_posterior.npz``
     - P1800370 (GWTC-1)
     - IMRPhenomPv2_NRTidal
     - below 0.89
     - 4,041

**Independent analyses**

Additionally, we provide support for an independent analysis using the more up-to-date waveform model IMRPhenomXP_NRTidalv3, which was not available at the time of the original LVK releases. This analysis is from Wouters et al. (2025), using a low-spin prior, and can be downloaded here: `<https://github.com/ThibeauWouters/neural_priors/tree/main/final_results/GW170817/bns/default>`_

.. list-table::
   :header-rows: 1
   :widths: 40 20 15 15 10

   * - File
     - Release
     - Waveform
     - Spin prior
     - Samples
   * - ``gw170817_xp_nrtv3.npz``
     - Wouters et al. (2025)
     - IMRPhenomXP_NRTidalv3
     - low
     - 23,138

**Example normalizing flow**

Below, we show an example extracted posterior samples on the source-frame masses and tidal deformabilities and overlay a set of samples drawn from the normalizing flow trained on these samples.

.. plot:: overview/likelihoods/gw_corner_gw170817.py

   Corner plot for GW170817 (GWTC-1, low-spin prior). Blue shows the original posterior samples; red shows samples drawn from the trained normalizing flow.
   The diagonal panels display marginal KDE curves; the off-diagonal panels show 68% (darker) and 90% (lighter) filled credible-interval contours.

GW190425
--------

The processed files live in ``jesterTOV/inference/data/gw190425/``. Two sets of posteriors are available: the original GWTC-2 release (seven posteriors covering three waveform models and two spin-prior choices each, plus one from IMRPhenomXP_NRTidalv3), and the updated GWTC-2.1 release (cosmological and non-cosmological variants from Zenodo record 6513631).

**LVK analyses**

Two sets of posteriors are available from the LVK collaboration, taken from the following sources:

* Discovery paper PE release — `<https://dcc.ligo.org/LIGO-P2000026/public>`_
* Zenodo 6513631 (GWTC-2.1) — `<https://zenodo.org/records/6513631>`_

The table below shows the posteriors from the discovery paper. 

.. list-table::
   :header-rows: 1
   :widths: 42 25 18 15

   * - File
     - Waveform
     - Spin prior
     - Samples
   * - ``gw190425_phenompnrt-ls_posterior.npz``
     - IMRPhenomPv2_NRTidal
     - low spin
     - 46,709
   * - ``gw190425_phenompnrt-hs_posterior.npz``
     - IMRPhenomPv2_NRTidal
     - high spin
     - 226,598
   * - ``gw190425_phenomdnrt-ls_posterior.npz``
     - IMRPhenomD_NRTidal
     - low spin
     - 53,945
   * - ``gw190425_phenomdnrt-hs_posterior.npz``
     - IMRPhenomD_NRTidal
     - high spin
     - 58,713
   * - ``gw190425_taylorf2-ls_posterior.npz``
     - TaylorF2
     - low spin
     - 23,364
   * - ``gw190425_taylorf2-hs_posterior.npz``
     - TaylorF2
     - high spin
     - 28,025

The table below shows the posteriors from the GWTC-2.1 release. (For details, refer to the Zenodo record and paper).

.. list-table::
   :header-rows: 1
   :widths: 60 25 15

   * - File
     - Waveform
     - Variant
   * - ``gw190425_gwtc2p1_mixed_cosmo_imrphenompv2_nrtidal_highspin_posterior.npz``
     - IMRPhenomPv2_NRTidal
     - cosmo / high-spin
   * - ``gw190425_gwtc2p1_mixed_cosmo_imrphenompv2_nrtidal_lowspin_posterior.npz``
     - IMRPhenomPv2_NRTidal
     - cosmo / low-spin
   * - ``gw190425_gwtc2p1_mixed_nocosmo_imrphenompv2_nrtidal_highspin_posterior.npz``
     - IMRPhenomPv2_NRTidal
     - nocosmo / high-spin
   * - ``gw190425_gwtc2p1_mixed_nocosmo_imrphenompv2_nrtidal_lowspin_posterior.npz``
     - IMRPhenomPv2_NRTidal
     - nocosmo / low-spin


**Independent analyses** 

Additionally, we provide support for an independent analysis using the more up-to-date waveform model IMRPhenomXP_NRTidalv3, which was not available at the time of the original LVK releases. This analysis is from Wouters et al. (2025), using a low-spin prior, and can be downloaded here: `<https://github.com/ThibeauWouters/neural_priors/tree/main/final_results/GW190425/bns/default>`_

.. list-table::
   :header-rows: 1
   :widths: 42 25 18 15

   * - File
     - Waveform
     - Spin prior
     - Samples
   * - ``gw190425_xp_nrtv3.npz``
     - IMRPhenomXP_NRTidalv3
     - low
     - 25,715
  
**Example normalizing flow**

Below, we show another example extracted posterior samples on the source-frame masses and tidal deformabilities and overlay a set of samples drawn from the normalizing flow trained on these samples.

.. plot:: overview/likelihoods/gw_corner_gw190425.py

   Corner plot for GW190425 (IMRPhenomPv2_NRTidal, low-spin prior). Blue shows the
   original posterior samples; red shows samples drawn from the trained
   normalizing flow.

----

Trained flows
-------------

Normalizing flows for all datasets listed above have already been trained and are shipped with JESTER — no separate training step is required. The trained models live under ``jesterTOV/inference/flows/models/gw_maf/gw170817/`` and ``jesterTOV/inference/flows/models/gw_maf/gw190425/``, with one subdirectory per dataset. When you use the :class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood` without specifying a ``model_dir``, JESTER automatically loads a sensible default for each event. The defaults are:

* **GW170817**: the GWTC-1 low-spin posterior (``gw170817_gwtc1_lowspin``), based on IMRPhenomPv2_NRTidal with a spin prior below 0.05.
* **GW190425**: the discovery-paper low-spin posterior (``gw190425_phenompnrt_ls``), based on IMRPhenomPv2_NRTidal with a low-spin prior.

You can override the default by passing the path to any of the trained subdirectories via the ``nf_model_dir`` argument (see :doc:`../../inference/analyze_bns`), allowing you to switch datasets or spin priors without any retraining.

The default model directories are: set in ``jester/jesterTOV/inference/likelihoods/factory.py``::

  GW_EVENT_PRESETS = {
      "GW170817": "flows/models/gw_maf/gw170817/gw170817_gwtc1_lowspin",
      "GW190425": "flows/models/gw_maf/gw190425/gw190425_phenompnrt_ls",
  }

Need more information? Check out:

* :doc:`../../inference/analyze_bns`: more information on how to postprocess your GW posterior into a trained flow for ``jester`` inference
* :doc:`../../inference/training_flows`: more details on training the flows in ``jester``

----

.. admonition:: Current scope: BNS only
   :class: important

   JESTER currently provides normalizing flows only for **binary neutron star (BNS)** events (GW170817 and GW190425). While neutron star-black hole (NSBH) mergers also carry tidal information from the neutron star component, they are usually quite uninformative when it comes to the tidal posteriors, and therefore only weakly constrain the EOS. Such events are not supported yet, but might be in a future update.

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
