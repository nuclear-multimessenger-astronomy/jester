.. _likelihood-gw:

Gravitational wave constraints from binary neutron star mergers
=================================================================

.. note::

   Running some of the scripts mentioned here require an installation of ``bilby`` to convert masses to source frame. This is an optional dependency that is not listed in the JESTER repo, so you may need to install it manually with ``uv pip install bilby`` or ``uv sync --extra bilby`` in your environment.

Binary neutron star (BNS) mergers observed with gravitational waves provide direct
constraints on tidal deformability. During the inspiral phase, each neutron star is
tidally deformed by its companion's gravitational field, and this deformation leaves
an imprint on the gravitational waveform. The tidal deformability
:math:`\Lambda = \frac{2}{3} k_2 \left(\frac{R}{M}\right)^5`
depends sensitively on the equation of state through both the radius :math:`R` and the
Love number :math:`k_2`.

JESTER currently supports two confirmed BNS events from the LIGO-Virgo catalogue:

.. TODO: put proper citations here

* **GW170817** — the first BNS merger (Abbott et al. 2017).
  Multiple posterior datasets are available covering both GWTC-1 and updated analyses,
  with low-spin and high-spin prior choices, as well as a posterior from the
  IMRPhenomXP_NRTidalv3 waveform model (Wouters et al. 2025).
* **GW190425** — the second BNS event (Abbott et al. 2020).
  Several waveform models are supported (IMRPhenomPNRT, IMRPhenomDNRT, TaylorF2,
  IMRPhenomXP_NRTidalv3).

Posterior samples for both events are stored as compressed NumPy archives (``.npz``) under
``jesterTOV/inference/data/``. Each file contains the four parameters required by
the likelihood: ``mass_1_source``, ``mass_2_source``, ``lambda_1``, and ``lambda_2``,
all in source frame (solar masses for masses, dimensionless for tidal deformabilities),
together with a ``metadata`` dictionary recording the data source, waveform model, and
sample count.

For each dataset, JESTER has a trained normalizing flow on the GW posterior samples.
The flow learns the joint density over component source-frame masses and tidal
deformabilities :math:`(m_1, m_2, \Lambda_1, \Lambda_2)`, and
:class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood` evaluates the EOS
likelihood by querying this flow. The corner plots below show the original
posterior samples (blue) and samples drawn from the trained flow (red), illustrating
how faithfully the flow captures the posterior.

----

GW170817
--------

The processed files live in ``jesterTOV/inference/data/gw170817/``. Five posterior sets
are included, drawn from two LIGO data releases and one independent analysis:

.. list-table::
   :header-rows: 1
   :widths: 40 20 15 15 10

   * - File
     - Release
     - Waveform
     - Spin prior
     - Samples
   * - ``gw170817_low_spin_posterior.npz``
     - P1800061
     - IMRPhenomPNRT
     - low (:math:`|\chi| \leq 0.05`)
     - 3,952
   * - ``gw170817_high_spin_posterior.npz``
     - P1800061
     - IMRPhenomPNRT
     - high (:math:`|\chi| \leq 0.89`)
     - 9,117
   * - ``gw170817_gwtc1_lowspin_posterior.npz``
     - P1800370 (GWTC-1)
     - IMRPhenomPv2NRT
     - low (:math:`|\chi| \leq 0.05`)
     - 8,078
   * - ``gw170817_gwtc1_highspin_posterior.npz``
     - P1800370 (GWTC-1)
     - IMRPhenomPv2NRT
     - high
     - 4,041
   * - ``gw170817_xp_nrtv3.npz``
     - Wouters et al. (2025)
     - IMRPhenomXP_NRTidalv3
     - low
     - 23,138

**Data sources:**

* P1800061 — `<https://dcc.ligo.org/LIGO-P1800061/public>`_
* P1800370 (GWTC-1) — `<https://dcc.ligo.org/LIGO-P1800370/public>`_
* Wouters et al. (2025) — `<https://github.com/ThibeauWouters/neural_priors/tree/main/final_results/GW170817/bns/default>`_

**Download script:** ``jesterTOV/inference/data/gw170817/download_gw170817.py``

The script downloads the raw ``.dat.gz`` and ``.hdf5`` files from the LIGO Document Control
Center, converts detector-frame masses to source frame using
``bilby.gw.conversion.luminosity_distance_to_redshift``, and saves the four-parameter
excerpts as ``.npz`` archives. Run it with:

.. code-block:: bash

   uv run python jesterTOV/inference/data/gw170817/download_gw170817.py


Below, we show the extracted posterior samples on the source-frame masses and tidal deformabilities for the GWTC-1 posterior (data) and samples generated from the normalizing flow trained on these samples. 

.. plot:: overview/likelihoods/gw_corner_gw170817.py

   Corner plot for GW170817 (GWTC-1, low-spin prior). Blue shows the original
   posterior samples; red shows samples drawn from the trained normalizing flow.
   The diagonal panels display marginal KDE curves; the off-diagonal panels show
   68% (darker) and 90% (lighter) filled credible-interval contours.

GW190425
--------

The processed files live in ``jesterTOV/inference/data/gw190425/``. Two sets of
posteriors are available: the original GWTC-2 release (seven posteriors covering three
waveform models and two spin-prior choices each, plus one from IMRPhenomXP_NRTidalv3),
and the updated GWTC-2.1 release (cosmological and non-cosmological variants from
Zenodo record 6513631).

**GWTC-2 posteriors** (P2000026; `GWOSC GWTC-2.1 v1 <https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW190425/v1>`_):

.. list-table::
   :header-rows: 1
   :widths: 42 25 18 15

   * - File
     - Waveform
     - Spin prior
     - Samples
   * - ``gw190425_phenompnrt-ls_posterior.npz``
     - IMRPhenomPNRT
     - low spin
     - 46,709
   * - ``gw190425_phenompnrt-hs_posterior.npz``
     - IMRPhenomPNRT
     - high spin
     - 226,598
   * - ``gw190425_phenomdnrt-ls_posterior.npz``
     - IMRPhenomDNRT
     - low spin
     - 53,945
   * - ``gw190425_phenomdnrt-hs_posterior.npz``
     - IMRPhenomDNRT
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
   * - ``gw190425_xp_nrtv3.npz``
     - IMRPhenomXP_NRTidalv3
     - low
     - 25,715

**GWTC-2.1 posteriors** (Zenodo 6513631; `GWOSC GWTC-2.1 v3 <https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW190425/v3/>`_).
The GWTC-2.1 release provides two variants of the mixed posterior: ``cosmo`` applies
cosmological corrections when converting to source-frame quantities (masses already in
source frame), while ``nocosmo`` does not (masses converted from detector frame using
the luminosity distance). For each variant, posteriors from IMRPhenomPv2_NRTidal with
both low-spin and high-spin priors are included:

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

**Data sources:**

* P2000026 — `<https://dcc.ligo.org/LIGO-P2000026/public>`_
* Zenodo 6513631 — `<https://zenodo.org/records/6513631>`_
* Wouters et al. (2025) — `<https://github.com/ThibeauWouters/neural_priors/tree/main/final_results/GW190425/bns/default>`_

**Download script:** ``jesterTOV/inference/data/gw190425/download_gw190425.py``

The script downloads both the GWTC-2 DCC file and the GWTC-2.1 Zenodo files (controlled
by the ``DOWNLOAD_GWTC2P1`` constant at the top). Run it with:

.. code-block:: bash

   uv run python jesterTOV/inference/data/gw190425/download_gw190425.py

.. plot:: overview/likelihoods/gw_corner_gw190425.py

   Corner plot for GW190425 (IMRPhenomPNRT, low-spin prior). Blue shows the
   original posterior samples; red shows samples drawn from the trained
   normalizing flow.

----

.. admonition:: Current scope: BNS only
   :class: important

   JESTER currently provides normalizing flows only for **binary neutron star (BNS)**
   events (GW170817 and GW190425). While neutron star-black hole (NSBH) mergers such a also carry tidal information,
   they are usually quite uninformative when it comes to the tidal posteriors. Such events are not supported yet, but might be in a future update.