.. _likelihood-nicer:

NICER constraints
=================

.. note::

   This page is a placeholder. The detailed content below is to be written soon.

NICER (Neutron star Interior Composition Explorer) is a NASA X-ray telescope on the
International Space Station that measures X-ray pulse profiles from millisecond pulsars.
By modelling the anisotropic X-ray emission from hot spots on the neutron star surface,
NICER constrains the stellar mass and radius simultaneously, providing direct input for
equation-of-state inference.

JESTER currently supports four pulsars observed by NICER:

* **PSR J0030+0451**
* **PSR J0437-4715**
* **PSR J0614-3329**
* **PSR J0740+6620**
  analyses (Miller et al. 2021); NICER+XMM-Newton data.

The figure below shows the mass-radius posteriors for one representative analysis
per pulsar, with filled contours at the 68% and 90% credible intervals. 
Below, we provide more details on the datasets that are available for each pulsar.

.. plot:: overview/likelihoods/nicer_mr_plot.py

   Mass-radius posteriors from NICER for the four supported pulsars.
   Filled contours show the 68% (darker) and 90% (lighter) credible intervals
   from kernel density estimation of the published posterior samples.
   One representative analysis is plotted per pulsar.

----

Data
----

Mass-radius posterior samples for all supported pulsars are stored as compressed NumPy
archives (``.npz``) under ``jesterTOV/inference/data/NICER/``. Each file contains
``radius`` (km), ``mass`` (solar masses), and a ``metadata`` dictionary with the
pulsar name, analysis group, hotspot model, Zenodo record URL, and paper reference.

The raw Zenodo archives (which can be several gigabytes) are downloaded into
``jesterTOV/inference/data/NICER/zenodo_data/`` (gitignored). Only the lightweight
extracted ``.npz`` files are version-controlled.

**Download script:** ``jesterTOV/inference/data/NICER/download_nicer.py``

This single script handles the full pipeline — downloading Zenodo archives, extracting
mass-radius samples from raw text files and tar archives, and downsampling large
posteriors to at most 100,000 samples. It requires the ``zenodo-get`` package:

.. code-block:: bash

   uv pip install zenodo-get
   uv run python jesterTOV/inference/data/NICER/download_nicer.py

.. warning::

   Downloading the raw Zenodo archives can take a significant amount of time. Several
   records contain full posterior chains that are multiple gigabytes in size. Make sure
   you have sufficient disk space and a stable internet connection before running the
   script. The processed ``.npz`` files (the only outputs you need for inference) are
   already version-controlled in the repository, so you only need to run this script
   if you want to reproduce the extraction from scratch.

Zenodo record identifiers and paper metadata are maintained in
``jesterTOV/inference/data/NICER/zenodo_downloader.py``.

PSR J0030+0451
^^^^^^^^^^^^^^

First millisecond pulsar observed by NICER with sufficient quality for mass-radius inference,
analyzed independently by the Amsterdam (X-PSI) and Maryland groups from the same 2017-2018
NICER data.

**Amsterdam group — Riley et al. 2019** (`Zenodo 3524457 <https://zenodo.org/records/3524457>`_)

Five hotspot geometries are available:

* ``J00300451_amsterdam_ST_S_NICER_only_Riley2019.npz`` — ST symmetric
* ``J00300451_amsterdam_ST_U_NICER_only_Riley2019.npz`` — ST unrestricted
* ``J00300451_amsterdam_CDT_U_NICER_only_Riley2019.npz`` — CDT unrestricted
* ``J00300451_amsterdam_ST_EST_NICER_only_Riley2019.npz`` — ST+EST
* ``J00300451_amsterdam_ST_PST_NICER_only_Riley2019.npz`` — ST+PST (recommended)

**Maryland group — Miller et al. 2019** (`Zenodo 3473464 <https://zenodo.org/records/3473464>`_)

Two hotspot geometries × two prior choices:

* ``J00300451_maryland_2spot_NICER_only_RM.npz`` — 2-spot, restricted-model prior
* ``J00300451_maryland_2spot_NICER_only_full.npz`` — 2-spot, full prior
* ``J00300451_maryland_3spot_NICER_only_RM.npz`` — 3-spot, restricted-model prior
* ``J00300451_maryland_3spot_NICER_only_full.npz`` — 3-spot, full prior

PSR J0437−4715
^^^^^^^^^^^^^^

The nearest and brightest millisecond pulsar, observed with NICER only.

**Amsterdam group — Choudhury et al. 2024** (`Zenodo 13766753 <https://zenodo.org/records/13766753>`_)

Headline result using CST+PDT hotspot model with 3C50 background:

* ``J04374715_amsterdam_CST_PDT_NICER_only_Choudhury2024.npz``

PSR J0614−3329
^^^^^^^^^^^^^^

A 1.4 solar-mass edge-on pulsar observed with NICER only.

**Amsterdam group — Dittmann et al. 2025** (`Zenodo 17380576 <https://zenodo.org/records/17380576>`_)

Headline result using ST+PDT hotspot model:

* ``J06143329_amsterdam_ST_PDT_NICER_only_Dittmann2025.npz``

PSR J0740+6620
^^^^^^^^^^^^^^

The most massive millisecond pulsar with a precise radio timing mass, providing
high-density EOS constraints. Analyzed jointly with XMM-Newton data.

**Amsterdam group — Salmi et al. 2024** (`Zenodo 10519473 <https://zenodo.org/records/10519473>`_)

Most recent analysis using gamma hotspot model with NICER+XMM-Newton data:

* ``J07406620_amsterdam_gamma_NICERXMM_equal_weights_recent.npz``

**Maryland group — Miller et al. 2021** (`Zenodo 4670689 <https://zenodo.org/records/4670689>`_)

Three dataset combinations x two prior choices:

* ``J07406620_maryland_unknown_NICER_only_RM.npz``
* ``J07406620_maryland_unknown_NICER_only_full.npz``
* ``J07406620_maryland_unknown_NICERXMM_RM.npz``
* ``J07406620_maryland_unknown_NICERXMM_full.npz``
* ``J07406620_maryland_unknown_NICERXMM_relative_RM.npz``
* ``J07406620_maryland_unknown_NICERXMM_relative_full.npz``

"NICER+XMM" indicates joint analysis; "relative" includes relative calibration
between instruments; "RM" uses a restricted-model prior.

Loading the data
^^^^^^^^^^^^^^^^

Each ``.npz`` file can be loaded directly:

.. code-block:: python

   import numpy as np

   data = np.load("NICER/J07406620_amsterdam_gamma_NICERXMM_equal_weights_recent.npz",
                  allow_pickle=True)
   radius = data["radius"]   # km, shape: (n_samples,)
   mass   = data["mass"]     # solar masses
   meta   = data["metadata"].item()   # dict: psr, group, zenodo_record, paper, …

In normal usage the likelihood classes load the data automatically based on your
YAML configuration.
