.. _likelihood-gw-fisher:

Gravitational-wave Fisher forecasts (simulated BNS populations)
=================================================================

The Fisher-forecast likelihood constrains the EOS using `gwfast <https://github.com/CosmoStatGW/gwfast>`_ Fisher-matrix forecasts for a simulated population of binary neutron star (BNS) sources, rather than a trained normalizing-flow posterior for a single real event (compare :doc:`gw`). This is the relevant likelihood for forecasting studies with next-generation detectors such as the Einstein Telescope, where the analysis target is a mock catalogue of hundreds to thousands of simulated detections rather than one or two published events. Each source in such a catalogue only has a Fisher-matrix estimate of its measurement uncertainty — a marginalized (diagonal) 1-sigma error per parameter — not a sampled posterior, so :class:`~jesterTOV.inference.likelihoods.gw_fisher.GWFisherLikelihood` is built directly around that data product instead.

Likelihood
----------

For each source, the observational information is reduced to an approximate 2D Gaussian over the effective tidal deformability and the mass ratio, :math:`P(\tilde{\Lambda}, q)`, together with the source-frame chirp mass :math:`m_c`. Because :math:`q` and :math:`m_c` are determined purely by the two component masses, with no dependence on the EOS, the only place the choice of EOS can enter is through :math:`\tilde{\Lambda}`. This licenses marginalizing the likelihood over :math:`q` :cite:`Montefusco:2024xrx`:

.. math::
    :label: gw_fisher_likelihood

    \log \mathcal{L}_{\rm source}(X) = \log \int_{q_{\rm min}}^{q_{\rm max}}
        P\big(\tilde{\Lambda}_X(q, m_c),\, q\big)\, dq \, .

For a candidate EOS :math:`X` and a trial mass ratio :math:`q = m_2/m_1 \leq 1`, the component masses :math:`m_1(m_c, q) \geq m_2(m_c, q)` follow from the standard chirp-mass/mass-ratio inversion, and :math:`\Lambda_1 = \Lambda_X(m_1)`, :math:`\Lambda_2 = \Lambda_X(m_2)` are read directly off that EOS's own mass-tidal-deformability curve — for a single EOS, :math:`\Lambda` is a deterministic, one-to-one function of mass, so there is no :math:`\Lambda_1`/:math:`\Lambda_2` degeneracy at this stage. The two are combined into the usual mass-weighted effective tidal deformability, originally introduced by :cite:`Flanagan:2007ix` and refined to the 5PN-consistent form used here (including the :math:`\eta^2` correction terms) by :cite:`Wade:2014vqa`

.. math::
    :label: gw_fisher_lambda_tilde

    \tilde{\Lambda}_X(q, m_c) = \frac{8}{13} \Big[
        (1 + 7\eta - 31\eta^2)(\Lambda_1 + \Lambda_2)
        + \sqrt{1 - 4\eta}\,(1 + 9\eta - 11\eta^2)(\Lambda_1 - \Lambda_2)
    \Big] \, , \qquad \eta = \frac{q}{(1+q)^2} \, ,

tracing out a predicted :math:`\tilde{\Lambda}_X(q)` curve as :math:`q` varies. Evaluating the observed :math:`P(\tilde{\Lambda}, q)` along this curve and integrating over :math:`q` (Eq. :eq:`gw_fisher_likelihood`) tests the EOS's prediction against the data, sidestepping the underdetermined inversion of the observation into :math:`(\Lambda_1, \Lambda_2)` directly. This is the same approach used to fold the LVK :math:`P(\tilde{\Lambda}, q)` posterior for GW170817 (originally reported in :cite:`LIGOScientific:2018hze`) into an EOS constraint.

The integral is evaluated numerically as a trapezoidal quadrature over a fixed, deterministic mass-ratio grid built once at initialization from ``q_min``, ``q_max``, and ``dq`` — not by Monte Carlo averaging over pre-sampled points, unlike :class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood` and :class:`~jesterTOV.inference.likelihoods.mock_mr.MockMassRadiusLikelihood`. There is therefore no random seed and no sampling noise involved in evaluating a single source's contribution. The total log-likelihood sums the per-source contributions over every retained source (independent events).

Per-source Gaussian fit
------------------------

Each source's :math:`P(\tilde{\Lambda}, q)` is approximated as a 2D Gaussian, built analytically from the Fisher errors reported for that source. gwfast fits :math:`\tilde{\Lambda}` directly as its own waveform parameter, so an ``err_LambdaTilde`` Fisher error is available natively; it does not fit :math:`q` directly, so the mass-ratio variance is obtained from the reported component-mass errors ``err_m1_src``/``err_m2_src`` via standard linear error propagation on :math:`q = m_2/m_1`.

Because gwfast only stores *marginalized* (diagonal) Fisher errors — no cross term between ``err_LambdaTilde`` and ``err_m1_src``/``err_m2_src`` is ever stored — the cross-covariance :math:`\mathrm{Cov}(\tilde{\Lambda}, q)` is exactly zero under that same diagonal assumption, not merely approximately so. The fit is therefore fully closed-form: no random sampling and no extra hyperparameters are needed to build it.

Data files
----------

Two HDF5 files are required, both supplied by the user (there are no bundled defaults, unlike :doc:`gw`, since the relevant catalogue is specific to each forecasting study):

* ``gwfast_result_file``: one row per *detected* source, with ``err_LambdaTilde``, ``err_m1_src``, ``err_m2_src`` (1-sigma marginalized Fisher errors), ``idx_det_in_cat`` (index of each detected source into the injection catalog), and ``snrs`` (SNR of every *injected* source, detected or not).
* ``injection_catalog_file``: true/injected values for every injected source, with ``m1_src``, ``m2_src`` (source-frame component masses), ``Mc`` (detector-frame chirp mass), ``z`` (redshift), ``Lambda1``, ``Lambda2``, ``eta``.

``m1_src`` is assumed to be the heavier component throughout (:math:`q = m_2/m_1 \leq 1`, the standard LVK/bilby convention); :class:`~jesterTOV.inference.likelihoods.gw_fisher.GWFisherLikelihood` checks this at construction time and raises a clear error naming the offending source if it is ever violated.

Sources are filtered by SNR at construction time: only sources with SNR at or above ``snr_threshold`` are retained. This is an *additional* cut on top of whatever detection threshold is already baked into ``gwfast_result_file`` at the time the Fisher forecast was computed.

Scaling to many sources
-------------------------

Unlike :class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood`/:class:`~jesterTOV.inference.likelihoods.gw.StackedGWLikelihood`, which are built around 1-2 real events, a forecasting study can put hundreds to thousands of simulated sources through this likelihood. All sources are evaluated as a single batched computation (nested ``jax.lax.map``: an outer map over sources, an inner map over the mass-ratio grid), controlled by two independent batch-size knobs, ``source_batch_size`` and ``q_batch_size``. Both default to ``1`` (a plain sequential scan), which keeps memory flat under the outer particle ``vmap`` used by e.g. the SMC sampler; raise either for faster standalone (non-sampler) evaluation. ``source_batch_size`` is the knob that matters most here, since the source axis is the one that scales with catalogue size.

Usage
-----

.. code-block:: yaml

   likelihoods:
     - type: "gw_fisher"
       enabled: true
       gwfast_result_file: "./gwfast_results_Delta_BNS_SFHo_snrth-8.h5"
       injection_catalog_file: "./BNS_cat_5yr_LVKpop_SFHo.h5"
       snr_threshold: 12.0
       q_min: 0.4
       q_max: 1.0
       dq: 0.01

See ``examples/inference/smc_random_walk/gw_fisher/`` for a complete working configuration using a small synthetic catalogue.

Further resources
-------------------

* API reference: :class:`jesterTOV.inference.likelihoods.gw_fisher.GWFisherLikelihood`
* Config class for usage in Bayesian inference workflows: :class:`jesterTOV.inference.config.schemas.likelihoods.GWFisherLikelihoodConfig`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
