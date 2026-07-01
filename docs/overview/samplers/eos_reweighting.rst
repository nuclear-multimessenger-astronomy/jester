.. _sampler-eos-reweighting:

EOS Reweighting
===============

Unlike the other samplers on this page, EOS reweighting does not sample a parametric equation-of-state model at all. Instead, it takes a fixed, discrete set of tabulated EOS curves: for example, a prior sample produced elsewhere, or a family of curves from a nuclear-theory calculation, or even a non-parametric method such as Gaussian processes.
Given these EOSs, we can evaluate jester's GPU-accelerated likelihoods (GW, NICER, radio, :math:`\chi`\ EFT, ...) directly on each curve. The result is a discrete posterior over the input EOS set. 

This is quite similar (and in fact, heavily inspired by) the ``lwp`` EOS reweighting pipeline (available on the LIGO GitLab https://git.ligo.org/reed.essick/lwp), but using the ``jester`` functionalities, bridging between ``jester`` and any external EOS-generation pipelines. Any codebase that can produce a set of :math:`(M, \Lambda, R)` curves can have those curves scored by ``jester``'s likelihood stack without needing to reimplement, e.g., the GW and NICER likelihoods.

How it works
=============

An input NPZ file supplies a set of :math:`N` EOS curves as arrays of mass, radius and tidal deformability. All curves are resampled onto a common mass grid spanning ``m_min`` to ``m_max`` (or, if ``m_max`` is not given, the maximum TOV mass across the input set, capped at 3 solar masses), with :math:`\Lambda` and :math:`R` set to zero above each curve's own maximum mass. The combined likelihood is then evaluated once per curve using :func:`jax.lax.map` with a fixed batch size, with progress logged after each batch. Given the per-curve log-likelihoods, and assuming uniform prior weights, the effective sample size and evidence are computed as well, which allow us to compare different EOS sets.

Configuration
-------------

.. code-block:: yaml

   sampler:
     type: "eos-reweighting"
     eos_file: "path/to/eos.npz"    # NPZ with keys: masses, lambdas, radii
     n_grid: 200                    # mass-grid points (default: 200)
     m_min: 1.0                     # lower mass bound in M_sun (default: 0.5)
     m_max: null                    # upper bound; null -> max(M_TOV) across curves, capped at 3.0 M_sun
     batch_size: 1000                # lax.map batch size (default: 1000); tune to fit memory requirements
     output_dir: "outdir/eos_reweighting/"

Because the EOS is supplied as tabulated curves rather than sampled, this sampler does not require an ``eos``, ``tov``, or ``prior`` section in the YAML config — see the dedicated top-level schema :class:`~jesterTOV.inference.config.schemas.eos_reweighting.EOSReweightingInferenceConfig`. Full field-by-field documentation, including the expected NPZ layout and the ``result.h5`` output structure, is in the :doc:`YAML reference </inference/yaml_reference>`.

A complete working example, including a notebook walking through the reweighting workflow, is in ``examples/inference/reweighting/``.

API reference
-------------

* :class:`jesterTOV.inference.samplers.eos_reweighting.EOSReweightingSampler`
* :class:`~jesterTOV.inference.config.schema.EOSReweightingConfig`
