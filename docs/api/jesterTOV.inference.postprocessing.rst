``jesterTOV.inference.postprocessing`` module
=============================================

.. currentmodule:: jesterTOV.inference.postprocessing.postprocessing

Visualization and analysis tools for EOS inference results.
After a sampling run completes, the postprocessing module produces a standard suite of diagnostic
and publication-quality plots from the HDF5 result file written by
:class:`~jesterTOV.inference.result.InferenceResult`.  The CLI entry point
``run_jester_postprocessing`` (see :func:`main`) can be called with the same ``config.yaml``
used for inference.

Configuration
-------------

Postprocessing behaviour is controlled through the ``postprocessing`` block of the inference
configuration file, parsed into :class:`~jesterTOV.inference.config.schema.PostprocessingConfig`.

Data Loading
------------

.. autosummary::
   :toctree: _autosummary/

   load_eos_data
   load_prior_data
   load_injection_eos

Plotting Functions
------------------

The main entry point is :func:`generate_all_plots`, which calls each individual plot function in
sequence.  Individual functions can also be called directly for custom workflows.

.. autosummary::
   :toctree: _autosummary/

   generate_all_plots
   make_cornerplot
   make_mass_radius_plot
   make_mass_lambda_plot
   make_mass_lambda_ratio_plot
   make_pressure_density_plot
   make_cs2_plot
   make_parameter_histograms
   make_contour_radii_plot
   make_contour_pressures_plot

Utilities
---------

.. autosummary::
   :toctree: _autosummary/

   setup_matplotlib
   report_credible_interval
   run_from_config
