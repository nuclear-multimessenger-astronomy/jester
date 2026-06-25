``jesterTOV.inference.run_inference`` module
============================================

.. currentmodule:: jesterTOV.inference.run_inference

Main entry point for running Bayesian inference on neutron star equation of state parameters.

This module provides the complete inference pipeline from configuration loading through sampling to result generation.

Functions
---------

.. autosummary::
   :toctree: _autosummary/

   determine_keep_names
   setup_prior
   setup_transform
   setup_likelihood
   run_sampling
   generate_eos_samples
   main
   cli_entry_point
