``jesterTOV.inference.samplers`` module
========================================

.. currentmodule:: jesterTOV.inference.samplers


MCMC and nested sampling algorithms for Bayesian inference.

Submodules
----------

The samplers taken from ``blackjax`` are implemented in a separate submodule.
This contains the sequential Monte Carlo sampler with a Gaussian random walk MCMC kernel, and the ``blackjax`` nested sampler with acceptance walk method.

Detailed documentations can be found in the following pages:

.. toctree::
   :maxdepth: 1

   jesterTOV.inference.samplers.blackjax

Sampler Classes
---------------

These refer to the base class (``JesterSampler``) and the output class (``SamplerOutput``) for all samplers implemented in ``jesterTOV.inference.samplers``.
Moreover, the ``EOSReweightingSampler`` (likelihood reweighting of a fixed, tabulated EOS set) is documented here as well.

.. autosummary::
   :toctree: _autosummary

   jester_sampler.JesterSampler
   jester_sampler.SamplerOutput
   eos_reweighting.EOSReweightingSampler

Sampler Functions
-----------------

.. autosummary::
   :toctree: _autosummary

   eos_reweighting.resample_eos_posterior
