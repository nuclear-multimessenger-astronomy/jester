``jesterTOV.inference.samplers`` module
========================================

.. currentmodule:: jesterTOV.inference.samplers


MCMC and nested sampling algorithms for Bayesian inference.

Submodules
----------

The samplers taken from ``blackjax`` are implemented in a separate submodule. 
This contains the sequential Monte Carlo sampler, with Gaussian random walk MCMC kernel and NUTS MCMC kernel (note: NUTS is experimental), and the ``blackjax`` nested sampler with acceptance walk method.

Detailed documentations can be found in the following pages:

.. toctree::
   :maxdepth: 1

   jesterTOV.inference.samplers.blackjax

Sampler Classes
---------------

These refer to the base class (``JesterSampler``) and the output class (``SamplerOutput``) for all samplers implemented in ``jesterTOV.inference.samplers``.
Moreover, the ``flowMC`` sampler is documented here as well.

.. autosummary::
   :toctree: _autosummary

   jester_sampler.JesterSampler
   jester_sampler.SamplerOutput
   flowmc.FlowMCSampler
