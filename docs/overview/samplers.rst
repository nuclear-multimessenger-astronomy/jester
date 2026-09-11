.. _overview-samplers:

Samplers
========

JESTER provides modern Bayesian sampling algorithms optimized for EOS inference with JAX acceleration. All samplers support GPU hardware and automatic differentiation.

Sequential Monte Carlo (SMC)
-----------------------------

Adaptive tempering with a Gaussian Random Walk kernel. Recommended default.

:doc:`samplers/smc`

Nested Sampling (NS-AW)
------------------------

Acceptance Walk variant for evidence computation and parameter estimation.

:doc:`samplers/nested_sampling`

EOS Reweighting
----------------

Evaluates jester's likelihoods on a fixed, tabulated set of EOS curves via importance sampling, rather than sampling a parametric EOS model.

:doc:`samplers/eos_reweighting`

.. toctree::
   :hidden:

   samplers/smc
   samplers/nested_sampling
   samplers/eos_reweighting
