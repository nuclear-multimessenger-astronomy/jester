.. _overview-samplers:

Samplers
========

JESTER provides modern Bayesian sampling algorithms optimized for EOS inference with JAX acceleration. All samplers support GPU hardware and automatic differentiation.

Sequential Monte Carlo (SMC)
-----------------------------

Adaptive tempering with Random Walk or NUTS kernels. Recommended default.

:doc:`samplers/smc`

Nested Sampling (NS-AW)
------------------------

Acceptance Walk variant for evidence computation and parameter estimation.

:doc:`samplers/nested_sampling`

FlowMC
------

Normalizing flow-enhanced MCMC for efficient exploration of complex posteriors.

:doc:`samplers/flowmc`

.. toctree::
   :hidden:

   samplers/smc
   samplers/nested_sampling
   samplers/flowmc
