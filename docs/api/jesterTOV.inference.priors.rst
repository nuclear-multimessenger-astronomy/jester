``jesterTOV.inference.priors`` module
======================================

.. currentmodule:: jesterTOV.inference.priors

Prior specification and parsing for Bayesian inference.

Prior Classes
-------------

Prior distributions are defined in the :mod:`jesterTOV.inference.base.prior` module
and used when writing ``.prior`` files for inference runs.

.. currentmodule:: jesterTOV.inference.base.prior

.. autosummary::
   :toctree: _autosummary

   UniformPrior
   MultivariateGaussianPrior
   Fixed
   CombinePrior
   Prior

Parser Functions
----------------

.. currentmodule:: jesterTOV.inference.priors

.. autosummary::
   :toctree: _autosummary

   parser.parse_prior_file
   parser.ParsedPrior
