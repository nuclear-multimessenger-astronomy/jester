``jesterTOV.inference.config`` module
=====================================

.. currentmodule:: jesterTOV.inference.config


Configuration parsing and validation for Bayesian inference runs.

Schema Classes
--------------

.. autosummary::
   :toctree: _autosummary

   schema.InferenceConfig
   schema.TransformConfig
   schema.PriorConfig
   schema.LikelihoodConfig
   schema.BaseSamplerConfig
   schema.FlowMCSamplerConfig
   schema.BlackJAXNSAWConfig
   schema.SMCRandomWalkSamplerConfig
   schema.SMCNUTSSamplerConfig
   schema.PostprocessingConfig

Parser Functions
----------------

.. autosummary::
   :toctree: _autosummary

   parser.load_config
