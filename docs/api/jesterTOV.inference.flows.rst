``jesterTOV.inference.flows`` module
=====================================

.. currentmodule:: jesterTOV.inference.flows

Normalizing flow models for gravitational wave and other posterior density estimation.
JESTER uses normalizing flows to build smooth density estimates from discrete posterior samples —
for example from bilby GW analyses — so they can be evaluated at arbitrary points during inference.

Flow Model
----------

The :class:`~jesterTOV.inference.flows.flow.Flow` class is the central object: it wraps a trained
`flowjax <https://flowjax.readthedocs.io/>`_ normalizing flow and exposes a simple ``log_prob``
interface that handles data standardization automatically.

.. autosummary::
   :toctree: _autosummary/

   flow.Flow
   flow.load_model

Training
--------

Flows are trained on posterior samples (stored as ``.npz`` files) using
:func:`~jesterTOV.inference.flows.train_flow.train_flow_from_config`, which is driven by
:class:`~jesterTOV.inference.flows.config.FlowTrainingConfig`.  The CLI entry point
``jester_train_flow`` invokes :func:`~jesterTOV.inference.flows.train_flow.main` directly.

.. autosummary::
   :toctree: _autosummary/

   config.FlowTrainingConfig
   train_flow.load_posterior
   train_flow.train_flow
   train_flow.train_flow_from_config
   train_flow.save_model

Bilby Posterior Extraction
--------------------------

When starting from a bilby HDF5 result file (e.g. a published GW analysis), use
:func:`~jesterTOV.inference.flows.bilby_extract.extract_gw_posterior_from_bilby` to extract
the relevant posterior columns into a ``.npz`` file that the flow-training pipeline can consume.
The CLI entry point ``jester_extract_gw_posterior_bilby`` exposes this function without requiring
a full bilby installation.

.. autosummary::
   :toctree: _autosummary/

   bilby_extract.extract_gw_posterior_from_bilby
