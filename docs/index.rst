JESTER Documentation
====================

.. image:: https://img.shields.io/badge/arXiv-2504.15893-b31b1b.svg
   :target: https://arxiv.org/abs/2504.15893
   :alt: arXiv Paper

.. image:: https://img.shields.io/badge/arXiv-2507.13039-b31b1b.svg
   :target: https://arxiv.org/abs/2507.13039
   :alt: arXiv Paper

JAX-accelerated equation of state inference and TOV solvers

``jester`` is a package to perform inference on the equation of state (EOS) with Bayesian inference and accelerates the TOV solver calls and the entire sampling procedure by using GPU hardware through ``jax``. 

Currently, ``jester`` supports the following EOS parametrizations:

* **Metamodel**: Taylor expansion of the energy density.
* **Metamodel+CSE**: Metamodel up to breakdown density (varied on-the-fly), and speed-of-sound extrapolation above the breakdown density parametrized by linear interpolation through a grid of speed of sound values.
* **Metamodel+peakCSE**: Metamodel up to breakdown density (varied on-the-fly), and speed-of-sound extrapolation above the breakdown density parametrized to have a Gaussian peak.
* **Spectral expansion**: 4-parameter spectral expansion from Lindblom 2010

Moreover, the following samplers are supported:

* **Sequential Monte Carlo** (Recommended): Implemented with `blackjax <https://github.com/blackjax-devs/blackjax>`_
* **Nested sampling**: Implemented in ``blackjax`` in `this specific fork <https://github.com/handley-lab/blackjax>`_
* **flowMC** (`GitHub <https://github.com/kazewong/flowMC>`_): Normalizing flow-enhanced MCMC sampling


Installation
=============

The latest stable release version can be installed with ``pip``::

    pip install jesterTOV

To run Bayesian inference, make sure to install support for CUDA or upgrade ``jax`` according to the
`jax documentation page <https://docs.jax.dev/en/latest/installation.html>`_::

    pip install "jax[cuda12]"

For developers, we recommend installing locally with ``uv``::

    git clone https://github.com/nuclear-multimessenger-astronomy/jester
    cd jester
    uv sync

Extra dependencies can be installed as follows::

    uv sync --extra cuda12   # For GPU support (fast sampling)
    uv sync --extra docs     # To work on documentation locally
    uv sync --extra dev      # To run tests locally


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   inference_index
   inference_quickstart
   inference
   citing

.. toctree::
   :maxdepth: 2
   :caption: Reference

   inference_yaml_reference

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/jesterTOV

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   inference_workflow
   inference_architecture
   inference_documentation_guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

