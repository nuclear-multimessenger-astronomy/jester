JESTER Documentation
====================

JAX-accelerated equation of state inference and TOV solvers

``jester`` is a package to perform inference on the equation of state (EOS) with Bayesian inference and accelerates the TOV solver calls and the entire sampling procedure by using GPU hardware through ``jax``. 

Currently, ``jester`` supports the following EOS parametrizations:

* **Metamodel**
* **Metamodel + speed-of-sound**
* **Metamodel+peakCSE** 
* **Spectral expansion**

``jester`` can supports the following TOV solver:

* **GR**
* **GR modified with pressure anisotropy**
* **Scalar-tensor theory of gravity** (work in progress!)

Moreover, the following samplers are supported:

* **Sequential Monte Carlo** 
* **Nested sampling**
* **flowMC** 


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
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples 

.. toctree::
   :maxdepth: 2
   :caption: Inference Guide

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
   inference_documentation_guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

