JESTER Documentation
====================

JAX-accelerated equation of state inference and TOV solvers

``jester`` performs Bayesian inference on neutron star equations of state using GPU-accelerated TOV solvers through ``jax``.

What's in JESTER?
-----------------

JESTER combines flexible EOS parametrizations with GPU-accelerated TOV solvers and modern Bayesian samplers.
See the :doc:`overview` guide for detailed explanations.

.. grid:: 2
    :class-container: component-grid

    .. grid-item:: 🧮 :doc:`EOS Models <overview/eos>`

       Parametrize the equation of state of nuclear matter

       - :doc:`Metamodel <overview/eos/metamodel>`
       - :doc:`Metamodel + CSE <overview/eos/metamodel_cse>`
       - :doc:`Spectral expansion <overview/eos/spectral>`

    .. grid-item:: ⚙️ :doc:`TOV Solvers <overview/tov_solvers>`

       Integrate neutron star structure equations

       - :doc:`General Relativity <overview/tov/gr>`
       - :doc:`Modified gravity theories <overview/tov/scalar_tensor>`
       - :doc:`Pressure anisotropy <overview/tov/anisotropy>`

    .. grid-item:: 🔭 :doc:`Likelihood Constraints <overview/likelihoods>`

       Multi-messenger astronomical observations

       - :doc:`Gravitational waves <overview/likelihoods/gw>`
       - :doc:`NICER mass-radius <overview/likelihoods/nicer>`
       - :doc:`Radio timing (mass measurements) <overview/likelihoods/radio>`
       - :doc:`Nuclear experiments (chiEFT) <overview/likelihoods/chieft>`

    .. grid-item:: 📊 :doc:`Samplers <overview/samplers>`

       GPU-accelerated Bayesian inference

       - :doc:`Sequential Monte Carlo <overview/samplers/smc>`
       - :doc:`Nested Sampling <overview/samplers/nested_sampling>`
       - :doc:`FlowMC (normalizing flows) <overview/samplers/flowmc>` 


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
    uv sync --extra dev      # For developers (work on documentation, run tests,...)


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Overview

   overview

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

