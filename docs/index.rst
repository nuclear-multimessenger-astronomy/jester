JESTER Documentation
====================

JAX-accelerated equation of state inference and TOV solvers

``jester`` performs Bayesian inference on neutron star equations of state using GPU-accelerated TOV solvers through ``jax``.

What's in JESTER?
-----------------

JESTER combines flexible EOS parametrizations with GPU-accelerated TOV solvers and modern Bayesian samplers.
See the :doc:`getting_started` guide for detailed explanations.

.. grid:: 2
    :class-container: component-grid

    .. grid-item:: 🧮 :doc:`EOS Models <getting_started/eos>`

       Parametrize the equation of state of nuclear matter

       - :doc:`Metamodel <getting_started/eos/metamodel>`
       - :doc:`Metamodel + CSE <getting_started/eos/metamodel_cse>`
       - :doc:`Spectral expansion <getting_started/eos/spectral>`

    .. grid-item:: ⚙️ :doc:`TOV Solvers <getting_started/tov_solvers>`

       Integrate neutron star structure equations

       - :doc:`General Relativity <getting_started/tov/gr>`
       - :doc:`Modified gravity theories <getting_started/tov/scalar_tensor>`
       - :doc:`Pressure anisotropy <getting_started/tov/anisotropy>`

    .. grid-item:: 🔭 :doc:`Likelihood Constraints <getting_started/likelihoods>`

       Multi-messenger astronomical observations

       - :doc:`Gravitational waves <getting_started/likelihoods/gw>`
       - :doc:`NICER mass-radius <getting_started/likelihoods/nicer>`
       - :doc:`Radio timing (mass measurements) <getting_started/likelihoods/radio>`
       - :doc:`Nuclear experiments (chiEFT) <getting_started/likelihoods/chieft>`

    .. grid-item:: 📊 :doc:`Samplers <getting_started/samplers>`

       GPU-accelerated Bayesian inference

       - :doc:`Sequential Monte Carlo <getting_started/samplers/smc>`
       - :doc:`Nested Sampling <getting_started/samplers/nested_sampling>`
       - :doc:`FlowMC (normalizing flows) <getting_started/samplers/flowmc>` 


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

