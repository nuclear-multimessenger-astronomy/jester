JESTER Documentation
====================

JAX-accelerated equation of state inference and TOV solvers

``jester`` performs Bayesian inference on neutron star equations of state using GPU-accelerated TOV solvers through ``jax``.

.. note::

   **Documentation is work in progress!** Some sections may be incomplete or under active development. We appreciate your patience as we improve the documentation.

What's in JESTER?
-----------------

JESTER combines flexible EOS parametrizations with GPU-accelerated TOV solvers and modern Bayesian samplers.
See the :doc:`overview/` guide for detailed explanations about what is implemented in the code.

.. grid:: 2
    :class-container: component-grid

    .. grid-item:: 🧮 :doc:`EOS models <overview/eos>`

       Parametrize the equation of state of nuclear matter

       - :doc:`Metamodel <overview/eos/metamodel>`
       - :doc:`Metamodel + CSE <overview/eos/metamodel_cse>`
       - :doc:`Spectral expansion <overview/eos/spectral>`

    .. grid-item:: ⚙️ :doc:`TOV solvers <overview/tov_solvers>`

       Integrate neutron star structure equations

       - :doc:`General Relativity <overview/tov/gr>`
       - :doc:`Modified gravity theories <overview/tov/scalar_tensor>`
       - :doc:`Pressure anisotropy <overview/tov/anisotropy>`

    .. grid-item:: 🔭 :doc:`Likelihood constraints <overview/likelihoods>`

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

``jester`` depends on a `specific fork of blackjax <https://github.com/handley-lab/blackjax>`_ for nested sampling support,
which prevents publishing to PyPI. Install the latest version by cloning the repository::

    git clone https://github.com/nuclear-multimessenger-astronomy/jester
    cd jester
    uv sync

Extra dependencies can be installed as follows::

    uv sync --extra cuda12   # For GPU support (fast sampling)
    uv sync --extra dev      # For developers (work on documentation, run tests,...)

To run Bayesian inference, make sure to install support for CUDA or upgrade ``jax`` according to the
`jax documentation page <https://docs.jax.dev/en/latest/installation.html>`_::

    uv sync --extra cuda12


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Overview

   overview/eos
   overview/tov_solvers
   overview/samplers
   overview/likelihoods

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/eos_tov/eos_tov
   examples/eos_tov/automatic_differentiation
   examples/eos_tov/eos_STtov
   examples/inference/result

.. toctree::
   :maxdepth: 2
   :caption: Inference Guide

   inference/quickstart
   inference/yaml_reference
   inference/workflow
   inference/analyze_bns

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/jesterTOV

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/adding_new_eos
   developer_guide/adding_new_tov
   developer_guide/adding_new_likelihood
   inference/documentation_guide
   citing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

