JESTER documentation
====================

*JAX-accelerated equation of state inference and TOV solvers*

``jester`` performs fast and accurate Bayesian inference on the neutron star equation of state (EOS) using GPU-accelerated TOV solvers and samplers through ``jax``.

Try it out!
============

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/nuclear-multimessenger-astronomy/jester/blob/main/examples/google_colab/GW170817_Google_Colab.ipynb
   :alt: Open In Colab

Try ``jester`` right away in your browser and infer the neutron star equation of state from GW170817 within 15 minutes — no installation required!

Want to run locally? Check the installation instructions below.


What's in ``jester``?
=======================

``jester`` combines flexible EOS parametrizations with GPU-accelerated TOV solvers and modern Bayesian samplers.
See the :doc:`overview/` guide for detailed explanations about what is implemented in the code.

.. grid:: 2
    :class-container: component-grid

    .. grid-item:: 🧮 :doc:`EOS models <overview/eos>`

       Parametrize the equation of state of neutron star matter

       - :doc:`Metamodel <overview/eos/metamodel>`
       - :doc:`Metamodel + CSE <overview/eos/metamodel_cse>`
       - :doc:`Spectral expansion <overview/eos/spectral>`

    .. grid-item:: ⚙️ :doc:`TOV solvers <overview/tov_solvers>`

       Integrate neutron star structure equations

       - :doc:`General Relativity <overview/tov/gr>`
       - :doc:`Pressure anisotropy <overview/tov/anisotropy>`
       - :doc:`Modified gravity theories <overview/tov/scalar_tensor>`

    .. grid-item:: 🔭 :doc:`Likelihood constraints <overview/likelihoods>`

       Multi-messenger astronomical observations

       - :doc:`Nuclear experiments (chiEFT) <overview/likelihoods/chieft>`
       - :doc:`Radio timing (mass measurements) <overview/likelihoods/radio>`
       - :doc:`NICER mass-radius <overview/likelihoods/nicer>`
       - :doc:`Gravitational waves <overview/likelihoods/gw>`

    .. grid-item:: 📊 :doc:`Samplers <overview/samplers>`

       GPU-accelerated Bayesian inference

       - :doc:`Sequential Monte Carlo <overview/samplers/smc>`
       - :doc:`Nested Sampling <overview/samplers/nested_sampling>`
       - :doc:`FlowMC (normalizing flows) <overview/samplers/flowmc>` 


Curious for more?
==================

* Check out the :doc:`examples/eos_tov/eos_tov` to get familiar with using ``jester`` for generating EOSs and solving TOV equations.
* Get familiar with the Bayesian inference workflow, starting from the :doc:`inference/quickstart` guide.
* Dive into the code itself in the API reference of :doc:`api/jesterTOV`

Installation
=============

``jester`` depends on a `specific fork of blackjax <https://github.com/handley-lab/blackjax>`_ for nested sampling support,
which prevents publishing to PyPI. Instead, install the latest version by cloning the repository::

    git clone https://github.com/nuclear-multimessenger-astronomy/jester

We recommend using ``uv`` for managing the Python environment and installing the package. Once ``uv`` is installed, create a virtual environment e.g. as follows::

   uv venv --python=3.12            # Specify the Python version, if desired (optional)
   source .venv/bin/activate        # Activate the virtual environment to access the installed dependencies 

The package can then be installed directly or in editable mode in case you want to modify the code or contribute to the development of ``jester``::

    cd jester
    uv pip install -e .             # Basic install for the core functionality (CPU-only)

Extra dependencies can be installed as follows::

    uv pip install -e ".[cuda12]"   # For GPU support (fast sampling)
    uv pip install -e ".[dev]"      # For developers (work on documentation, run tests,...)

Or using ``uv`` also as follows::

    uv sync
    uv sync --extra cuda12   # For GPU support (fast sampling)
    uv sync --extra dev      # For developers (work on documentation, run tests,...)

To run Bayesian inference, make sure to install support for CUDA or upgrade ``jax`` according to the
`jax documentation page <https://docs.jax.dev/en/latest/installation.html>`_.
This should work fine by specifying the appropriate extra (``.[cuda12]``) when installing with ``uv``.

Having trouble? Check out the :doc:`developer_guide/faq`.


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Overview

   overview/eos
   overview/tov_solvers
   overview/likelihoods
   overview/samplers

.. toctree::
   :maxdepth: 2
   :caption: Basic examples

   examples/eos_tov/eos_tov
   examples/eos_tov/automatic_differentiation
   examples/eos_tov/prior_predictive

.. toctree::
   :maxdepth: 2
   :caption: Bayesian inference guide

   inference/quickstart
   inference/yaml_reference
   inference/workflow
   inference/analyze_bns
   inference/training_flows
   examples/inference/result

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/jesterTOV

.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   developer_guide/faq
   developer_guide/adding_new_eos
   developer_guide/adding_new_tov
   developer_guide/adding_new_likelihood
   developer_guide/documentation_guide

.. toctree::
   :maxdepth: 2
   :caption: Miscellaneous

   citing
   acknowledgements

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

