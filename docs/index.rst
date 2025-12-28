JESTER Documentation
====================

**J**\ ax-based **E**\ o**S** and **T**\ ov solv\ **ER**

JESTER is a scientific computing library for solving the Tolman-Oppenheimer-Volkoff (TOV) equations for neutron star physics. It provides hardware-accelerated computations via JAX with automatic differentiation capabilities.

.. image:: https://img.shields.io/badge/arXiv-2024.12345-b31b1b.svg
   :target: https://arxiv.org/abs/2024.12345
   :alt: arXiv Paper

.. image:: https://codecov.io/gh/your-username/jester/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/your-username/jester
   :alt: Code Coverage

Key Features
============

* **JAX-Accelerated**: Hardware acceleration with GPU/TPU support
* **Automatic Differentiation**: Built-in gradients for parameter studies
* **Neutron Star Physics**: Complete TOV equation solver with realistic equations of state
* **Extensible**: Modular design for custom equation of state models
* **Well-Tested**: Comprehensive test suite with 95+ tests

Quick Start
===========

Install JESTER:

.. code-block:: bash

   pip install jester-tov
   
   # For GPU support
   pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Basic usage:

.. code-block:: python

   import jesterTOV as jtov
   
   # Create equation of state
   eos = jtov.eos.MetaModel_EOS_model()
   
   # Solve TOV equations
   masses, radii = eos.M_R_curve()
   
   print(f"Maximum mass: {max(masses):.2f} solar masses")

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   inference_index
   inference_quickstart
   inference

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/jesterTOV
   api/inference

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   inference_architecture

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

