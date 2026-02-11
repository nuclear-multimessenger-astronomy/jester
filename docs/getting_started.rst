Welcome to JESTER! This guide introduces the core concepts and components of the library.

Getting Started
---------------

TO MOVE

Currently, jester supports the following EOS parametrizations:

Metamodel: Taylor expansion of the energy density.

Metamodel+CSE: Metamodel up to breakdown density (varied on-the-fly), and speed-of-sound extrapolation above the breakdown density parametrized by linear interpolation through a grid of speed of sound values.

Metamodel+peakCSE: Metamodel up to breakdown density (varied on-the-fly), and speed-of-sound extrapolation above the breakdown density parametrized to have a Gaussian peak.

Spectral expansion: 4-parameter spectral expansion from Lindblom 2010

Moreover, the following samplers are supported:

Sequential Monte Carlo (Recommended): Implemented with blackjax

Nested sampling: Implemented in blackjax in this specific fork

flowMC (GitHub): Normalizing flow-enhanced MCMC sampling

**Equation of State (EOS) Models**
   Overview of EOS parametrizations available in JESTER.

   :doc:`getting_started/eos`

**TOV Solvers**
   Introduction to Tolman-Oppenheimer-Volkoff equation solvers.

   :doc:`getting_started/tov_solvers`

**Samplers**
   Bayesian sampling methods for EOS inference.

   :doc:`getting_started/samplers`

**Likelihood Constraints**
   Observational constraints from multi-messenger astronomy.

   :doc:`getting_started/likelihoods`

.. toctree::
   :hidden:

   getting_started/eos
   getting_started/tov_solvers
   getting_started/samplers
   getting_started/likelihoods
