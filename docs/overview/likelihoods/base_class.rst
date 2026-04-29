.. _likelihood-base_class:

Base likelihood class
======================

All likelihoods in JESTER inherit from :class:`jesterTOV.inference.base.likelihood.LikelihoodBase`,
an abstract base class that defines the common interface. Each likelihood stores its observational
data internally at construction time, rather than receiving it at evaluation time. The single method
every likelihood must implement is ``evaluate``:

.. code-block:: python

    def evaluate(self, params: dict[str, Float]) -> Float:
        """Return the log-likelihood for the given parameter dictionary."""
        ...

The ``params`` dictionary maps parameter names (e.g. ``"K_sat"``, ``"L_sym"``) to their current
values as JAX scalars. The method must return a single scalar log-likelihood value.

The base class also provides ``model`` and ``data`` properties that concrete subclasses populate
during ``__init__``. These give uniform access to the theoretical model and the stored
observational data, which is useful for debugging and postprocessing.

In practice you do not instantiate likelihoods directly: the factory function
``jesterTOV.inference.likelihoods.factory.create_likelihoods`` reads the inference configuration
and returns the appropriate :class:`~jesterTOV.inference.likelihoods.combined.CombinedLikelihood`
that wraps all enabled likelihoods.

Further resources
-----------------

* API reference: :class:`jesterTOV.inference.base.likelihood.LikelihoodBase`
* Config class for usage in Bayesian inference workflows: :class:`jesterTOV.inference.config.schema.LikelihoodConfig`
