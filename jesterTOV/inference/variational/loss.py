r"""Reverse-KL (ELBO) loss for neural variational inference.

The variational family :math:`Q(\lambda; \phi)` (a normalizing flow) is fit to an
unnormalized target density :math:`P(\lambda) \propto \mathrm{prior}(\lambda) \cdot
\mathrm{likelihood}(\lambda)` by minimizing the reverse KL divergence

.. math::

    \mathcal{L}(\phi) = \mathbb{E}_{\lambda \sim Q(\cdot;\phi)}
        \left[ \log Q(\lambda;\phi) - \log P(\lambda) \right],

estimated via reparameterized Monte Carlo samples from the flow itself
(:func:`~flowjax.distributions.AbstractDistribution.sample_and_log_prob`). This mirrors
``gwax/gwax/variational.py::trainer``'s inner ``loss_fn``.
"""

from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray

LogTargetFn = Callable[[Array], Float[Array, " batch"]]


def make_elbo_loss_fn(
    static: Any,
    log_target_fn: LogTargetFn,
    batch_size: int,
) -> Callable[[Any, PRNGKeyArray], Float]:
    """Build the reverse-KL loss closure used inside the training scan.

    Parameters
    ----------
    static : Any
        The "static" half of an ``equinox.partition`` split of the flow (the
        non-trainable pytree leaves, e.g. architecture/bijection structure).
    log_target_fn : Callable[[Array], Array]
        Function mapping a batch of samples, shape ``(batch_size, dim)``, to their
        unnormalized log target density, shape ``(batch_size,)``. Built by
        :func:`jesterTOV.inference.variational.targets.make_prior_likelihood_target`
        or :func:`jesterTOV.inference.variational.targets.make_flow_prior_target`.
    batch_size : int
        Number of reparameterized samples drawn from the flow per gradient step.

    Returns
    -------
    Callable[[Any, PRNGKeyArray], Float]
        ``loss_fn(params, key) -> scalar loss``, where ``params`` is the trainable
        half of the same ``equinox.partition`` split as ``static``.
    """

    def loss_fn(params: Any, key: PRNGKeyArray) -> Float:
        flow = eqx.combine(params, static)
        samples, log_flows = flow.sample_and_log_prob(key, (batch_size,))
        log_targets = log_target_fn(samples)
        return jnp.mean(log_flows - log_targets)

    return loss_fn
