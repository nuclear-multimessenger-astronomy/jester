r"""Neural variational inference (NVI) trainer.

Fits a normalizing flow :math:`Q(\lambda;\phi)` to an unnormalized target density by
minimizing the reverse-KL loss from :mod:`jesterTOV.inference.variational.loss`, following
the training-loop structure of ``gwax/gwax/variational.py::trainer`` (``equinox.partition``
/ ``equinox.combine`` around a ``jax.lax.scan`` of ``equinox.filter_jit``-wrapped,
``equinox.filter_value_and_grad``-based update steps).

Training operates on the **raw flowjax object**, not jester's ``Flow`` wrapper — the latter
carries non-pytree ``metadata``/``flow_kwargs`` dict attributes that are unsuitable for
``equinox.partition``/``lax.scan``. Callers should wrap the returned flow into a ``Flow``
(with ``standardize=False``) only when they need to save it or use ``Flow.sample``/
``Flow.log_prob`` outside this module.

Two training modes, selected by passing exactly one of ``prior`` or ``prev_flow``:

- ``prior`` given: ordinary single-stage VI, target = ``prior x likelihood``.
- ``prev_flow`` given: sequential Bayesian update — the previous flow's density stands in
  for the prior, and the new flow is warm-started from ``prev_flow``'s trained weights
  (see :func:`sequential_vi_update`).
"""

from dataclasses import dataclass
from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array
from jaxtyping import Float, PRNGKeyArray
from paramax.wrappers import NonTrainable

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.base.prior import Prior
from jesterTOV.inference.base.transform import NtoMTransform
from jesterTOV.inference.flows.flow import create_flow
from jesterTOV.inference.variational.loss import make_elbo_loss_fn
from jesterTOV.inference.variational.targets import (
    make_flow_prior_target,
    make_prior_likelihood_target,
)


@dataclass
class ELBOTrainingResult:
    """Result of an NVI training run.

    Attributes
    ----------
    flow : Any
        The trained, raw flowjax ``AbstractDistribution``.
    losses : Array
        Per-step reverse-KL loss values, shape ``(steps,)``.
    final_params : Any
        The trainable half of the final ``equinox.partition`` split (mainly useful for
        diagnostics/inspection; ``flow`` already has these combined in).
    """

    flow: Any
    losses: Float[Array, " steps"]
    final_params: Any


def _resolve_flow(flow: Any) -> Any:
    """Unwrap a jester ``Flow`` to its underlying raw flowjax object, if needed."""
    return flow.flow if hasattr(flow, "flow") else flow


def train_vi(
    key: PRNGKeyArray,
    likelihood: LikelihoodBase,
    likelihood_transforms: Sequence[NtoMTransform],
    parameter_names: Sequence[str],
    *,
    prior: Prior | None = None,
    prev_flow: Any | None = None,
    dim: int,
    flow: Any | None = None,
    flow_kwargs: dict[str, Any] | None = None,
    batch_size: int = 512,
    steps: int = 20_000,
    learning_rate: float = 1e-3,
    use_cosine_schedule: bool = True,
    progress_every: int = 500,
) -> ELBOTrainingResult:
    """Train a normalizing flow against ``prior x likelihood`` (or a sequential-update target).

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    likelihood : LikelihoodBase
        Likelihood to target (e.g. ``MockMassRadiusLikelihood``, or a ``CombinedLikelihood``).
    likelihood_transforms : Sequence[NtoMTransform]
        Likelihood transforms applied before ``likelihood.evaluate`` (e.g. ``JesterTransform``).
    parameter_names : Sequence[str]
        Names of the sampled parameters, in the order matching ``dim``.
    prior : Prior, optional
        The prior to target. Mutually exclusive with ``prev_flow``.
    prev_flow : Any, optional
        A previously trained flow (jester ``Flow`` or raw flowjax object) whose density
        replaces the prior, for a sequential Bayesian update. Mutually exclusive with
        ``prior``. If ``flow`` is not separately given, training warm-starts from this flow's
        weights.
    dim : int
        Dimensionality of the flow (number of sampled parameters).
    flow : Any, optional
        An existing (untrained or partially trained) raw flowjax object to continue training
        from. If ``None``, a fresh flow is built via ``create_flow`` (unless ``prev_flow`` is
        given, in which case its weights are used as the warm start).
    flow_kwargs : dict, optional
        Extra kwargs forwarded to ``create_flow`` when building a fresh flow.
    batch_size : int
        Number of reparameterized flow samples per gradient step.
    steps : int
        Number of gradient steps.
    learning_rate : float
        Adam learning rate (peak learning rate if ``use_cosine_schedule``).
    use_cosine_schedule : bool
        If True, decay the learning rate from ``learning_rate`` to 0 via
        ``optax.cosine_decay_schedule`` over ``steps`` steps.
    progress_every : int
        Print the mean loss over the last ``progress_every`` steps at that cadence
        (post-hoc, read off the ``losses`` array — no in-scan callback).

    Returns
    -------
    ELBOTrainingResult
    """
    if (prior is None) == (prev_flow is None):
        raise ValueError("Exactly one of `prior` or `prev_flow` must be given.")

    flow_kwargs = {} if flow_kwargs is None else flow_kwargs

    if flow is None:
        if prev_flow is not None:
            flow = _resolve_flow(prev_flow)
        else:
            key, flow_key = jax.random.split(key)
            flow = create_flow(flow_key, dim=dim, **flow_kwargs)

    if prior is not None:
        log_target_fn = make_prior_likelihood_target(
            prior, likelihood, likelihood_transforms, parameter_names
        )
    else:
        log_target_fn = make_flow_prior_target(
            prev_flow, likelihood, likelihood_transforms, parameter_names
        )

    params, static = eqx.partition(
        flow,
        filter_spec=eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )

    loss_fn = make_elbo_loss_fn(static, log_target_fn, batch_size)

    if use_cosine_schedule:
        schedule = optax.cosine_decay_schedule(learning_rate, steps)
        optimizer = optax.adam(schedule)
    else:
        optimizer = optax.adam(learning_rate)

    state = optimizer.init(params)

    @eqx.filter_jit
    def update(carry, _step):
        step_key, step_params, step_state = carry
        step_key, subkey = jax.random.split(step_key)
        loss, grad = eqx.filter_value_and_grad(loss_fn)(step_params, subkey)
        updates, step_state = optimizer.update(grad, step_state, step_params)
        step_params = eqx.apply_updates(step_params, updates)
        return (step_key, step_params, step_state), loss

    # Run the scan in chunks of `progress_every` steps (rather than one scan over
    # all `steps`) so a Python-side print lands after each chunk. A single
    # all-at-once scan gives no feedback until the *entire* run (JIT compile +
    # all steps) finishes, which on a laptop-scale EOS+TOV forward pass can look
    # indistinguishable from a hang for the first (compiling) chunk.
    chunk_size = progress_every if progress_every > 0 else steps
    n_chunks = -(-steps // chunk_size)  # ceil division
    print(
        f"[train_vi] starting training: {steps} steps, batch_size={batch_size}, "
        f"{n_chunks} chunk(s) of up to {chunk_size} steps (first chunk includes "
        "JIT compilation and may take a while)"
    )
    loss_chunks = []
    steps_done = 0
    for _chunk_idx in range(n_chunks):
        this_chunk = min(chunk_size, steps - steps_done)
        (key, params, state), chunk_losses = jax.lax.scan(
            update, (key, params, state), jnp.arange(this_chunk)
        )
        loss_chunks.append(chunk_losses)
        steps_done += this_chunk
        print(
            f"[train_vi] steps {steps_done}/{steps}: mean loss over last "
            f"{this_chunk} steps = {float(jnp.mean(chunk_losses)):.4f}"
        )

    losses = jnp.concatenate(loss_chunks)
    trained_flow = eqx.combine(params, static)

    return ELBOTrainingResult(flow=trained_flow, losses=losses, final_params=params)


def sequential_vi_update(
    key: PRNGKeyArray,
    prev_flow: Any,
    likelihood: LikelihoodBase,
    likelihood_transforms: Sequence[NtoMTransform],
    parameter_names: Sequence[str],
    **train_vi_kwargs: Any,
) -> ELBOTrainingResult:
    """Sequentially update ``prev_flow`` on new data, warm-started from its trained weights.

    This is the neural-Bayesian-update step from papers 2504.07197 / 2602.20277: treat the
    previous flow's density as the new prior, and fit a fresh flow (initialized from the
    previous flow's weights) against ``likelihood x prev_flow`` via reverse KL. Explicit,
    discoverable wrapper around :func:`train_vi(..., prev_flow=prev_flow, ...) <train_vi>` so
    the "previous posterior becomes the new prior" substitution can't be assembled incorrectly
    by hand.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    prev_flow : Any
        The previously trained flow (jester ``Flow`` or raw flowjax object) to update.
    likelihood : LikelihoodBase
        Likelihood for the *new* data only (not the previous data — that information is
        already encoded in ``prev_flow``'s density).
    likelihood_transforms : Sequence[NtoMTransform]
        Likelihood transforms for the new likelihood.
    parameter_names : Sequence[str]
        Names of the sampled parameters, in the order matching ``prev_flow``'s dimensionality.
    **train_vi_kwargs : Any
        Forwarded to :func:`train_vi` (e.g. ``dim``, ``steps``, ``batch_size``,
        ``learning_rate``). Do not pass ``prior``, ``prev_flow``, or ``flow`` here.

    Returns
    -------
    ELBOTrainingResult
    """
    for reserved in ("prior", "prev_flow", "flow"):
        if reserved in train_vi_kwargs:
            raise ValueError(
                f"sequential_vi_update sets `{reserved}` internally; do not pass it explicitly."
            )
    train_vi_kwargs.setdefault("dim", len(parameter_names))
    return train_vi(
        key,
        likelihood,
        likelihood_transforms,
        parameter_names,
        prev_flow=prev_flow,
        **train_vi_kwargs,
    )
