r"""Batched log density of a masked autoregressive flow with rational-quadratic-spline transformers.

flowjax evaluates ``flow.log_prob`` one point at a time; under ``vmap`` XLA then stores the spline parameters
point-major, uses a transposed matrix product for the network's output layer, and finds the spline bin with a
``searchsorted`` loop and a gather. For the GW likelihood (5000 particles x 500 mass samples per call) that is most of
the cost. :func:`batched_log_prob` computes the same quantity for a whole batch with every intermediate laid out as
``[features, points]``: the network becomes plain matrix products ``W @ H``, and the bin is found by counting knots
below the input and read with a one-hot sum, which fuse into elementwise kernels. The arithmetic of the spline is
flowjax's, so the result agrees with ``flow.log_prob`` to round-off.

Supported: ``Transformed(base, Invert(Scan(Chain([MaskedAutoregressive, Permute]))))`` with an unconditional
:class:`flowjax.bijections.RationalQuadraticSpline` transformer and ReLU activations, i.e. what
``create_flow(flow_type="masked_autoregressive_flow", transformer_type="rational_quadratic_spline", invert=True)``
builds for ``dim >= 3``. Check with :func:`supports_batched_log_prob` first.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import paramax
from flowjax.bijections import (
    Chain,
    Invert,
    MaskedAutoregressive,
    Permute,
    RationalQuadraticSpline,
    Scan,
)
from flowjax.distributions import AbstractDistribution, Transformed
from jaxtyping import Array, Bool, Float


def _layer_parts(flow: Transformed) -> tuple[Scan, int]:
    scan = flow.bijection.bijection  # type: ignore[attr-defined]
    n_layers = jax.tree_util.tree_leaves(eqx.filter(scan.bijection, eqx.is_array))[0].shape[0]
    return scan, n_layers


def supports_batched_log_prob(flow: AbstractDistribution) -> bool:
    """Whether :func:`batched_log_prob` reproduces ``flow.log_prob`` for this flow."""
    if not isinstance(flow, Transformed):
        return False
    bijection = flow.bijection
    if not (isinstance(bijection, Invert) and isinstance(bijection.bijection, Scan)):
        return False
    layer = bijection.bijection.bijection
    if not (isinstance(layer, Chain) and len(layer.bijections) == 2):
        return False
    maf, perm = layer.bijections
    if not (isinstance(maf, MaskedAutoregressive) and isinstance(perm, Permute)):
        return False
    if maf.cond_shape is not None or len(maf.shape) != 1:
        return False
    mlp = maf.masked_autoregressive_mlp
    if mlp.activation is not jax.nn.relu or not _is_identity(mlp.final_activation):
        return False
    out_size = mlp.out_size
    if not isinstance(out_size, int) or out_size % maf.shape[0]:
        return False
    transformer = maf.transformer_constructor(jnp.zeros(out_size // maf.shape[0]))
    return isinstance(transformer, RationalQuadraticSpline)


def _is_identity(fn) -> bool:
    x = jnp.array([-1.5, 0.0, 2.0])
    try:
        return bool(jnp.array_equal(fn(x), x))
    except Exception:  # noqa: BLE001
        return False


def _pick(arr: Array, onehot: Bool[Array, "..."], axis: int) -> Array:
    """``arr`` at the index marked in ``onehot`` along ``axis`` (exact: the other terms are zeros)."""
    return jnp.sum(jnp.where(onehot, arr, 0.0), axis=axis)


def rqs_transform_and_log_det(
    x: Array,
    x_pos: Array,
    y_pos: Array,
    derivatives: Array,
    interval: tuple[float, float],
    axis: int = -1,
) -> tuple[Array, Array]:
    r"""flowjax ``RationalQuadraticSpline.transform_and_log_det``, elementwise for many splines at once.

    Args:
        x: Inputs, one per spline.
        x_pos, y_pos, derivatives: Knot positions and derivatives, with the knot axis at ``axis`` and the other axes
            matching ``x``.
        interval: Spline interval; outside it the transform is the identity.
        axis: Knot axis of ``x_pos``, ``y_pos`` and ``derivatives``.

    Returns:
        The transformed values and :math:`\log |dy/dx|`, both shaped like ``x``.
    """
    lower, upper = interval
    axis = axis % x_pos.ndim
    in_bounds = (x >= lower) & (x <= upper)
    x_clipped = jnp.where(x < lower, lower, jnp.where(x > upper, upper, x))
    n_bins = x_pos.shape[axis] - 1
    # jnp.searchsorted(x_pos, x_clipped) - 1 (side="left"): the number of knots strictly below the input, minus one.
    k = jnp.sum(x_pos < jnp.expand_dims(x_clipped, axis), axis=axis) - 1
    k = jnp.clip(k, 0, n_bins - 1)
    knot_index = jnp.arange(n_bins + 1).reshape([-1 if i == axis else 1 for i in range(x_pos.ndim)])
    at_k = knot_index == jnp.expand_dims(k, axis)
    at_k1 = knot_index == jnp.expand_dims(k + 1, axis)
    xk, xk1 = _pick(x_pos, at_k, axis), _pick(x_pos, at_k1, axis)
    yk, yk1 = _pick(y_pos, at_k, axis), _pick(y_pos, at_k1, axis)
    dk, dk1 = _pick(derivatives, at_k, axis), _pick(derivatives, at_k1, axis)
    # From here on the same expressions as flowjax (Durkan et al. 2019, eqs. 4 and 5).
    xi = (x_clipped - xk) / (xk1 - xk)
    sk = (yk1 - yk) / (xk1 - xk)
    num = (yk1 - yk) * (sk * xi**2 + dk * xi * (1 - xi))
    den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
    y = yk + num / den
    y = jnp.where(y < lower, lower, jnp.where(y > upper, upper, y))
    y = jnp.where(in_bounds, y, x)
    deriv_num = sk**2 * (dk1 * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
    deriv_den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2
    deriv = jnp.where(in_bounds, deriv_num / deriv_den, 1.0)
    return y, jnp.log(deriv)


def batched_log_prob(flow: Transformed, x: Float[Array, "n dim"]) -> Float[Array, " n"]:
    """``jax.vmap(flow.log_prob)(x)`` for a flow accepted by :func:`supports_batched_log_prob`."""
    scan, n_layers = _layer_parts(flow)
    arrays, static = eqx.partition(scan.bijection, eqx.is_array)
    z = x.T  # [dim, n]
    dim, n = z.shape
    log_det = jnp.zeros(n, x.dtype)
    for index in range(n_layers):
        layer = eqx.combine(jax.tree_util.tree_map(lambda a: a[index], arrays), static)
        maf, perm = layer.bijections
        mlp = paramax.unwrap(maf.masked_autoregressive_mlp)  # applies the autoregressive masks
        h = z
        for i, linear in enumerate(mlp.layers):
            h = linear.weight @ h + linear.bias[:, None]
            if i < len(mlp.layers) - 1:
                h = jax.nn.relu(h)
        params = h.reshape(dim, -1, n)  # [dim, params per dimension, n], as flowjax reshapes to (dim, -1)

        def build_spline(p: Array) -> RationalQuadraticSpline:
            return paramax.unwrap(maf.transformer_constructor(p))

        spline = eqx.filter_vmap(
            eqx.filter_vmap(build_spline, in_axes=1, out_axes=eqx.if_array(-1))
        )(params)  # knot arrays [dim, knots, n]
        y, log_deriv = rqs_transform_and_log_det(
            z, spline.x_pos, spline.y_pos, spline.derivatives, spline.interval, axis=1
        )
        log_det = log_det + jnp.sum(log_deriv, axis=0)
        z = y[perm.permutation]  # a tuple of index arrays, as in flowjax's Permute
    log_prob = flow.base_dist.log_prob(z.T) + log_det
    return jnp.where(jnp.isnan(log_prob), -jnp.inf, log_prob)  # as flowjax's Transformed._log_prob
