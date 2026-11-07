"""Tests for the batched MAF + rational-quadratic-spline log_prob (jesterTOV/inference/flows/batched.py).

It must reproduce flowjax's ``flow.log_prob`` (the reference) for every shipped GW flow, for flows with several layers,
and for points outside the spline interval; and StackedGWLikelihood must give the same log L with it as without it.
"""

import glob
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jesterTOV
from jesterTOV.inference.flows.batched import batched_log_prob, supports_batched_log_prob
from jesterTOV.inference.flows.flow import Flow, create_flow, disable_x64
from jesterTOV.inference.likelihoods.gw import StackedGWLikelihood

from .test_likelihoods import _save_toy_flow

SHIPPED_FLOWS = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(jesterTOV.__file__),
            "inference/flows/models/gw_maf/*/*/",
        )
    )
)


def _test_points(n: int, seed: int = 0, interval: float = 5.0) -> jnp.ndarray:
    """Standard-normal-ish points plus points outside and exactly on the spline interval."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)) * 1.5
    x[: n // 10] *= 4.0
    x[n // 10 : n // 10 + 5, 2] = interval
    x[n // 10 + 5 : n // 10 + 10, 3] = -interval
    return jnp.asarray(x)


def _randomise(flow, seed: int, scale: float = 0.3):
    """Perturb every float leaf, so that untrained toy flows are far from the identity."""
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    leaves = [
        leaf + scale * jax.random.normal(k, leaf.shape, leaf.dtype)
        for leaf, k in zip(leaves, keys)
    ]
    return eqx.combine(jax.tree_util.tree_unflatten(treedef, leaves), static)


def test_shipped_flows_are_found():
    assert len(SHIPPED_FLOWS) >= 10


@pytest.mark.parametrize(
    "model_dir", SHIPPED_FLOWS, ids=lambda d: os.path.basename(d.rstrip("/"))
)
def test_matches_flowjax_on_shipped_gw_flows(model_dir):
    flow = Flow.from_directory(model_dir).flow
    assert supports_batched_log_prob(flow)
    x = _test_points(400)
    reference = np.asarray(jax.vmap(flow.log_prob)(x))
    result = np.asarray(eqx.filter_jit(batched_log_prob)(flow, x))
    assert np.all(np.isfinite(reference))
    np.testing.assert_allclose(result, reference, rtol=0, atol=1e-10)


@pytest.mark.parametrize("flow_layers", [1, 3])
@pytest.mark.parametrize("knots", [4, 10])
def test_matches_flowjax_on_random_flows(flow_layers, knots):
    flow = create_flow(
        key=jax.random.key(flow_layers * 100 + knots),
        dim=4,
        nn_depth=2,
        nn_width=16,
        flow_layers=flow_layers,
        transformer_type="rational_quadratic_spline",
        transformer_knots=knots,
        transformer_interval=4.0,
    )
    flow = _randomise(flow, seed=knots)
    assert supports_batched_log_prob(flow)
    x = _test_points(400, seed=1, interval=4.0)
    reference = np.asarray(jax.vmap(flow.log_prob)(x))
    result = np.asarray(eqx.filter_jit(batched_log_prob)(flow, x))
    np.testing.assert_allclose(result, reference, rtol=0, atol=1e-10)


def test_matches_flowjax_in_float32():
    model_dir = SHIPPED_FLOWS[0]
    flow = Flow.from_directory(model_dir, dtype="float32").flow
    with disable_x64():
        x = _test_points(400).astype(jnp.float32)
        reference = np.asarray(jax.vmap(flow.log_prob)(x))
        result = eqx.filter_jit(batched_log_prob)(flow, x)
        assert result.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(result), reference, rtol=0, atol=2e-4)


def test_affine_flows_are_not_supported():
    flow = create_flow(key=jax.random.key(0), dim=4, transformer_type="affine")
    assert not supports_batched_log_prob(flow)


def _stacked(model_dirs, batch_size, use_float32=False, **kwargs):
    return StackedGWLikelihood(
        event_names=[f"event_{i}" for i in range(len(model_dirs))],
        model_dirs=[str(d) for d in model_dirs],
        N_masses_evaluation=20,
        N_masses_batch_size=batch_size,
        event_batch_size=2,
        seed=42,
        use_float32=use_float32,
        **kwargs,
    )


@pytest.fixture
def spline_flow_dirs(tmp_path):
    return [
        _save_toy_flow(
            tmp_path / f"event_{i}",
            seed=i,
            standardize=True,
            transformer_type="rational_quadratic_spline",
        )
        for i in range(2)
    ]


PARAMS = {
    "masses_EOS": jnp.linspace(1.0, 1.5, 100),  # M_TOV inside the mass range: penalties apply
    "Lambdas_EOS": jnp.linspace(2000.0, 10.0, 100),
}


def test_stacked_likelihood_uses_batched_flow_for_spline_flows(spline_flow_dirs, tmp_path):
    assert _stacked(spline_flow_dirs, 5).use_batched_flow
    affine = [_save_toy_flow(tmp_path / "affine", seed=0)]
    assert not _stacked(affine, 5).use_batched_flow


@pytest.mark.parametrize("batch_size", [1, 7, 20])
def test_stacked_likelihood_same_with_and_without_batched_flow(spline_flow_dirs, batch_size):
    """Same log L, including a batch size that does not divide N_masses_evaluation."""
    fast = _stacked(spline_flow_dirs, batch_size, penalty_value=-50.0)
    slow = _stacked(spline_flow_dirs, batch_size, penalty_value=-50.0, use_batched_flow=False)
    assert fast.use_batched_flow and not slow.use_batched_flow
    a, b = fast.evaluate(PARAMS), slow.evaluate(PARAMS)
    assert jnp.isfinite(a)
    np.testing.assert_allclose(float(a), float(b), rtol=0, atol=1e-10)


def test_stacked_likelihood_batched_flow_float32_under_jit_vmap(spline_flow_dirs):
    fast = _stacked(spline_flow_dirs, 7, use_float32=True)
    slow = _stacked(spline_flow_dirs, 7, use_float32=True, use_batched_flow=False)
    masses = jnp.stack([PARAMS["masses_EOS"], PARAMS["masses_EOS"] * 1.1])
    lambdas = jnp.stack([PARAMS["Lambdas_EOS"], PARAMS["Lambdas_EOS"] * 0.8])

    def run(likelihood):
        fn = jax.jit(
            jax.vmap(lambda m, l: likelihood.evaluate({"masses_EOS": m, "Lambdas_EOS": l}))
        )
        return fn(masses, lambdas)

    a, b = run(fast), run(slow)
    assert a.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(a))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-4)
