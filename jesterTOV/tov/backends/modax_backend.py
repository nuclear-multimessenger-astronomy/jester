r"""modax-backed :class:`~jesterTOV.tov.backends.base.OdeBackend`.

modax's installable package is named ``solvers`` (the project is called
"modax", but ``pyproject.toml``'s ``[tool.setuptools.packages.find]`` only
ships the ``solvers`` subpackage) — imported here as ``import solvers.*``,
not ``import modax``. It's an optional dependency (``jesterTOV[modax]``), so
the import happens lazily inside :meth:`ModaxBackend.solve` rather than at
module load time; selecting ``backend="diffrax"`` never requires modax to be
installed.

Three adaptations are needed to bridge modax's API to jester's diffrax-style
RHS functions:

1. **Argument order.** diffrax RHS functions are ``rhs(t, y, args)``; modax
   expects ``ode_fn(y, t, params)``. Flipped by a thin closure below.
2. **Direction and a *fixed* ``t_span``.** modax requires ``t_span`` to be
   strictly increasing, but TOV solvers integrate enthalpy *downward*
   (``t0=h_center > t1=0``) — and, under ``jax.vmap`` (``construct_family``
   vmaps ``solve`` over a whole central-pressure grid), ``t0`` differs per
   star since ``h_center`` depends on the central pressure. We reparametrize
   to ``u = (t - t0) / (t1 - t0)``, which always runs over the fixed,
   compile-time-constant interval ``[0, 1]`` regardless of ``t0``/``t1``. By
   the chain rule, ``dy/du = (t1 - t0) * rhs(t(u), y, args)`` where
   ``t(u) = t0 + u * (t1 - t0)``.
**Known limitation — no reverse-mode autodiff.** modax's pure-JAX solvers
build the adaptive step loop with ``jax.lax.while_loop`` using a
data-dependent trip count, and register no ``custom_vjp``. JAX does not
support reverse-mode AD (``jax.grad``) through such a loop
(``ValueError: Reverse-mode differentiation does not work for
lax.while_loop or lax.fori_loop with dynamic start/stop values.``).
Forward-mode AD (``jax.jvp``) works fine. This means ``backend="modax"``
is unusable today with jester's gradient-based samplers (SMC-NUTS,
FlowMC); it is only safe with gradient-free samplers (e.g. SMC-RW) until
modax adds a custom VJP upstream.

3. **`t0`/`t1` must flow through `params`, not a closure.** modax's
   ``custom_vmap`` batching rule asserts that *no value closed over by the
   RHS may be batched* (``assert not any(tree_leaves(consts_batched))`` in
   ``jax.custom_batching``) — only genuine positional arguments
   (``y0``/``t_span``/``params``) may vary per vmap lane. Since ``t0``/``t1``
   differ per star under ``construct_family``'s vmap, they cannot be
   closed over by ``modax_rhs`` like a fixed constant; they are packed into
   modax's ``params`` array instead (``params = [t0, t1]``), which modax's
   batching *does* support. ``eos_dict`` (``args``), by contrast, is the
   same for every star in a single ``construct_family`` call (only ``pc``
   is vmapped) — it is genuinely unbatched, so closing over it is safe.
   This all lives entirely inside this adapter — physics-model files keep
   writing RHS functions in terms of the original variable ``h``, unaware
   of the reparametrization or of modax's `params`-vs-closure distinction.
"""

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.tov.backends.base import OdeBackend, OdeResult

_ALGORITHM_MODULES = {
    "Tsit5": "solvers.tsit5jax",
    "Rodas5P": "solvers.rodas5Pjax",
    "KenCarp5": "solvers.kencarp5jax",
}
# Stiff (implicit) solvers accept an extra `lu_precision` kwarg; Tsit5 does not.
_STIFF_ALGORITHMS = frozenset({"Rodas5P", "KenCarp5"})


class ModaxBackend(OdeBackend):
    """Wraps modax's ``solvers.<algorithm>jax.solve`` behind the common :class:`OdeBackend` interface."""

    SUPPORTED_ALGORITHMS = frozenset(_ALGORITHM_MODULES)

    def solve(
        self,
        rhs: Callable,
        y0: tuple[Float[Array, ""], ...],
        t0: float | Float[Array, ""],
        t1: float | Float[Array, ""],
        dt0: float | Float[Array, ""],
        args: dict,
        algorithm: str,
        rtol: float,
        atol: float,
        max_steps: int,
    ) -> OdeResult:
        if algorithm not in _ALGORITHM_MODULES:
            raise ValueError(
                f"ModaxBackend does not support algorithm={algorithm!r}; "
                f"choose one of {sorted(_ALGORITHM_MODULES)}"
            )
        try:
            import importlib

            solver_module = importlib.import_module(_ALGORITHM_MODULES[algorithm])
        except ImportError as exc:
            raise ImportError(
                "backend='modax' requires the optional modax dependency. "
                "Install it with `uv pip install -e '.[modax]'`."
            ) from exc

        # t_span is always [0, 1] — a compile-time constant, independent of
        # the (possibly batched) t0/t1 — so it never becomes a per-lane
        # value under jax.vmap.
        t_span = jnp.array([0.0, 1.0], dtype=jnp.float64)
        # First step as a fraction of the normalized [0, 1] domain, matching
        # the ~1e-3 relative first step TOV solvers use in physical h-space.
        first_step = 1e-3

        n_vars = len(y0)
        # t0/t1 vary per star under jax.vmap (e.g. h_center depends on the
        # central pressure); they must flow through modax's traced `params`
        # argument rather than a Python closure — see module docstring.
        modax_params = jnp.array([t0, t1], dtype=jnp.float64)

        def modax_rhs(y, u, params_arr):
            t0_, t1_ = params_arr[0], params_arr[1]
            scale = t1_ - t0_
            t = t0_ + u * scale
            dydt = rhs(t, tuple(y[i] for i in range(n_vars)), args)
            return scale * jnp.stack(dydt)

        solve_kwargs: dict[str, object] = dict(
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
            return_stats=True,
        )
        if algorithm in _STIFF_ALGORITHMS:
            solve_kwargs["lu_precision"] = "fp64"

        sol_arr, stats = solver_module.solve(
            modax_rhs,
            jnp.stack(y0),
            t_span,
            modax_params,
            **solve_kwargs,
        )

        ys_final_arr = sol_arr[0, -1, :]
        accepted_steps = stats["accepted_steps"][0]
        success = jnp.all(jnp.isfinite(ys_final_arr)) & (accepted_steps > 0)
        ys_final_arr = jnp.where(success, ys_final_arr, jnp.nan)
        ys_final = tuple(ys_final_arr[i] for i in range(n_vars))

        return OdeResult(
            ys_final=ys_final, success=jnp.asarray(success, dtype=jnp.float64)
        )
