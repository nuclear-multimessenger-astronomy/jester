r"""diffrax-backed :class:`~jesterTOV.tov.backends.base.OdeBackend`.

Reproduces exactly what ``gr.py``/``anisotropy.py`` did before the backend
refactor: ``diffeqsolve`` with ``scan_kind="bounded"``, a ``PIDController``,
``SaveAt(t1=True)``, and ``throw=False`` for graceful (NaN-propagating)
failure handling.
"""

from typing import Callable

import jax.numpy as jnp
from diffrax import RESULTS, Dopri5, Dopri8, ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
from jaxtyping import Array, Float

from jesterTOV.tov.backends.base import OdeBackend, OdeResult

_SOLVERS = {
    "Dopri5": Dopri5,
    "Dopri8": Dopri8,
    "Tsit5": Tsit5,
}


class DiffraxBackend(OdeBackend):
    """Wraps diffrax's ``diffeqsolve`` behind the common :class:`OdeBackend` interface."""

    SUPPORTED_ALGORITHMS = frozenset(_SOLVERS)

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
        if algorithm not in _SOLVERS:
            raise ValueError(
                f"DiffraxBackend does not support algorithm={algorithm!r}; "
                f"choose one of {sorted(_SOLVERS)}"
            )

        sol = diffeqsolve(
            ODETerm(rhs),
            _SOLVERS[algorithm](scan_kind="bounded"),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=args,
            saveat=SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            throw=False,
        )

        # diffrax always populates `ys` with throw=False; NaN-poisoned on failure.
        ys_final = tuple(y[-1] for y in sol.ys)  # type: ignore[union-attr]
        success = jnp.asarray(sol.result == RESULTS.successful, dtype=jnp.float64)
        ys_final = tuple(jnp.where(success, y, jnp.nan) for y in ys_final)

        return OdeResult(ys_final=ys_final, success=success)
