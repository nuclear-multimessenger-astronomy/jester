r"""
Pluggable ODE backends for TOV solvers.

The only symbol physics-model files (``gr.py``, ``anisotropy.py``, ...) need
from this package is :func:`solve_ode`. Backend selection (diffrax vs modax)
happens via a single dict lookup in :data:`BACKEND_REGISTRY` — physics files
never branch on backend name themselves. Adding a new backend later means
writing one ``<name>_backend.py`` implementing
:class:`~jesterTOV.tov.backends.base.OdeBackend` and adding one line here.
"""

from typing import Callable

from jaxtyping import Array, Float

from jesterTOV.tov.backends.base import OdeBackend, OdeResult
from jesterTOV.tov.backends.diffrax_backend import DiffraxBackend

# modax ("solvers" package) is an optional dependency; importing it lazily
# inside ModaxBackend.solve keeps `backend="diffrax"` usage free of any
# import-time cost or hard dependency on modax being installed.
from jesterTOV.tov.backends.modax_backend import ModaxBackend

BACKEND_REGISTRY: dict[str, OdeBackend] = {
    "diffrax": DiffraxBackend(),
    "modax": ModaxBackend(),
}

# diffrax's own default (diffrax.diffeqsolve(..., max_steps=4096)) — used
# here so solve_ode()'s default reproduces pre-refactor behavior exactly.
DEFAULT_MAX_STEPS = 4096


def solve_ode(
    rhs: Callable,
    y0: tuple[Float[Array, ""], ...],
    t0: float | Float[Array, ""],
    t1: float | Float[Array, ""],
    dt0: float | Float[Array, ""],
    args: dict,
    backend: str,
    algorithm: str,
    rtol: float,
    atol: float,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> OdeResult:
    r"""Integrate ``rhs`` from ``t0`` to ``t1`` using the selected backend.

    This is the single entrypoint every TOV physics-model solver calls for
    ODE integration. See :meth:`OdeBackend.solve` for the full argument
    reference.

    Raises:
        KeyError: If ``backend`` is not a registered backend name.
    """
    if backend not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown ODE backend {backend!r}; choose one of {sorted(BACKEND_REGISTRY)}"
        )
    return BACKEND_REGISTRY[backend].solve(
        rhs, y0, t0, t1, dt0, args, algorithm, rtol, atol, max_steps
    )
