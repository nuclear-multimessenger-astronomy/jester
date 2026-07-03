r"""
Shared interface for pluggable ODE backends used by TOV solvers.

A "backend" wraps a specific ODE-integration library (diffrax, modax, ...)
behind one common call signature, so that physics-model solvers (``gr.py``,
``anisotropy.py``, ...) never branch on which library is doing the
integration. Backend selection happens once, in :data:`BACKEND_REGISTRY`
(see ``backends/__init__.py``).
"""

from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

from jaxtyping import Array, Float


class OdeResult(NamedTuple):
    """Result of a single ODE integration, backend-agnostic.

    Attributes:
        ys_final: Final state, same structure/order as the ``y0`` passed in.
        success: 1.0 if the integration succeeded, 0.0 otherwise (a float
            mask rather than a Python bool, so it stays ``jax.vmap``/
            ``jax.grad``-safe). ``ys_final`` is already NaN-poisoned when
            ``success`` is 0.0, so callers do not need to branch on it to get
            the existing "graceful failure via NaN propagation" behavior.
    """

    ys_final: tuple[Float[Array, ""], ...]
    success: Float[Array, ""]


class OdeBackend(ABC):
    """Common interface every ODE backend adapter implements."""

    #: Algorithm names this backend knows how to run, e.g. {"Dopri5", "Dopri8"}.
    SUPPORTED_ALGORITHMS: frozenset[str] = frozenset()

    @abstractmethod
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
        r"""Integrate ``rhs`` from ``t0`` to ``t1``.

        Args:
            rhs: Right-hand side in diffrax convention, ``rhs(t, y, args) -> dy/dt``.
                This is the canonical convention inside jester; backends that
                need a different argument order (e.g. modax's ``(y, t,
                params)``) adapt it internally.
            y0: Initial state tuple.
            t0: Independent-variable value at the start of integration.
            t1: Independent-variable value at the end of integration.
            dt0: Initial step size (signed — negative when integrating from
                a larger ``t0`` down to a smaller ``t1``, as TOV solvers do
                integrating enthalpy from the stellar center to the surface).
            args: Extra arguments passed to ``rhs`` (e.g. ``eos_dict``).
            algorithm: Name of the specific algorithm to use within this
                backend (must be in :data:`SUPPORTED_ALGORITHMS`).
            rtol: Relative tolerance for adaptive step-size control.
            atol: Absolute tolerance for adaptive step-size control.
            max_steps: Maximum number of integration steps.

        Returns:
            OdeResult: final state and a success flag.
        """
        raise NotImplementedError
