r"""General Relativity TOV solver running on the modax GPU ODE kernels.

This solver integrates the same equations as :class:`~jesterTOV.tov.gr.GRTOVSolver`,
with the same tolerances, but all stars of a batch go through one modax Tsit5 launch in
which every star has its own adaptive step sequence. In a vmapped diffrax solve all stars
share one loop, which runs until the slowest star of the batch has finished.

modax runs on a CUDA GPU only and is imported when the solver is created.

The enthalpy is replaced by :math:`s = 1 - h / h_0`, so all stars integrate over
:math:`s \in [0, 1]` as modax requires one save grid for the whole ensemble.
"""

import functools
import math
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap
from jaxtyping import Array, Float

from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, FamilyData, TOVSolution
from jesterTOV.tov.gr import _calc_k2, _initial_conditions

# One parameter row per EOS: the packed table, ``N_COLUMNS`` values per node, nothing else. All stars of an
# EOS read the same row (modax ``params_group_size``), so the table is stored and read once per EOS.
# A coarse index in front of the table (bins in h**(1/4) holding a start node for a short search) was tried: same
# brackets, but no faster than the full branchless search of the right-hand side, so the row holds only the table.
N_HEADER = 0
N_COLUMNS = 5
_LOG_H, _LOG_E, _LOG_P, _H, _DLOGE_DLOGP = 0, 1, 2, 3, 4

# State of a star: (r, m, H, b) and its start enthalpy h0, which is constant. h0 lives in the state
# because it differs per star while the row is shared by the stars of an EOS.
N_STATE = 5
# modax divides the error norm by the number of components. h0 gets weight zero, and the four physical
# components sqrt(5/4), so the norm is the same as for the four-component system.
ERROR_WEIGHTS = (math.sqrt(5.0 / 4.0),) * 4 + (0.0,)

RTOL = 1e-5
ATOL = 1e-6
# The same budget of attempted steps as the default of diffrax.diffeqsolve.
MAX_STEPS = 4096
# Equal to the first step of the diffrax solver, |dh| = 1e-3 hc, in units of s.
FIRST_STEP = 1e-3


def pack_table(eos_data: EOSData) -> Float[Array, " n_row"]:
    r"""Pack the EOS columns the right-hand side needs into one row-major array.

    Row-major puts the values of one node next to each other, so a lookup touches few cache lines.

    Args:
        eos_data: EOS quantities in geometric units.

    Returns:
        Array of length ``N_COLUMNS * n_table``.
    """
    return jnp.stack(
        [
            jnp.log(eos_data.hs),
            jnp.log(eos_data.es),
            jnp.log(eos_data.ps),
            eos_data.hs,
            eos_data.dloge_dlogps,
        ],
        axis=1,
    ).ravel()


def _search_steps(n_table: int) -> tuple[int, ...]:
    """Powers of two from the largest one below ``n_table - 1`` down to 1: the steps of the bracket search."""
    top = 1 << (max(n_table - 2, 1).bit_length() - 1)
    steps = []
    while top > 0:
        steps.append(top)
        top //= 2
    return tuple(steps)


def _bracket_lo(n_table: int, stride: int = 1, column: int = 0) -> Callable[..., int]:
    r"""Search of the right-hand side, as a plain function for testing: the largest node index ``lo <= n_table - 2``
    with ``p[stride * lo + column] <= h``, i.e. the bracket ``[lo, lo + 1]`` of ``h`` (assumes ``h`` >= the first node).

    It visits a fixed number of candidates (``_search_steps``) and selects instead of branching, so the GPU kernel
    has no data-dependent loop. It returns the same bracket as plain bisection for every ``h``, including nodes and
    enthalpies at or above the last node.
    """
    steps = _search_steps(n_table)
    last_lo = n_table - 2

    def search(p, h):
        lo = 0
        for step in steps:
            cand = lo + step
            idx = cand if cand < last_lo else last_lo  # always a valid index
            take = (cand <= last_lo) & (p[stride * idx + column] <= h)
            lo = cand if take else lo
        return lo

    return search


@functools.cache
def make_tov_rhs(n_table: int) -> Callable[..., tuple[float, ...]]:
    r"""Build the right-hand side for EOS tables of ``n_table`` nodes.

    It is the same system as :func:`~jesterTOV.tov.gr._tov_ode` multiplied by
    :math:`dh/ds = -h_0`, with :math:`h_0` carried as a fifth, constant state component. The
    function is plain Python using ``math``, so it compiles as CUDA device code and can also be
    called on the host. One binary search per call serves all three lookups. The result is cached,
    because modax caches its kernels on the identity of this function.

    Args:
        n_table: Number of nodes in the EOS table.

    Returns:
        A function ``(y, s, p) -> (dr/ds, dm/ds, dH/ds, db/ds, 0)``.
    """
    stride = N_COLUMNS
    offset = N_HEADER
    four_pi = 4.0 * math.pi
    steps = _search_steps(n_table)
    last_lo = n_table - 2

    def tov_rhs(y, s, p):
        h0 = y[4]
        # An invalid star has h0 = 0: stop before reading the table, which may hold NaN.
        if h0 == 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        r = y[0]
        m = y[1]
        H = y[2]
        b = y[3]

        # Below the first node the derivatives are frozen at their value on that node, as in
        # _tov_ode. A step ends at s = 1 at the latest, so h >= 0 here.
        h = h0 * (1.0 - s)
        h_first = p[offset + _H]
        if h < h_first:
            h = h_first

        # Bracket [lo, lo + 1] of h: a fixed number of candidates and a select instead of a branch, identical to
        # bisection (see _bracket_lo, which is this loop on its own). A data-dependent while loop here cost two
        # thirds of the time per step on the GPU (divergence, dependent loads, local memory).
        lo = 0
        for step in steps:
            cand = lo + step
            idx = cand if cand < last_lo else last_lo
            take = (cand <= last_lo) & (p[offset + stride * idx + _H] <= h)
            lo = cand if take else lo
        ilo = offset + stride * lo
        ihi = ilo + stride

        # Linear in log(h) for e and p, linear in h for dlog(e)/dlog(p), as in _tov_ode.
        frac_log = (math.log(h) - p[ilo + _LOG_H]) / (p[ihi + _LOG_H] - p[ilo + _LOG_H])
        frac = (h - p[ilo + _H]) / (p[ihi + _H] - p[ilo + _H])
        e = math.exp(p[ilo + _LOG_E] + frac_log * (p[ihi + _LOG_E] - p[ilo + _LOG_E]))
        pres = math.exp(p[ilo + _LOG_P] + frac_log * (p[ihi + _LOG_P] - p[ilo + _LOG_P]))
        dloge = p[ilo + _DLOGE_DLOGP] + frac * (
            p[ihi + _DLOGE_DLOGP] - p[ilo + _DLOGE_DLOGP]
        )
        dedp = e / pres * dloge

        A = 1.0 / (1.0 - 2.0 * m / r)
        C1 = 2.0 / r + A * (2.0 * m / (r * r) + four_pi * r * (pres - e))
        term = 2.0 * (m + four_pi * r * r * r * pres) / (r * (r - 2.0 * m))
        C0 = (
            A
            * (
                -6.0 / (r * r)
                + four_pi * (e + pres) * dedp
                + four_pi * (5.0 * e + 9.0 * pres)
            )
            - term * term
        )

        drdh = -r * (r - 2.0 * m) / (m + four_pi * r * r * r * pres)
        dmdh = four_pi * r * r * e * drdh
        dHdh = b * drdh
        dbdh = -(C0 * H + C1 * b) * drdh

        return -h0 * drdh, -h0 * dmdh, -h0 * dHdh, -h0 * dbdh, 0.0

    return tov_rhs


def _solve_flat(
    build_rhs: Callable[[int], Callable[..., tuple[float, ...]]],
    n_columns: int,
    group: int,
    y0: Float[Array, "n n_state"],
    tables: Float[Array, "n_eos n_row"],
) -> Float[Array, "n n_out"]:
    """One modax launch for ``n = n_eos * group`` stars; star ``i`` uses table row ``i // group``.

    The last state component is the start enthalpy h0. A star whose state, h0 or table is not finite
    is invalid: it starts from a harmless state with h0 = 0, so its right-hand side is zero and it
    finishes after a few steps (a NaN state would be rejected on every step up to ``MAX_STEPS`` and hold
    up its warp). Returns the surface state without h0: NaN for an invalid star, inf for one that ran
    out of steps (as diffrax with ``throw=False``).
    """
    from modax.tsit5 import solve  # type: ignore[import-not-found]  # optional GPU dependency

    n_table = (tables.shape[1] - N_HEADER) // n_columns
    table_ok = jnp.repeat(jnp.all(jnp.isfinite(tables), axis=1), group)
    valid = table_ok & jnp.all(jnp.isfinite(y0), axis=1) & (y0[:, -1] > 0.0)
    safe_y0 = jnp.where(valid[:, None], y0, 1.0).at[:, -1].set(
        jnp.where(valid, y0[:, -1], 0.0)
    )
    hist, stats = solve(
        build_rhs(n_table),
        safe_y0,
        jnp.array([0.0, 1.0]),
        tables,
        rtol=RTOL,
        atol=ATOL,
        first_step=FIRST_STEP,
        max_steps=MAX_STEPS,
        # A NumPy constant: modax converts the weights on the host while tracing.
        error_weights=np.asarray(ERROR_WEIGHTS),
        return_stats=True,
        params_group_size=group,
    )
    surface: Array = hist[:, -1, :-1]
    unfinished = stats["loop_steps"] >= MAX_STEPS
    surface = cast(Array, jnp.where(unfinished[:, None], jnp.inf, surface))
    return cast(Array, jnp.where(valid[:, None], surface, jnp.nan))


@functools.cache
def _batch_solver(
    build_rhs: Callable[[int], Callable[..., tuple[float, ...]]],
    n_columns: int,
    group: int,
) -> custom_vmap:
    """A solver of the stars of some EOS that turns any vmap on top of it into one larger launch."""

    @custom_vmap
    def solve_batch(y0: Array, tables: Array) -> Float[Array, "n n_out"]:
        return _solve_flat(build_rhs, n_columns, group, y0, tables)

    @solve_batch.def_vmap
    def solve_batch_vmap(
        axis_size: int, in_batched: list[bool], y0: Array, tables: Array
    ) -> tuple[Array, bool]:
        def flatten(arg: Array, batched: bool) -> Array:
            full = arg if batched else jnp.broadcast_to(arg, (axis_size,) + arg.shape)
            return full.reshape((-1,) + full.shape[2:])

        # Stars and table rows are both EOS-major, so star i still reads row i // group.
        # Through solve_batch and not _solve_flat, so a further vmap level is flattened as well.
        out = solve_batch(flatten(y0, in_batched[0]), flatten(tables, in_batched[1]))
        return out.reshape((axis_size, -1) + out.shape[1:]), True

    return solve_batch


def solve_stars(
    build_rhs: Callable[[int], Callable[..., tuple[float, ...]]],
    n_columns: int,
    y0: Float[Array, "n n_vars"],
    h0: Float[Array, " n"],
    table: Float[Array, " n_row"],
) -> Float[Array, "n n_vars"]:
    r"""Integrate ``n`` stars of one EOS from :math:`s = 0` to :math:`s = 1` in one modax launch.

    The table is passed once and shared by all stars. A vmap over this function, for instance over
    EOS, becomes one launch of all stars together, still with one table row per EOS.

    Args:
        build_rhs: Cached function returning the device right-hand side for a table size; it takes
            the state with h0 appended.
        n_columns: Number of columns per node in ``table``.
        y0: Initial states.
        h0: Enthalpy at the start of each star.
        table: Packed EOS table, with ``n_columns`` values per node.

    Returns:
        State of each star at the surface: NaN for an invalid star and inf for one that ran out of steps.
    """
    y0_full = jnp.concatenate([y0, h0[:, None]], axis=1)
    return _batch_solver(build_rhs, n_columns, y0.shape[0])(y0_full, table[None, :])


class GRTOVModaxSolver(TOVSolverBase):
    """General Relativity TOV solver using the modax Tsit5 GPU kernel.

    It has the same equations and tolerances as :class:`~jesterTOV.tov.gr.GRTOVSolver`, and is selected with
    ``backend: modax`` in the ``tov`` section of a config. It supports jit and vmap but no gradients: modax
    differentiates through Enzyme, which is not wired into this right-hand side, so use it with
    gradient-free samplers such as ``smc-rw``.
    """

    def __init__(self) -> None:
        try:
            import modax.tsit5  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "The modax TOV backend needs modax-solvers and a CUDA GPU."
            ) from error

    def solve(
        self, eos_data: EOSData, pc: float, tov_params: dict[str, float] = {}
    ) -> TOVSolution:
        """Solve the TOV equations for one central pressure.

        Args:
            eos_data: EOS quantities in geometric units.
            pc: Central pressure [geometric units].
            tov_params: Not used.

        Returns:
            TOVSolution: Mass, radius and Love number in geometric units.
        """
        pcs = jnp.asarray(pc)[None]
        masses, radii, k2s = self._solve_many(eos_data, pcs)
        # Scalar arrays, as in GRTOVSolver.solve: TOVSolution is annotated with float.
        return TOVSolution(M=masses[0], R=radii[0], k2=k2s[0])  # type: ignore[arg-type]

    def _solve_many(
        self, eos_data: EOSData, pcs: Float[Array, " n"]
    ) -> tuple[Float[Array, " n"], Float[Array, " n"], Float[Array, " n"]]:
        """Solve for several central pressures of one EOS in one launch."""
        h0, _, (r0, m0, H0, b0) = jax.vmap(
            lambda pc: _initial_conditions(eos_data, pc)
        )(pcs)
        y0 = jnp.stack([r0, m0, H0, b0], axis=1)
        surface = solve_stars(make_tov_rhs, N_COLUMNS, y0, h0, pack_table(eos_data))
        R, M, H, b = surface[:, 0], surface[:, 1], surface[:, 2], surface[:, 3]
        return M, R, _calc_k2(R, M, H, b)

    def construct_family(
        self,
        eos_data: EOSData,
        ndat: int,
        min_nsat: float,
        tov_params: dict[str, Any] = {},
    ) -> FamilyData:
        r"""Construct M-R-:math:`\Lambda` curves, with all central pressures in one launch.

        Args:
            eos_data: EOS quantities in geometric units.
            ndat: Number of points in the central pressure grid.
            min_nsat: Minimum central density in units of saturation density.
            tov_params: Not used.

        Returns:
            FamilyData: Mass-radius-tidal curves in physical units.
        """
        is_valid_eos = (
            jnp.all(eos_data.ps > 0)
            & jnp.all(eos_data.hs > 0)
            & jnp.all(eos_data.es > 0)
            & jnp.all(eos_data.cs2 >= 0)
            & jnp.all(eos_data.cs2 <= 1)
        )
        eos_data = jax.tree_util.tree_map(
            lambda x: jnp.where(is_valid_eos, x, jnp.nan), eos_data
        )
        pc_min = self._get_pc_min(eos_data, min_nsat)
        pc_max = self._get_pc_max(eos_data)
        pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)
        masses, radii, k2s = self._solve_many(eos_data, pcs)
        return self._create_family_data(pcs, masses, radii, k2s, ndat)

    def get_required_parameters(self) -> list[str]:
        """GR TOV requires no additional parameters beyond the EOS.

        Returns:
            list[str]: Empty list.
        """
        return []
