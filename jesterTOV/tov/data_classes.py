"""
JAX-compatible dataclasses for EOS and TOV data structures.

Uses NamedTuple for immutability and automatic JAX pytree compatibility.
No additional dependencies required beyond JAX and jaxtyping.
"""

from typing import Any, NamedTuple, Optional
from jaxtyping import Float, Int, Array


class EOSData(NamedTuple):
    """
    Immutable container for EOS quantities in geometric units.

    NamedTuple is automatically JAX pytree-compatible, no extra dependencies needed.
    All arrays represent physical quantities sampled over a density/pressure grid.
    """

    ns: Float[Array, "n_points"]  # Number density [geometric units]
    ps: Float[Array, "n_points"]  # Pressure [geometric units]
    hs: Float[Array, "n_points"]  # Specific enthalpy [geometric units]
    es: Float[Array, "n_points"]  # Energy density [geometric units]
    dloge_dlogps: Float[Array, "n_points"]  # d(ln eps)/d(ln p)
    cs2: Float[Array, "n_points"]  # Speed of sound squared
    mu: Optional[Float[Array, "n_points"]] = None  # Chemical potential
    extra_constraints: Optional[dict[str, Any]] = None
    # EOS-specific constraint violation magnitudes or counts.
    # Values must be JAX arrays (not Python float()) when constructed inside jax.vmap.
    # Convention: Keys use "n_*_violations" or "n_*" format.
    # Examples: {"n_gamma_violations": jnp.maximum(0.0, 0.1 - gamma_min)} for spectral EOS


class TOVSolution(NamedTuple):
    """
    Single neutron star solution from TOV equations.

    When vmapped, fields become batched arrays:
        solutions = jax.vmap(solve)(pcs)
        # solutions.M is array [M1, M2, ..., Mn]
    """

    M: float  # Mass [geometric units]
    R: float  # Radius [geometric units]
    k2: float  # Second Love number (dimensionless)


class FamilyData(NamedTuple):
    r"""
    Mass-radius-tidal family curves in physical units.

    Points are stored sorted by central pressure :math:`p_c` (ascending).
    This guarantees that masses are strictly monotone within any contiguous
    run of the same ``branch_ids`` value, which is required for safe
    branch-wise interpolation.

    ``branch_ids`` encodes which stable branch each point belongs to:

    - Values ``0, 1, ..., N_MAX_BRANCHES-1``: point is on that stable branch.
    - Value ``N_MAX_BRANCHES``: sentinel indicating an unstable configuration.

    Use :func:`jesterTOV.utils.interp_family_multi_branch` to interpolate
    observables (radii, tidal deformabilities) at a given mass while correctly
    handling multiple stable branches.
    """

    log10pcs: Float[Array, "ndat"]  # Log10 central pressure [geometric units]
    masses: Float[Array, "ndat"]  # Masses [M_sun]
    radii: Float[Array, "ndat"]  # Radii [km]
    lambdas: Float[Array, "ndat"]  # Dimensionless tidal deformability
    branch_ids: Int[
        Array, "ndat"
    ]  # Stable-branch label (N_MAX_BRANCHES = unstable sentinel)
