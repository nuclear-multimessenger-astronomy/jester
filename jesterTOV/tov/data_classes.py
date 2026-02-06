"""
JAX-compatible dataclasses for EOS and TOV data structures.

Uses NamedTuple for immutability and automatic JAX pytree compatibility.
No additional dependencies required beyond JAX and jaxtyping.
"""

from typing import NamedTuple, Optional
from jaxtyping import Float, Array


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
    extra_constraints: Optional[dict[str, float]] = None
    # EOS-specific constraint violation counts
    # Convention: Keys use "n_*_violations" or "n_*" format
    # Examples: {"n_gamma_violations": 5.0} for spectral EOS


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
    """
    Mass-radius-tidal family curves in physical units.

    Represents a sequence of neutron star solutions across different
    central pressures, forming M-R-Λ curves for inference.
    """

    log10pcs: Float[Array, "ndat"]  # Log10 central pressure [geometric units]
    masses: Float[Array, "ndat"]  # Masses [M_sun]
    radii: Float[Array, "ndat"]  # Radii [km]
    lambdas: Float[Array, "ndat"]  # Dimensionless tidal deformability
