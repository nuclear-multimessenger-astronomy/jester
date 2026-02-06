r"""
Base class for TOV equation solvers.

This module defines the abstract interface that all TOV solvers must implement,
whether for General Relativity, modified gravity, or scalar-tensor theories.
"""

from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from jesterTOV.tov.data_classes import EOSData, TOVSolution, FamilyData
from jesterTOV import utils
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class TOVSolverBase(ABC):
    """
    Abstract base class for TOV equation solvers.

    All TOV solvers must implement:
    - solve(): Solve TOV equations for a given central pressure
    - get_required_parameters(): Return list of additional parameters beyond EOS
    """

    @abstractmethod
    def solve(self, eos_data: EOSData, pc: float, **kwargs) -> TOVSolution:
        r"""
        Solve TOV equations for given central pressure.

        Args:
            eos_data: EOS quantities (type-safe dataclass)
            pc: Central pressure [geometric units]
            **kwargs: Additional solver-specific parameters

        Returns:
            TOVSolution: Mass, radius, and Love number k2 [geometric units]
        """
        pass

    def get_required_parameters(self) -> list[str]:
        """
        Return additional parameters needed beyond EOS.

        Examples:
            - GR TOV: [] (no extra params)
            - Anisotropic TOV: ["gamma"] (anisotropy parameter)
            - Scalar-tensor: ["beta_ST", "phi_c", "nu_c"]

        Returns:
            list[str]: Parameter names
        """
        return []

    def construct_family(
        self,
        eos_data: EOSData,
        ndat: int,
        min_nsat: float,
        **kwargs,
    ) -> FamilyData:
        r"""
        Construct M-R-Λ curves by solving for multiple central pressures.

        The central pressure grid spans from a minimum based on min_nsat to
        the maximum pressure where the EOS remains causal (:math:`c_s^2 < 1`).

        Args:
            eos_data: EOS quantities in geometric units
            ndat: Number of points in central pressure grid
            min_nsat: Minimum central density in units of saturation density
                     (assumed to be 0.16 :math:`\mathrm{fm}^{-3}`)
            **kwargs: Additional solver-specific parameters

        Returns:
            FamilyData: Mass-radius-tidal curves in physical units
        """
        # Create central pressure grid
        pc_min = self._get_pc_min(eos_data, min_nsat)
        pc_max = self._get_pc_max(eos_data)
        pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

        # Solve TOV for each pc using vmap
        # NOTE: vmap batches TOVSolution fields into arrays
        def solve_single_pc(pc):
            return self.solve(eos_data, pc, **kwargs)

        solutions = jax.vmap(solve_single_pc)(pcs)

        # Extract batched results (vmap converts scalar fields to arrays)
        masses: Float[Array, "ndat"] = solutions.M  # type: ignore[assignment]
        radii: Float[Array, "ndat"] = solutions.R  # type: ignore[assignment]
        k2s: Float[Array, "ndat"] = solutions.k2  # type: ignore[assignment]

        # Convert to physical units and compute tidal deformability
        return self._create_family_data(pcs, masses, radii, k2s, ndat)

    def _get_pc_min(self, eos_data: EOSData, min_nsat: float) -> Float[Array, ""]:
        """
        Calculate minimum central pressure from minimum density.

        Args:
            eos_data: EOS quantities
            min_nsat: Minimum density in units of saturation density

        Returns:
            Scalar Array: Minimum central pressure [geometric units]
        """
        min_n_geometric = min_nsat * 0.16 * utils.fm_inv3_to_geometric
        pc_min = utils.interp_in_logspace(min_n_geometric, eos_data.ns, eos_data.ps)
        return pc_min

    def _get_pc_max(self, eos_data: EOSData) -> Float[Array, ""]:
        """
        Calculate maximum causal central pressure.

        The maximum pressure is where the EOS becomes non-causal (cs2 >= 1).
        If the EOS is everywhere causal, use the maximum tabulated pressure.

        Args:
            eos_data: EOS quantities

        Returns:
            Scalar Array: Maximum central pressure [geometric units]
        """
        # Find first non-causal point
        mask = eos_data.cs2 >= 1.0
        any_noncausal = jnp.any(mask)
        indices = jnp.arange(len(eos_data.cs2))
        masked_indices = jnp.where(mask, indices, len(eos_data.cs2))
        first_noncausal_idx = jnp.min(masked_indices)

        # Use first non-causal point or last point if all causal
        idx = jnp.where(any_noncausal, first_noncausal_idx, len(eos_data.ps) - 1)
        pc_max = eos_data.ps[idx]
        return pc_max

    def _create_family_data(
        self,
        pcs: Float[Array, "ndat"],
        masses: Float[Array, "ndat"],
        radii: Float[Array, "ndat"],
        k2s: Float[Array, "ndat"],
        ndat: int,
    ) -> FamilyData:
        """
        Shared post-processing: unit conversion, compactness limits, interpolation.

        Args:
            pcs: Central pressures [geometric units]
            masses: Masses [geometric units]
            radii: Radii [geometric units]
            k2s: Love numbers [dimensionless]
            ndat: Number of points for output grid

        Returns:
            FamilyData: Processed family curves in physical units
        """
        # Calculate compactness
        compactness = masses / radii

        # Convert to physical units
        masses_solar = masses / utils.solar_mass_in_meter
        radii_km = radii / 1e3

        # Calculate tidal deformability
        lambdas = 2.0 / 3.0 * k2s * jnp.power(compactness, -5.0)

        # Limit masses to be below MTOV (removes unstable branch)
        pcs_lim, masses_lim, radii_lim, lambdas_lim = utils.limit_by_MTOV(
            pcs, masses_solar, radii_km, lambdas
        )

        # Get a mass grid and interpolate, since we might have some duplicate points
        mass_grid = jnp.linspace(jnp.min(masses_lim), jnp.max(masses_lim), ndat)
        radii_interp = jnp.interp(mass_grid, masses_lim, radii_lim)
        lambdas_interp = jnp.interp(mass_grid, masses_lim, lambdas_lim)
        pcs_interp = jnp.interp(mass_grid, masses_lim, pcs_lim)
        log10pcs = jnp.log10(pcs_interp)

        return FamilyData(
            log10pcs=log10pcs,
            masses=mass_grid,
            radii=radii_interp,
            lambdas=lambdas_interp,
        )
