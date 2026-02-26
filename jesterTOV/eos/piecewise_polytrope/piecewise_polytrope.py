r"""
Piecewise polytrope equation of state model.

Implements the Read et al. (2009) 4-parameter piecewise polytrope EOS, matching
the LALSuite implementation. The model uses an analytical SLy4 crust (4-piece fit)
stitched to a user-specified 3-piece high-density core.

Reference:
    Read et al., Physical Review D 79, 124032 (2009)
    LALSuite: lalsimulation/lib/LALSimNeutronStarEOSPiecewisePolytrope.c
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.logging_config import get_logger
from jesterTOV.tov.data_classes import EOSData

logger = get_logger("jester")

# ---- SLy4 crust parameters (LALSuite values, Read et al. 2009) ----
# 4-piece fit to the low-density outer and inner crust.
# Extracted from LALSimNeutronStarEOSPiecewisePolytrope.c lines 427–431.

_SLY_CRUST_RHO_SI: Float[Array, "4"] = jnp.array(
    [0.0, 2.44033979e10, 3.78358138e14, 2.62780487e15]
)  # Starting rest-mass densities [kg/m³]

_SLY_CRUST_K_SI: Float[Array, "4"] = jnp.array(
    [
        1.0801158752700761e7,
        1.311359898998385e10,
        6.507604807550857e19,
        3.053461077133694e8,
    ]
)  # Polytropic constants [Pa · (kg/m³)^(−Γ)]

_SLY_CRUST_GAMMA: Float[Array, "4"] = jnp.array(
    [1.58424999, 1.28732904, 0.62223344, 1.35692395]
)  # Adiabatic indices (dimensionless)

# LALSuite physical constants (used to match the reference implementation)
_LAL_G_SI: float = 6.67430e-11  # Gravitational constant [m³/(kg·s²)]
_LAL_C_SI: float = 299792458.0  # Speed of light [m/s]
_LAL_G_C2_SI: float = _LAL_G_SI / _LAL_C_SI**2  # G/c² [m/kg]


class PiecewisePolytrope_EOS_model(Interpolate_EOS_model):
    r"""
    4-parameter piecewise polytrope equation of state model.

    Parametrizes the neutron star EOS as a piecewise power law
    :math:`p = K_i \rho^{\Gamma_i}` in each density interval. The model
    consists of:

    - Low-density SLy4 crust: 4 analytical pieces from the LALSuite fit.
    - High-density core: 3 user-specified pieces with adiabatic indices
      :math:`\Gamma_1, \Gamma_2, \Gamma_3`.

    The free parameters are:

    - :math:`\log_{10} p_1^\mathrm{SI}` — :math:`\log_{10}` of the pressure
      in Pa at the reference density :math:`\rho_1 = 10^{17.7}` kg/m³.
    - :math:`\Gamma_1, \Gamma_2, \Gamma_3` — adiabatic indices for the three
      high-density pieces.

    The implementation always builds 8-element piece tables for JAX
    compatibility: the 7-piece case (typical) is padded with a dummy last
    element, while the 8-piece case (extra joining piece when the natural
    crust-core junction falls outside the expected range) uses all 8 slots.
    Branch selection is performed with :func:`jnp.where` so the code is fully
    JAX-traceable.

    Parameters
    ----------
    n_points : int
        Number of pressure grid points in the EOS table (default: 500).

    References
    ----------
    Read et al., Physical Review D 79, 124032 (2009).

    Examples
    --------
    >>> from jesterTOV.eos.piecewise_polytrope import PiecewisePolytrope_EOS_model
    >>> model = PiecewisePolytrope_EOS_model(n_points=500)
    >>> params = {"logp1_si": 34.384, "gamma1": 3.005, "gamma2": 2.988, "gamma3": 2.851}
    >>> eos_data = model.construct_eos(params)
    >>> print(eos_data.ps.shape)  # (500,)
    """

    # Maximum pressure in geometric units [m⁻²].
    _P_MAX_GEOM: float = 2e-9

    def __init__(self, n_points: int = 500) -> None:
        r"""
        Initialise the piecewise polytrope EOS model.

        Parameters
        ----------
        n_points : int
            Number of log-spaced pressure grid points (default: 500).
        """
        super().__init__()
        self.n_points = n_points
        logger.info(
            f"Initialized PiecewisePolytrope_EOS_model with n_points={n_points}"
        )

    def get_required_parameters(self) -> list[str]:
        r"""
        Return the parameter names required by this EOS.

        Returns
        -------
        list[str]
            ``["logp1_si", "gamma1", "gamma2", "gamma3"]``
        """
        return ["logp1_si", "gamma1", "gamma2", "gamma3"]

    def construct_eos(self, params: dict[str, float]) -> EOSData:
        r"""
        Construct the full EOS table from piecewise polytrope parameters.

        The method:

        1. Builds 8-element piece tables with :meth:`_build_piece_tables`.
        2. Generates a log-spaced pressure grid from the minimum piece pressure
           to :attr:`_P_MAX_GEOM`.
        3. Computes energy density and rest-mass density analytically from the
           polytrope formulas.
        4. Converts to nuclear units and calls :meth:`interpolate_eos` to obtain
           enthalpy and :math:`d\ln\varepsilon/d\ln p`.
        5. Computes the speed of sound squared.

        Parameters
        ----------
        params : dict[str, float]
            Must contain the keys ``logp1_si``, ``gamma1``, ``gamma2``,
            ``gamma3``.

        Returns
        -------
        EOSData
            Complete EOS in geometric units.
        """
        logp1_si: float = params["logp1_si"]
        gamma1: float = params["gamma1"]
        gamma2: float = params["gamma2"]
        gamma3: float = params["gamma3"]

        # Build piece tables (always 8 elements for JAX traceability)
        rho_tab, k_tab, gamma_tab, n_tab, a_tab, p_tab = self._build_piece_tables(
            logp1_si, gamma1, gamma2, gamma3
        )

        # Log-spaced pressure grid in geometric units [m⁻²]
        # Start from piece 1 (piece 0 starts at ρ=0, p=0)
        p_min_geom = p_tab[1]
        log_p_array = jnp.linspace(
            jnp.log(p_min_geom), jnp.log(self._P_MAX_GEOM), self.n_points
        )
        p_array_geom: Float[Array, "n_points"] = jnp.exp(log_p_array)

        # Determine which piece each pressure point belongs to
        idx = jnp.clip(jnp.searchsorted(p_tab, p_array_geom, side="right") - 1, 0, 7)

        # Extract piece parameters for each grid point
        k_i = k_tab[idx]
        n_i = n_tab[idx]
        a_i = a_tab[idx]

        # Rest-mass density: ρ = (p / K_i)^(n_i / (n_i + 1))
        rho_array: Float[Array, "n_points"] = (p_array_geom / k_i) ** (
            n_i / (n_i + 1.0)
        )

        # Energy density: ε = (1 + a_i) ρ + n_i p
        eps_array: Float[Array, "n_points"] = (
            1.0 + a_i
        ) * rho_array + n_i * p_array_geom

        # Convert to nuclear units for interpolate_eos
        # p, ε: geometric to MeV/fm³
        # rho: geometric to number density [fm⁻³]
        p_nuclear = p_array_geom / utils.MeV_fm_inv3_to_geometric
        eps_nuclear = eps_array / utils.MeV_fm_inv3_to_geometric
        n_array = rho_array / utils.MeV_fm_inv3_to_geometric / utils.m  # fm⁻³

        # Use base-class method for enthalpy and dloge/dlogp (geometric units)
        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(
            n_array, p_nuclear, eps_nuclear
        )

        # Speed of sound squared
        cs2 = ps / (es * dloge_dlogps)

        return EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=cs2,
            mu=None,
            extra_constraints=None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_piece_tables(
        self,
        logp1_si: float,
        gamma1: float,
        gamma2: float,
        gamma3: float,
    ) -> tuple[
        Float[Array, "8"],
        Float[Array, "8"],
        Float[Array, "8"],
        Float[Array, "8"],
        Float[Array, "8"],
        Float[Array, "8"],
    ]:
        r"""
        Build piece tables with always 8 elements (JAX-compatible).

        The 4-parameter PP EOS can have either 7 or 8 pieces depending on
        whether the crust-core joining density falls inside the expected range.
        Both cases are computed and combined with :func:`jnp.where` so the
        function remains fully JAX-traceable.

        Parameters
        ----------
        logp1_si : float
            :math:`\log_{10}(p_1 / \mathrm{Pa})` at :math:`\rho_1 = 10^{17.7}` kg/m³.
        gamma1, gamma2, gamma3 : float
            Adiabatic indices for the three high-density core pieces.

        Returns
        -------
        rho_tab : Float[Array, "8"]
            Starting rest-mass densities in geometric units [m⁻²].
        k_tab : Float[Array, "8"]
            Polytropic constants in geometric units.
        gamma_tab : Float[Array, "8"]
            Adiabatic indices.
        n_tab : Float[Array, "8"]
            Polytropic indices :math:`n_i = 1 / (\Gamma_i - 1)`.
        a_tab : Float[Array, "8"]
            Integration constants for the energy density.
        p_tab : Float[Array, "8"]
            Starting pressures in geometric units [m⁻²].
        """
        rho1_si: float = 10**17.7  # Reference density [kg/m³]
        rho2_si: float = 1e18  # Second transition density [kg/m³]

        # Pressure at rho1 [Pa]
        p1_si = 10.0**logp1_si

        # Polytropic constants K for the three core pieces [Pa·(kg/m³)^(−Γ)]
        k1_si = p1_si / rho1_si**gamma1
        k2_si = p1_si / rho1_si**gamma2
        k3_si = k2_si * rho2_si ** (gamma2 - gamma3)

        # Natural joining density where SLy crust piece 3 meets core piece 1
        # Continuity: K_crust[3] * ρ₀^Γ_crust[3] = K₁ * ρ₀^Γ₁
        rho0_si = (_SLY_CRUST_K_SI[3] / k1_si) ** (1.0 / (gamma1 - _SLY_CRUST_GAMMA[3]))

        # Extra joining piece needed when ρ₀ is outside [ρ_crust[3], ρ₁]
        need_extra = (rho0_si <= _SLY_CRUST_RHO_SI[3]) | (rho0_si >= rho1_si)

        # ---- 7-piece case, padded to 8 elements (dummy last piece) ----
        rho_7_si = jnp.concatenate(
            [
                _SLY_CRUST_RHO_SI,
                jnp.array([rho0_si, rho1_si, rho2_si, rho2_si * 10.0]),
            ]
        )
        k_7_si = jnp.concatenate(
            [_SLY_CRUST_K_SI, jnp.array([k1_si, k2_si, k3_si, k3_si])]
        )
        gamma_7 = jnp.concatenate(
            [_SLY_CRUST_GAMMA, jnp.array([gamma1, gamma2, gamma3, gamma3])]
        )

        # ---- 8-piece case (extra joining piece between crust and core) ----
        rho_j1_si: float = 5.0e15  # [kg/m³]
        rho_j2_si: float = 1.0e16  # [kg/m³]
        p_j1_si = _SLY_CRUST_K_SI[3] * rho_j1_si ** _SLY_CRUST_GAMMA[3]
        p_j2_si = k1_si * rho_j2_si**gamma1
        gamma_join = jnp.log(p_j2_si / p_j1_si) / jnp.log(rho_j2_si / rho_j1_si)
        k_join_si = p_j1_si / rho_j1_si**gamma_join

        rho_8_si = jnp.concatenate(
            [
                _SLY_CRUST_RHO_SI,
                jnp.array([rho_j1_si, rho_j2_si, rho1_si, rho2_si]),
            ]
        )
        k_8_si = jnp.concatenate(
            [_SLY_CRUST_K_SI, jnp.array([k_join_si, k1_si, k2_si, k3_si])]
        )
        gamma_8 = jnp.concatenate(
            [_SLY_CRUST_GAMMA, jnp.array([gamma_join, gamma1, gamma2, gamma3])]
        )

        # ---- Select between 7 and 8 piece case (JAX-compatible) ----
        rho_tab_si = jnp.where(need_extra, rho_8_si, rho_7_si)
        k_tab_si = jnp.where(need_extra, k_8_si, k_7_si)
        gamma_tab = jnp.where(need_extra, gamma_8, gamma_7)

        # Convert densities and K values to geometric units
        rho_tab = rho_tab_si * _LAL_G_C2_SI  # [m⁻²]
        # K_geom = K_SI · G^(1−Γ) · c^(2Γ−4)  (vectorised over 8 elements)
        k_tab = (
            k_tab_si
            * _LAL_G_SI ** (1.0 - gamma_tab)
            * _LAL_C_SI ** (2.0 * gamma_tab - 4.0)
        )

        # Polytropic index n_i = 1 / (Γ_i − 1)
        n_tab = 1.0 / (gamma_tab - 1.0)

        # Pressure at the start of each piece: p_i = K_i ρ_i^Γ_i
        p_tab = k_tab * rho_tab**gamma_tab

        # Integration constants a_i (sequential recurrence; safe in JAX for fixed size)
        # a_{i} = a_{i-1} + (n_{i-1} − n_i) * p_i / ρ_i
        a_tab: Float[Array, "8"] = jnp.zeros(8)
        for i in range(1, 8):
            a_tab = a_tab.at[i].set(
                a_tab[i - 1] + (n_tab[i - 1] - n_tab[i]) * p_tab[i] / rho_tab[i]
            )

        return rho_tab, k_tab, gamma_tab, n_tab, a_tab, p_tab
