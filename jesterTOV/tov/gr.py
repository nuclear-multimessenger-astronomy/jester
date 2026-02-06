r"""
General Relativity TOV equation solver.

This module implements the standard Tolman-Oppenheimer-Volkoff equations
for hydrostatic equilibrium in General Relativity.

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Reference:** NMMA code https://github.com/nuclear-multimessenger-astronomy/nmma/
"""

import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

from jesterTOV import utils
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, TOVSolution


def _tov_ode(h, y, eos):
    r"""
    TOV ordinary differential equation system.

    This function defines the coupled ODE system for the TOV equations plus
    tidal deformability. The system is solved in terms of the enthalpy h as
    the independent variable (decreasing from center to surface).

    The TOV equations are:

    .. math::
        \frac{dr}{dh} &= -\frac{r(r-2m)}{m + 4\pi r^3 p} \\
        \frac{dm}{dh} &= 4\pi r^2 \varepsilon \frac{dr}{dh} \\
        \frac{dH}{dh} &= \beta \frac{dr}{dh} \\
        \frac{d\beta}{dh} &= -(C_0 H + C_1 \beta) \frac{dr}{dh}

    where H and :math:`\beta` are auxiliary variables for tidal deformability.

    Parameters
    ----------
    h : float
        Enthalpy (independent variable)
    y : tuple
        State vector (r, m, H, β)
    eos : dict
        EOS interpolation data

    Returns
    -------
    tuple
        Derivatives (dr/dh, dm/dh, dH/dh, dβ/dh)
    """
    # Extract EOS interpolation arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    # Extract current state variables
    r, m, H, b = y
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)

    # Metric coefficient A = 1/(1-2m/r)
    A = 1.0 / (1.0 - 2.0 * m / r)
    # Tidal deformability coefficients
    C1 = 2.0 / r + A * (2.0 * m / (r * r) + 4.0 * jnp.pi * r * (p - e))
    C0 = A * (
        -6 / (r * r)
        + 4.0 * jnp.pi * (e + p) * dedp
        + 4.0 * jnp.pi * (5.0 * e + 9.0 * p)
    ) - jnp.power(2.0 * (m + 4.0 * jnp.pi * r * r * r * p) / (r * (r - 2.0 * m)), 2.0)

    # TOV equation derivatives
    drdh = -r * (r - 2.0 * m) / (m + 4.0 * jnp.pi * r * r * r * p)  # dr/dh
    dmdh = 4.0 * jnp.pi * r * r * e * drdh  # dm/dh
    dHdh = b * drdh  # dH/dh
    dbdh = -(C0 * H + C1 * b) * drdh  # dβ/dh

    return drdh, dmdh, dHdh, dbdh


def _calc_k2(R, M, H, b):
    r"""
    Calculate the second Love number k₂ for tidal deformability.

    The Love number k₂ relates the tidal deformability to the neutron star's
    mass and radius. It is computed from the auxiliary variables H and β
    obtained from the TOV integration.

    The tidal deformability is given by:

    .. math::
        \Lambda = \frac{2}{3} k_2 C^{-5}

    where :math:`C = M/R` is the compactness.

    Parameters
    ----------
    R : float
        Neutron star radius [geometric units]
    M : float
        Neutron star mass [geometric units]
    H : float
        Auxiliary tidal variable at surface
    b : float
        Auxiliary tidal variable β at surface

    Returns
    -------
    float
        Second Love number k₂
    """
    y = R * b / H
    C = M / R

    num = (
        (8.0 / 5.0)
        * jnp.power(1 - 2 * C, 2.0)
        * jnp.power(C, 5.0)
        * (2 * C * (y - 1) - y + 2)
    )
    den = (
        2
        * C
        * (
            4 * (y + 1) * jnp.power(C, 4)
            + (6 * y - 4) * jnp.power(C, 3)
            + (26 - 22 * y) * C * C
            + 3 * (5 * y - 8) * C
            - 3 * y
            + 6
        )
    )
    den -= (
        3
        * jnp.power(1 - 2 * C, 2)
        * (2 * C * (y - 1) - y + 2)
        * jnp.log(1.0 / (1 - 2 * C))
    )

    return num / den


class GRTOVSolver(TOVSolverBase):
    """
    Standard General Relativity TOV solver.

    Solves the TOV equations:

    .. math::
        \\frac{dr}{dh} &= -\\frac{r(r-2m)}{m + 4\\pi r^3 p} \\\\
        \\frac{dm}{dh} &= 4\\pi r^2 \\varepsilon \\frac{dr}{dh}

    plus the equations for tidal deformability.
    """

    def solve(self, eos_data: EOSData, pc: float, **kwargs) -> TOVSolution:
        r"""
        Solve TOV equations for given central pressure.

        This function integrates the TOV equations from the center of the star
        (where r=0, m=0) outward to the surface (where p=0), using the enthalpy
        as the integration variable. The integration starts slightly off-center
        to avoid singularities.

        The solver uses the Dormand-Prince 5th order adaptive method (Dopri5)
        with proper error control for numerical stability.

        Args:
            eos_data: EOS quantities in geometric units
            pc: Central pressure [geometric units]
            **kwargs: Not used for GR TOV (included for interface compatibility)

        Returns:
            TOVSolution: Mass, radius, and Love number in geometric units.
                        Returns NaN values on solver failure (JAX-compatible).

        Notes:
            The integration is performed from center to surface, with the enthalpy
            decreasing from h_center to 0. Initial conditions are set using
            series expansions valid near the center.
        """
        # Convert EOSData to dictionary for ODE solver
        eos_dict = {
            "p": eos_data.ps,
            "h": eos_data.hs,
            "e": eos_data.es,
            "dloge_dlogp": eos_data.dloge_dlogps,
        }

        # Extract EOS arrays
        ps = eos_data.ps
        hs = eos_data.hs
        es = eos_data.es
        dloge_dlogps = eos_data.dloge_dlogps

        # Central values and initial conditions
        hc = utils.interp_in_logspace(pc, ps, hs)
        ec = utils.interp_in_logspace(hc, hs, es)
        dedp_c = ec / pc * jnp.interp(hc, hs, dloge_dlogps)
        dhdp_c = 1.0 / (ec + pc)
        dedh_c = dedp_c / dhdp_c

        # Initial values using series expansion near center
        dh = -1e-3 * hc
        h0 = hc + dh
        r0 = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
        r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
        m0 = 4.0 * jnp.pi * ec * jnp.power(r0, 3.0) / 3.0
        m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
        H0 = r0 * r0
        b0 = 2.0 * r0

        y0 = (r0, m0, H0, b0)

        sol = diffeqsolve(
            ODETerm(_tov_ode),
            Dopri5(scan_kind="bounded"),
            t0=h0,
            t1=0,
            dt0=dh,
            y0=y0,
            args=eos_dict,
            saveat=SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
            throw=False,
        )

        # Extract solution values
        # Note: diffrax always returns arrays, even on failure
        R = sol.ys[0][-1]
        M = sol.ys[1][-1]
        H = sol.ys[2][-1]
        b = sol.ys[3][-1]

        k2 = _calc_k2(R, M, H, b)

        # # FIXME: might remove this
        # # Use jnp.where for JAX-compatible conditional
        # # If solver failed (result != 0), return NaN
        # success = sol.result == 0
        # M_out = jnp.where(success, M, jnp.nan)
        # R_out = jnp.where(success, R, jnp.nan)
        # k2_out = jnp.where(success, k2, jnp.nan)

        return TOVSolution(M=M, R=R, k2=k2)

    def get_required_parameters(self) -> list[str]:
        """
        GR TOV requires no additional parameters beyond EOS.

        Returns:
            list[str]: Empty list (no extra parameters)
        """
        return []
