r"""
Post-TOV equation solver with beyond-GR corrections.

This module extends the standard TOV equations to include phenomenological
modifications that parameterize deviations from General Relativity. The
modifications are implemented through additional sigma terms in the pressure
gradient equation.

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Reference:** Yagi & Yunes, Phys. Rev. D 88, 023009 (2013)
"""

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

from jesterTOV import utils
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, TOVSolution


def _sigma_func(p, e, m, r, lambda_BL, lambda_DY, lambda_HB, gamma, alpha, beta):
    r"""
    Compute the non-GR correction term sigma for post-TOV equations.

    This function implements various phenomenological modifications to
    General Relativity that appear as additional terms in the TOV equations.
    The corrections are parameterized by several coupling constants.

    The sigma function includes:

    - **Brans-Dicke-like**: :math:`\sigma_{\mathrm{BL}} = -\frac{\lambda_{\mathrm{BL}} r^2}{3}(\varepsilon + 3p)(\varepsilon + p)A`
    - **Dynamical Chern-Simons**: :math:`\sigma_{\mathrm{DY}} = \lambda_{\mathrm{DY}} \frac{2m}{r} p`
    - **Horava-like**: :math:`\sigma_{\mathrm{HB}} = -(\frac{1}{\lambda_{\mathrm{HB}}} - 1) \frac{r}{2} \frac{dp}{dr}`
    - **Post-Newtonian**: :math:`\sigma_{\mathrm{PN}} = \gamma \frac{2m}{r} p \tanh(\alpha(\frac{m}{r} - \beta))`

    Parameters
    ----------
    p : float
        Pressure at current radius
    e : float
        Energy density at current radius
    m : float
        Enclosed mass at current radius
    r : float
        Current radius
    lambda_BL : float
        Bowers-Liang coupling parameter
    lambda_DY : float
        Doneva-Yazadjiev coupling parameter
    lambda_HB : float
        Herrera-Barreto coupling parameter
    gamma : float
        Post-Newtonian amplitude parameter
    alpha : float
        Post-Newtonian steepness parameter
    beta : float
        Post-Newtonian transition point parameter

    Returns
    -------
    float
        Total sigma correction term
    """
    # Metric coefficient A = 1/(1-2m/r)
    A = 1.0 / (1.0 - 2.0 * m / r)
    dpdr = -(e + p) * (m + 4.0 * jnp.pi * r * r * r * p) / r / r * A
    sigma = 0.0
    # models reviewed in https://doi.org/10.1140/epjc/s10052-020-8361-4
    # in Eq. 12, the power of epsilon should be 2
    sigma += -lambda_BL * r * r / 3.0 * (e + 3.0 * p) * (e + p) * A
    sigma += lambda_DY * 2.0 * m / r * p
    sigma += -(1.0 / lambda_HB - 1.0) * r / 2.0 * dpdr
    # post-Newtonian modification
    sigma += gamma * 2.0 * m / r * p * jnp.tanh(alpha * (m / r - beta))
    return sigma


def _tov_ode(h, y, eos):
    r"""
    Post-TOV ordinary differential equation system.

    Includes beyond-GR corrections through the sigma function.

    Parameters
    ----------
    h : float
        Enthalpy (independent variable)
    y : tuple
        State vector (r, m, H, β)
    eos : dict
        EOS interpolation data plus modification parameters

    Returns
    -------
    tuple
        Derivatives (dr/dh, dm/dh, dH/dh, dβ/dh)
    """
    # fetch the eos arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    cs2s = eos["cs2"]
    # actual equations
    r, m, H, b = y
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = 1.0 / utils.interp_in_logspace(h, hs, cs2s)

    # evalute the sigma and dsigmadp
    sigma = _sigma_func(
        p,
        e,
        m,
        r,
        eos["lambda_BL"],
        eos["lambda_DY"],
        eos["lambda_HB"],
        eos["gamma"],
        eos["alpha"],
        eos["beta"],
    )
    dsigmadp_fn = jax.grad(_sigma_func, argnums=0)  # Gradient w.r.t. p
    dsigmade_fn = jax.grad(_sigma_func, argnums=1)  # Gradient w.r.t. e
    dsigmadp = dsigmadp_fn(
        p,
        e,
        m,
        r,
        eos["lambda_BL"],
        eos["lambda_DY"],
        eos["lambda_HB"],
        eos["gamma"],
        eos["alpha"],
        eos["beta"],
    )
    dsigmadp += dedp * dsigmade_fn(
        p,
        e,
        m,
        r,
        eos["lambda_BL"],
        eos["lambda_DY"],
        eos["lambda_HB"],
        eos["gamma"],
        eos["alpha"],
        eos["beta"],
    )

    A = 1.0 / (1.0 - 2.0 * m / r)
    # terms for radius and mass
    dpdr = -(e + p) * (m + 4.0 * jnp.pi * r * r * r * p) / r / r * A
    dpdr += -2.0 * sigma / r  # adding non-GR contribution
    # terms for tidal deformability
    C1 = 2.0 / r + A * (2.0 * m / (r * r) + 4.0 * jnp.pi * r * (p - e))
    C0 = A * (
        -6 / (r * r)
        + 4.0 * jnp.pi * (e + p) * (1.0 + dedp) / (1.0 - dsigmadp)
        + 4.0 * jnp.pi * (4.0 * e + 8.0 * p)
        + 16.0 * jnp.pi * sigma
    ) - jnp.power(2.0 * (m + 4.0 * jnp.pi * r * r * r * p) / (r * (r - 2.0 * m)), 2.0)

    drdh = (e + p) / dpdr
    dmdh = 4.0 * jnp.pi * r * r * e * drdh
    dHdh = b * drdh
    dbdh = -(C0 * H + C1 * b) * drdh

    dydt = drdh, dmdh, dHdh, dbdh

    return dydt


def _calc_k2(R, M, H, b):
    r"""
    Calculate the second Love number k₂ for tidal deformability.

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


class PostTOVSolver(TOVSolverBase):
    """
    Post-TOV solver with phenomenological beyond-GR corrections.

    Solves post-TOV equations with correction term sigma:

    .. math::
        \\frac{dp}{dr} = -\\frac{[\\varepsilon(r) + p(r)][m(r) + 4\\pi r^3 p(r)]}{r[r - 2m(r)]} - \\frac{2\\sigma(r)}{r}

    The sigma function includes:
        - Brans-Dicke-like corrections (lambda_BL)
        - Dynamical Chern-Simons corrections (lambda_DY)
        - Horava-like corrections (lambda_HB)
        - Post-Newtonian corrections (gamma, alpha, beta)
    """

    def solve(self, eos_data: EOSData, pc: float, **kwargs) -> TOVSolution:
        r"""
        Solve post-TOV equations for given central pressure.

        This function integrates the post-TOV equations that include beyond-GR
        corrections. The integration procedure is identical to the standard TOV case,
        but the differential equations include additional sigma terms.

        Args:
            eos_data: EOS quantities in geometric units
            pc: Central pressure [geometric units]
            **kwargs: Must contain theory modification parameters:
                - lambda_BL: Bowers-Liang coupling parameter
                - lambda_DY: Doneva-Yazadjiev coupling parameter
                - lambda_HB: Herrera-Barreto coupling parameter
                - gamma: Post-Newtonian amplitude parameter
                - alpha: Post-Newtonian steepness parameter
                - beta: Post-Newtonian transition point parameter

        Returns:
            TOVSolution: Mass, radius, and Love number in geometric units.
                        Returns NaN values on solver failure (JAX-compatible).

        Notes:
            The modifications affect the stellar structure but the same integration
            method and boundary conditions as the standard TOV case are used.
        """
        # Extract modification parameters from kwargs
        lambda_BL = kwargs.get("lambda_BL", 0.0)
        lambda_DY = kwargs.get("lambda_DY", 0.0)
        lambda_HB = kwargs.get("lambda_HB", 1.0)  # Default 1.0 means no correction
        gamma = kwargs.get("gamma", 0.0)
        alpha = kwargs.get("alpha", 0.0)
        beta = kwargs.get("beta", 0.0)

        # Convert EOSData to dictionary for ODE solver
        eos_dict = {
            "p": eos_data.ps,
            "h": eos_data.hs,
            "e": eos_data.es,
            "cs2": eos_data.cs2,
            "dloge_dlogp": eos_data.dloge_dlogps,
            # Add modification parameters
            "lambda_BL": lambda_BL,
            "lambda_DY": lambda_DY,
            "lambda_HB": lambda_HB,
            "gamma": gamma,
            "alpha": alpha,
            "beta": beta,
        }

        # Extract EOS arrays
        ps = eos_data.ps
        hs = eos_data.hs
        es = eos_data.es
        cs2s = eos_data.cs2

        # Central values and initial conditions
        hc = utils.interp_in_logspace(pc, ps, hs)
        ec = utils.interp_in_logspace(hc, hs, es)
        dedp_c = 1.0 / utils.interp_in_logspace(hc, hs, cs2s)
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

        # Handle solver failure gracefully (JAX-compatible, no asserts)
        if sol.ys is None or sol.result != 0:
            # Return NaN on failure - constraint checking will catch this
            return TOVSolution(M=jnp.nan, R=jnp.nan, k2=jnp.nan)

        R = sol.ys[0][-1]
        M = sol.ys[1][-1]
        H = sol.ys[2][-1]
        b = sol.ys[3][-1]

        k2 = _calc_k2(R, M, H, b)

        return TOVSolution(M=M, R=R, k2=k2)

    def get_required_parameters(self) -> list[str]:
        """
        Post-TOV requires 6 additional theory parameters.

        Returns:
            list[str]: ["lambda_BL", "lambda_DY", "lambda_HB", "gamma", "alpha", "beta"]
        """
        return ["lambda_BL", "lambda_DY", "lambda_HB", "gamma", "alpha", "beta"]
