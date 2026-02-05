r"""
Scalar-Tensor TOV equation solver.

This module implements TOV equations for scalar-tensor theories of gravity,
where the gravitational interaction includes both a metric tensor and a scalar field.

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Reference:** Stephanie M. Brown 2023 ApJ 958 125
"""

import jax.numpy as jnp
from jax import lax
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

from jesterTOV import utils
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, TOVSolution


def _tov_ode_iter(h, y, eos):
    """
    Scalar-tensor TOV ODE system for interior solution.

    Parameters
    ----------
    h : float
        Enthalpy (independent variable)
    y : tuple
        State vector (r, m, nu, psi, phi)
    eos : dict
        EOS data plus scalar-tensor parameters

    Returns
    -------
    tuple
        Derivatives (dr/dh, dm/dh, dnu/dh, dpsi/dh, dphi/dh)
    """
    # EOS quantities
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]

    # scalar-tensor parameters
    beta_ST = eos["beta_ST"]

    r, m, _, psi, phi = y
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)

    # scalar coupling function
    A_phi = jnp.exp(0.5 * beta_ST * jnp.power(phi, 2))
    alpha_phi = beta_ST * phi

    # dpdr
    dpdr = -(e + p) * (
        (m + 4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 3) * p)
        / (r * (r - 2.0 * m))
        + 0.5 * r * jnp.power(psi, 2)
        + alpha_phi * psi
    )

    # chain rule derivatives
    drdh = (e + p) / dpdr
    dmdh = (
        4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 2) * e
        + 0.5 * r * (r - 2.0 * m) * jnp.power(psi, 2)
    ) * drdh
    dnudh = (
        2
        * (m + 4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 3) * p)
        / (r * (r - 2.0 * m))
        + r * jnp.power(psi, 2)
    ) * drdh
    dpsidh = (
        (
            4.0
            * jnp.pi
            * jnp.power(A_phi, 4)
            * r
            / (r - 2.0 * m)
            * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi)
        )
        - (2.0 * (r - m) / (r * (r - 2.0 * m)) * psi)
    ) * drdh

    dphidh = psi * drdh
    return drdh, dmdh, dnudh, dpsidh, dphidh


def _SText_ode(r, y, eos):
    """
    Scalar-tensor ODE system for exterior solution.

    Parameters
    ----------
    r : float
        Radius (independent variable)
    y : tuple
        State vector (m, nu, phi, psi)
    eos : dict
        Not used but required for interface

    Returns
    -------
    tuple
        Derivatives (dm/dr, dnu/dr, dphi/dr, dpsi/dr)
    """
    m, _, _, psi = y
    dmdr = 0.5 * r * (r - 2.0 * m) * jnp.square(psi)
    dnudr = (2.0 * m) / (r * (r - 2.0 * m)) + r * jnp.square(psi)
    dphidr = psi
    dpsidr = -2.0 * (r - m) / (r * (r - 2.0 * m)) * psi
    return dmdr, dnudr, dphidr, dpsidr


class ScalarTensorTOVSolver(TOVSolverBase):
    """
    Scalar-tensor theory TOV solver.

    Solves modified TOV equations that include scalar field coupling.
    The solution requires iterative solving to match boundary conditions
    at the star surface and spatial infinity.

    Note:
        Tidal deformability calculation has not been implemented for
        scalar-tensor theory. The k2 Love number is currently set to 0.
    """

    def solve(self, eos_data: EOSData, pc: float, **kwargs) -> TOVSolution:
        r"""
        Solve scalar-tensor TOV equations for given central pressure.

        The solver uses an iterative procedure to find values of nu_c and phi_c
        (central metric coefficient and scalar field) that satisfy boundary
        conditions at spatial infinity.

        Args:
            eos_data: EOS quantities in geometric units
            pc: Central pressure [geometric units]
            **kwargs: Must contain scalar-tensor parameters:
                - beta_ST: Scalar field coupling parameter
                - nu_c: Initial guess for central metric coefficient
                - phi_c: Initial guess for central scalar field value

        Returns:
            TOVSolution: Mass, radius, and Love number in geometric units.
                        Returns NaN values on solver failure (JAX-compatible).
                        Note: k2 is currently set to 0 (not yet implemented).
        """
        # Extract scalar-tensor parameters from kwargs
        beta_ST = kwargs.get("beta_ST", 0.0)
        nu0 = kwargs.get("nu_c", 0.0)
        phi0 = kwargs.get("phi_c", 0.0)

        # Convert EOSData to dictionary for ODE solver
        eos_dict = {
            "p": eos_data.ps,
            "h": eos_data.hs,
            "e": eos_data.es,
            "dloge_dlogp": eos_data.dloge_dlogps,
            # Add scalar-tensor parameters
            "beta_ST": beta_ST,
            "nu_c": nu0,
            "phi_c": phi0,
        }

        # Extract EOS arrays
        ps = eos_data.ps
        hs = eos_data.hs
        es = eos_data.es

        # Central values and initial conditions
        hc = utils.interp_in_logspace(pc, ps, hs)
        ec = utils.interp_in_logspace(hc, hs, es)
        dloge_dlogps = eos_data.dloge_dlogps
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
        psi0 = 0.0

        # Iteration parameters
        damping = 0.8
        max_iterations = 2000
        tol = 1e-6

        # ------------------------------
        # Function with iteration inside
        # ------------------------------
        def run_iteration(nu0_init, phi0_init):
            # Initial carry
            init_state = (0, nu0_init, phi0_init, 0.0, 0.0, 1.0, 1.0)

            def cond_fun(state):
                (
                    i,
                    nu0_local,
                    phi0_local,
                    R_final,
                    M_inf_final,
                    nu_inf_final,
                    phi_inf_final,
                ) = state
                return (i < max_iterations) & (
                    (jnp.abs(nu_inf_final) >= tol * 1e2)
                    | (jnp.abs(phi_inf_final) >= tol)
                )

            def body_fun(state):
                i, nu0_local, phi0_local, _, _, _, _ = state

                # Interior
                y0 = (r0, m0, nu0_local, psi0, phi0_local)
                sol_iter = diffeqsolve(
                    ODETerm(_tov_ode_iter),
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

                # Handle solver failure
                if sol_iter.ys is None or sol_iter.result != 0:
                    # Return state with large values to exit loop
                    return (
                        max_iterations,
                        nu0_local,
                        phi0_local,
                        jnp.nan,
                        jnp.nan,
                        1e10,
                        1e10,
                    )

                R = sol_iter.ys[0][-1]
                M_s = sol_iter.ys[1][-1]
                nu_s = sol_iter.ys[2][-1]
                psi_s = sol_iter.ys[3][-1]
                phi_s = sol_iter.ys[4][-1]

                # Exterior
                y_surf = (M_s, nu_s, phi_s, psi_s)
                r_max = 4 * 128 * 4.0 * jnp.power(3.0 / (4.0 * jnp.pi * ec), 1.0 / 3.0)
                sol_ext = diffeqsolve(
                    ODETerm(_SText_ode),
                    Dopri5(scan_kind="bounded"),
                    t0=R,
                    t1=r_max,
                    dt0=1e-9,
                    y0=y_surf,
                    saveat=SaveAt(t1=True),
                    stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
                    throw=False,
                )

                # Handle solver failure
                if sol_ext.ys is None or sol_ext.result != 0:
                    return (max_iterations, nu0_local, phi0_local, R, M_s, 1e10, 1e10)

                M_inf = sol_ext.ys[0][-1]
                nu_inf = sol_ext.ys[1][-1]
                phi_inf = sol_ext.ys[2][-1]

                nu0_local = nu0_local - damping * nu_inf
                phi0_local = phi0_local - damping * phi_inf

                return (i + 1, nu0_local, phi0_local, R, M_inf, nu_inf, phi_inf)

            final_state = lax.while_loop(cond_fun, body_fun, init_state)
            (
                i_final,
                nu0_final,
                phi0_final,
                R_final,
                M_inf_final,
                nu_inf_final,
                phi_inf_final,
            ) = final_state

            # After iteration done, recalculate again for final structure
            # Interior
            y0 = (r0, m0, nu0_final, psi0, phi0_final)
            sol_iter = diffeqsolve(
                ODETerm(_tov_ode_iter),
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

            if sol_iter.ys is None or sol_iter.result != 0:
                return jnp.nan, jnp.nan

            R = sol_iter.ys[0][-1]
            M_s = sol_iter.ys[1][-1]
            nu_s = sol_iter.ys[2][-1]
            psi_s = sol_iter.ys[3][-1]
            phi_s = sol_iter.ys[4][-1]

            y_surf = (M_s, nu_s, phi_s, psi_s)
            r_max = 4 * 128 * 4.0 * jnp.power(3.0 / (4.0 * jnp.pi * ec), 1.0 / 3.0)
            sol_ext_final = diffeqsolve(
                ODETerm(_SText_ode),
                Dopri5(scan_kind="bounded"),
                t0=R_final,
                t1=r_max,
                dt0=1e-9,
                y0=y_surf,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
                throw=False,
            )

            if sol_ext_final.ys is None or sol_ext_final.result != 0:
                return R_final, jnp.nan

            M_inf_final = sol_ext_final.ys[0][-1]

            return R_final, M_inf_final

        R, M_inf = run_iteration(nu0, phi0)

        # FIXME: Tidal deformability calculation has not been implemented
        # Return k2 = 0 temporarily
        k2 = 0.0

        return TOVSolution(M=M_inf, R=R, k2=k2)

    def get_required_parameters(self) -> list[str]:
        """
        Scalar-tensor TOV requires 3 additional parameters.

        Returns:
            list[str]: ["beta_ST", "nu_c", "phi_c"]
        """
        return ["beta_ST", "nu_c", "phi_c"]
