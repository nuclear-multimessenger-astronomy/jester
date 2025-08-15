r"""
Post-TOV (modified TOV) equation solver in the scalar tensor theory.

This module modify the standard TOV equations to calculate stellar structure solution in the scalar tensor theory. 
# TODO: Explain methods

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Reference:** Stephanie M. Brown 2023 ApJ 958 125
"""

from . import utils
import jax
import jax.numpy as jnp
from jax import lax
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

def tov_ode_iter(h, y, eos):
    # EOS quantities
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]

    # scalar-tensor parameters
    beta_ST = eos["beta_ST"]

    r, m, _, psi, phi = y
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    
    # FIXME speed of sound term will be used in tidal deformability calculations
    #dedp = e / p * jnp.interp(h, hs, dloge_dlogps)

    #scalar coupling function
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
    dmdh = (4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 2) * e 
            + 0.5 * r * (r - 2.0 * m) * jnp.power(psi, 2)) * drdh
    dnudh = (2 * (m + 4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 3) * p)
             / (r * (r - 2.0 * m)) + r * jnp.power(psi, 2)) * drdh
    dpsidh = (
        (4.0 * jnp.pi * jnp.power(A_phi, 4) * r / (r - 2.0 * m)
         * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi))
        - (2.0 * (r - m) / (r * (r - 2.0 * m)) * psi)
    ) * drdh

    dphidh = psi * drdh
    return drdh, dmdh, dnudh, dpsidh, dphidh


def SText_ode(r, y, eos):
    m, _, _, psi = y
    dmdr = 0.5 * r * (r - 2.0 * m) * jnp.square(psi)
    dnudr = (2.0 * m) / (r * (r - 2.0 * m)) + r * jnp.square(psi)
    dphidr = psi
    dpsidr = -2.0 * (r - m) / (r * (r - 2.0 * m)) * psi
    return dmdr, dnudr, dphidr, dpsidr


def tov_solver(eos, pc):
    r"""
    Solve the Scalar Tensor TOV equations for a given central pressure.
    """
    # Extract EOS interpolation arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]

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
    psi0 = 0.0

    ###########################################################
    # initial guess values
    nu0 = eos["nu_c"]
    phi0 = eos["phi_c"]
    damping = 0.8
    max_iterations = 2000
    tol = 1e-6

    # ------------------------------
    # Function with iteration inside 
    # ------------------------------
    def run_iteration(nu0_init, phi0_init):
        # Initial carry tanpa simpan Solution
        init_state = (0, nu0_init, phi0_init, 0.0, 0.0, 1.0, 1.0)

        def cond_fun(state):
            i, nu0_local, phi0_local, R_final, M_inf_final, nu_inf_final, phi_inf_final = state
            return (i < max_iterations) & ((jnp.abs(nu_inf_final) >= tol*1e2) | (jnp.abs(phi_inf_final) >= tol))

        def body_fun(state):
            i, nu0_local, phi0_local, _, _, _, _ = state

            # Interior
            y0 = (r0, m0, nu0_local, psi0, phi0_local)
            sol_iter = diffeqsolve(
                ODETerm(tov_ode_iter),
                Dopri5(scan_kind="bounded"),
                t0=h0,
                t1=0,
                dt0=dh,
                y0=y0,
                args=eos,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
            )

            R = sol_iter.ys[0][-1]
            M_s = sol_iter.ys[1][-1]
            nu_s = sol_iter.ys[2][-1]
            psi_s = sol_iter.ys[3][-1]
            phi_s = sol_iter.ys[4][-1]

            # Exterior
            y_surf = (M_s, nu_s, phi_s, psi_s)
            r_max = 4*128 * 4.0 * jnp.power(3.0 / (4.0 * jnp.pi * ec), 1.0 / 3.0)
            sol_ext = diffeqsolve(
                ODETerm(SText_ode),
                Dopri5(scan_kind="bounded"),
                t0=R,
                t1=r_max,
                dt0=1e-9,
                y0=y_surf,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6)
            )

            M_inf = sol_ext.ys[0][-1]
            nu_inf = sol_ext.ys[1][-1]
            phi_inf = sol_ext.ys[2][-1]

            nu0_local = nu0_local - damping * nu_inf
            phi0_local = phi0_local - damping * phi_inf


            return (i + 1, nu0_local, phi0_local, R, M_inf, nu_inf, phi_inf)

        final_state = lax.while_loop(cond_fun, body_fun, init_state)
        i_final, nu0_final, phi0_final, R_final, M_inf_final, nu_inf_final, phi_inf_final = final_state

        # After iteration done, recalculate again for final structure.
        # Interior
        y0 = (r0, m0, nu0_final, psi0, phi0_final)
        sol_iter = diffeqsolve(
            ODETerm(tov_ode_iter),
            Dopri5(scan_kind="bounded"),
            t0=h0,
            t1=0,
            dt0=dh,
            y0=y0,
            args=eos,
            saveat=SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
        )

        R = sol_iter.ys[0][-1]
        M_s = sol_iter.ys[1][-1]
        nu_s = sol_iter.ys[2][-1]
        psi_s = sol_iter.ys[3][-1]
        phi_s = sol_iter.ys[4][-1]
        
        y_surf = (M_s, nu_s, phi_s, psi_s)
        r_max = 4*128 * 4.0 * jnp.power(3.0 / (4.0 * jnp.pi * ec), 1.0 / 3.0)
        sol_ext_final = diffeqsolve(
            ODETerm(SText_ode),
            Dopri5(scan_kind="bounded"),
            t0=R_final,
            t1=r_max,
            dt0=1e-9,
            y0=y_surf,
            saveat=SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-6)
        )

        return R_final, M_inf_final, nu_inf_final, phi_inf_final, sol_ext_final

    R, M_inf, nu_inf, phi_inf, sol_ext = run_iteration(nu0, phi0)
    
    # FIXME Tidal deformability calculation has not been implemented
    # Return k2 = 0 temporarily
    k2 = 0
    return M_inf, R, k2

#For diagnostic, used also in demonstration example file.
def tov_solver_printsol(eos, pc):
    r"""
    Solve the Scalar Tensor TOV equations for a given central pressure, and return solution array.
    """
    # Extract EOS interpolation arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]

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
    psi0 = 0.0

    ###########################################################
    # initial guess values
    nu0 = eos["nu_c"]
    phi0 = eos["phi_c"]
    damping = 0.8
    max_iterations = 2000
    tol = 1e-6

    # ------------------------------
    # Function with iteration inside
    # ------------------------------
    def run_iteration(nu0_init, phi0_init):
        # Initial carry tanpa simpan Solution
        init_state = (0, nu0_init, phi0_init, 0.0, 0.0, 1.0, 1.0)

        def cond_fun(state):
            i, nu0_local, phi0_local, R_final, M_inf_final, nu_inf_final, phi_inf_final = state
            return (i < max_iterations) & ((jnp.abs(nu_inf_final) >= tol*1e2) | (jnp.abs(phi_inf_final) >= tol))

        def body_fun(state):
            i, nu0_local, phi0_local, _, _, _, _ = state

            # Interior
            y0 = (r0, m0, nu0_local, psi0, phi0_local)
            sol_iter = diffeqsolve(
                ODETerm(tov_ode_iter),
                Dopri5(scan_kind="bounded"),
                t0=h0,
                t1=0,
                dt0=dh,
                y0=y0,
                args=eos,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
            )

            R = sol_iter.ys[0][-1]
            M_s = sol_iter.ys[1][-1]
            nu_s = sol_iter.ys[2][-1]
            psi_s = sol_iter.ys[3][-1]
            phi_s = sol_iter.ys[4][-1]

            # Exterior
            y_surf = (M_s, nu_s, phi_s, psi_s)
            r_max = 4*128 * 4.0 * jnp.power(3.0 / (4.0 * jnp.pi * ec), 1.0 / 3.0)
            sol_ext = diffeqsolve(
                ODETerm(SText_ode),
                Dopri5(scan_kind="bounded"),
                t0=R,
                t1=r_max,
                dt0=1e-11,
                y0=y_surf,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6)
            )

            M_inf = sol_ext.ys[0][-1]
            nu_inf = sol_ext.ys[1][-1]
            phi_inf = sol_ext.ys[2][-1]

            #damped iteration
            nu0_local = nu0_local - damping * nu_inf
            phi0_local = phi0_local - damping * phi_inf

            jax.debug.print("Iteration {i}: ν∞={nu}, φ∞={phi},νc={nu0}, φc={phi0}, M={M_inf}", i=i, nu=nu_inf, phi=phi_inf,nu0 = nu0_local, phi0=phi0_local, M_inf = M_inf/utils.solar_mass_in_meter)

            return (i + 1, nu0_local, phi0_local, R, M_inf, nu_inf, phi_inf)

        final_state = lax.while_loop(cond_fun, body_fun, init_state)
        i_final, nu0_final, phi0_final, R_final, M_inf_final, nu_inf_final, phi_inf_final = final_state


        # After iteration done, recalculate again for final structure.
        # Interior
        # phi0_final = -0.1
        y0 = (r0, m0, nu0_final, psi0, phi0_final)
        sol_iter = diffeqsolve(
            ODETerm(tov_ode_iter),
            Dopri5(scan_kind="bounded"),
            t0=h0,
            t1=0,
            dt0=dh,
            y0=y0,
            args=eos,
            #saveat=SaveAt(t1=True),
            saveat=SaveAt(ts=jnp.linspace(h0, 0, 500)),
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
        )

        R = sol_iter.ys[0][-1]
        M_s = sol_iter.ys[1][-1]
        nu_s = sol_iter.ys[2][-1]
        psi_s = sol_iter.ys[3][-1]
        phi_s = sol_iter.ys[4][-1]
        
        y_surf = (M_s, nu_s, phi_s, psi_s) 
        r_max = 4*128 * 4.0 * jnp.power(3.0 / (4.0 * jnp.pi * ec), 1.0 / 3.0) 
        sol_ext_final = diffeqsolve(
            ODETerm(SText_ode),
            Dopri5(scan_kind="bounded"),
            t0=R_final,
            t1=r_max,
            dt0=1e-11,
            y0=y_surf,
            saveat=SaveAt(ts=jnp.linspace(R_final, r_max, 500)), #if you want to see curve
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-6)
        )

        return R_final, M_inf_final, nu_inf_final, phi_inf_final, sol_iter, sol_ext_final

    R, M_inf, nu_inf, phi_inf, sol_iter, sol_ext = run_iteration(nu0, phi0)
    
    # FIXME Tidal deformability calculation has not been implemented.
    # Return k2 = 0 temporarily
    k2 = 0
    return M_inf, R, k2, sol_iter, sol_ext
