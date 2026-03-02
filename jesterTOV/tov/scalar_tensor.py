r"""
Scalar-Tensor TOV equation solver.

This module implements TOV equations for scalar-tensor theories of gravity,
where the gravitational interaction includes both a metric tensor and a scalar field.

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Reference:** G. Creci et al Phys.Rev.D 111 (2025) 8, 089901 (erratum)
"""

import jax.numpy as jnp
from jax import lax
from diffrax import diffeqsolve, ODETerm, Dopri8, SaveAt, PIDController, Event

from jesterTOV import utils
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, TOVSolution
from jesterTOV.tov.scalar_tensor_utils import (
    build_exterior_basis,
    build_exterior_basis_autodiff,
    coeff_solver,
    compute_tidal_deformabilities,
)


def _tov_ode_iter(h, y, eos):
    r"""
    Scalar-tensor TOV ODE system for interior solution.
    Used for iterating scalar field matching condition.

    Parameters
    ----------
    h : float
        Enthalpy (independent variable).
    y : tuple
        State vector :math:`(r, m, \\nu, \\psi, \\phi)` where:
        - :math:`r`: radial coordinate
        - :math:`m`: mass enclosed
        - :math:`\\nu`: metric function
        - :math:`\\psi`: scalar field derivative :math:`d\\phi/dr`
        - :math:`\\phi`: scalar field
    eos : dict
        Dictionary containing EOS arrays and scalar-tensor parameters:
        - ``p``: pressure array
        - ``h``: enthalpy array
        - ``e``: energy density array
        - ``dloge_dlogp``: logarithmic derivative :math:`d\\log e/d\\log p`
        - ``beta_ST``: scalar-tensor coupling parameter

    Returns
    -------
    tuple
        Derivatives :math:`(dr/dh, dm/dh, d\\nu/dh, d\\psi/dh, d\\phi/dh)`.
    """
    # EOS quantities
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]

    # scalar-tensor parameters
    beta_ST = eos["beta_ST"]

    r, m, nu, psi, phi = y

    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)

    # scalar coupling function
    A_phi = jnp.exp(0.5 * beta_ST * jnp.power(phi, 2))
    alpha_phi = beta_ST * phi

    # Regularization parameter
    EPS = 1e-25

    # Modified dpdr to avoid division by zero
    dpdr = -(e + p) * (
        (m + 4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 3) * p)
        / (r * (r - 2.0 * m + EPS))  # Regularize denominator
        + 0.5 * r * jnp.power(psi, 2)
        + alpha_phi * psi
    )

    # Safe division for drdh (handles dpdr ≈ 0)
    safe_dpdr = jnp.where(
        jnp.abs(dpdr) < EPS, jnp.copysign(EPS, dpdr), dpdr  # Preserve sign
    )
    drdh = (e + p) / safe_dpdr  # Numerically stable division

    # Remaining equations with regularized denominators
    dmdh = (
        4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 2) * e
        + 0.5 * r * (r - 2.0 * m) * jnp.power(psi, 2)
    ) * drdh

    dnudh = (
        2
        * (m + 4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 3) * p)
        / (r * (r - 2.0 * m + EPS))  # Regularized
        + r * jnp.power(psi, 2)
    ) * drdh

    dpsidh = (
        (
            4.0
            * jnp.pi
            * jnp.power(A_phi, 4)
            * r
            / (r - 2.0 * m + EPS)  # Regularized
            * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi)
        )
        - (2.0 * (r - m) / (r * (r - 2.0 * m + EPS)) * psi)  # Regularized
    ) * drdh

    dphidh = psi * drdh
    return drdh, dmdh, dnudh, dpsidh, dphidh


def _tov_ode_iter_tidal(h, y, eos):
    r"""
    Scalar-tensor TOV ODE system for interior solution with tidal deformability.

    Parameters
    ----------
    h : float
        Enthalpy (independent variable).
    y : tuple
        State vector :math:`(r, m, \\nu, \\psi, \\phi, H_0, H_0', \\delta\\phi, \\delta\\phi')` where:
        - :math:`r`: radial coordinate
        - :math:`m`: mass enclosed
        - :math:`\\nu`: metric function
        - :math:`\\psi`: scalar field derivative :math:`d\\phi/dr`
        - :math:`\\phi`: scalar field
        - :math:`H_0`: metric perturbation (tidal field)
        - :math:`H_0'`: derivative of :math:`H_0`
        - :math:`\\delta\\phi`: scalar field perturbation
        - :math:`\\delta\\phi'`: derivative of :math:`\\delta\\phi`
    eos : dict
        Dictionary containing EOS arrays and scalar-tensor parameters:
        - ``p``: pressure array
        - ``h``: enthalpy array
        - ``e``: energy density array
        - ``dloge_dlogp``: logarithmic derivative :math:`d\\log e/d\\log p`
        - ``beta_ST``: scalar-tensor coupling parameter

    Returns
    -------
    tuple
        Derivatives :math:`(dr/dh, dm/dh, d\\nu/dh, d\\psi/dh, d\\phi/dh, dH_0/dh, dH_0'/dh, d\\delta\\phi/dh, d\\delta\\phi'/dh)`.
    """
    # EOS quantities
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    beta_ST = eos["beta_ST"]  # scalar-tensor parameter

    r, m, nu, psi, phi, H0, H0_prime, delta_phi, delta_phi_prime = y
    EPS = 1e-25  # small value to avoid zero division error

    # Interpolate EOS
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)

    # Scalar field terms
    A_phi = jnp.exp(0.5 * beta_ST * jnp.power(phi, 2))
    # Note that there is a alpha_phi definition difference between Brown (2023) and Creci (2023)
    # Here we use Brown (2023) definition for TOV solver, and use alpha_phi(Creci) = - alpha_phi(Brown) for tidal deformability calcs
    alpha_phi = beta_ST * phi
    A_phi4 = jnp.power(A_phi, 4)
    four_pi_Aphi4 = 4.0 * jnp.pi * A_phi4
    r2 = r * r
    r3 = r2 * r

    # Core equations -----------------------------------------------------------
    # dpdr with regularization
    denom_non_tidal = r - 2.0 * m + EPS
    dpdr = -(e + p) * (
        (m + four_pi_Aphi4 * r3 * p) / (r * denom_non_tidal)
        + 0.5 * r * jnp.power(psi, 2)
        + alpha_phi * psi
    )

    # Safe division for drdh
    safe_dpdr = jnp.where(jnp.abs(dpdr) < EPS, jnp.copysign(EPS, dpdr), dpdr)
    drdh = (e + p) / safe_dpdr

    # Remaining derivatives
    dmdh = (four_pi_Aphi4 * r2 * e + 0.5 * r * (r - 2.0 * m) * jnp.power(psi, 2)) * drdh

    dnudh = (
        2 * (m + four_pi_Aphi4 * r3 * p) / (r * denom_non_tidal) + r * jnp.power(psi, 2)
    ) * drdh

    dpsidh = (
        four_pi_Aphi4
        * r
        / denom_non_tidal
        * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi)
        - 2.0 * (r - m) / (r * denom_non_tidal) * psi
    ) * drdh

    dphidh = psi * drdh

    # TOV should be exactly same with Brown (2023) and Pani (2014) paper
    # Tidal deformabilities (ℓ=2) ----------------------------------------------
    comp = m / r
    denom_pert = r - 2.0 * m + EPS

    # Coefficients for H0 equation
    F1 = (4.0 * jnp.pi * jnp.power(r, 3) * A_phi4 * (p - e) + 2.0 * (r - m)) / (
        r * denom_pert
    )

    F0_num = (
        4.0
        * jnp.pi
        * jnp.power(r, 3)
        * p
        * A_phi4
        * (r * (dedp + 9.0) - 2.0 * m * (dedp + 13.0))
        + 4.0 * jnp.pi * jnp.power(r, 3) * e * A_phi4 * (dedp + 5.0) * (r - 2.0 * m)
        - 4.0
        * jnp.power(r, 2)
        * (r - 2.0 * m)
        * jnp.power(psi, 2)
        * (4.0 * jnp.pi * jnp.power(r, 3) * p * A_phi4 + m)
        - 64.0
        * jnp.power(jnp.pi, 2)
        * jnp.power(r, 6)
        * jnp.power(p, 2)
        * jnp.power(A_phi4, 2)
        - 6.0 * r * (r - 2.0 * m)  # ℓ(ℓ+1) = 6 for ℓ=2
        - jnp.power(r, 4) * jnp.power(r - 2.0 * m, 2) * jnp.power(psi, 4)
        - 4.0 * jnp.power(m, 2)
    )
    F0 = F0_num / (jnp.power(r, 2) * jnp.power(r - 2.0 * m, 2))

    Fs_num = (
        4.0
        * jnp.power(r, 2)
        * (
            2.0
            * jnp.pi
            * A_phi4
            * (
                -alpha_phi
                * (
                    (dedp - 9.0) * p + (dedp - 1.0) * e
                )  # changed alpha-phi definition to follow Creci et al (2023)
                + 4.0 * r * p * psi
            )
            + (r - 2.0 * m) * jnp.power(psi, 3)
        )
        + 8.0 * m * psi
    )
    Fs = Fs_num / (r * (r - 2.0 * m))

    # Coefficients for dphi equation
    G1 = F1  # Same as F1
    G0 = (
        4.0
        * jnp.pi
        * r
        * A_phi4
        / (r - 2.0 * m)
        * (
            jnp.power(alpha_phi, 2) * ((dedp + 9.0) * p + (dedp - 7.0) * e)
            + (e - 3.0 * p)
            * (-beta_ST)  # α' = - beta for DEF model, Creci et al notation
        )
        - 6.0 / (r * (r - 2.0 * m))  # ℓ(ℓ+1) = 6 for ℓ=2
        - 4.0 * jnp.power(psi, 2)
    )
    Gs = Fs / 4.0  # As defined in paper

    # Perturbation derivatives
    dH0dh = H0_prime * drdh
    dH0_primedh = (-F1 * H0_prime - F0 * H0 + Fs * delta_phi) * drdh
    ddelta_phidh = delta_phi_prime * drdh
    ddelta_phi_primedh = (-G1 * delta_phi_prime - G0 * delta_phi + Gs * H0) * drdh

    return (
        drdh,
        dmdh,
        dnudh,
        dpsidh,
        dphidh,
        dH0dh,
        dH0_primedh,
        ddelta_phidh,
        ddelta_phi_primedh,
    )


# --------------------------------------------
# This part, define matrix multiplication to solve for matching conditions (as well as it's derivative)
# [ H0OnlyQT(M,q,R)   H0OnlyET(M,q,R)   H0OnlyQS(M,q,R)   H0OnlyES(M,q,R) ]   [ cQT ]   [ H0_int(M,q,R) ]
# [ H0OnlyQT'(M,q,R)  H0OnlyET'(M,q,R)  H0OnlyQS'(M,q,R)  H0OnlyES'(M,q,R) ]  [ cET ] = [ H0'_int(M,q,R) ]
# [ φpOnlyQT(M,q,R)   φpOnlyET(M,q,R)   φpOnlyQS(M,q,R)   φpOnlyES(M,q,R) ]  [ cQS ]   [ φp_int(M,q,R) ]
# [ φpOnlyQT'(M,q,R)  φpOnlyET'(M,q,R)  φpOnlyQS'(M,q,R)  φpOnlyES'(M,q,R) ]  [ cES ]   [ φp'_int(M,q,R) ]
# Left matrix from infinity expansion, right matrix from tov solver, and c matrix is what we solve for to determine lambdas.


class ScalarTensorTOVSolver(TOVSolverBase):
    r"""
    Scalar-tensor theory TOV solver.

    Solves modified TOV equations that include scalar field coupling.
    The solution requires iterative solving to match boundary conditions
    at the star surface and spatial infinity.

    Implements the scalar-tensor TOV equations with tidal deformability
    following Creci et al. (2023) Phys.Rev.D 111 (2025) 8, 089901 (erratum).

    Parameters
    ----------
    beta_ST : float, optional
        Scalar-tensor coupling parameter :math:`\\beta_{\\mathrm{ST}}`.
    phi_inf_tgt : float, optional
        Target asymptotic value of the scalar field at infinity.
    phi_c : float, optional
        Central value of the scalar field.

    Notes
    -----
    The solver computes both the stellar structure and tidal deformabilities
    (tensor :math:`\\Lambda_T`, scalar :math:`\\Lambda_S`, and mixed
    :math:`\\Lambda_{\\mathrm{ST}}`) using matched asymptotic expansions.
    """

    def solve(
        self, eos_data: EOSData, pc: float, tov_params: dict[str, float]
    ) -> TOVSolution:
        r"""
        Solve scalar-tensor TOV equations for given EOS and central pressure.

        Parameters
        ----------
        eos_data : EOSData
            Equation of state data containing pressure, enthalpy, energy density,
            and logarithmic derivatives.
        pc : float
            Central pressure (geometric units).
        tov_params : dict[str, float]
            Scalar-tensor theory parameters with keys ``beta_ST``, ``nu_c``, ``phi_c``.

        Returns
        -------
        TOVSolution
            Solution containing mass, radius, and Love number k2 in Jordan frame.

        Raises
        ------
        ValueError
            If iteration fails to converge.

        Notes
        -----
        The solver uses iterative matching to find the central scalar field value
        that yields the desired asymptotic value at infinity. Tidal deformabilities
        are computed using two particular interior solutions combined with exterior
        basis functions.
        """
        beta_ST = tov_params["beta_ST"]
        phi_inf_target = tov_params["phi_inf_tgt"]
        phi0 = tov_params["phi_c"]
        # Extract EOS interpolation arrays
        # Convert EOSData to dictionary for ODE solver
        eos_dict = {
            "p": eos_data.ps,
            "h": eos_data.hs,
            "e": eos_data.es,
            "dloge_dlogp": eos_data.dloge_dlogps,
            # Add scalar-tensor parameters
            "beta_ST": beta_ST,
            "phi_c": phi0,
            "phi_inf_target": phi_inf_target,
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

        # Initial values using series expansion near center, GR approximation

        dh = -1e-3 * hc
        h0 = hc + dh
        r0 = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
        r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
        m0 = 4.0 * jnp.pi * ec * jnp.power(r0, 3.0) / 3.0
        m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
        psi0 = 0.0

        H0_center = jnp.power(r0, 2)  # ~r^2 for l=2
        H0_prime_center = 2.0 * r0  # derivative
        delta_phi_center = jnp.power(r0, 2)
        delta_phi_prime_center = 2.0 * r0

        nu0 = 0.0
        damping = 0.5
        max_iterations = 1000
        tol = 1e-5

        def run_iteration(phi0_init):
            big = 1e9
            init_state = (
                0,  # iteration count
                phi0_init,  # phi0_local
                0.0,
                0.0,  # R_final, M_inf_final
                big,  # phi_inf_final
                jnp.array([phi0_init], dtype=jnp.float64),  # prev_x
                jnp.array([big], dtype=jnp.float64),  # prev_F
            )

            # Keep forward_solver as a single function (called once per iteration)
            def forward_solver(params):
                phi0_trial = params[0]
                y0 = (r0, m0, nu0, psi0, phi0_trial)

                # ------ Stop if mass > 20 Msun, should not affect iteration result
                M_limit = 20.0 * utils.solar_mass_in_meter

                def mass_event(t, y, args, **kwargs):
                    return y[1] > M_limit

                # ------
                sol_iter = diffeqsolve(
                    ODETerm(_tov_ode_iter),
                    Dopri8(scan_kind="bounded"),
                    t0=h0,
                    t1=0,
                    dt0=dh,
                    y0=y0,
                    args=eos_dict,
                    saveat=SaveAt(t1=True),
                    stepsize_controller=PIDController(rtol=1e-7, atol=1e-8),
                    event=Event(mass_event),
                    throw=False,
                )
                # In this iteration, failed solver will still be useful for the next iteration.
                # More iteration = more converge to physical value.
                R = sol_iter.ys[0][-1]  # type: ignore[index]
                M_s = sol_iter.ys[1][-1]  # type: ignore[index]
                nu_s = sol_iter.ys[2][-1]  # type: ignore[index]
                psi_s = sol_iter.ys[3][-1]  # type: ignore[index]
                phi_s = sol_iter.ys[4][-1]  # type: ignore[index]

                EPS = 1e-25
                nu_s_prime = 2 * M_s / (R * (R - 2.0 * M_s)) + R * jnp.power(psi_s, 2)

                front = (
                    2
                    * psi_s
                    / jnp.sqrt(jnp.power(nu_s_prime, 2) + 4 * jnp.power(psi_s, 2))
                )
                inside_tanh = jnp.sqrt(
                    jnp.power(nu_s_prime, 2) + 4 * jnp.power(psi_s, 2)
                ) / (nu_s_prime + 2 / R)
                phi_inf = phi_s + front * jnp.arctanh(inside_tanh)
                # Return shifted value (phi_inf - target) instead of just phi_inf
                return jnp.array([phi_inf - phi_inf_target]), (R, M_s)

            # Define core step function for scan
            def step_func(state, _):
                i, phi0, R_prev, M_prev, phi_inf_prev, prev_x, prev_F = state

                x_curr = jnp.array([phi0])
                F_curr, (R, M) = forward_solver(x_curr)

                # Choose step type based on iteration count
                def damped_step():
                    step = -damping * F_curr
                    return x_curr + step, x_curr, F_curr

                def linearized_step():
                    dx = x_curr - prev_x
                    dF = F_curr - prev_F
                    J = dF / (dx + 1e-12)
                    step = -0.8 * F_curr / (J + 1e-12)
                    return x_curr + jnp.clip(step, -1e6, 1e6), x_curr, F_curr

                x_next, new_prev_x, new_prev_F = lax.cond(
                    i < 10, lambda _: damped_step(), lambda _: linearized_step(), None
                )

                return (i + 1, x_next[0], R, M, F_curr[0], new_prev_x, new_prev_F), None

            # Run phases until convergence
            def phase_loop(state):
                # First run 50 iterations mixing damped/linearized
                state, _ = lax.scan(step_func, state, None, 50)

                # Then run 25-step linearized phases until converged
                def cond(state):
                    i, _, _, _, phi_inf, _, _ = state
                    return (i < max_iterations) & (jnp.abs(phi_inf) >= tol)

                state = lax.while_loop(
                    cond, lambda s: lax.scan(step_func, s, None, 25)[0], state
                )
                return state

            final_state = phase_loop(init_state)
            i_final, phi0_final, R_final, M_inf_final, phi_inf_final, _, _ = final_state

            # Return NaN if max iteration reached or enclosed mass reached 20 M_sun
            too_big_mass = (M_inf_final / utils.solar_mass_in_meter) > 20.0
            too_many_iters = i_final >= max_iterations
            returnNAN = too_big_mass | too_many_iters

            def nan_branch(_):
                return (jnp.nan,) * 15

            # Calculate tidal deformability using converged phi0 value
            def compute_branch(_):
                # Interior solve
                # There are two interior particular solutions
                # Below start with case 1, H0 = 0 and normalize by setting cET = 0 (2nd case dphi0=0 with cES=0)
                y0_case1 = (
                    r0,
                    m0,
                    nu0,
                    psi0,
                    phi0_final,
                    0.0,
                    0.0,
                    delta_phi_center,
                    delta_phi_prime_center,
                )
                sol_iter_1 = diffeqsolve(
                    ODETerm(_tov_ode_iter_tidal),
                    Dopri8(scan_kind="bounded"),
                    t0=h0,
                    t1=0,
                    dt0=dh,
                    y0=y0_case1,
                    args=eos_dict,
                    saveat=SaveAt(t1=True),
                    stepsize_controller=PIDController(rtol=1e-7, atol=1e-8),
                    throw=False,
                )

                R = sol_iter_1.ys[0][-1]  # type: ignore[index]
                M_s = sol_iter_1.ys[1][-1]  # type: ignore[index]
                nu_s = sol_iter_1.ys[2][-1]  # type: ignore[index]
                psi_s = sol_iter_1.ys[3][-1]  # type: ignore[index]
                phi_s = sol_iter_1.ys[4][-1]  # type: ignore[index]
                H0_surface_1 = sol_iter_1.ys[5][-1]  # type: ignore[index]
                H0_prime_surface_1 = sol_iter_1.ys[6][-1]  # type: ignore[index]
                delta_phi_surface_1 = sol_iter_1.ys[7][-1]  # type: ignore[index]
                delta_phi_prime_surface_1 = sol_iter_1.ys[8][-1]  # type: ignore[index]

                # case 2
                y0_case2 = (
                    r0,
                    m0,
                    nu0,
                    psi0,
                    phi0_final,
                    H0_center,
                    H0_prime_center,
                    0.0,
                    0.0,
                )
                # y0_case2 = (r0, m0, nu0, psi0, phi0_final, H0_center, H0_prime_center,delta_phi_center, delta_phi_prime_center)
                sol_iter_2 = diffeqsolve(
                    ODETerm(_tov_ode_iter_tidal),
                    Dopri8(scan_kind="bounded"),
                    t0=h0,
                    t1=0,
                    dt0=dh,
                    y0=y0_case2,
                    args=eos_dict,
                    saveat=SaveAt(t1=True),
                    stepsize_controller=PIDController(rtol=1e-7, atol=1e-8),
                    throw=False,
                )
                H0_surface_2 = sol_iter_2.ys[5][-1]  # type: ignore[index]
                H0_prime_surface_2 = sol_iter_2.ys[6][-1]  # type: ignore[index]
                delta_phi_surface_2 = sol_iter_2.ys[7][-1]  # type: ignore[index]
                delta_phi_prime_surface_2 = sol_iter_2.ys[8][-1]  # type: ignore[index]
                return (
                    R_final,
                    M_inf_final,
                    nu_s,
                    phi_inf_final,
                    psi_s,
                    phi_s,
                    M_s,
                    H0_surface_1,
                    H0_prime_surface_1,
                    delta_phi_surface_1,
                    delta_phi_prime_surface_1,
                    H0_surface_2,
                    H0_prime_surface_2,
                    delta_phi_surface_2,
                    delta_phi_prime_surface_2,
                )

            return lax.cond(returnNAN, nan_branch, compute_branch, operand=None)

        (
            R,
            M_inf,
            nu_s,
            phi_inf,
            psi_s,
            phi_s,
            M_s,
            H0_surface_1,
            H0_prime_surface_1,
            delta_phi_surface_1,
            delta_phi_prime_surface_1,
            H0_surface_2,
            H0_prime_surface_2,
            delta_phi_surface_2,
            delta_phi_prime_surface_2,
        ) = run_iteration(phi0)
        # define scalar charge q (Eq. 4.18 & 4.19)
        nu_s_prime = 2 * M_s / (R * (R - 2 * M_s)) + R * psi_s * psi_s
        q = 2 * psi_s / nu_s_prime

        exterior_basis_matrix = build_exterior_basis(M_inf, q, R)
        exterior_basis_matrix_prime = build_exterior_basis_autodiff(M_inf, q, R)

        # The idea: we have 6 coefficients with 4 equations. To reduce coefficients, we set two particular cases
        # Case 1: H0 = 0 so cET = 0
        # Case 2: dphi = 0 so cES = 0
        # And then we normalize with one of interior solution coefficient (here case 2)
        # the ratios between coefficients are then used to calculate tidal deformability
        # therefore, normalization has to be consistent for all matrix.

        # CASE 1 (Scalar deformability)
        # Set cET = 0, so replace second lhs matrix column with -H01 or -dphi01
        interior_sol = (
            H0_surface_2,
            H0_prime_surface_2,
            delta_phi_surface_2,
            delta_phi_prime_surface_2,
        )
        exterior_basis_matrix_1 = exterior_basis_matrix
        exterior_basis_matrix_prime_1 = exterior_basis_matrix_prime

        mat1_p0 = jnp.array(exterior_basis_matrix_1[0])
        mat1_p1 = jnp.array(exterior_basis_matrix_1[1])
        mat1_prime_p0 = jnp.array(exterior_basis_matrix_prime_1[0])
        mat1_prime_p1 = jnp.array(exterior_basis_matrix_prime_1[1])
        mat1_p0 = mat1_p0.at[1].set(-H0_surface_1)
        mat1_p1 = mat1_p1.at[1].set(-delta_phi_surface_1)
        mat1_prime_p0 = mat1_prime_p0.at[1].set(-H0_prime_surface_1)
        mat1_prime_p1 = mat1_prime_p1.at[1].set(-delta_phi_prime_surface_1)
        exterior_basis_matrix_1 = (mat1_p0, mat1_p1)
        exterior_basis_matrix_prime_1 = (mat1_prime_p0, mat1_prime_p1)

        coeffs_1 = coeff_solver(
            interior_sol, exterior_basis_matrix_1, exterior_basis_matrix_prime_1
        )
        cQT1, c2, cQS1, cES = coeffs_1

        # CASE 2 (tensor deformability)
        # Setting cES = 0
        # so change the coeffs into cQT, cET, cQS, c2, where c2 is particular soluion coeff
        # and replace equation relating to ES to be -H01 or -dphi01
        interior_sol = (
            H0_surface_2,
            H0_prime_surface_2,
            delta_phi_surface_2,
            delta_phi_prime_surface_2,
        )
        exterior_basis_matrix_2 = exterior_basis_matrix
        exterior_basis_matrix_prime_2 = exterior_basis_matrix_prime

        mat2_part0 = jnp.array(exterior_basis_matrix_2[0])
        mat2_part1 = jnp.array(exterior_basis_matrix_2[1])
        mat2_prime_part0 = jnp.array(exterior_basis_matrix_prime_2[0])
        mat2_prime_part1 = jnp.array(exterior_basis_matrix_prime_2[1])
        mat2_part0 = mat2_part0.at[3].set(-H0_surface_1)
        mat2_part1 = mat2_part1.at[3].set(-delta_phi_surface_1)
        mat2_prime_part0 = mat2_prime_part0.at[3].set(-H0_prime_surface_1)
        mat2_prime_part1 = mat2_prime_part1.at[3].set(-delta_phi_prime_surface_1)

        exterior_basis_matrix_2 = (mat2_part0, mat2_part1)
        exterior_basis_matrix_prime_2 = (mat2_prime_part0, mat2_prime_part1)

        coeffs_2 = coeff_solver(
            interior_sol, exterior_basis_matrix_2, exterior_basis_matrix_prime_2
        )
        cQT2, cET, cQS2, c2 = coeffs_2

        # Final coefficients
        coeffs = cQT1, cQT2, cET, cQS1, cQS2, cES

        lambda_T, lambda_S, lambda_ST1, lambda_ST2 = compute_tidal_deformabilities(
            coeffs
        )
        # lambda_ST1 should have same value with lambda_ST2

        # Jordan frame conversion
        A_phi_inf = jnp.exp(0.5 * beta_ST * jnp.power(phi_inf_target, 2))
        A_phi_s = jnp.exp(0.5 * beta_ST * jnp.power(phi_s, 2))
        R_jordan = A_phi_s * R
        M_inf_jordan = (1 / A_phi_inf) * (
            M_inf + (beta_ST * phi_inf_target * (-q * M_inf))
        )
        # Tidal deforms dimensionless, multiplied by ADM mass ^-5
        Lambda_T_J = lambda_T * jnp.power(M_inf, -5.0)
        Lambda_S_J = (
            (
                jnp.exp(2 * beta_ST * jnp.power(phi_inf_target, 2))
                / (4 * beta_ST * beta_ST * phi_inf_target * phi_inf_target)
            )
            * lambda_S
            * jnp.power(M_inf, -5.0)
        )
        Lambda_ST1_J = (
            (
                -jnp.exp(beta_ST * jnp.power(phi_inf_target, 2))
                / (2 * beta_ST * phi_inf_target)
            )
            * lambda_ST1
            * jnp.power(M_inf, -5.0)
        )  # or lambda_ST2, must be same
        Lambda_ST2_J = (
            (
                -jnp.exp(beta_ST * jnp.power(phi_inf_target, 2))
                / (2 * beta_ST * phi_inf_target)
            )
            * lambda_ST2
            * jnp.power(M_inf, -5.0)
        )  # or lambda_ST1, must be same
        # @TODO: add function to return other tidal deformabilities or quantities
        return TOVSolution(M=M_inf_jordan, R=R_jordan, k2=3 / 2 * lambda_T / jnp.power(R_jordan, 5))  # type: ignore[arg-type]

    def get_required_parameters(self) -> list[str]:
        r"""
        Return additional parameters required by scalar-tensor TOV solver.

        Returns
        -------
        list[str]
            List of parameter names: ``["beta_ST", "phi_inf_tgt", "phi_c"]``.

        Notes
        -----
        These parameters correspond to:
        - ``beta_ST``: Scalar-tensor coupling constant.
        - ``phi_inf_tgt``: Target asymptotic scalar field value at infinity.
        - ``phi_c``: Central scalar field value.
        """
        return ["beta_ST", "phi_inf_tgt", "phi_c"]
