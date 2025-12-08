r"""
Post-TOV (modified TOV) equation solver in the scalar tensor theory.

This module modify the standard TOV equations to calculate stellar structure solution in the scalar tensor theory. 
# TODO: Explain methods

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Reference:** Stephanie M. Brown 2023 ApJ 958 125
"""

from . import utils
import jax
from jax import lax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, Event, Dopri8


# Scalar-Tensor TOV-solver and tidal-deformability calculation from Astrophys.J. 958 (2023) 2, 125
# Needs to be optimized on GPU
# Use as it is, corectness of the result still questionable
def calc_k2_ST(R, M, y_R):
    r"""
    Calculate the even-parity perturbation tidal Love number k₂ in scalar-tensor gravity theories.
    
    This implements a Eq. 56a from Brown, Astrophys.J. 958 (2023) 2, 125, with minor correction.
    
    Parameters
    ----------
    R : float or array_like
        Neutron star radius in geometric units (same units as M)
    M : float or array_like  
        Neutron star mass in geometric units (same units as R)
    y_R : float or array_like
        Value of the logarithmic derivative y(r) = r * (dH/dr) / H evaluated at the surface r = R,
        where H is the even-parity tensor perturbation.
    
    Returns
    -------
    k2 : float or array_like
        Dimensionless tidal Love number k₂. Returns the same shape as input parameters.
    
    Notes
    -----
    - The compactness is defined as C = M/R
    - A sign error is suspected in the original paper's logarithmic term. This implementation
      uses a positive sign (3 * (1-2C)²(...) * log(1-2C)) instead of the negative sign in Brown's paper.
    - Valid for 0 < C < 0.5 (sub-compactness limit)
    - The expression becomes singular as C → 0.5 (black hole limit)
    
    """
    C = M / R  # compactness

    num = (
        8.0
        * jnp.power(2.0 * C - 1.0, 2.0)
        * jnp.power(C, 5.0)
        * (2.0 + 2.0 * C * (y_R - 1.0) - y_R)
    )

    den = 5.0 * (
        2.0
        * C
        * (
            6.0
            - 3.0 * y_R
            + jnp.power(C, 2) * (26.0 - 22.0 * y_R)
            + 4.0 * jnp.power(C, 4) * (1.0 + y_R)
            + 3.0 * C * (-8.0 + 5.0 * y_R)
            + jnp.power(C, 3) * (6.0 * y_R - 4.0)
        )
        + 3.0 #flipped to plus, possible typo in Brown's paper
        * jnp.power(1.0 - 2.0 * C, 2.0)
        * (2.0 + 2.0 * C * (y_R - 1.0) - y_R)
        * jnp.log(1.0 - 2.0 * C)
    )

    return num / den

def calc_kappa2_ST(R, M, w_R):
    r"""
    Calculate the scalar perturbation tidal Love number κ₂ in scalar-tensor gravity theories.
    
    This implements a Eq. 56a from Brown, Astrophys.J. 958 (2023) 2, 125, with minor correction.
    
    Parameters
    ----------
    R : float or array_like
        Neutron star radius in geometric units (same units as M)
    M : float or array_like  
        Neutron star mass in geometric units (same units as R)
    w_R : float or array_like
        Value of the logarithmic derivative w(r) = r * (dφ/dr) / φ evaluated at the surface r = R,
        where φ is the scalar field. This characterizes the scalar field configuration.
    
    Returns
    -------
    k2 : float or array_like
        Dimensionless scalar perturbation tidal Love number κ₂. Returns the same shape as input parameters.
    
    Notes
    -----
    - The compactness is defined as C = M/R
    - Valid for 0 < C < 0.5 (sub-compactness limit)
    - The expression becomes singular as C → 0.5 (black hole limit)
    
    """
    C = M / R  # compactness

    num = (
        4.0
        * jnp.power(C, 5.0)
        * (2.0 * C - 1.0)
        * (2.0 * C * C * w_R - 6.0 * C * w_R + 3.0 * w_R + 6.0 * C - 6.0)
    )

    den = (
        45.0
        * (
            2.0 * C * (C * (12.0 - 9.0 * w_R) + 3.0 * (-2.0 + w_R) + C * C * (-2.0 + 6.0 * w_R))
            - (2.0 * C - 1.0) #flipped to negative
            * (-6.0 + 6.0 * C + 3.0 * w_R - 6.0 * C * w_R + 2.0 * C * C * w_R)
            * jnp.log(1.0 - 2.0 * C)
        )
    )

    return num / den

    
def tov_ode_iter(h, y, eos):
    r"""
    Solve the Tolman-Oppenheimer-Volkoff (TOV) equations in scalar-tensor gravity without calculating tidal deformability.
    
    This implements the modified TOV equations for neutron stars in scalar-tensor theories, 
    including scalar field coupling effects but excluding tidal perturbation calculations for efficiency in root finding.
    This implement Appendix A of Brown, Astrophys.J. 958 (2023) 2, 125 and will be used for iteration.
    
    Args:
        h: float
            Enthalpy variable (integration variable)
        y: tuple[float, float, float, float, float]
            State vector containing:
            - r: Radial coordinate in geometric units
            - m: Mass function in geometric units
            - nu: Metric function ν(r) related to time-time metric component
            - psi: Scalar field derivative ψ = dφ/dr
            - phi: Scalar field φ
        eos: dict
            Equation of state dictionary containing:
            - p: Pressure array
            - h: Enthalpy array  
            - e: Energy density array
            - dloge_dlogp: Derivative d(log e)/d(log p)
            - beta_ST: Scalar-tensor coupling parameter β
            
    Returns:
        tuple[float, float, float, float, float]: Derivatives of state variables with respect to h:
            - dr/dh: Radial coordinate derivative
            - dm/dh: Mass function derivative  
            - dnu/dh: Metric function derivative
            - dpsi/dh: Scalar field second-derivative 
            - dphi/dh: Scalar field derivative
    
    Note:
        - Uses regularization (EPS = 1e-99) to handle singularities at r = 2m
        - Scalar coupling function: A(φ) = exp(0.5 * β * φ²)
        - Valid for static, spherically symmetric configurations
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

    #scalar coupling function
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
        jnp.abs(dpdr) < EPS, 
        jnp.copysign(EPS, dpdr),  # Preserve sign
        dpdr
    )
    drdh = (e + p) / safe_dpdr  # Numerically stable division
    
    # Remaining equations with regularized denominators
    dmdh = (
        4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 2) * e 
        + 0.5 * r * (r - 2.0 * m) * jnp.power(psi, 2)
    ) * drdh
    
    dnudh = (
        2 * (m + 4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 3) * p) 
        / (r * (r - 2.0 * m + EPS))  # Regularized
        + r * jnp.power(psi, 2)
    ) * drdh
    
    dpsidh = (
        (
            4.0 * jnp.pi * jnp.power(A_phi, 4) * r 
            / (r - 2.0 * m + EPS)  # Regularized
            * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi)
        )
        - (
            2.0 * (r - m) 
            / (r * (r - 2.0 * m + EPS))  # Regularized
            * psi
        )
    ) * drdh

    dphidh = psi * drdh
    return drdh, dmdh, dnudh, dpsidh, dphidh

def tov_ode_iter_tidal(h, y, eos):
    r"""
    Solve the TOV equations in scalar-tensor gravity including tidal deformability calculations.
    
    This extends the standard TOV equations to compute both tensor (even-parity) and scalar 
    tidal Love numbers by solving the coupled perturbation equations. 
    This implement Appendix A of Brown, Astrophys.J. 958 (2023) 2, 125, with minor correction.
    
    
    Args:
        h: float
            Enthalpy variable (integration variable)
        y: tuple[float, float, float, float, float, float, float]
            State vector containing:
            - r: Radial coordinate in geometric units
            - m: Mass function in geometric units  
            - nu: Metric function ν(r)
            - psi: Scalar field derivative ψ = dφ/dr
            - phi: Scalar field φ
            - y_even: Even-parity tensor perturbation variable y(r) = r(dH/dr)/H
            - y_scalar: Scalar perturbation variable w(r) = r(dφ/dr)/φ
        eos: dict
            Equation of state dictionary containing:
            - p: Pressure array
            - h: Enthalpy array
            - e: Energy density array
            - dloge_dlogp: Derivative d(log e)/d(log p)
            - beta_ST: Scalar-tensor coupling parameter β
            
    Returns:
        tuple[float, float, float, float, float, float, float]: Derivatives of state variables:
            - dr/dh, dm/dh, dnu/dh, dpsi/dh, dphi/dh: Core TOV equations in scalar-tensor
            - dy_even/dh: Even-parity tensor perturbation evolution
            - dw_scalar/dh: Scalar perturbation evolution
    
    Note:
        - Implements tidal perturbation equations based on Brown (2023) with corrections
        - Uses regularization (EPS = 1e-99) to handle singularities
        - Computes both k₂ (tensor) and κ₂ (scalar) tidal Love numbers
        - Suspected typos in Eq. A7b in original reference have been corrected
    """
    # EOS quantities
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    beta_ST = eos["beta_ST"]  # scalar-tensor parameter

    r, m, nu, psi, phi, y_even, y_scalar = y
    EPS = 1e-25  # small value to avoid zero division error
    
    # Interpolate EOS
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)
    
    # Scalar field terms
    A_phi = jnp.exp(0.5 * beta_ST * jnp.power(phi, 2))
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
    dmdh = (
        four_pi_Aphi4 * r2 * e 
        + 0.5 * r * (r - 2.0 * m) * jnp.power(psi, 2)
    ) * drdh
    
    dnudh = (
        2 * (m + four_pi_Aphi4 * r3 * p) / (r * denom_non_tidal) 
        + r * jnp.power(psi, 2)
    ) * drdh
    
    dpsidh = (
        four_pi_Aphi4 * r / denom_non_tidal * (
            alpha_phi * (e - 3.0 * p) 
            + r * (e - p) * psi
        )
        - 2.0 * (r - m) / (r * denom_non_tidal) * psi
    ) * drdh

    dphidh = psi * drdh
    
    # Tidal deformabilities (ℓ=2) ----------------------------------------------
    comp = m / r
    inv_1_minus_2comp = 1.0 / (1.0 - 2.0 * comp + EPS)  # Regularized tidal denominator
    
    # Even parity tensor perturbations
    F_even = inv_1_minus_2comp * (1.0 + four_pi_Aphi4 * r2 * (p - e))
    term_sub = 2*inv_1_minus_2comp/r*(m + four_pi_Aphi4 * r3 * p) + r2 * jnp.power(psi, 2) #this fix consistent to GR limit, where this term propto (nu')^2
    Q_even = (
        inv_1_minus_2comp * (
            -6.0 #(ℓ=2)
            + four_pi_Aphi4 * r2 * ((e + p) * dedp + 9.0 * p + 5.0 * e)
        )
        - jnp.power(term_sub, 2)
    ) / r2
    dydh = (-(y_even**2 + y_even * F_even + r2 * Q_even) / r) * drdh
    
    # Scalar perturbations
    F_scalar = F_even  # Same definition as F_even
    Q_scalar = inv_1_minus_2comp * (
        four_pi_Aphi4 * r2 * (1.0 + 4.0 * phi * alpha_phi) * (3.0 * p - e) * beta_ST 
        - 6.0
    ) / r2
    dwdh = (-(y_scalar**2 + y_scalar * F_scalar + r2 * Q_scalar) / r) * drdh

    return drdh, dmdh, dnudh, dpsidh, dphidh, dydh, dwdh

   
def SText_ode(r, y, eos):
    r"""
    Solve the exterior field equations in scalar-tensor gravity (vacuum region).
    
    This implements the vacuum field equations outside the neutron star surface,
    propagating the metric and scalar field solutions from the stellar surface to infinity.
    
    Args:
        r: float
            Radial coordinate in geometric units
        y: tuple[float, float, float, float]
            State vector containing:
            - m: Mass function in geometric units
            - nu: Metric function ν(r) related to time-time metric component
            - phi: Scalar field φ
            - psi: Scalar field derivative ψ = dφ/dr
        eos: dict
            Equation of state dictionary (unused in vacuum, kept for interface consistency)
            
    Returns:
        tuple[float, float, float, float]: Derivatives of state variables with respect to r:
            - dm/dr: Mass function evolution
            - dnu/dr: Metric function evolution
            - dphi/dr: Scalar field evolution (equals ψ)
            - dpsi/dr: Scalar field derivative evolution
    
    Note:
        - Uses regularization (eps = 1e-25) to handle coordinate singularities
        - Assumes vacuum (zero energy density and pressure) outside the star
        - Used to compute asymptotic values for ν and φ
    """
    m, nu, phi, psi = y
    # Use existing safeguarded terms from previous implementation
    eps = 1e-25
    r_safe = r + eps
    denom1 = r_safe * (r_safe - 2.0 * m) + eps  # Reused from prior safeguards

    dmdr = 0.5 * r_safe * (r_safe - 2.0 * m) * jnp.square(psi)
    dnudr = (2.0 * m) / denom1 + r_safe * jnp.square(psi)  # Protected denominator
    dphidr = psi  # No protection needed - direct assignment
    dpsidr = -2.0 * (r_safe - m) / denom1 * psi  # Protected denominator
    return dmdr, dnudr, dphidr, dpsidr

def tov_solver(eos, pc):
    r"""
    Solve the complete scalar-tensor TOV system for neutron star structure and tidal deformability.
    
    This function computes the full neutron star solution including metric functions, scalar field,
    and tidal Love numbers using a shooting method with adaptive iteration strategies.
    
    Args:
        eos: dict
            Equation of state dictionary containing:
            - p: Pressure array
            - h: Enthalpy array  
            - e: Energy density array
            - dloge_dlogp: Derivative d(log e)/d(log p)
            - beta_ST: Scalar-tensor coupling parameter β
            - nu_c: Initial guess for central metric function ν₀
            - phi_c: Initial guess for central scalar field φ₀
        pc: float
            Central pressure in geometric units
            
    Returns:
        tuple[float, float, float]: A tuple containing:
            - M_inf: ADM mass at infinity in geometric units
            - R: Neutron star radius in geometric units
            - k2_jordan: Total tidal Love number in Jordan frame (k₂ + κ₂)
    
    Note:
        - Uses single-phase iterative shooting method with adaptive step selection
        - Implicit boundary conditions determined by `tol`
        - Returns NaN values if solution fails (mass > 20 M☉ or max iterations exceeded)
        - Combines tensor (k₂) and scalar (κ₂) Love numbers for Jordan frame result
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

    # Initial values using series expansion near center, GR approximation
    
    dh = -1e-3 * hc
    h0 = hc + dh
    r0 = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
    r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
    m0 = 4.0 * jnp.pi * ec * jnp.power(r0, 3.0) / 3.0
    m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
    psi0 = 0.0

    y0_tidal = 2.
    w0_tidal = 2.

    ###########################################################
    # initial guess values
    phi_inf_target = eos["phi_inf_tgt"]
    phi0 = eos["phi_c"]
    nu0=0.0
    damping = 0.5
    max_iterations = 1000
    tol = 1e-5


    def run_iteration(phi0_init):
        big = 1e9
        init_state = (
            0,                              # iteration count
            phi0_init,                      # phi0_local
            0.0, 0.0,                       # R_final, M_inf_final
            big,                            # phi_inf_final
            jnp.array([phi0_init], dtype=jnp.float64),  # prev_x
            jnp.array([big], dtype=jnp.float64),        # prev_F
        )

        # Keep forward_solver as a single function (called once per iteration)
        def forward_solver(params):
            phi0_trial = params[0]
            y0 = (r0, m0, nu0, psi0, phi0_trial)

            #------ Stop if mass > 20 Msun
            M_limit = 20.0 * utils.solar_mass_in_meter

            def mass_event(t, y, args, **kwargs):
                return y[1] > M_limit
            #------
            sol_iter = diffeqsolve(
                ODETerm(tov_ode_iter),
                Dopri8(scan_kind="bounded"),
                t0=h0,
                t1=0,
                dt0=dh,
                y0=y0,
                args=eos,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-7, atol=1e-8),
                event=Event(mass_event),
                throw=False,
            )
            
            R = sol_iter.ys[0][-1]
            M_s = sol_iter.ys[1][-1]
            nu_s = sol_iter.ys[2][-1]
            psi_s = sol_iter.ys[3][-1]
            phi_s = sol_iter.ys[4][-1]

            EPS = 1e-99
            nu_s_prime =  2 *M_s / (R * (R - 2.0 * M_s)) + R * jnp.power(psi_s, 2)
            
            front = 2 * psi_s / jnp.sqrt(jnp.power(nu_s_prime,2) + 4 * jnp.power(psi_s,2))
            inside_tanh = jnp.sqrt(jnp.power(nu_s_prime,2) + 4 * jnp.power(psi_s,2)) / (nu_s_prime + 2 / R)
            phi_inf = phi_s + front * jnp.arctanh(inside_tanh)
            # MODIFIED: Return shifted value (phi_inf - target) instead of just phi_inf
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
                i < 10,
                lambda _: damped_step(),
                lambda _: linearized_step(),
                None
            )
    
            return (
                i+1, x_next[0], R, M, F_curr[0], 
                new_prev_x, new_prev_F
            ), None
    
        # Run phases until convergence
        def phase_loop(state):
            # First run 50 iterations mixing damped/linearized
            state, _ = lax.scan(step_func, state, None, 50)
            
            # Then run 25-step linearized phases until converged
            def cond(state):
                i, _, _, _, phi_inf, _, _ = state
                return (i < max_iterations) & (jnp.abs(phi_inf) >= tol)
                
            state = lax.while_loop(
                cond,
                lambda s: lax.scan(step_func, s, None, 25)[0],
                state
            )
            return state
    
        final_state = phase_loop(init_state)
        i_final, phi0_final, R_final, M_inf_final, phi_inf_final, _, _ = final_state

        # MODIFIED: Add target back for debugging output
        # jax.debug.print(
        #     "i_final={i_final:.3e},phi_inf_final={phi_inf_final:.3e}, accuracy = {accuracy:.3e}",
        #     i_final = i_final, phi_inf_final=phi_inf_final + phi_inf_target, accuracy = phi_inf_final)
        # Return NaN if max iteration reached or enclosed mass reached 20 M_sun
        too_big_mass = (M_inf_final / utils.solar_mass_in_meter) > 20.0
        too_many_iters = i_final >= max_iterations
        returnNAN = too_big_mass | too_many_iters

        def nan_branch(_):
            return (jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan)

        # Calculate tidal deformability using converged phi0 value
        def compute_branch(_):
            # Interior solve
            y0 = (r0, m0, nu0, psi0, phi0_final, y0_tidal, w0_tidal)
            sol_iter = diffeqsolve(
                ODETerm(tov_ode_iter_tidal),
                Dopri8(scan_kind="bounded"),
                t0=h0,
                t1=0,
                dt0=dh,
                y0=y0,
                args=eos,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-7, atol=1e-8),
                throw=False,
            )

            R = sol_iter.ys[0][-1]
            M_s = sol_iter.ys[1][-1]
            nu_s = sol_iter.ys[2][-1]
            psi_s = sol_iter.ys[3][-1]
            phi_s = sol_iter.ys[4][-1]
            yR_tidal = sol_iter.ys[5][-1]
            wR_tidal = sol_iter.ys[6][-1]

            return (R_final, M_inf_final, phi_inf_final, yR_tidal, wR_tidal, psi_s, phi_s, M_s)

        return lax.cond(returnNAN, nan_branch, compute_branch, operand=None)
    
    R, M_inf, phi_inf, yR_tidal, wR_tidal, psi_s, phi_s, M_s = run_iteration(phi0)

    #Jordan frame conversion
    beta_ST = eos["beta_ST"]
    A_phi_inf = jnp.exp(0.5 * beta_ST * jnp.power(phi_inf_target, 2))
    R_jordan = A_phi_inf*R
    nu_s_prime =  2 *M_s / (R * (R - 2.0 * M_s)) + R * jnp.power(psi_s, 2)
    M_inf_jordan = (1/A_phi_inf)*(M_inf + (beta_ST*phi_inf_target*psi_s/(jnp.sqrt(jnp.pi)*nu_s_prime)))
    k2_jordan = calc_k2_ST(R, M_inf, yR_tidal) + calc_kappa2_ST(R, M_inf, wR_tidal) 
    # k2_jordan = calc_kappa2_ST(R, M_inf, wR_tidal) 
    return M_inf_jordan, R_jordan, k2_jordan

# For diagnostic, used also in demonstration example file.
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
                (jnp.abs(nu_inf_final) >= tol * 1e2) | (jnp.abs(phi_inf_final) >= tol)
            )

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
            r_max = 4 * 128 * 4.0 * jnp.power(3.0 / (4.0 * jnp.pi * ec), 1.0 / 3.0)
            sol_ext = diffeqsolve(
                ODETerm(SText_ode),
                Dopri5(scan_kind="bounded"),
                t0=R,
                t1=r_max,
                dt0=1e-11,
                y0=y_surf,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
            )

            M_inf = sol_ext.ys[0][-1]
            nu_inf = sol_ext.ys[1][-1]
            phi_inf = sol_ext.ys[2][-1]

            # damped iteration
            nu0_local = nu0_local - damping * nu_inf
            phi0_local = phi0_local - damping * phi_inf

            jax.debug.print(
                "Iteration {i}: ν∞={nu}, φ∞={phi},νc={nu0}, φc={phi0}, M={M_inf}",
                i=i,
                nu=nu_inf,
                phi=phi_inf,
                nu0=nu0_local,
                phi0=phi0_local,
                M_inf=M_inf / utils.solar_mass_in_meter,
            )

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
            # saveat=SaveAt(t1=True),
            saveat=SaveAt(ts=jnp.linspace(h0, 0, 500)),
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
        )

        R = sol_iter.ys[0][-1]
        M_s = sol_iter.ys[1][-1]
        nu_s = sol_iter.ys[2][-1]
        psi_s = sol_iter.ys[3][-1]
        phi_s = sol_iter.ys[4][-1]

        y_surf = (M_s, nu_s, phi_s, psi_s)
        r_max = 4 * 128 * 4.0 * jnp.power(3.0 / (4.0 * jnp.pi * ec), 1.0 / 3.0)
        sol_ext_final = diffeqsolve(
            ODETerm(SText_ode),
            Dopri5(scan_kind="bounded"),
            t0=R_final,
            t1=r_max,
            dt0=1e-11,
            y0=y_surf,
            saveat=SaveAt(
                ts=jnp.linspace(R_final, r_max, 500)
            ),  # if you want to see curve
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
        )

        return (
            R_final,
            M_inf_final,
            nu_inf_final,
            phi_inf_final,
            sol_iter,
            sol_ext_final,
        )

    R, M_inf, nu_inf, phi_inf, sol_iter, sol_ext = run_iteration(nu0, phi0)

    # FIXME: Tidal deformability calculation has not been implemented.
    # Return k2 = 0 temporarily
    k2 = 0
    return M_inf, R, k2, sol_iter, sol_ext
