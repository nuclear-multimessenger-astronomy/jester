r"""
Scalar-Tensor Theory (STT) TOV equation solver for neutron star structure.

This module provides functions to solve the TOV equations in scalar-tensor theories
of gravity, which describe the hydrostatic equilibrium of a spherically symmetric,
static star in modified gravity theories. The implementation follows the approach
used in scalar-tensor gravity with the coupling function A(φ) = exp(βφ²/2).

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Physics Background:**
In scalar-tensor theories, gravity is mediated by both the metric tensor (as in 
General Relativity) and a scalar field φ. The Einstein frame spacetime metric is:

.. math::
    ds² = -e^\nu dt^2 + e^\lambda dr^2 + r^2(d\theta^2 + sin^2\theta d\varphi^2)

The system of coupled differential equations includes:
- Pressure P(r) and energy density ε(r) evolution
- Mass function m(r) 
- Metric functions \nu(r), \lambda(r)
- Scalar field \varphi(r) and its radial derivative \psi(r) = d\varphi/dr

**Reference:** Based on the formulation in jester_dev/mg/pyTOV-STT/tov_stt_solver.py
"""

from . import utils
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
import jax.scipy.optimize as jopt


def A_of_phi(phi, beta):
    r"""
    Compute the scalar field coupling function A(φ).
    
    Parameters:
    -----------
    phi : float or array
        Scalar field value
    beta : float
        Scalar-tensor coupling parameter
        
    Returns:
    --------
    float or array
        Coupling function A(φ) = exp(βφ²/2)
        
    Physics:
    --------
    This function controls how the scalar field couples to matter.
    - A(φ) = 1 + O(φ²) for weak fields
    - The exponential form ensures A(φ) > 0 always
    - β < 0 typical for viable scalar-tensor theories
    """
    return jnp.exp(0.5 * beta * phi**2)


def alpha_of_phi(phi, beta):
    r"""
    Compute the scalar field coupling derivative \alpha(\varphi) = dA/d\varphi / A.
    
    Parameters:
    -----------
    phi : float or array
        Scalar field value
    beta : float
        Scalar-tensor coupling parameter
        
    Returns:
    --------
    float or array
        Coupling derivative \alpha(\varphi) = \beta\varphi
        
    Physics:
    --------
    This appears in the TOV equations as the coupling between
    scalar field gradients and matter. It determines how strongly
    the scalar field affects stellar structure.
    """
    return beta * phi


def stt_tov_ode(h, y, eos):
    r"""
    STT TOV ordinary differential equation system.
    
    This function defines the coupled ODE system for the STT TOV equations.
    The system includes pressure, mass, metric function, scalar field, scalar
    field derivative, and auxiliary variables for tidal deformability.
    
    The STT TOV equations are based on the original Python implementation:
    
    .. math::
        \frac{dr}{dh} &= -\frac{r(r-2m)}{m + 4\pi A^4 r^3 P + \frac{1}{2}r^2(r-2m)\psi^2 + \alpha r^2(P+\varepsilon)\psi} \\
        \frac{dm}{dh} &= 4\pi r^2 \varepsilon A^4 \frac{dr}{dh} + \frac{1}{2}r(r-2m)\psi^2 \frac{dr}{dh} \\
        \frac{d\nu}{dh} &= \frac{dr}{dh} \left( \frac{2(m + 4\pi A^4 r^3 P)}{r(r-2m)} + r\psi^2 \right) \\
        \frac{d\phi}{dh} &= \psi \frac{dr}{dh} \\
        \frac{d\psi}{dh} &= \frac{dr}{dh} \left( \frac{4\pi A^4 r}{r-2m}[\alpha(\varepsilon-3P) + r(\varepsilon-P)\psi] - \frac{2(r-m)\psi}{r(r-2m)} \right)
        
    Args:
        h (float): Enthalpy (independent variable).
        y (tuple): State vector (r, m, \nu, \varphi, \psi, H, \beta) for interior integration.
        eos (dict): EOS interpolation data with additional STT parameters.
        
    Returns:
        tuple: Derivatives (dr/dh, dm/dh, d\nu/dh, d\varphi/dh, d\psi/dh, dH/dh, d\beta/dh).
    """
    # Extract EOS interpolation arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    
    # Extract STT parameters
    beta = eos["beta"]
    
    # FIXME: nu is not used
    # Extract current state variables
    r, m, nu, phi, psi, H, b = y
    
    # Get energy density and pressure from EOS
    eps = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = eps / p * jnp.interp(h, hs, dloge_dlogps)
    
    # Compute scalar field coupling functions
    A = A_of_phi(phi, beta)
    alpha = alpha_of_phi(phi, beta)
    
    # STT TOV equation derivatives following the Python implementation
    # 
    # The Python implementation uses r as independent variable with:
    # dP/dr = -(ε+P) * [(m + 4πA⁴r³P)/(r(r-2m)) + 0.5*r*ψ² + α*ψ]
    # 
    # To convert to h-based integration, we use: 
    # dh/dP = 1/(ε+P), so dr/dh = (dr/dP) * (dP/dh) = (dr/dP) * (ε+P)
    # Since dP/dr = -(ε+P) * factor, we have dr/dP = -1/((ε+P) * factor)
    # Therefore: dr/dh = -1/factor
    # 
    # From Python: factor = (m + 4πA⁴r³P)/(r(r-2m)) + 0.5*r*ψ² + α*ψ
    # So: dr/dh = -r(r-2m) / [m + 4πA⁴r³P + 0.5*r*r*(r-2m)*ψ² + α*r*(r-2m)*ψ]
    
    denominator = (m + 4.0*jnp.pi*A**4*r**3*p + 
                   0.5*r*r*(r - 2.0*m)*psi**2 + 
                   alpha*r*(r - 2.0*m)*psi)
    drdh = -r*(r - 2.0*m) / denominator
    
    # dm/dh (mass continuity with scalar field energy)
    dmdh = (4.0*jnp.pi*r**2*eps*A**4 + 0.5*r*(r - 2.0*m)*psi**2) * drdh
    
    # dν/dh (metric function evolution)
    dnudh = drdh * (2.0*(m + 4.0*jnp.pi*A**4*r**3*p) / (r*(r - 2.0*m)) + r*psi**2)
    
    # dφ/dh (scalar field evolution)
    dphidh = psi * drdh
    
    # dψ/dh (scalar field derivative evolution)
    dpsidh = drdh * (4.0*jnp.pi*A**4*r/(r - 2.0*m) * 
                     (alpha*(eps - 3.0*p) + r*(eps - p)*psi) - 
                     2.0*(r - m)*psi/(r*(r - 2.0*m)))
    
    # Auxiliary variables for tidal deformability (simplified for now)
    # These follow the same pattern as in the standard TOV solver
    # but with STT modifications that need to be implemented carefully
    
    # FIXME: change this!!!
    # For now, use simplified versions 
    dHdh = b * drdh
    
    # Simplified dbdh (needs full STT tidal calculation)
    metric_coeff = 1.0 / (1.0 - 2.0 * m / r)
    C1 = 2.0 / r + metric_coeff * (2.0 * m / (r * r) + 4.0 * jnp.pi * r * (p - eps))
    C0 = metric_coeff * (
        -6 / (r * r)
        + 4.0 * jnp.pi * (eps + p) * dedp
        + 4.0 * jnp.pi * (5.0 * eps + 9.0 * p)
    ) - jnp.power(2.0 * (m + 4.0 * jnp.pi * r * r * r * p) / (r * (r - 2.0 * m)), 2.0)
    
    dbdh = -(C0 * H + C1 * b) * drdh
    
    return drdh, dmdh, dnudh, dphidh, dpsidh, dHdh, dbdh


def stt_tov_ode_exterior(h, y, eos):
    r"""
    STT TOV ordinary differential equation system for vacuum exterior.
    
    This function defines the coupled ODE system for the STT TOV equations
    in the vacuum region outside the star where P = 0.
    
    In the exterior vacuum region:
    - Pressure P = 0 (no matter)
    - Mass m(r) changes only due to scalar field energy
    - Metric and scalar field evolve according to vacuum field equations
    
    Args:
        h (float): Enthalpy (independent variable) - should be 0 in vacuum.
        y (tuple): State vector (P, m, \nu, \varphi, \psi).
        eos (dict): EOS interpolation data with additional STT parameters.
        
    Returns:
        tuple: Derivatives (dP/dh, dm/dh, d\nu/dh, d\varphi/dh, d\psi/dh).
    """
    # FIXME: these variables are unused
    # Extract STT parameters
    beta = eos["beta"]
    
    # Extract current state variables
    P, m, nu, phi, psi = y
    
    # In vacuum, pressure is zero # FIXME: correct?
    P = 0.0
    
    # Estimate radius (this needs refinement in full implementation) # FIXME: do this
    r = jnp.sqrt(3.0 * m / (4.0 * jnp.pi))  # Rough estimate for vacuum region
    
    # Vacuum STT equations
    dPdh = 0.0  # No pressure in vacuum
    
    # Only scalar field energy contributes to mass in vacuum
    drdh = 1.0  # This needs proper calculation from integration history
    dmdh = drdh * 0.5*r*(r - 2*m)*psi**2
    
    dnudh = drdh * (2.0*m/(r*(r - 2.0*m)) + r*psi**2)
    
    dphidh = drdh * psi
    
    dpsidh = drdh * (-2*(r - m)*psi/(r*(r - 2.0*m)))
    
    return dPdh, dmdh, dnudh, dphidh, dpsidh


def calc_k2_stt(R, M, H, b):
    r"""
    Calculate the second Love number k₂ for tidal deformability in STT.
    
    The Love number calculation in STT follows the same general approach
    as in GR, but the auxiliary variables H and β are computed using
    the modified TOV equations that include scalar field effects.
    
    Args:
        R (float): Neutron star radius [geometric units].
        M (float): Neutron star mass [geometric units].
        H (float): Auxiliary tidal variable at surface.
        b (float): Auxiliary tidal variable β at surface.

    Returns:
        float: Second Love number k₂.
    """
    # For now, use the same formula as in GR
    # Future refinement may include STT-specific corrections
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


def _trial_solution(eos, pc, beta, params):
    """
    Trial solution for boundary condition optimization.
    
    This function integrates the STT equations with given central boundary
    conditions and returns the asymptotic deviation from physical behavior.
    
    Args:
        eos (dict): EOS interpolation data
        pc (float): Central pressure 
        beta (float): Scalar-tensor coupling parameter
        params (tuple): Central values (nu_c, phi_c) to optimize
        
    Returns:
        float: Objective function value (deviation from asymptotic conditions)
    """
    nu_c, phi_c = params
    
    # Extract EOS interpolation arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    
    # Add STT parameters to eos dict
    eos_stt = eos.copy()
    eos_stt["beta"] = beta
    
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
    
    # Use optimized boundary conditions
    psi0 = 0.0  # Scalar field derivative starts at zero by symmetry
    
    # Auxiliary variables for tidal deformability
    H0 = r0 * r0
    b0 = 2.0 * r0

    # Initial state vector: [r, m, ν, φ, ψ, H, β]
    y0 = (r0, m0, nu_c, phi_c, psi0, H0, b0)

    # Integrate the STT TOV equations to surface
    sol = diffeqsolve(
        ODETerm(stt_tov_ode),
        Dopri5(scan_kind="bounded"),
        t0=h0,
        t1=0,  # Integrate until surface (h=0)
        dt0=dh,
        y0=y0,
        args=eos_stt,
        saveat=SaveAt(ts=jnp.linspace(h0, 0, 500)),  # Save at regular intervals
        stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    )
    
    # Find surface index where pressure drops to near zero
    r_vals, m_vals, nu_vals, phi_vals, psi_vals, _, _ = sol.ys
    
    # Continue integration into vacuum exterior region
    # Get surface conditions
    surface_idx = len(r_vals) - 1
    r_surface = r_vals[surface_idx]
    m_surface = m_vals[surface_idx]
    nu_surface = nu_vals[surface_idx]
    phi_surface = phi_vals[surface_idx]
    psi_surface = psi_vals[surface_idx]
    
    # Integrate exterior vacuum equations to large radius
    r_max = 10.0 * r_surface  # Extend far into asymptotic region
    
    # Define exterior vacuum ODE system
    def exterior_ode(r, y, args):
        # FIXME: some variables are unused
        m, nu, phi, psi = y
        
        # Vacuum STT equations (no matter)
        dm_dr = 0.5 * r * (r - 2.0*m) * psi**2
        dnu_dr = 2.0*m/(r*(r - 2.0*m)) + r*psi**2
        dphi_dr = psi
        dpsi_dr = -2.0*(r - m)*psi/(r*(r - 2.0*m))
        
        return dm_dr, dnu_dr, dphi_dr, dpsi_dr
    
    # Initial conditions for exterior integration
    y_ext_0 = (m_surface, nu_surface, phi_surface, psi_surface)
    
    # Integrate exterior
    sol_ext = diffeqsolve(
        ODETerm(exterior_ode),
        Dopri5(scan_kind="bounded"),
        t0=r_surface,
        t1=r_max,
        dt0=(r_max - r_surface) / 1000,
        y0=y_ext_0,
        args=eos_stt,
        saveat=SaveAt(ts=jnp.array([r_max])),  # Just save final point
        stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    )
    
    # Extract asymptotic values (convert arrays to scalars)
    _, nu_final, phi_final, _ = sol_ext.ys
    nu_final = nu_final.at[0].get()  # Extract scalar from array
    phi_final = phi_final.at[0].get()  # Extract scalar from array
    
    # Objective: minimize deviation from asymptotic conditions
    # In STT, we expect ν(∞) → 0 and φ(∞) → 0 for physical solutions
    objective = jnp.sqrt(nu_final**2 + phi_final**2)
    
    return objective

# FIXME: this function is unused
def _compute_asymptotic_mass(r_vals, nu_vals):
    """
    Compute the asymptotic gravitational mass from metric behavior.
    
    This function calculates the true gravitational mass using the 
    asymptotic behavior of the metric function ν(r).
    
    Args:
        r_vals (array): Radial coordinate values (non-uniform grid)
        nu_vals (array): Metric function ν(r) values
        
    Returns:
        float: Asymptotic gravitational mass M = dν/dr × r² × e^ν / 2
    """
    # Use values near the outer boundary but not exactly at boundary
    idx = -10 if len(r_vals) > 10 else -1
    r_val = r_vals[idx]
    nu_val = nu_vals[idx]
    
    # Compute derivative numerically for non-uniform grid
    if len(r_vals) > 1:
        # Use JAX gradient with non-uniform spacing
        dnu_dr = jnp.gradient(nu_vals, r_vals)[idx]
    else:
        dnu_dr = 0.0
    
    # Asymptotic mass formula: M = dν/dr × r² × e^ν / 2
    mass = dnu_dr * r_val**2 * jnp.exp(nu_val) / 2.0
    
    return mass


def _find_accurate_surface(r_vals, sol_ts, eos):
    """
    Find stellar surface more accurately using interpolation.
    
    This function uses interpolation to find the precise radius where
    the specific enthalpy h = (ε + P)/ρ - 1 drops to zero.
    
    Args:
        r_vals (array): Radial coordinate values 
        sol_ts (array): Enthalpy values from integration
        eos (dict): EOS interpolation data
        
    Returns:
        float: Accurate stellar radius
    """
    # Find approximate surface index (where enthalpy approaches zero)
    surface_idx = len(r_vals) - 1
    
    # Extract data near the surface for interpolation
    n_points = min(4, len(r_vals))
    start_idx = max(0, surface_idx - n_points + 1)
    
    r_data = r_vals[start_idx:surface_idx+1]
    h_data = sol_ts[start_idx:surface_idx+1]
    
    # Use linear interpolation to find where h = 0
    # (More sophisticated interpolation could be added later)
    if len(r_data) >= 2 and h_data[0] > 0 and h_data[-1] <= 0:
        # Linear interpolation between last positive and first non-positive h
        for i in range(len(h_data) - 1):
            if h_data[i] > 0 and h_data[i+1] <= 0:
                # Linear interpolation
                alpha = -h_data[i] / (h_data[i+1] - h_data[i])
                r_surface = r_data[i] + alpha * (r_data[i+1] - r_data[i])
                return r_surface
    
    # Fallback to grid-based surface detection
    return r_vals[surface_idx]


def stt_tov_solver(eos, pc, beta=-4.5, optimize_bc=True):
    r"""
    Solve the STT TOV equations for a given central pressure and coupling parameter.

    This function integrates the STT TOV equations from the center of the star
    outward to the surface, using the enthalpy as the integration variable.
    The scalar-tensor coupling is controlled by the parameter β.

    Args:
        eos (dict): EOS interpolation data containing:

            - **p**: Pressure array [geometric units]
            - **h**: Enthalpy array [geometric units]
            - **e**: Energy density array [geometric units]
            - **dloge_dlogp**: Logarithmic derivative array

        pc (float): Central pressure [geometric units].
        beta (float, optional): Scalar-tensor coupling parameter. Default is -4.5.
                               β = 0 corresponds to General Relativity.
        optimize_bc (bool, optional): Whether to optimize boundary conditions. Default is True.

    Returns:
        tuple: A tuple containing:

            - **M**: Gravitational mass [geometric units]
            - **R**: Circumferential radius [geometric units]
            - **k2**: Second Love number for tidal deformability

    Note:
        This implementation includes boundary condition optimization to satisfy
        asymptotic conditions, matching the Python reference solver approach.
    """
    if optimize_bc:
        # Optimize boundary conditions for physical asymptotic behavior
        initial_guess = jnp.array([-1.0, -0.106])  # (nu_c, phi_c)
        
        # Define objective function for optimization
        def objective(params):
            return _trial_solution(eos, pc, beta, params)
        
        # Optimize using JAX scipy
        result = jopt.minimize(objective, initial_guess, method='BFGS')
        nu0, phi0 = result.x
        
    else:
        # Use simple initial guesses (fallback to original behavior)
        nu0 = -1.0
        phi0 = -0.106
    
    # Extract EOS interpolation arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    
    # Add STT parameters to eos dict
    eos_stt = eos.copy()
    eos_stt["beta"] = beta
    
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
    
    # Use optimized boundary conditions
    psi0 = 0.0  # Scalar field derivative starts at zero by symmetry
    
    # Auxiliary variables for tidal deformability
    H0 = r0 * r0
    b0 = 2.0 * r0

    # Initial state vector: [r, m, ν, φ, ψ, H, β]
    y0 = (r0, m0, nu0, phi0, psi0, H0, b0)

    # Integrate the STT TOV equations with full solution storage
    sol = diffeqsolve(
        ODETerm(stt_tov_ode),
        Dopri5(scan_kind="bounded"),
        t0=h0,
        t1=0,  # Integrate until surface (h=0)
        dt0=dh,
        y0=y0,
        args=eos_stt,
        saveat=SaveAt(ts=jnp.linspace(h0, 0, 1000)),  # Save at regular intervals
        stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    )

    # Extract solution arrays
    r_vals, M_vals, nu_vals, phi_vals, psi_vals, H_vals, b_vals = sol.ys
    
    # Find accurate surface radius using interpolation
    R = _find_accurate_surface(r_vals, sol.ts, eos_stt)
    
    # Get surface values (using last grid point as approximation)
    H_surface = H_vals[-1] 
    b_surface = b_vals[-1]
    
    # Continue integration into vacuum exterior for asymptotic mass calculation
    r_max = 10.0 * R  # Extend far into asymptotic region
    
    # Define exterior vacuum ODE system for mass calculation
    def exterior_ode_mass(r, y, args):
        # FIXME: some variables are unused
        m, nu, phi, psi = y
        
        # Vacuum STT equations (no matter)
        dm_dr = 0.5 * r * (r - 2.0*m) * psi**2
        dnu_dr = 2.0*m/(r*(r - 2.0*m)) + r*psi**2
        dphi_dr = psi
        dpsi_dr = -2.0*(r - m)*psi/(r*(r - 2.0*m))
        
        return dm_dr, dnu_dr, dphi_dr, dpsi_dr
    
    # Initial conditions for exterior integration
    y_ext_0 = (M_vals[-1], nu_vals[-1], phi_vals[-1], psi_vals[-1])
    
    # Integrate exterior for asymptotic mass
    sol_ext = diffeqsolve(
        ODETerm(exterior_ode_mass),
        Dopri5(scan_kind="bounded"),
        t0=R,
        t1=r_max,
        dt0=(r_max - R) / 1000,
        y0=y_ext_0,
        args=eos_stt,
        saveat=SaveAt(ts=jnp.linspace(R, r_max, 500)),  # Save at regular intervals
        stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    )
    
    # Extract exterior solution
    _, nu_ext_vals, _, _ = sol_ext.ys
    r_ext_vals = sol_ext.ts
    
    # FIXME: these variables are unused
    # Combine interior and exterior solutions for mass calculation
    r_combined = jnp.concatenate([r_vals, r_ext_vals[1:]])  # Skip duplicate at boundary
    nu_combined = jnp.concatenate([nu_vals, nu_ext_vals[1:]])
    
    # Compute asymptotic mass using metric behavior at the outermost point
    # Use the analytical derivative from the ODE system instead of finite differences
    idx = -10 if len(r_ext_vals) > 10 else -1
    r_asymptotic = r_ext_vals[idx]
    nu_asymptotic = nu_ext_vals[idx]
    m_asymptotic = sol_ext.ys[0][idx]  # Mass from exterior solution
    psi_asymptotic = sol_ext.ys[3][idx]  # ψ from exterior solution
    
    # Analytical derivative from exterior ODE: dν/dr = 2m/(r(r-2m)) + r·ψ²
    dnu_dr_analytical = (2.0 * m_asymptotic / (r_asymptotic * (r_asymptotic - 2.0 * m_asymptotic)) + 
                        r_asymptotic * psi_asymptotic**2)
    
    # Asymptotic mass formula: M = dν/dr × r² × e^ν / 2
    M_asymptotic = dnu_dr_analytical * r_asymptotic**2 * jnp.exp(nu_asymptotic) / 2.0
    
    # Calculate k2 using auxiliary variables
    k2 = calc_k2_stt(R, M_asymptotic, H_surface, b_surface)

    return M_asymptotic, R, k2