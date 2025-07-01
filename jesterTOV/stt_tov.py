r"""
Scalar-Tensor Theory (STT) TOV equation solver for neutron star structure.

This module provides functions to solve the TOV equations in scalar-tensor theories
of gravity, which describe the hydrostatic equilibrium of a spherically symmetric,
static star in modified gravity theories. The implementation follows the approach
used in scalar-tensor gravity with the coupling function A(φ) = exp(βφ²/2).

**ARCHITECTURE**: This is a complete rewrite following the Python reference solver
at jester_dev/mg/pyTOV-STT/tov_stt_solver.py. Key changes from previous implementation:

1. **r-based integration** (not enthalpy-based) matching Python reference
2. **Two-stage interior/exterior integration** with proper boundary matching  
3. **Correct state vector**: [P, m, ν, φ, ψ] exactly as in Python reference
4. **Proper surface detection** using specific enthalpy condition
5. **Accurate asymptotic mass** calculation from metric behavior

**Units:** All calculations are performed in geometric units where G = c = 1.

**Physics Background:**
In scalar-tensor theories, gravity is mediated by both the metric tensor (as in 
General Relativity) and a scalar field φ. The Einstein frame spacetime metric is:

.. math::
    ds² = -e^ν dt² + e^λ dr² + r²(dθ² + sin²θ dφ²)

The system of coupled differential equations includes:
- Pressure P(r) and energy density ε(r) evolution
- Mass function m(r) 
- Metric functions ν(r), λ(r)
- Scalar field φ(r) and its radial derivative ψ(r) = dφ/dr

**Reference:** Based exactly on jester_dev/mg/pyTOV-STT/tov_stt_solver.py
"""

from . import utils
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
import jax.scipy.optimize as jopt
from scipy.interpolate import PchipInterpolator
from scipy import optimize
import numpy as np


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
    Compute the scalar field coupling derivative α(φ) = dA/dφ / A.
    
    Parameters:
    -----------
    phi : float or array
        Scalar field value
    beta : float
        Scalar-tensor coupling parameter
        
    Returns:
    --------
    float or array
        Coupling derivative α(φ) = βφ
        
    Physics:
    --------
    This appears in the TOV equations as the coupling between
    scalar field gradients and matter. It determines how strongly
    the scalar field affects stellar structure.
    """
    return beta * phi


def stt_tov_interior(r, y, eos_dict, beta):
    r"""
    STT TOV interior equations using radius as independent variable.
    
    This implements the complete STT-TOV system exactly as in the Python reference.
    The equations describe stellar interior with matter and scalar field.
    
    **Key Physics**:
    
    1. **Hydrostatic equilibrium** (modified by scalar field):
       dP/dr = -(ε+P)[(m + 4πA⁴r³P)/(r(r-2m)) + 0.5*r*ψ² + α*ψ]
       
    2. **Mass continuity** (includes scalar field energy):
       dm/dr = 4πr²εA⁴ + 0.5*r*(r-2m)*ψ²
       
    3. **Metric function evolution**:
       dν/dr = 2(m + 4πA⁴r³P)/(r(r-2m)) + r*ψ²
       
    4. **Scalar field evolution**:
       dφ/dr = ψ
       
    5. **Scalar field derivative evolution**:
       dψ/dr = 4πA⁴r/(r-2m)[α(ε-3P) + r(ε-P)ψ] - 2(r-m)ψ/(r(r-2m))
    
    Args:
        r (float): Radial coordinate (independent variable)
        y (tuple): State vector [P, m, ν, φ, ψ]
        eos_dict (dict): EOS interpolation data
        beta (float): Scalar-tensor coupling parameter
        
    Returns:
        tuple: Derivatives [dP/dr, dm/dr, dν/dr, dφ/dr, dψ/dr]
    """
    P, m, _, phi, psi = y
    
    # Get energy density from EOS
    eps = utils.interp_in_logspace(P, eos_dict["p"], eos_dict["e"])
    
    # Compute scalar field coupling functions
    A = A_of_phi(phi, beta)
    alpha = alpha_of_phi(phi, beta)
    
    # STT TOV equations - exactly following Python reference
    dP_dr = (-(eps + P) * 
             ((m + 4.0*jnp.pi*A**4*r**3*P) / (r*(r - 2.0*m)) + 
              0.5*r*psi**2 + alpha*psi))
    
    dm_dr = (4*jnp.pi*A**4*r**2*eps + 
             0.5*r*(r - 2*m)*psi**2)
    
    dnu_dr = (2.0*(m + 4.0*jnp.pi*A**4*r**3*P) / (r*(r - 2.0*m)) + 
              r*psi**2)
    
    dphi_dr = psi
    
    dpsi_dr = (4*jnp.pi*A**4*r/(r - 2.0*m) * 
               (alpha*(eps - 3.0*P) + r*(eps - P)*psi) - 
               2*(r - m)*psi/(r*(r - 2.0*m)))
    
    return jnp.array([dP_dr, dm_dr, dnu_dr, dphi_dr, dpsi_dr])


def stt_tov_exterior(r, y, beta=None):
    r"""
    STT TOV exterior vacuum equations using radius as independent variable.
    
    In the vacuum region outside the star, pressure P = 0 and the equations
    simplify to pure vacuum field equations. The scalar field still contributes
    to the mass and metric evolution.
    
    **Vacuum equations**:
    
    1. dP/dr = 0 (no matter)
    2. dm/dr = 0.5*r*(r-2m)*ψ² (scalar field energy only)  
    3. dν/dr = 2m/(r(r-2m)) + r*ψ²
    4. dφ/dr = ψ
    5. dψ/dr = -2(r-m)*ψ/(r(r-2m))
    
    Args:
        r (float): Radial coordinate (independent variable)
        y (tuple): State vector [P, m, ν, φ, ψ]
        beta (float, optional): Scalar-tensor coupling parameter (unused in vacuum)
        
    Returns:
        tuple: Derivatives [dP/dr, dm/dr, dν/dr, dφ/dr, dψ/dr]
    """
    # TODO: phi is not used?
    _, m, _, phi, psi = y
    
    # Vacuum STT equations - exactly following Python reference
    dP_dr = 0.0  # No pressure in vacuum
    
    dm_dr = 0.5*r*(r - 2*m)*psi**2  # Only scalar field energy contributes
    
    dnu_dr = 2.0*m/(r*(r - 2.0*m)) + r*psi**2
    
    dphi_dr = psi
    
    dpsi_dr = -2*(r - m)*psi/(r*(r - 2.0*m))
    
    return jnp.array([dP_dr, dm_dr, dnu_dr, dphi_dr, dpsi_dr])


def _find_accurate_surface(r_vals, P_vals, eos_dict):
    """
    Find stellar surface using PCHIP interpolation and root finding.
    
    The stellar surface is defined where the specific enthalpy h = (ε+P)/ρ - 1 = 0.
    This follows the exact method from the Python reference solver.
    
    Args:
        r_vals (array): Radial grid points
        P_vals (array): Pressure values 
        eos_dict (dict): EOS interpolation data
        
    Returns:
        float: Accurate stellar radius
    """
    # Find approximate surface index (where pressure drops to near zero)
    surface_idx = len(r_vals) - 1
    for i in range(len(P_vals)):
        if P_vals[i] <= 0:
            surface_idx = i
            break
    
    # Extract data near the surface for interpolation
    n_points = min(4, surface_idx + 1)
    start_idx = max(0, surface_idx - n_points + 1)
    
    r_data = np.array(r_vals[start_idx:surface_idx+1])
    P_data = np.array(P_vals[start_idx:surface_idx+1])
    
    # Compute specific enthalpy h = (ε+P)/ρ - 1
    h_data = np.zeros_like(P_data)
    for i in range(len(P_data)):
        if P_data[i] > 0:
            eps_i = utils.interp_in_logspace(P_data[i], eos_dict["p"], eos_dict["e"])
            rho_i = utils.interp_in_logspace(P_data[i], eos_dict["p"], eos_dict["rho"])
            h_data[i] = (eps_i + P_data[i]) / rho_i - 1.0
        else:
            h_data[i] = 0.0
    
    # Use PCHIP interpolation and root finding
    if len(r_data) >= 2 and np.any(h_data > 0):
        try:
            h_interp = PchipInterpolator(r_data, h_data)
            radius = optimize.brentq(h_interp, r_data[0], r_data[-1], xtol=1e-16)
            return radius
        except:
            pass
    
    # Fallback to grid-based detection
    return r_vals[surface_idx]


def _compute_asymptotic_mass(r_vals, nu_vals):
    """
    Compute the asymptotic gravitational mass from metric behavior.
    
    At large r, the metric component ν(r) behaves as:
    ν(r) ≈ ln(1 - 2M/r) + O(1/r²)
    
    So M can be extracted from: M = r²(dν/dr)e^ν/2
    
    This follows the exact method from the Python reference solver.
    
    Args:
        r_vals (array): Radial coordinate values
        nu_vals (array): Metric function ν(r) values
        
    Returns:
        float: Asymptotic gravitational mass
    """
    # Use values near the outer boundary
    idx = -10 if len(r_vals) > 10 else -1
    r_val = r_vals[idx]
    nu_val = nu_vals[idx]
    
    # Compute derivative numerically
    dnu_dr = np.gradient(np.array(nu_vals), np.array(r_vals))[idx]
    
    # Asymptotic mass formula from Python reference
    mass = dnu_dr * r_val**2 * np.exp(nu_val) / 2.0
    
    return mass


def _compute_scalar_charge(r_vals, phi_vals):
    """
    Compute the scalar charge from asymptotic scalar field behavior.
    
    At large r, the scalar field behaves as:
    φ(r) ≈ ω/r + O(1/r²)
    
    The scalar charge is ω = -r²(dφ/dr)|_{r→∞}
    
    Args:
        r_vals (array): Radial coordinate values
        phi_vals (array): Scalar field φ(r) values
        
    Returns:
        float: Scalar charge
    """
    # Use values near the outer boundary
    idx = -10 if len(r_vals) > 10 else -1
    r_val = r_vals[idx]
    
    # Compute derivative numerically
    dphi_dr = np.gradient(np.array(phi_vals), np.array(r_vals))[idx]
    
    # Scalar charge formula
    scalar_charge = -dphi_dr * r_val**2
    
    return scalar_charge


def _trial_solution(params, eos_dict, P_c, beta, r_max, N):
    """
    Trial solution for boundary condition optimization.
    
    This function integrates the STT equations with given central boundary
    conditions and returns the asymptotic deviation from physical behavior.
    Follows the exact approach from the Python reference solver.
    
    Args:
        params (tuple): Central values (nu_c, phi_c) to optimize
        eos_dict (dict): EOS interpolation data
        P_c (float): Central pressure
        beta (float): Scalar-tensor coupling parameter
        r_max (float): Maximum integration radius
        N (int): Number of grid points
        
    Returns:
        float: Objective function value (deviation from asymptotic conditions)
    """
    nu_c, phi_c = params
    
    # Set up radial grid
    dr = r_max / (N - 1)
    r_grid = jnp.linspace(dr, r_max, N)  # Start from dr to avoid r=0 singularity
    
    # Initial conditions at r = dr (near center)
    y0 = jnp.array([P_c, 0.0, nu_c, phi_c, 0.0])
    
    # Define ODE function for diffrax
    def ode_func(r, y, args=None):
        return stt_tov_interior(r, y, eos_dict, beta)
    
    # Integrate interior until surface
    sol = diffeqsolve(
        ODETerm(ode_func),
        Dopri5(),
        t0=dr,
        t1=r_max,
        dt0=dr,
        y0=y0,
        args=None,
        saveat=SaveAt(ts=r_grid),
        stepsize_controller=PIDController(rtol=1e-6, atol=1e-8),
        max_steps=50000, # TODO: can this be lowered?
    )
    
    # Find surface (where pressure drops to zero)
    P_vals = sol.ys[0]
    surface_idx = len(P_vals) - 1
    for i in range(len(P_vals)):
        if P_vals[i] <= 0:
            surface_idx = i
            break
    
    # Get surface conditions
    r_surface = r_grid[surface_idx]
    surface_conditions = jnp.array([sol.ys[j][surface_idx] for j in range(5)])
    
    # Continue integration into vacuum exterior
    r_ext_max = 10.0 * r_surface
    r_ext_grid = jnp.linspace(r_surface, r_ext_max, 500)
    
    def ode_func_ext(r, y, args=None):
        return stt_tov_exterior(r, y, beta)
    
    # Integrate exterior
    sol_ext = diffeqsolve(
        ODETerm(ode_func_ext),
        Dopri5(),
        t0=r_surface,
        t1=r_ext_max,
        dt0=(r_ext_max - r_surface) / 500,
        y0=surface_conditions,
        args=None,
        saveat=SaveAt(ts=r_ext_grid),
        stepsize_controller=PIDController(rtol=1e-6, atol=1e-8),
        max_steps=50000, # TODO: can this be lowered?
    )
    
    # Extract asymptotic values
    nu_final = sol_ext.ys[2][-1]
    phi_final = sol_ext.ys[3][-1]
    
    # Objective: minimize deviation from asymptotic conditions
    # In STT, we expect ν(∞) → 0 and φ(∞) → 0 for physical solutions
    objective = jnp.sqrt(nu_final**2 + phi_final**2)
    
    return objective


def stt_tov_solver(eos_dict, P_c, beta=-4.5, optimize_bc=True, N=16001):
    r"""
    Solve the STT TOV equations for a given central pressure and coupling parameter.
    
    This function integrates the STT TOV equations from the center of the star
    outward to the surface using radius as the integration variable, exactly
    following the architecture of the Python reference solver.
    
    **Two-Stage Integration**:
    1. **Interior**: Uses matter TOV equations from center to surface
    2. **Exterior**: Uses vacuum field equations from surface to asymptotic region
    
    **Boundary Condition Optimization**:
    Optimizes (ν_c, φ_c) to satisfy asymptotic conditions ν(∞)=0, φ(∞)=0
    
    Args:
        eos_dict (dict): EOS interpolation data containing:
            - **p**: Pressure array [geometric units]
            - **e**: Energy density array [geometric units]  
            - **rho**: Rest mass density array [geometric units]
        P_c (float): Central pressure [geometric units]
        beta (float, optional): Scalar-tensor coupling parameter. Default is -4.5.
                               β = 0 corresponds to General Relativity.
        optimize_bc (bool, optional): Whether to optimize boundary conditions. Default is True.
        N (int, optional): Number of grid points. Default is 16001.
        
    Returns:
        tuple: A tuple containing:
            - **M**: Gravitational mass [geometric units]
            - **R**: Circumferential radius [geometric units]  
            - **scalar_charge**: Scalar charge [geometric units]
            
    Note:
        This implementation follows the exact architecture of the Python reference
        solver with proper r-based integration and two-stage interior/exterior
        boundary matching.
    """
    # Set up radial grid - extend to several stellar radii for asymptotic region
    eps_c = utils.interp_in_logspace(P_c, eos_dict["p"], eos_dict["e"])
    r_max = 128 * 4.0 * (3.0/(4.0*jnp.pi*eps_c))**(1.0/3.0)
    
    if optimize_bc:
        # Optimize boundary conditions for physical asymptotic behavior
        initial_guess = jnp.array([-1.0, -0.106])  # (nu_c, phi_c)
        
        # Define objective function for optimization
        def objective(params):
            return _trial_solution(params, eos_dict, P_c, beta, r_max, N)
        
        # Optimize using JAX scipy with multiple methods for robustness
        try:
            result = jopt.minimize(objective, initial_guess, method='Nelder-Mead')
            if not result.success:
                # Try BFGS as backup
                result = jopt.minimize(objective, initial_guess, method='BFGS')
            nu_c, phi_c = result.x
        except:
            # Fallback to fixed boundary conditions
            nu_c, phi_c = -1.0, -0.106
    else:
        # Use fixed boundary conditions
        nu_c, phi_c = -1.0, -0.106
    
    # Set up final integration with optimized boundary conditions
    dr = r_max / (N - 1)
    r_grid = jnp.linspace(dr, r_max, N)
    
    # Initial conditions at r = dr
    y0 = jnp.array([P_c, 0.0, nu_c, phi_c, 0.0])
    
    # Define ODE function for interior
    def ode_func_interior(r, y, args=None):
        return stt_tov_interior(r, y, eos_dict, beta)
    
    # Integrate interior
    sol_interior = diffeqsolve(
        ODETerm(ode_func_interior),
        Dopri5(),
        t0=dr,
        t1=r_max,
        dt0=dr,
        y0=y0,
        args=None,
        saveat=SaveAt(ts=r_grid),
        stepsize_controller=PIDController(rtol=1e-6, atol=1e-8),
        max_steps=50000, # TODO: can this be lowered?
    )
    
    # Find surface
    P_vals = sol_interior.ys[0]
    surface_idx = len(P_vals) - 1
    for i in range(len(P_vals)):
        if P_vals[i] <= 0:
            surface_idx = i
            break
    
    # Get accurate surface radius
    R = _find_accurate_surface(r_grid[:surface_idx+1], P_vals[:surface_idx+1], eos_dict)
    
    # Get surface conditions for exterior integration
    r_surface = r_grid[surface_idx]
    surface_conditions = jnp.array([sol_interior.ys[j][surface_idx] for j in range(5)])
    
    # Integrate exterior for asymptotic mass calculation
    r_ext_max = 10.0 * R
    r_ext_grid = jnp.linspace(r_surface, r_ext_max, 1000)
    
    def ode_func_exterior(r, y, args=None):
        return stt_tov_exterior(r, y, beta)
    
    sol_exterior = diffeqsolve(
        ODETerm(ode_func_exterior),
        Dopri5(),
        t0=r_surface,
        t1=r_ext_max,
        dt0=(r_ext_max - r_surface) / 1000,
        y0=surface_conditions,
        args=None,
        saveat=SaveAt(ts=r_ext_grid),
        stepsize_controller=PIDController(rtol=1e-6, atol=1e-8),
        max_steps=50000, # TODO: can this be lowered?
    )
    
    # Combine interior and exterior solutions
    r_combined = jnp.concatenate([r_grid[:surface_idx+1], r_ext_grid[1:]])
    nu_combined = jnp.concatenate([sol_interior.ys[2][:surface_idx+1], sol_exterior.ys[2][1:]])
    phi_combined = jnp.concatenate([sol_interior.ys[3][:surface_idx+1], sol_exterior.ys[3][1:]])
    
    # Compute asymptotic mass and scalar charge
    M = _compute_asymptotic_mass(r_combined, nu_combined)
    scalar_charge = _compute_scalar_charge(r_combined, phi_combined)
    
    return M, R, scalar_charge


def calc_k2_stt(R, M, eos_dict=None, P_c=None, beta=-4.5):
    r"""
    Calculate the second Love number k₂ for tidal deformability in STT.
    
    **NOTE**: This is a placeholder implementation. The full STT tidal calculation
    requires solving additional auxiliary equations for the tidal perturbations,
    which is not yet implemented. For now, this returns a simplified estimate.
    
    Args:
        R (float): Neutron star radius [geometric units]
        M (float): Neutron star mass [geometric units]
        eos_dict (dict, optional): EOS interpolation data (unused in placeholder)
        P_c (float, optional): Central pressure [geometric units] (unused in placeholder)
        beta (float, optional): Scalar-tensor coupling parameter (unused in placeholder)
        
    Returns:
        float: Second Love number k₂ (simplified estimate)
    """
    # Simplified compactness-based estimate
    # TODO: Implement full STT tidal calculation using eos_dict, P_c, beta
    C = M / R
    
    # Basic Love number formula (GR limit)
    y_approx = 2.0  # Simplified assumption
    
    num = (8.0 / 5.0) * (1 - 2*C)**2 * C**5 * (2*C*(y_approx - 1) - y_approx + 2)
    den = (2*C*(4*(y_approx + 1)*C**4 + (6*y_approx - 4)*C**3 + 
                (26 - 22*y_approx)*C**2 + 3*(5*y_approx - 8)*C - 3*y_approx + 6) - 
           3*(1 - 2*C)**2 * (2*C*(y_approx - 1) - y_approx + 2) * jnp.log(1.0/(1 - 2*C)))
    
    return num / den