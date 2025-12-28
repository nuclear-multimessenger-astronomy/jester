r"""
Utility functions for neutron star physics calculations.

This module provides essential utility functions for equation of state
interpolation, unit conversions, numerical integration, and auxiliary
calculations needed for TOV equation solving.

**Units:** The module defines conversion factors between different unit
systems commonly used in neutron star physics.
"""

from jax import vmap
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float
from interpax._spline import interp1d as interpax_interp1d

from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController

#################################
### PHYSICAL CONSTANTS AND UNIT CONVERSIONS ###
#################################

# Fundamental constants (SI units)
eV = 1.602176634e-19  # Elementary charge [C]
c = 299792458.0  # Speed of light [m/s]
G = 6.6743e-11  # Gravitational constant [m³/kg/s²]
Msun = 1.988409870698051e30  # Solar mass [kg]
hbarc = 197.3269804593025  # Reduced Planck constant × c [MeV⋅fm]

# Particle masses [MeV]
m_p = 938.2720881604904  # Proton mass
m_n = 939.5654205203889  # Neutron mass
m_e = 0.510998  # Electron mass
m = (m_p + m_n) / 2.0  # Average nucleon mass (Margueron et al.)

# Derived constants
hbar = hbarc  # Alias for compatibility
solar_mass_in_meter = Msun * G / c / c  # Solar mass in geometric units [m]

# Basic unit conversions
fm_to_m = 1e-15  # Femtometer to meter
MeV_to_J = 1e6 * eV  # MeV to Joule
m_to_fm = 1.0 / fm_to_m  # Meter to femtometer
J_to_MeV = 1.0 / MeV_to_J  # Joule to MeV

# Number density conversions
fm_inv3_to_SI = 1.0 / fm_to_m**3  # fm⁻³ to m⁻³
number_density_to_geometric = 1  # Number density scaling factor
fm_inv3_to_geometric = fm_inv3_to_SI * number_density_to_geometric

SI_to_fm_inv3 = 1.0 / fm_inv3_to_SI
geometric_to_fm_inv3 = 1.0 / fm_inv3_to_geometric

# Pressure and energy density conversions
MeV_fm_inv3_to_SI = MeV_to_J * fm_inv3_to_SI  # MeV/fm³ to Pa
SI_to_MeV_fm_inv3 = 1.0 / MeV_fm_inv3_to_SI  # Pa to MeV/fm³
pressure_SI_to_geometric = G / c**4  # SI pressure to geometric units
MeV_fm_inv3_to_geometric = MeV_fm_inv3_to_SI * pressure_SI_to_geometric

# Additional useful conversions
dyn_cm2_to_MeV_fm_inv3 = 1e-1 * J_to_MeV / m_to_fm**3  # dyn/cm² to MeV/fm³
g_cm_inv3_to_MeV_fm_inv3 = 1e3 * c**2 * J_to_MeV / m_to_fm**3  # g/cm³ to MeV/fm³

# Inverse conversions
geometric_to_SI = 1.0 / pressure_SI_to_geometric
SI_to_MeV_fm_inv3 = 1.0 / MeV_fm_inv3_to_SI
geometric_to_MeV_fm_inv3 = 1.0 / MeV_fm_inv3_to_geometric


#########################
### UTILITY FUNCTIONS ###
#########################

# Vectorized polynomial root finding
roots_vmap = vmap(partial(jnp.roots, strip_zeros=False), in_axes=0, out_axes=0)


@vmap
def cubic_root_for_proton_fraction(coefficients):
    r"""
    Solve cubic equation for proton fraction in beta-equilibrium.

    This function solves the cubic equation that arises from the
    beta-equilibrium condition in neutron star matter:

    .. math::
        ax^3 + bx^2 + cx + d = 0

    where the coefficients are related to the symmetry energy and
    electron chemical potential. Uses Cardano's formula for exact
    analytical solution.

    Parameters
    ----------
    coefficients : Array
        Cubic polynomial coefficients [a, b, c, d]

    Returns
    -------
    Array
        Three roots of the cubic equation (may be complex)

    Notes
    -----
    This function is vectorized to handle multiple coefficient sets
    simultaneously for different densities.
    """
    a, b, c, d = coefficients

    # Cardano's formula implementation
    f = ((3.0 * c / a) - ((b**2) / (a**2))) / 3.0
    g = (((2.0 * (b**3)) / (a**3)) - ((9.0 * b * c) / (a**2)) + (27.0 * d / a)) / 27.0
    g_squared = g**2
    f_cubed = f**3
    h = g_squared / 4.0 + f_cubed / 27.0

    R = -(g / 2.0) + jnp.sqrt(h)
    S = jnp.cbrt(R)
    T = -(g / 2.0) - jnp.sqrt(h)
    U = jnp.cbrt(T)

    # Three roots of the cubic equation
    x1 = (S + U) - (b / (3.0 * a))
    x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * jnp.sqrt(3.0) * 0.5j
    x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * jnp.sqrt(3.0) * 0.5j

    return jnp.array([x1, x2, x3])


def cumtrapz(y, x):
    r"""
    Cumulatively integrate y(x) using the composite trapezoidal rule.

    This function performs cumulative integration using the trapezoidal rule,
    which is essential for computing thermodynamic quantities like enthalpy
    and chemical potential from EOS data.

    The trapezoidal rule approximates:

    .. math::
        \int_{x_0}^{x_i} y(x) dx \approx \sum_{j=1}^{i} \frac{\Delta x_j}{2}(y_{j-1} + y_j)

    Parameters
    ----------
    y : Array
        Values to integrate
    x : Array
        The coordinate to integrate along

    Returns
    -------
    Array
        The result of cumulative integration of y along x

    Notes
    -----
    The result array has the same length as the input, with the first
    element set to a small value (1e-30) to avoid logarithm issues.
    """
    # Validate input arrays
    assert y.shape == x.shape, "Input arrays must have matching shapes"
    assert len(y.shape) == 1, "Input arrays must be one-dimensional"
    assert len(x.shape) == 1, "Input arrays must be one-dimensional"

    # Apply trapezoidal rule for cumulative integration
    dx = jnp.diff(x)
    res = jnp.cumsum(dx * (y[1:] + y[:-1]) / 2.0)

    # Prepend small value to avoid log(0) issues in subsequent calculations
    res = jnp.concatenate((jnp.array([1e-30]), res))

    return res


def interp_in_logspace(x, xs, ys):
    r"""
    Perform logarithmic interpolation.

    This function performs interpolation in logarithmic space, which is
    more appropriate for quantities that span many orders of magnitude
    (like pressure and density in neutron stars).

    The interpolation is performed as:

    .. math::
        \log y(x) = \text{interp}(\log x, \log x_s, \log y_s)

    Parameters
    ----------
    x : float
        Point at which to evaluate the interpolation
    xs : Array
        Known x-coordinates (must be positive)
    ys : Array
        Known y-coordinates (must be positive)

    Returns
    -------
    float
        Interpolated value at x

    Notes
    -----
    All input values must be positive since logarithms are taken.
    """
    # Perform interpolation in log space and convert back
    logx = jnp.log(x)
    logxs = jnp.log(xs)
    logys = jnp.log(ys)
    return jnp.exp(jnp.interp(logx, logxs, logys))


def limit_by_MTOV(
    pc: Array, m: Array, r: Array, l: Array
) -> tuple[Array, Array, Array, Array]:
    r"""
    Truncate neutron star family at maximum TOV mass.

    This function limits the mass-radius relation to the stable branch
    by truncating at the maximum TOV mass (MTOV). Points beyond MTOV
    correspond to unstable configurations and are replaced with duplicates
    of the MTOV values to maintain array shape for JIT compilation.

    The maximum mass occurs when:

    .. math::
        \frac{dM}{dp_c} = 0

    Parameters
    ----------
    pc : Array
        Central pressure array
    m : Array
        Gravitational mass array
    r : Array
        Radius array
    l : Array
        Tidal deformability array

    Returns
    -------
    pc : Array
        Truncated central pressure array
    m : Array
        Truncated mass array
    r : Array
        Truncated radius array
    l : Array
        Truncated tidal deformability array

    Notes
    -----
    This approach maintains static array shapes required for JAX JIT
    compilation while effectively removing unstable configurations.
    """

    # Identify maximum TOV mass and corresponding index
    m_at_TOV = jnp.max(m)
    idx_TOV = jnp.argmax(m)

    # Extract values at maximum mass point
    pc_at_TOV = pc[idx_TOV]
    r_at_TOV = r[idx_TOV]
    l_at_TOV = l[idx_TOV]

    # Identify stable (mass-increasing) configurations
    m_is_increasing = jnp.diff(m) > 0
    m_is_increasing = jnp.insert(m_is_increasing, idx_TOV, True)

    # Mask out configurations beyond maximum mass
    m_is_increasing = jnp.where(jnp.arange(len(m)) > idx_TOV, False, m_is_increasing)

    # Replace unstable configurations with MTOV values
    pc_new = jnp.where(m_is_increasing, pc, pc_at_TOV)
    m_new = jnp.where(m_is_increasing, m, m_at_TOV)
    r_new = jnp.where(m_is_increasing, r, r_at_TOV)
    l_new = jnp.where(m_is_increasing, l, l_at_TOV)

    # Sort by increasing mass for consistency
    sort_idx = jnp.argsort(m_new)
    pc_new = pc_new[sort_idx]
    m_new = m_new[sort_idx]
    r_new = r_new[sort_idx]
    l_new = l_new[sort_idx]

    return pc_new, m_new, r_new, l_new


###################
### SPLINES etc ###
###################


def cubic_spline(xq: Float[Array, "n"], xp: Float[Array, "n"], fp: Float[Array, "n"]):
    r"""
    Cubic spline interpolation using interpax.

    This function creates a cubic spline interpolator through the given
    data points and evaluates it at the query points. Cubic splines
    provide smooth interpolation with continuous first and second derivatives.

    Parameters
    ----------
    xq : Float[Array, "n"]
        Query points for evaluation
    xp : Float[Array, "n"]
        Known x-coordinates of data points
    fp : Float[Array, "n"]
        Known y-coordinates, i.e., fp = f(xp)

    Returns
    -------
    Array
        Interpolated values at query points xq

    Notes
    -----
    Uses the interpax library for JAX-compatible spline interpolation.
    See: https://github.com/f0uriest/interpax
    """
    return interpax_interp1d(xq, xp, fp, method="cubic")


def sigmoid(x: Array) -> Array:
    r"""
    Sigmoid activation function.

    Computes the sigmoid function:

    .. math::
        \sigma(x) = \frac{1}{1 + e^{-x}}

    Parameters
    ----------
    x : Array
        Input values

    Returns
    -------
    Array
        Sigmoid function values in range (0, 1)
    """
    return 1.0 / (1.0 + jnp.exp(-x))


def calculate_rest_mass_density(e: Float[Array, "n"], p: Float[Array, "n"]):
    r"""
    Compute rest-mass density from energy density and pressure.

    This function solves the first law of thermodynamics to obtain the
    rest-mass density (baryon density) from the energy density and pressure.
    The relation is given by:

    .. math::
        \frac{d\rho}{d\varepsilon} = \frac{\rho}{p + \varepsilon}

    where :math:`\rho` is the rest-mass density, :math:`\varepsilon` is the
    energy density, and :math:`p` is the pressure.

    Parameters
    ----------
    e : Float[Array, "n"]
        Energy density array [geometric units]
    p : Float[Array, "n"]
        Pressure array [geometric units]

    Returns
    -------
    Array
        Rest-mass density array [geometric units]

    Notes
    -----
    This function uses diffrax for ODE integration and may have
    compatibility issues with some diffrax versions. The initial
    condition assumes :math:`\rho(\varepsilon_0) = \varepsilon_0`.
    """

    # Define pressure interpolation function
    def p_interp(e_val):
        return jnp.interp(e_val, e, p)

    # Define the ODE: dρ/dε = ρ/(p + ε)
    def rhs(t, rho, args):
        p_val = p_interp(t)
        return rho / (p_val + t)

    # Initial condition: assume ρ(ε_0) = ε_0
    rho0 = e[0]

    # Set up ODE integration using diffrax
    term = ODETerm(rhs)
    solver = Tsit5()

    # Integrate from e[0] to e[-1]
    solution = diffeqsolve(
        term,
        solver,
        t0=e[0],  # Initial energy density
        t1=e[-1],  # Final energy density
        dt0=1e-8,  # Initial step size
        y0=rho0,  # Initial rest-mass density
        saveat=SaveAt(ts=e),  # Save at input grid points
        stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
        throw=False,
    )

    return solution.ys
