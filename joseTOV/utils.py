# this script contain utility functions
from jax import vmap
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float
from interpax import interp1d as interpax_interp1d

#################################
### CONSTANTS AND CONVERSIONS ###
#################################

# to avoid additional dependecy on scipy
eV = 1.602176634e-19
c = 299792458.0
G = 6.6743e-11
Msun = 1.988409870698051e30
hbarc = 197.3269804593025  # in MeV fm
hbar = hbarc # TODO: check if must be updated, this is just taken from Rahul's code
m_p = 938.2720881604904  # in MeV
m_n = 939.5654205203889  # in MeV
m = (m_p + m_n) / 2.0  # in MeV, average nucleonic mass defined by Margueron et al
m_e = 0.510998 # mass electron in MeV
solar_mass_in_meter = Msun * G / c / c # solar mass in geometric unit

# simple conversions
fm_to_m = 1e-15
MeV_to_J = 1e6 * eV
m_to_fm = 1.0 / fm_to_m
J_to_MeV = 1.0 / MeV_to_J

# number density
fm_inv3_to_SI = 1.0 / fm_to_m**3
number_density_to_geometric = 1
fm_inv3_to_geometric = fm_inv3_to_SI * number_density_to_geometric

SI_to_fm_inv3 = 1.0 / fm_inv3_to_SI
geometric_to_fm_inv3 = 1.0 / fm_inv3_to_geometric

# pressure and energy density
MeV_fm_inv3_to_SI = MeV_to_J * fm_inv3_to_SI
SI_to_MeV_fm_inv3 = 1.0 / MeV_fm_inv3_to_SI
pressure_SI_to_geometric = G / c**4
MeV_fm_inv3_to_geometric = MeV_fm_inv3_to_SI * pressure_SI_to_geometric
dyn_cm2_to_MeV_fm_inv3 = 1e-1 * J_to_MeV / m_to_fm**3
g_cm_inv3_to_MeV_fm_inv3 = 1e3 * c**2 * J_to_MeV / m_to_fm**3

geometric_to_SI = 1.0 / pressure_SI_to_geometric
SI_to_MeV_fm_inv3 = 1.0 / MeV_fm_inv3_to_SI
geometric_to_MeV_fm_inv3 = 1.0 / MeV_fm_inv3_to_geometric


#########################
### UTILITY FUNCTIONS ###
#########################

# vmapped jnp.roots function
roots_vmap = vmap(partial(jnp.roots, strip_zeros=False), in_axes=0, out_axes=0)


@vmap
def cubic_root_for_proton_fraction(coefficients):

    a, b, c, d = coefficients
    
    f = ((3.0 * c / a) - ((b ** 2) / (a ** 2))) / 3.0
    g = (
        ((2.0 * (b ** 3)) / (a ** 3)) - ((9.0 * b * c) / (a ** 2)) + (27.0 * d / a)
    ) / 27.0
    g_squared = g ** 2
    f_cubed = f ** 3
    h = g_squared / 4.0 + f_cubed / 27.0

    R = -(g / 2.0) + jnp.sqrt(h)
    S = jnp.cbrt(R)
    T = -(g / 2.0) - jnp.sqrt(h)
    U = jnp.cbrt(T)

    x1 = (S + U) - (b / (3.0 * a))
    x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * jnp.sqrt(3.0) * 0.5j
    x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * jnp.sqrt(3.0) * 0.5j

    return jnp.array([x1, x2, x3])


def cumtrapz(y, x):
    """
    Cumulatively integrate y(x) using the composite trapezoidal rule.

    Parameters
    ----------
    y : jax.numpy.ndarray
        Values to integrate.
    x : jax.numpy.ndarray
        The coordinate to integrate along.

    Returns
    -------
    res : jax.numpy.ndarray
        The result of cumulative integration of `y` along `x`.
    """
    # check the shape of y and x
    assert y.shape == x.shape, "Not matching shape between y and x"
    assert len(y.shape) == 1, "y is expected to be one-dimensional array"
    assert len(x.shape) == 1, "x is expected to be one-dimensional array"

    # get the step size of x
    dx = jnp.diff(x)
    res = jnp.cumsum(dx * (y[1::] + y[:-1:]) / 2.0)
    res = jnp.concatenate(
        (
            jnp.array(
                [
                    1e-30,
                ]
            ),
            res,
        )
    )

    return res


def interp_in_logspace(x, xs, ys):
    logx = jnp.log(x)
    logxs = jnp.log(xs)
    logys = jnp.log(ys)
    return jnp.exp(jnp.interp(logx, logxs, logys))

def limit_by_MTOV(pc: Array,
                  m: Array, 
                  r: Array, 
                  l: Array) -> tuple[Array, Array, Array]:
    """
    Limits the M, R and Lambda curves to be below MTOV in a jit-friendly manner (i.e., static shape sizes).
    The idea now is to feed this into some routine that creates an interpolation out of this, which then uses jnp.unique to get rid of these duplicates
    NOTE: this assumes that the M curve increases up to a point and potentially decreases after that point. In case the EOS has some weird features and the M curve increases again, this function will return weird results.
    TODO: generalize this for weird EOS or check if we do not have those weird EOS when sampling NEPs.
    
    Args:
        pcs (Array["npoints"]): Original pressure
        m (Array["npoints"]): Original mass curve
        r (Array["npoints"]): Original radius curve
        l (Array["npoints"]): Original lambdas curve
        
    Returns:
        tuple[Array["npoints"], Array["npoints"], Array["npoints"]]: Tuple of new mass, radius and lambdas curves, where the part of the curves where mass decreases is replaced with duplication of the first entry of the M, R and Lambda arrays.
    """
    
    # Fetch the MTOV, we will use it to dump duplicates of it wherever the NS family is unphysical
    m_at_TOV = jnp.max(m)
    idx_TOV = jnp.argmax(m)
    
    pc_at_TOV = pc[idx_TOV]
    r_at_TOV = r[idx_TOV]
    l_at_TOV = l[idx_TOV]
    
    # Find out where the mass array is increasing, and insert True at the TOV index to pad length of the array correctly
    m_is_increasing = jnp.diff(m) > 0
    m_is_increasing = jnp.insert(m_is_increasing, idx_TOV, True)
    
    pc_new = jnp.where(m_is_increasing, pc, pc_at_TOV)
    m_new  = jnp.where(m_is_increasing, m, m_at_TOV)
    r_new  = jnp.where(m_is_increasing, r, r_at_TOV)
    l_new  = jnp.where(m_is_increasing, l, l_at_TOV)
    
    # Sort in increasing values of M for plotting etc
    sort_idx = jnp.argsort(m_new)
    
    pc_new = pc_new[sort_idx]
    m_new  = m_new[sort_idx]
    r_new  = r_new[sort_idx]
    l_new  = l_new[sort_idx]
    
    return pc_new, m_new, r_new, l_new

###################
### SPLINES etc ###
###################

def cubic_spline(xq: Float[Array, "n"],
                 xp: Float[Array, "n"],
                 fp: Float[Array, "n"]):
    """
    Create a cubic spline interpolating function through (xp, fp) with interpax (https://github.com/f0uriest/interpax)
    Args:
        xq (Float[Array, "n"]): x values at which we are going to evaluate the spline interpolator
        xp (Float[Array, "n"]): x values of the data points
        fp (Float[Array, "n"]): y values of the data points, i.e. fp = f(xp)
    """
    return interpax_interp1d(xq, xp, fp, method = "cubic")

def sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-x))