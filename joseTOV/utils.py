# this script contain utility functions
from jax import vmap
import jax.numpy as jnp
from functools import partial

# to avoid additional dependecy on scipy
eV = 1.602176634e-19
c = 299792458.0
G = 6.6743e-11
Msun = 1.988409870698051e30
hbarc = 197.3269804593025  # in MeV fm
m_p = 938.2720881604904  # in MeV
m_n = 939.5654205203889  # in MeV

fm_to_m = 1e-15
MeV_to_J = 1e6 * eV

# number density
fm_inv3_to_SI = 1.0 / fm_to_m**3
number_density_to_geometric = 1
fm_inv3_to_geometric = fm_inv3_to_SI * number_density_to_geometric

# pressure and energy density
MeV_fm_inv3_to_SI = MeV_to_J * fm_inv3_to_SI
pressure_SI_to_geometric = G / c**4
MeV_fm_inv3_to_geometric = MeV_fm_inv3_to_SI * pressure_SI_to_geometric

# solar mass in geometric unit
solar_mass_in_meter = Msun * G / c / c

# vmapped jnp.roots function
roots_vmap = vmap(
    partial(jnp.roots, strip_zeros=False),
    in_axes=0,
    out_axes=0
)

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

    return res
