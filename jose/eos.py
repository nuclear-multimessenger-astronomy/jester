import jax.numpy as jnp
from jax.scipy.optimize import minimize

from . import utils, tov


class Interpolate_EOS_model(object):
    def __init__(self, n, p, e): 
        # the expected units are
        # n[fm^-3], p[MeV / m^3], e[MeV / fm^3] 
        self.p = jnp.array(p * utils.MeV_fm_inv3_to_geometric)
        self.e = jnp.array(e * utils.MeV_fm_inv3_to_geometric)
        self.n = jnp.array(n * utils.fm_inv3_to_geometric)

        # calculate the pseudo enthalpy
        self.h = utils.cumtrapz(self.p / (self.e + self.p), self.logp)

        # pre-calculate quantities
        self.logp = jnp.log(self.p)
        self.loge = jnp.log(self.e)
        self.logn = jnp.log(self.n)
        self.logh = jnp.log(self.h)
        self.dloge_dlogp = jnp.gradient(self.loge, self.logp)

    def energy_density_from_pseudo_enthalpy(self, h):
        loge_of_h = jnp.interp(jnp.log(h), self.logh, self.loge)
        return jnp.exp(loge_of_h)

    def pressure_from_pseudo_enthalpy(self, h):
        logp_of_h = jnp.interp(jnp.log(h), self.logh, self.logp)
        return jnp.exp(logp_of_h)
    
    def dloge_dlogp_from_pressure(self, p):
        return jnp.interp(jnp.log(p), self.dloge_dlogp, self.logp)

    def pseudo_enthalpy_from_pressure(self, p):
        logh_of_p = jnp.interp(jnp.log(p), self.logp, self.logh)
        return jnp.exp(logh_of_p)

    def pressure_from_number_density(self, n):
        logp_of_n = jnp.interp(n, n, self.logp)
        return jnp.exp(logp_of_n)

def construct_family(eos, ndat=50):
    # start at pc at 0.1nsat
    pc_min = eos.pressure_from_number_density(
        0.1 * 0.16 * utils.fm_inv3_to_geometric
    )
    # end at pc at pmax
    pc_max = eos.p[-1]

    pcs = jnp.logspace(
        jnp.log10(pc_min),
        jnp.log10(pc_max),
        num=ndat
    )

    ms, rs, ks, pcs = jnp.vectorize(
            tov.tov_solver,
            excluded=['eos',])(eos, pcs)

    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass
    ms /= utils.solar_mass_in_meter
    # convert the radius to km
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2. / 3. * ks * jnp.power(cs, -5.)

    return jnp.log(pcs), ms, rs, lambdas
