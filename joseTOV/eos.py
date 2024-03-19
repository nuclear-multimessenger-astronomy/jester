import jax.numpy as jnp

from . import utils, tov


class Interpolate_EOS_model(object):
    def __init__(self, n, p, e):
        # the expected units are
        # n[fm^-3], p[MeV / m^3], e[MeV / fm^3]
        self.p = jnp.array(p * utils.MeV_fm_inv3_to_geometric)
        self.e = jnp.array(e * utils.MeV_fm_inv3_to_geometric)
        self.n = jnp.array(n * utils.fm_inv3_to_geometric)

        # calculate the pseudo enthalpy
        h = utils.cumtrapz(self.p / (self.e + self.p), jnp.log(self.p))
        # just a random small number
        self.h = jnp.concatenate(
            (
                jnp.array(
                    [
                        1e-30,
                    ]
                ),
                h,
            )
        )
        # pre-calculate quantities
        self.logp = jnp.log(self.p)
        self.loge = jnp.log(self.e)
        self.logn = jnp.log(self.n)
        self.logh = jnp.log(self.h)
        dloge_dlogp = jnp.diff(self.loge) / jnp.diff(self.logp)
        dloge_dlogp = jnp.concatenate(
            (
                jnp.array(
                    [
                        dloge_dlogp.at[0].get(),
                    ]
                ),
                dloge_dlogp,
            )
        )
        self.dloge_dlogp = dloge_dlogp

    def energy_density_from_pseudo_enthalpy(self, h):
        loge_of_h = jnp.interp(jnp.log(h), self.logh, self.loge)
        return jnp.exp(loge_of_h)

    def pressure_from_pseudo_enthalpy(self, h):
        logp_of_h = jnp.interp(jnp.log(h), self.logh, self.logp)
        return jnp.exp(logp_of_h)

    def dloge_dlogp_from_pseudo_enthalpy(self, h):
        return jnp.interp(h, self.h, self.dloge_dlogp)

    def pseudo_enthalpy_from_pressure(self, p):
        logh_of_p = jnp.interp(jnp.log(p), self.logp, self.logh)
        return jnp.exp(logh_of_p)

    def pressure_from_number_density(self, n):
        logp_of_n = jnp.interp(n, self.n, self.logp)
        return jnp.exp(logp_of_n)


def construct_family(eos, ndat=50):
    # start at pc at 0.1nsat
    pc_min = eos.pressure_from_number_density(2 * 0.16 * utils.fm_inv3_to_geometric)
    # end at pc at pmax
    pc_max = eos.p[-1]

    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    ms, rs, ks = jnp.vectorize(
        tov.tov_solver,
        excluded=[
            0,
        ],
    )(eos, pcs)

    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass
    ms /= utils.solar_mass_in_meter
    # convert the radius to km
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    return jnp.log(pcs), ms, rs, lambdas
