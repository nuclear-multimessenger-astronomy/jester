import jax
import jax.numpy as jnp
from jax.scipy.special import factorial

from . import utils, tov

import os
# get the crust
DEFAULT_DIR = os.path.join(os.path.dirname(__file__))
crust = jnp.load(f'{DEFAULT_DIR}/crust/BPS.npz')


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


class MetaModel_EOS_model(Interpolate_EOS_model):
    def __init__(self, coefficient_sat, coefficient_sym, nsat=0.16):
        # add the first derivative coefficient in Esat to
        # make it work with jax.numpy.polyval
        coefficient_sat = jnp.insert(coefficient_sat, 1, 0.)
        # get the coefficents index array
        index_sat = jnp.arange(len(coefficient_sat))
        index_sym = jnp.arange(len(coefficient_sym))

        coeff_sat = coefficient_sat / factorial(index_sat)
        coeff_sym = coefficient_sym / factorial(index_sym)
        self.coefficient_sat = coeff_sat
        self.coefficient_sym = coeff_sym
        self.nsat = nsat
        # calculate the coefficient for the derivative
        coeff_sat_grad = coefficient_sat * index_sat / factorial(index_sat)
        coeff_sym_grad = coefficient_sym * index_sym / factorial(index_sym)
        self.coefficient_sat_grad = coeff_sat_grad[1:]
        self.coefficient_sym_grad = coeff_sym_grad[1:]

        ns = jnp.logspace(
            -1,
            jnp.log10(12 * nsat),
            num=1500
        )
        ps = self.pressure_from_number_density_nuclear_unit(ns)
        es = self.energy_density_from_number_density_nuclear_unit(ns)
        ns = jnp.concatenate((crust['n'], ns))
        ps = jnp.concatenate((crust['p'], ps))
        es = jnp.concatenate((crust['e'], es))

        super().__init__(ns, ps, es)

    def esym(self, n):
        x = (n - self.nsat) / (3. * self.nsat)
        return jnp.polyval(self.coefficient_sym[::-1], x)

    def esat(self, n):
        x = (n - self.nsat) / (3. * self.nsat)
        return jnp.polyval(self.coefficient_sat[::-1], x)

    def proton_fraction(self, n):
        # chemical potential of electron
        # mu_e = hbarc * pow(3 * pi**2 * x * n, 1. / 3.)
        #      = hbarc * pow(3 * pi**2 * n, 1. / 3.) * y (y = x**1./3.)
        # mu_p - mu_n = dEdx
        #             = -4 * Esym * (1. - 2. * x)
        #             = -4 * Esym + 8 * Esym * y**3
        # at beta equilibrium, the polynominal is given by
        # mu_e(y) + dEdx(y) - (m_n - m_p) = 0
        # p_0 = -4 * Esym - (m_n - m_p)
        # p_1 = hbarc * pow(3 * pi**2 * n, 1. / 3.)
        # p_2 = 0
        # p_3 = 8 * Esym
        Esym = self.esym(n)
        coeffs = jnp.array([
            8. * Esym,
            jnp.zeros(shape=n.shape),
            utils.hbarc * jnp.power(3. * jnp.pi**2 * n, 1. / 3.),
            -4. * Esym - (utils.m_n - utils.m_p),
        ]).T
        ys = utils.roots_vmap(coeffs)
        # only take the ys in [0, 1] and real
        idx = (ys.imag == 0.) * (ys.real >= 0.) * (ys.real <= 1.)
        physical_ys = ys.at[idx].get().real
        x = jnp.power(physical_ys, 1. / 3.)
        import pdb; pdb.set_trace()
        return x

    def energy_per_particle_nuclear_unit(self, n):
        x = self.proton_fraction(n)
        delta = 1. - 2. * x
        dynamic_part = self.esat(n) + self.esym(n) * delta**2. 
        static_part = x * utils.m_n + (1. - x) * utils.m_p
        return dynamic_part + static_part

    def energy_per_particle_grad_nuclear_unit(self, n):
        delta = 1. - 2. * self.proton_fraction(n)
        x = (n - self.nsat) / (3. * self.nsat)
        Esat_grad = jnp.polyval(self.coefficient_sat_grad[::-1], x)
        Esym_grad = jnp.polyval(self.coefficient_sym_grad[::-1], x)
        return (Esat_grad + Esym_grad * delta**2) / 3. / self.nsat

    def energy_density_from_number_density_nuclear_unit(self, n):
        return n * self.energy_per_particle_nuclear_unit(n)

    def pressure_from_number_density_nuclear_unit(self, n):
        return n**2 * self.energy_per_particle_grad_nuclear_unit(n)


def construct_family(eos, ndat=50):
    # start at pc at 0.1nsat
    pc_min = eos.pressure_from_number_density(
        2 * 0.16 * utils.fm_inv3_to_geometric
    )
    # end at pc at pmax
    pc_max = eos.p[-1]

    pcs = jnp.logspace(
        jnp.log10(pc_min),
        jnp.log10(pc_max),
        num=ndat
    )

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
