import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
from jaxtyping import Array, Float

from . import utils, tov

import os

# get the crust
DEFAULT_DIR = os.path.join(os.path.dirname(__file__))
crust = jnp.load(f"{DEFAULT_DIR}/crust/BPS.npz")


class Interpolate_EOS_model(object):
    def __init__(
        self,
        n: Float[Array, "n_points"],
        p: Float[Array, "n_points"],
        e: Float[Array, "n_points"],
    ):
        # the expected units are
        # n[fm^-3], p[MeV / m^3], e[MeV / fm^3]
        self.p = jnp.array(p * utils.MeV_fm_inv3_to_geometric)
        self.e = jnp.array(e * utils.MeV_fm_inv3_to_geometric)
        self.n = jnp.array(n * utils.fm_inv3_to_geometric)

        # calculate the pseudo enthalpy
        self.h = utils.cumtrapz(self.p / (self.e + self.p), jnp.log(self.p))
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

    def energy_density_from_pseudo_enthalpy(self, h: Float):
        loge_of_h = jnp.interp(jnp.log(h), self.logh, self.loge)
        return jnp.exp(loge_of_h)

    def pressure_from_pseudo_enthalpy(self, h: Float):
        logp_of_h = jnp.interp(jnp.log(h), self.logh, self.logp)
        return jnp.exp(logp_of_h)

    def dloge_dlogp_from_pseudo_enthalpy(self, h: Float):
        return jnp.interp(h, self.h, self.dloge_dlogp)

    def pseudo_enthalpy_from_pressure(self, p: Float):
        logh_of_p = jnp.interp(jnp.log(p), self.logp, self.logh)
        return jnp.exp(logh_of_p)

    def pressure_from_number_density(self, n: Float):
        logp_of_n = jnp.interp(n, self.n, self.logp)
        return jnp.exp(logp_of_n)


class MetaModel_EOS_model(Interpolate_EOS_model):
    def __init__(
        self,
        coefficient_sat: Float[Array, "n_sat_coeff"],
        coefficient_sym: Float[Array, "n_sym_coeff"],
        nsat=0.16,
        nmax=12 * 0.16,
        ndat=1000,
        fix_proton_fraction=False,
        fix_proton_fraction_val=0.,
    ):
        # add the first derivative coefficient in Esat to
        # make it work with jax.numpy.polyval
        coefficient_sat = jnp.insert(coefficient_sat, 1, 0.0)
        # get the coefficents index array
        index_sat = jnp.arange(len(coefficient_sat))
        index_sym = jnp.arange(len(coefficient_sym))

        coeff_sat = coefficient_sat / factorial(index_sat)
        coeff_sym = coefficient_sym / factorial(index_sym)
        self.coefficient_sat = coeff_sat
        self.coefficient_sym = coeff_sym
        self.nsat = nsat
        self.fix_proton_fraction = fix_proton_fraction
        self.fix_proton_fraction_val = fix_proton_fraction_val
        # number densities in unit of fm^-3
        ns = jnp.logspace(-1, jnp.log10(nmax), num=ndat)
        es = self.energy_density_from_number_density_nuclear_unit(ns)
        ps = self.pressure_from_number_density_nuclear_unit(ns)

        ns = jnp.concatenate((crust["n"], ns))
        ps = jnp.concatenate((crust["p"], ps))
        es = jnp.concatenate((crust["e"], es))

        super().__init__(ns, ps, es)

    def esym(self, n: Float[Array, "n_points"]):
        x = (n - self.nsat) / (3.0 * self.nsat)
        return jnp.polyval(self.coefficient_sym[::-1], x)

    def esat(self, n: Float[Array, "n_points"]):
        x = (n - self.nsat) / (3.0 * self.nsat)
        return jnp.polyval(self.coefficient_sat[::-1], x)

    def proton_fraction(self, n: Float[Array, "n_points"]):
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
        coeffs = jnp.array(
            [
                8.0 * Esym,
                jnp.zeros(shape=n.shape),
                utils.hbarc * jnp.power(3.0 * jnp.pi**2 * n, 1.0 / 3.0),
                -4.0 * Esym - (utils.m_n - utils.m_p),
            ]
        ).T
        ys = utils.cubic_root_for_proton_fraction(coeffs)
        # only take the ys in [0, 1] and real
        physical_ys = jnp.where(
            (ys.imag == 0.0) * (ys.real >= 0.0) * (ys.real <= 1.0),
            ys.real,
            jnp.zeros_like(ys.real),
        ).sum(axis=1)
        x = jnp.power(physical_ys, 1.0 / 3.0)
        return x

    def energy_per_particle_nuclear_unit(self, n: Float[Array, "n_points"]):
        x = jax.lax.cond(
            self.fix_proton_fraction,
            lambda x: self.fix_proton_fraction_val * jnp.ones(n.shape),
            self.proton_fraction,
            n
        )
        self.proton_fraction(n)
        delta = 1.0 - 2.0 * x
        dynamic_part = self.esat(n) + self.esym(n) * delta**2.0
        static_part = x * utils.m_n + (1.0 - x) * utils.m_p
        return dynamic_part + static_part

    def energy_density_from_number_density_nuclear_unit(
        self, n: Float[Array, "n_points"]
    ):
        return n * self.energy_per_particle_nuclear_unit(n)

    def pressure_from_number_density_nuclear_unit(self, n: Float[Array, "n_points"]):
        p = n * n * jnp.diagonal(jax.jacfwd(self.energy_per_particle_nuclear_unit)(n))
        return p


class MetaModel_with_CSE_EOS_model(Interpolate_EOS_model):
    def __init__(
        self,
        # parameters for the MetaModel
        coefficient_sat: Float[Array, "n_sat_coeff"],
        coefficient_sym: Float[Array, "n_sym_coeff"],
        n_break: Float,
        # parameters for the CSE
        ngrids: Float[Array, "n_grid_point"],
        cs2grids: Float[Array, "n_grid_point"],
        nsat=0.16,
        nmax=25 * 0.16,
    ):

        # initializate the MetaModel part
        self.metamodel = MetaModel_EOS_model(
            coefficient_sat,
            coefficient_sym,
            nsat=nsat,
            nmax=n_break,
            ndat=50,
        )
        # calculate the chemical potential at the transition point
        self.n_break = n_break
        self.p_break = (
            self.metamodel.pressure_from_number_density_nuclear_unit(
                jnp.array(
                    [
                        n_break,
                    ]
                )
            )
            .at[0]
            .get()
        )
        self.e_break = (
            self.metamodel.energy_density_from_number_density_nuclear_unit(
                jnp.array(
                    [
                        n_break,
                    ]
                )
            )
            .at[0]
            .get()
        )
        self.mu_break = (self.p_break + self.e_break) / self.n_break
        self.cs2_break = (
            jnp.diff(self.metamodel.p).at[-1].get()
            / jnp.diff(self.metamodel.e).at[-1].get()
        )
        # define the speed-of-sound interpolation
        # of the extension portion
        self.ngrids = ngrids
        self.cs2grids = cs2grids
        self.cs2 = lambda n: jnp.interp(n, ngrids, cs2grids)
        # number densities in unit of fm^-3
        ns = jnp.logspace(jnp.log10(self.n_break), jnp.log10(nmax), num=1000)
        mus = self.mu_break * jnp.exp(utils.cumtrapz(self.cs2(ns) / ns, ns))
        ps = self.p_break + utils.cumtrapz(self.cs2(ns) * mus, ns)
        es = self.e_break + utils.cumtrapz(mus, ns)
        ns = jnp.concatenate((self.metamodel.n / utils.fm_inv3_to_geometric, ns))
        ps = jnp.concatenate((self.metamodel.p / utils.MeV_fm_inv3_to_geometric, ps))
        es = jnp.concatenate((self.metamodel.e / utils.MeV_fm_inv3_to_geometric, es))

        super().__init__(ns, ps, es)


def construct_family(eos, ndat=50, min_nsat=2):
    # constrcut the dictionary
    ns, ps, hs, es, dloge_dlogps = eos
    # calculate the pc_min
    pc_min = utils.interp_in_logspace(min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps)
    eos_dict = dict(p=ps, h=hs, e=es, dloge_dlogp=dloge_dlogps)

    # end at pc at pmax
    pc_max = eos_dict["p"][-1]

    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    ms, rs, ks = jnp.vectorize(
        tov.tov_solver,
        excluded=[
            0,
        ],
    )(eos_dict, pcs)

    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass
    ms /= utils.solar_mass_in_meter
    # convert the radius to km
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    return jnp.log(pcs), ms, rs, lambdas
