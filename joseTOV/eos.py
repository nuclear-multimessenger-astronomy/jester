import os
import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
from jaxtyping import Array, Float, Int
from functools import partial

from . import utils, tov

##############
### CRUSTS ###
##############

DEFAULT_DIR = os.path.join(os.path.dirname(__file__))
CRUST_DIR = f"{DEFAULT_DIR}/crust"

def load_crust(name: str) -> tuple[Array, Array, Array]:
    """
    Load a crust file from the default directory.

    Args:
        name (str): Name of the crust to load, or a filename if a file outside of jose is supplied.

    Returns:
        tuple[Array, Array, Array]: Number densities [fm^-3], pressures [MeV / fm^-3], and energy densities [MeV / fm^-3] of the crust.
    """
    
    # Get the available crust names
    available_crust_names = [f.split(".")[0] for f in os.listdir(CRUST_DIR) if f.endswith(".npz")]
    
    # If a name is given, but it is not a filename, load the crust from the jose directory
    if not name.endswith(".npz"):
        if name in available_crust_names:
            name = os.path.join(CRUST_DIR, f"{name}.npz")
        else:
            raise ValueError(f"Crust {name} not found in {CRUST_DIR}. Available crusts are {available_crust_names}")
    
    # Once the correct file is identified, load it
    crust = jnp.load(name)
    n, p, e = crust["n"], crust["p"], crust["e"]
    return n, p, e

class Interpolate_EOS_model(object):
    """
    Base class to interpolate EOS data. 
    """
    def __init__(
        self,
        n: Float[Array, "n_points"],
        p: Float[Array, "n_points"],
        e: Float[Array, "n_points"],
    ):
        """
        Initialize the EOS model with the provided data and compute auxiliary data.

        Args:
            n (Float[Array, n_points]): Number densities. Expected units are n[fm^-3]
            p (Float[Array, n_points]): Pressure values. Expected units are p[MeV / fm^3]
            e (Float[Array, n_points]): Energy densities. Expected units are e[MeV / fm^3]
        """
        
        # Save the provided data as attributes, make conversions
        self.n = jnp.array(n * utils.fm_inv3_to_geometric)
        self.p = jnp.array(p * utils.MeV_fm_inv3_to_geometric)
        self.e = jnp.array(e * utils.MeV_fm_inv3_to_geometric)
        
        # Pre-calculate quantities
        self.logn = jnp.log(self.n)
        self.logp = jnp.log(self.p)
        self.h = utils.cumtrapz(self.p / (self.e + self.p), jnp.log(self.p)) # enthalpy
        self.loge = jnp.log(self.e)
        self.logh = jnp.log(self.h)
        
        # TODO: might be better to use jnp.gradient?
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

    @partial(jax.jit, static_argnums=(0,))
    def energy_density_from_pseudo_enthalpy(self, h: Float):
        loge_of_h = jnp.interp(jnp.log(h), self.logh, self.loge)
        return jnp.exp(loge_of_h)

    @partial(jax.jit, static_argnums=(0,))
    def pressure_from_pseudo_enthalpy(self, h: Float):
        logp_of_h = jnp.interp(jnp.log(h), self.logh, self.logp)
        return jnp.exp(logp_of_h)

    @partial(jax.jit, static_argnums=(0,))
    def dloge_dlogp_from_pseudo_enthalpy(self, h: Float):
        return jnp.interp(h, self.h, self.dloge_dlogp)

    @partial(jax.jit, static_argnums=(0,))
    def pseudo_enthalpy_from_pressure(self, p: Float):
        logh_of_p = jnp.interp(jnp.log(p), self.logp, self.logh)
        return jnp.exp(logh_of_p)

    @partial(jax.jit, static_argnums=(0,))
    def pressure_from_number_density(self, n: Float):
        logp_of_n = jnp.interp(n, self.n, self.logp)
        return jnp.exp(logp_of_n)


class MetaModel_EOS_model(Interpolate_EOS_model):
    """
    MetaModel_EOS_model is a class to interpolate EOS data with a meta-model.

    Args:
        Interpolate_EOS_model (object): Base class of interpolation EOS data.
    """
    
    # TODO: decide whether these have to be attributes of Interpolate_EOS_model or MetaModel_EOS_model
    cs2: Array
    mu: Array
    
    def __init__(
        self,
        # Metamodel parameters
        NEP_dict: dict,
        kappas: tuple[Float, Float, Float, Float, Float, Float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        v_nq: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0],
        b_sat: Float = 17.0,
        b_sym: Float = 25.0,
        # density parameters
        nsat: Float = 0.16,
        nmin_nsat: Float = 0.1,
        nmax_nsat: Float = 12,
        ndat: Int = 200,
        # crust parameters
        crust: bool = "BPS",
        max_n_crust_nsat: Float = 0.5,
        ndat_spline: Int = 10
    ):
        """
        Initialize the MetaModel_EOS_model with the provided coefficients and compute auxiliary data.
        
        TODO: add documentation
        """
        
        # Save given attributes
        self.nsat = nsat
        self.v_nq = jnp.array(v_nq)
        self.b_sat = b_sat
        self.b_sym = b_sym
        self.N = 4 # TODO: this is fixed in the metamodeling papers, but we might want to extend this in the future
        
        # Set all the NEPs for the metamodel
        self.NEP_dict = NEP_dict
        
        self.E_sat = NEP_dict.get("E_sat", 0.0)
        self.K_sat = NEP_dict.get("K_sat", 0.0)
        self.Q_sat = NEP_dict.get("Q_sat", 0.0)
        self.Z_sat = NEP_dict.get("Z_sat", 0.0)

        self.E_sym = NEP_dict.get("E_sym", 0.0)
        self.L_sym = NEP_dict.get("L_sym", 0.0)
        self.K_sym = NEP_dict.get("K_sym", 0.0)
        self.Q_sym = NEP_dict.get("Q_sym", 0.0)
        self.Z_sym = NEP_dict.get("Z_sym", 0.0)

        # TODO: perhaps a bit cleaner but a bit less clear
        # for key in ["E_sat", "K_sat", "Q_sat", "Z_sat", "E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym"]:
        #     setattr(self, key, NEP_dict.get(key, 0.0))
        
        # TODO: clean up, not used so much?
        # Add the first derivative coefficient in Esat to make it work with jax.numpy.polyval
        coefficient_sat = jnp.array([self.E_sat,        0.0, self.K_sat, self.Q_sat, self.Z_sat])
        coefficient_sym = jnp.array([self.E_sym, self.L_sym, self.K_sym, self.Q_sym, self.Z_sym])
        
        # Get the coefficents index array and get coefficients
        index_sat = jnp.arange(len(coefficient_sat))
        index_sym = jnp.arange(len(coefficient_sym))

        self.coefficient_sat = coefficient_sat / factorial(index_sat)
        self.coefficient_sym = coefficient_sym / factorial(index_sym)
        
        # Preprocess the kappas
        assert len(kappas) == 6, "kappas must be a tuple of 6 values: kappa_sat, kappa_sat2, kappa_sat3, kappa_NM, kappa_NM2, kappa_NM3"
        self.kappa_sat, self.kappa_sat2, self.kappa_sat3, self.kappa_NM, self.kappa_NM2, self.kappa_NM3 = kappas
        self.kappa_sym = self.kappa_NM - self.kappa_sat
        self.kappa_sym2 = self.kappa_NM2 - self.kappa_sat2
        self.kappa_sym3 = self.kappa_NM3 - self.kappa_sat3
        
        # t_sat or TFGsat is the kinetic energy per nucleons in SM and at saturation, see just after eq (13) in the margueron paper
        self.t_sat = 3 * utils.hbar ** 2 / (10 * utils.m) * (3 * jnp.pi ** 2 * self.nsat / 2) ** (2/3)
        
        # Potential energy 
        # v_sat is defined in equations (22) - (26) in the Margueron et al. paper
        # TODO: there are more terms here, perhaps check the other reference that Rahul shared?
        v_sat_0 = self.E_sat -     self.t_sat * ( 1 +   self.kappa_sat +   self.kappa_sat2 +     self.kappa_sat3)
        v_sat_1 =            -     self.t_sat * ( 2 + 5*self.kappa_sat + 8 * self.kappa_sat2 + 11* self.kappa_sat3)
        v_sat_2 = self.K_sat - 2 * self.t_sat * (-1 + 5*self.kappa_sat + 20*self.kappa_sat2 + 44* self.kappa_sat3)
        v_sat_3 = self.Q_sat - 2 * self.t_sat * ( 4 - 5*self.kappa_sat + 40*self.kappa_sat2 + 220* self.kappa_sat3)
        v_sat_4 = self.Z_sat - 8 * self.t_sat * (-7 + 5*self.kappa_sat - 10*self.kappa_sat2 + 110* self.kappa_sat3) 
        
        self.v_sat = jnp.array([v_sat_0, v_sat_1, v_sat_2, v_sat_3, v_sat_4])
        
        # v_sym2 is defined in equations (27) to (31) in the Margueron et al. paper
        v_sym2_0 = self.E_sym -     self.t_sat * ( 2**(2/3)*( 1+  self.kappa_NM+   self.kappa_NM2+    self.kappa_NM3) - ( 1+  self.kappa_sat+   self.kappa_sat2+   self.kappa_sat3)  ) - v_nq[0]
        v_sym2_1 = self.L_sym -     self.t_sat * ( 2**(2/3)*( 2+5*self.kappa_NM+8* self.kappa_NM2+ 11*self.kappa_NM3) - ( 2+5*self.kappa_sat+8* self.kappa_sat2+11*self.kappa_sat3)  ) - v_nq[1]
        v_sym2_2 = self.K_sym - 2 * self.t_sat * ( 2**(2/3)*(-1+5*self.kappa_NM+20*self.kappa_NM2+ 44*self.kappa_NM3) - (-1+5*self.kappa_sat+20*self.kappa_sat2+44*self.kappa_sat3)  ) - v_nq[2]
        v_sym2_3 = self.Q_sym - 2 * self.t_sat * ( 2**(2/3)*( 4-5*self.kappa_NM+40*self.kappa_NM2+ 220*self.kappa_NM3) - ( 4-5*self.kappa_sat+40*self.kappa_sat2+220*self.kappa_sat3) ) - v_nq[3]
        v_sym2_4 = self.Z_sym - 8 * self.t_sat * ( 2**(2/3)*(-7+5*self.kappa_NM-10*self.kappa_NM2+ 110*self.kappa_NM3) - (-7+5*self.kappa_sat-10*self.kappa_sat2+110*self.kappa_sat3) ) - v_nq[4]
        
        self.v_sym2 = jnp.array([v_sym2_0, v_sym2_1, v_sym2_2, v_sym2_3, v_sym2_4])
        
        # Load and preprocess the crust
        ns_crust, ps_crust, es_crust = load_crust(crust)
        max_n_crust = max_n_crust_nsat * nsat
        mask = ns_crust <= max_n_crust
        ns_crust, ps_crust, es_crust = ns_crust[mask], ps_crust[mask], es_crust[mask]
        
        # FIXME: remove this once we discussed about this with Rahul
        # mu_lowest = (es_crust[0] + ps_crust[0]) / ns_crust[0]
        mu_lowest = 930.1193490245807
        
        cs2_crust = jnp.gradient(ps_crust, es_crust)
        
        # Make sure the metamodel starts above the crust
        max_n_crust = ns_crust[-1]
        nmin = nmin_nsat * self.nsat
        nmin = jnp.max(jnp.array([nmin, max_n_crust + 1e-3]))
        self.nmin = nmin
        
        # Create the density array
        self.nmax = nmax_nsat * self.nsat
        self.ndat = ndat
        
        # We first set the metamodel n array to self.n, to compute all auxiliary quantities
        n_metamodel = jnp.linspace(self.nmin, self.nmax, ndat)
        
        # Auxiliaries first
        x = self.compute_x(n_metamodel)
        proton_fraction = self.compute_proton_fraction(n_metamodel)
        delta = 1 - 2 * proton_fraction
        
        f_1 = self.compute_f_1(delta)
        f_star = self.compute_f_star(delta)
        f_star2 = self.compute_f_star2(delta)
        f_star3 = self.compute_f_star3(delta)
        v = self.compute_v(delta)
        b = self.compute_b(delta)
        
        # Other quantities
        p_metamodel = self.compute_pressure(x, f_1, f_star, f_star2, f_star3, b, v)
        e_metamodel = self.compute_energy(x, f_1, f_star, f_star2, f_star3, b, v)
        
        # Get cs2 for the metamodel
        cs2_metamodel = self.compute_cs2(n_metamodel, p_metamodel, e_metamodel, x, delta, f_1, f_star, f_star2, f_star3, b, v)
        
        # Spline for speed of sound for the connection region
        ns_spline = jnp.append(ns_crust, n_metamodel)
        cs2_spline = jnp.append(cs2_crust, cs2_metamodel)
        
        n_connection = jnp.linspace(max_n_crust, self.nmin, ndat_spline)
        cs2_connection = utils.cubic_spline(n_connection, ns_spline, cs2_spline)
        cs2_connection = jnp.clip(cs2_connection, 1e-5, 1.0)
        
        # Concatenate the arrays
        n = jnp.concatenate([ns_crust, n_connection, n_metamodel])
        cs2 = jnp.concatenate([cs2_crust, cs2_connection, cs2_metamodel])
        
        # Compute pressure and energy from chemical potential and initialize the parent class with it
        log_mu = utils.cumtrapz(cs2, jnp.log(n)) + jnp.log(mu_lowest)
        mu = jnp.exp(log_mu)
        p = utils.cumtrapz(cs2 * mu, n) + ps_crust[0]
        e = mu * n - p
        
        # TODO: this is perhaps best put in the top class but then how to do this for cs2 and mu?
        indices = jnp.where(jnp.diff(n) == 0.0)[0]
        n = jnp.delete(n, indices)
        p = jnp.delete(p, indices)
        e = jnp.delete(e, indices)
        cs2 = jnp.delete(cs2, indices)
        mu = jnp.delete(mu, indices)
        
        self.cs2 = cs2
        self.mu = mu
        
        super().__init__(n, p, e)
        
        
    #################
    ### AUXILIARY ###
    #################
        
    def u(self,
          x: Array,
          b: Array,
          alpha: Int):
        # TODO: documentation
        return 1 - ((-3 * x) ** (self.N + 1 - alpha) * jnp.exp(- b * (1 + 3 * x)))
        
    @partial(jax.jit, static_argnums=(0,))
    def compute_x(self,
                  n: Array):
        return (n - self.nsat) / (3 * self.nsat)
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_b(self,
                  delta: Array):
        return self.b_sat + self.b_sym * delta ** 2
        
    @partial(jax.jit, static_argnums=(0,))
    def compute_f_1(self,
                    delta: Array):
        return (1 + delta) ** (5/3) + (1 - delta) ** (5/3)
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_f_star(self,
                       delta: Array):
        return (self.kappa_sat + self.kappa_sym * delta) * (1 + delta) ** (5/3) + (self.kappa_sat - self.kappa_sym * delta) * (1 - delta) ** (5/3)
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_f_star2(self,
                        delta: Array):
        return (self.kappa_sat2 + self.kappa_sym2 * delta) * (1 + delta) ** (5/3) + (self.kappa_sat2 - self.kappa_sym2 * delta) * (1 - delta) ** (5/3)
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_f_star3(self,
                        delta: Array):
        return (self.kappa_sat3 + self.kappa_sym3 * delta) * (1 + delta) ** (5/3) + (self.kappa_sat3 - self.kappa_sym3 * delta) * (1 - delta) ** (5/3)
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_v(self,
                  delta: Array) -> Array:
        return jnp.array([self.v_sat[alpha] + self.v_sym2[alpha] * delta ** 2 + self.v_nq[alpha] * delta ** 4 for alpha in range(self.N + 1)])
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_energy(self,
                       x: Array,
                       f_1: Array,
                       f_star: Array,
                       f_star2: Array,
                       f_star3: Array,
                       b: Array,
                       v: Array) -> Array:
        
        prefac = self.t_sat / 2 * (1 + 3 * x) ** (2/3)
        linear = (1 + 3 * x) * f_star
        quadratic = (1 + 3 * x) ** 2 * f_star2
        cubic = (1 + 3 * x) ** 3 * f_star3
        
        kinetic_energy = prefac * (f_1 + linear + quadratic + cubic)
        
        # Potential energy
        # TODO: a bit cumbersome, find another way, jax tree map?
        potential_energy = 0
        for alpha in range(5):
            u = self.u(x, b, alpha)
            potential_energy += v.at[alpha].get() / (factorial(alpha)) * x ** alpha * u
        
        return kinetic_energy + potential_energy
    
    @partial(jax.jit, static_argnums=(0,))
    def esym(self,
             x: Array):
        # TODO: change this to be self-consistent: see Rahul's approach for that.
        return jnp.polyval(self.coefficient_sym[::-1], x)
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_pressure(self,
                         x: Array,
                         f_1: Array,
                         f_star: Array,
                         f_star2: Array,
                         f_star3: Array,
                         b: Array,
                         v: Array) -> Array:
        
        # TODO: currently only for ELFc!
        p_kin = 1/3 * self.nsat * self.t_sat * (1 + 3 * x) ** (5/3) *\
            (f_1 + 5/2 * (1 + 3 * x) * f_star + 4 * (1 + 3 * x) ** 2 * f_star2 \
             + 11/2 * (1 + 3 * x) ** 3 * f_star3)
    
        # TODO: cumbersome with jnp.array, find another way
        p_pot = 0
        for alpha in range(1, 5):
            u = self.u(x, b, alpha)
            fac1 = alpha * u
            fac2 = (self.N + 1 - alpha - 3 * b * x) * (u - 1)
            p_pot += v.at[alpha].get() / (factorial(alpha)) * x ** (alpha - 1) * (fac1 + fac2)
            
        p_pot = p_pot - v.at[0].get() * (-3) ** (self.N + 1) * x ** self.N * (self.N + 1 - 3 * b * x) * jnp.exp(- b * (1 + 3 * x)) 
        p_pot = p_pot * (1/3) * self.nsat * (1 + 3 * x) ** 2
        
        return p_pot + p_kin
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_cs2(self,
                    n: Array,
                    p: Array,
                    e: Array,
                    x: Array,
                    delta: Array,
                    f_1: Array,
                    f_star: Array,
                    f_star2: Array,
                    f_star3: Array,
                    b: Array,
                    v: Array):
        
        ### Compute incompressibility
        
        # Kinetic part
        K_kin = self.t_sat * (1 + 3 * x) ** (2/3) *\
            (-f_1 + 5 * (1 + 3 * x) * f_star + 20 * (1 + 3 * x) ** 2 * f_star2 \
             + 44 * (1 + 3 * x) ** 3 * f_star3)

        # Potential part
        K_pot = 0
        for alpha in range(2, self.N + 1):
            u = 1 -  ( (-3 * x) ** (self.N + 1 - alpha) * jnp.exp(- b * (1 + 3 * x)))
            x_up = (self.N + 1 - alpha - 3 * b * x) * (u - 1)
            x2_upp = (-(self.N + 1 - alpha) * (self.N - alpha) + 6 * b * x * (self.N + 1 - alpha) - 9 * x ** 2 * b ** 2) * (1 - u)
            
            K_pot = K_pot + v.at[alpha].get() / (factorial(alpha)) * x ** (alpha - 2) \
                * (alpha * (alpha - 1) * u + 2 * alpha * x_up + x2_upp)
                
        K_pot += v.at[0].get() * (-(self.N + 1) * (self.N) + 6 * b * x * (self.N + 1) - 9 * x ** 2 * b ** 2) *((-3) ** (self.N + 1) * x ** (self.N - 1) * jnp.exp(- b * (1 + 3 * x)))
        K_pot += 2 * v.at[1].get() * (self.N - 3 * b * x) * (-(-3) ** (self.N) * x ** (self.N - 1) * jnp.exp(- b * (1 + 3 * x)) )
        K_pot += v.at[1].get() * (-(self.N) * (self.N - 1) + 6 * b * x * (self.N) - 9 * x ** 2 * b ** 2) * ((-3)** (self.N) * x ** (self.N - 1) * jnp.exp(- b * (1 + 3 * x)))
        K_pot *= (1 + 3 * x) ** 2
        
        K = K_kin + K_pot + 18 / n * p
        
        # For electron
        
        K_Fb = (3. * jnp.pi**2 /2. * n) ** (1./3.) * utils.hbarc
        K_Fe = K_Fb * (1. - delta) ** (1./3.)  
        C = utils.m_e ** 4 / (8. * jnp.pi ** 2) / utils.hbarc ** 3
        x = K_Fe / utils.m_e
        f = x * (1 + 2 * x ** 2) * jnp.sqrt(1 + x ** 2) - jnp.arcsinh(x)
        
        e_electron = C*f
        p_electron = - e_electron + 8. / 3. * C * x ** 3 * jnp.sqrt(1 + x ** 2)
        K_electron = 8 * C / n * x ** 3 * (3 + 4 * x ** 2) / (jnp.sqrt(1 + x ** 2)) - 9 / n * (e_electron + p_electron)
        
        # Sum together:
        K_tot = K + K_electron
        
        # Finally, get cs2:
        chi = K_tot/9. 
        
        total_energy_density = (e + utils.m) * n + e_electron
        total_pressure = p + p_electron
        h_tot = (total_energy_density + total_pressure) / n
        
        cs2 = chi/h_tot
        
        return cs2
    
    def compute_proton_fraction(self,
                                n: Array) -> Float[Array, "n_points"]:
        """
        Computes the proton fraction for a given number density.

        Args:
            n (Float[Array, "n_points"]): Number density in fm^-3.

        Returns:
            Float[Array, "n_points"]: Proton fraction as a function of the number density.
        """
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
        # TODO: change this
        Esym = self.esym(n)
        
        a = 8.0 * Esym
        b = jnp.zeros(shape=n.shape)
        c = utils.hbarc * jnp.power(3.0 * jnp.pi**2 * n, 1.0 / 3.0)
        d = -4.0 * Esym - (utils.m_n - utils.m_p)
        
        coeffs = jnp.array(
            [
                a,
                b,
                c,
                d,
            ]
        ).T
        ys = utils.cubic_root_for_proton_fraction(coeffs)
        physical_ys = jnp.where(
            (ys.imag == 0.0) * (ys.real >= 0.0) * (ys.real <= 1.0),
            ys.real,
            jnp.zeros_like(ys.real),
        ).sum(axis=1)
        proton_fraction = jnp.power(physical_ys, 3)
        return proton_fraction

class MetaModel_with_CSE_EOS_model(Interpolate_EOS_model):
    """
    MetaModel_with_CSE_EOS_model is a class to interpolate EOS data with a meta-model and using the CSE.

    Args:
        Interpolate_EOS_model (object): Base class of interpolation EOS data.
    """
    def __init__(
        self,
        # parameters for the MetaModel
        NEP_dict: dict,
        nbreak_nsat: Float,
        # parameters for the CSE
        ngrids: Float[Array, "n_grid_point"],
        cs2grids: Float[Array, "n_grid_point"],
        # density parameters
        nsat: Float=  0.16,
        nmin_nsat: Float = 0.1,
        nmax_nsat: Float = 12,
        ndat_metamodel: Int = 100,
        ndat_CSE: Int = 100,
        **metamodel_kwargs
    ):
        """
        Initialize the MetaModel_with_CSE_EOS_model with the provided coefficients and compute auxiliary data.

        Args:
            coefficient_sat (Float[Array, "n_sat_coeff"]): The coefficients for the saturation part of the metamodel part of the EOS.
            coefficient_sym (Float[Array, "n_sym_coeff"]): The coefficients for the symmetry part of the metamodel part of the EOS.
            nbreak (Float): The number density at the transition point between the metamodel and the CSE part of the EOS.
            ngrids (Float[Array, "n_grid_point"]): The number densities for the CSE part of the EOS.
            cs2grids (Float[Array, "n_grid_point"]): The speed of sound squared for the CSE part of the EOS.
            nsat (Float, optional): Saturation density. Defaults to 0.16 fm^-3.
            nmin (Float, optional): Starting point of densities. Defaults to 0.1 fm^-3.
            nmax (Float, optional): End point of EOS. Defaults to 12*0.16 fm^-3, i.e. 12 nsat.
            ndat_metamodel (Int, optional): Number of datapoints to be used for the metamodel part of the EOS. Defaults to 1000.
            ndat_CSE (Int, optional): Number of datapoints to be used for the CSE part of the EOS. Defaults to 1000.
        """

        # TODO: align with new metamodel code
        nmax = nmax_nsat * nsat

        # Initializate the MetaModel part up to n_break
        self.metamodel = MetaModel_EOS_model(
            NEP_dict,
            nsat = nsat,
            nmin_nsat = nmin_nsat,
            nmax_nsat = nbreak_nsat,
            ndat = ndat_metamodel,
            **metamodel_kwargs
        )
        assert len(ngrids) == len(cs2grids), "ngrids and cs2grids must have the same length."
        # calculate the chemical potential at the transition point
        self.nbreak = nbreak_nsat * nsat
        
        # Convert units back for CSE initialization
        n_metamodel = self.metamodel.n / utils.fm_inv3_to_geometric
        p_metamodel = self.metamodel.p / utils.MeV_fm_inv3_to_geometric
        e_metamodel = self.metamodel.e / utils.MeV_fm_inv3_to_geometric
        
        # Get values at break density
        self.p_break   = jnp.interp(self.nbreak, n_metamodel, p_metamodel)
        self.e_break   = jnp.interp(self.nbreak, n_metamodel, e_metamodel)
        self.mu_break  = jnp.interp(self.nbreak, n_metamodel, self.metamodel.mu)
        self.cs2_break = jnp.interp(self.nbreak, n_metamodel, self.metamodel.cs2)
        
        # Define the speed-of-sound interpolation of the extension portion
        self.ngrids = jnp.concatenate((jnp.array([self.nbreak]), ngrids))
        self.cs2grids = jnp.concatenate((jnp.array([self.cs2_break]), cs2grids))
        self.cs2_extension_function = lambda n: jnp.interp(n, self.ngrids, self.cs2grids)
        
        # Compute n, p, e for CSE (number densities in unit of fm^-3)
        n_CSE = jnp.logspace(jnp.log10(self.nbreak), jnp.log10(nmax), num=ndat_CSE)
        cs2_CSE = self.cs2_extension_function(n_CSE)
        mu_CSE = self.mu_break * jnp.exp(utils.cumtrapz(cs2_CSE / n_CSE, n_CSE))
        p_CSE = self.p_break + utils.cumtrapz(cs2_CSE * mu_CSE, n_CSE)
        e_CSE = self.e_break + utils.cumtrapz(mu_CSE, n_CSE)
        
        # TODO: remove this, this is only saved to give to Rahul's TOV solver for cross-checking:
        self.n_CSE = n_CSE
        self.cs2_CSE = cs2_CSE
        
        # Combine metamodel and CSE data
        # TODO: converting units back and forth might be numerically unstable if conversion factors are large?
        n = jnp.concatenate((n_metamodel, n_CSE))
        p = jnp.concatenate((p_metamodel, p_CSE))
        e = jnp.concatenate((e_metamodel, e_CSE))
        
        cs2 = jnp.concatenate((self.metamodel.cs2, cs2_CSE))
        mu = jnp.concatenate((self.metamodel.mu, mu_CSE))
        
        # TODO: make less cumbersome, but this is needed since at this point p and e for sure have duplicate at nbreak due to cumtrapz first element being constant
        # TODO: perhaps it is best to also make cs2 and mu as part of the init. Then the base class can handle this kind of removal of duplicates
        for array_to_check in [n, p, e]:
            indices = jnp.where(jnp.diff(array_to_check) == 0.0)[0]
            
            n = jnp.delete(n, indices)
            p = jnp.delete(p, indices)
            e = jnp.delete(e, indices)
        
            cs2 = jnp.delete(cs2, indices)
            mu = jnp.delete(mu, indices)
            
        self.cs2 = cs2
        self.mu = mu

        super().__init__(n, p, e)
        

def construct_family(eos: tuple,
                     ndat: Int=50, 
                     min_nsat: Float=2) -> tuple[Float[Array, "ndat"], Float[Array, "ndat"], Float[Array, "ndat"], Float[Array, "ndat"]]:
    """
    Solve the TOV equations and generate the M, R and Lambda curves.

    Args:
        eos (tuple): Tuple of the EOS data (ns, ps, hs, es).
        ndat (int, optional): Number of datapoints used when constructing the central pressure grid. Defaults to 50.
        min_nsat (int, optional): Starting density for central pressure in numbers of nsat (assumed to be 0.16 fm^-3). Defaults to 2.

    Returns:
        tuple[Float[Array, "ndat"], Float[Array, "ndat"], Float[Array, "ndat"], Float[Array, "ndat"]]: log(pcs), masses in solar masses, radii in km, and dimensionless tidal deformabilities
    """
    # Construct the dictionary
    ns, ps, hs, es, dloge_dlogps = eos
    eos_dict = dict(p=ps, h=hs, e=es, dloge_dlogp=dloge_dlogps)
    
    # calculate the pc_min
    pc_min = utils.interp_in_logspace(min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps)

    # end at pc at pmax
    pc_max = eos_dict["p"][-1]

    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    ### TODO: Check the timing with this vmap implementation, which also works
    # def solve_single_pc(pc):
    #     """Solve for single pc value"""
    #     return tov.tov_solver(eos_dict, pc)
    # ms, rs, ks = jax.vmap(solve_single_pc)(pcs)
    
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
    
    # TODO: perhaps put a boolean here to flag whether or not to do this, or do we always want to do this?
    # Limit masses to be below MTOV
    ms, rs, lambdas = utils.limit_by_MTOV(ms, rs, lambdas)

    return jnp.log(pcs), ms, rs, lambdas
