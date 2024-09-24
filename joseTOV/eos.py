import os
import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
from jaxtyping import Array, Float, Int

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
    def __init__(self):
        pass
    
    def interpolate_eos(self,
                        n: Float[Array, "n_points"],
                        p: Float[Array, "n_points"],
                        e: Float[Array, "n_points"]):
        """
        Given n, p and e, interpolate to obtain necessary auxiliary quantities. 

        Args:
            n (Float[Array, n_points]): Number densities. Expected units are n[fm^-3]
            p (Float[Array, n_points]): Pressure values. Expected units are p[MeV / fm^3]
            e (Float[Array, n_points]): Energy densities. Expected units are e[MeV / fm^3]
        """
        
        # Save the provided data as attributes, make conversions
        ns = jnp.array(n * utils.fm_inv3_to_geometric)
        ps = jnp.array(p * utils.MeV_fm_inv3_to_geometric)
        es = jnp.array(e * utils.MeV_fm_inv3_to_geometric)
        
        hs = utils.cumtrapz(ps / (es + ps), jnp.log(ps)) # enthalpy
        # TODO: might be better to use jnp.gradient?
        dloge_dlogps = jnp.diff(jnp.log(e)) / jnp.diff(jnp.log(p))
        dloge_dlogps = jnp.concatenate(
            (
                jnp.array(
                    [
                        dloge_dlogps.at[0].get(),
                    ]
                ),
                dloge_dlogps,
            )
        )
        return ns, ps, hs, es, dloge_dlogps

    # TODO: remove?
    # def energy_density_from_pseudo_enthalpy(self, h: Float):
    #     loge_of_h = jnp.interp(jnp.log(h), self.logh, self.loge)
    #     return jnp.exp(loge_of_h)

    # def pressure_from_pseudo_enthalpy(self, h: Float):
    #     logp_of_h = jnp.interp(jnp.log(h), self.logh, self.logp)
    #     return jnp.exp(logp_of_h)

    # def dloge_dlogp_from_pseudo_enthalpy(self, h: Float):
    #     return jnp.interp(h, self.h, self.dloge_dlogp)

    # def pseudo_enthalpy_from_pressure(self, p: Float):
    #     logh_of_p = jnp.interp(jnp.log(p), self.logp, self.logh)
    #     return jnp.exp(logh_of_p)

    # def pressure_from_number_density(self, n: Float):
    #     logp_of_n = jnp.interp(n, self.n, self.logp)
    #     return jnp.exp(logp_of_n)


class MetaModel_EOS_model(Interpolate_EOS_model):
    """
    MetaModel_EOS_model is a class to interpolate EOS data with a meta-model.

    Args:
        Interpolate_EOS_model (object): Base class of interpolation EOS data.
    """
    
    def __init__(
        self,
        kappas: tuple[Float, Float, Float, Float, Float, Float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        v_nq: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0],
        b_sat: Float = 17.0,
        b_sym: Float = 25.0,
        # density parameters
        nsat: Float = 0.16,
        nmin_MM_nsat: Float = 0.12 / 0.16, 
        nmax_nsat: Float = 12,
        ndat: Int = 200,
        # crust parameters
        crust_name: bool = "BPS",
        max_n_crust_nsat: Float = 0.5,
        ndat_spline: Int = 10
    ):
        """
        Initialize the MetaModel_EOS_model with the provided coefficients and compute auxiliary data.
        
        TODO: add documentation
        """
        
        # Save as attributes
        self.nsat = nsat
        self.v_nq = jnp.array(v_nq)
        self.b_sat = b_sat
        self.b_sym = b_sym
        self.N = 4 # TODO: this is fixed in the metamodeling papers, but we might want to extend this in the future
        
        self.nmin_MM_nsat = nmin_MM_nsat
        self.nmax_nsat = nmax_nsat
        self.ndat = ndat
        self.max_n_crust_nsat = max_n_crust_nsat
        self.ndat_spline = ndat_spline
        
        # Constructions
        assert len(kappas) == 6, "kappas must be a tuple of 6 values: kappa_sat, kappa_sat2, kappa_sat3, kappa_NM, kappa_NM2, kappa_NM3"
        self.kappa_sat, self.kappa_sat2, self.kappa_sat3, self.kappa_NM, self.kappa_NM2, self.kappa_NM3 = kappas
        self.kappa_sym = self.kappa_NM - self.kappa_sat
        self.kappa_sym2 = self.kappa_NM2 - self.kappa_sat2
        self.kappa_sym3 = self.kappa_NM3 - self.kappa_sat3
        
        # t_sat or TFGsat is the kinetic energy per nucleons in SM and at saturation, see just after eq (13) in the margueron paper
        self.t_sat = 3 * utils.hbar ** 2 / (10 * utils.m) * (3 * jnp.pi ** 2 * self.nsat / 2) ** (2/3)
        
        # v_sat is defined in equations (22) - (26) in the Margueron et al. paper
        self.v_sat_0_no_NEP = -self.t_sat * (1 + self.kappa_sat + self.kappa_sat2 + self.kappa_sat3)
        self.v_sat_1_no_NEP = -self.t_sat * (2 + 5 * self.kappa_sat + 8 * self.kappa_sat2 + 11 * self.kappa_sat3)
        self.v_sat_2_no_NEP = - 2 * self.t_sat * (-1 + 5 * self.kappa_sat + 20 * self.kappa_sat2 + 44 * self.kappa_sat3)
        self.v_sat_3_no_NEP = - 2 * self.t_sat * ( 4 - 5 * self.kappa_sat + 40 * self.kappa_sat2 + 220 * self.kappa_sat3)
        self.v_sat_4_no_NEP = - 8 * self.t_sat * (-7 + 5*self.kappa_sat - 10*self.kappa_sat2 + 110* self.kappa_sat3) 
        
        self.v_sym2_0_no_NEP = - self.t_sat * (2 ** (2/3) * (1 + self.kappa_NM + self.kappa_NM2 + self.kappa_NM3) - (1 + self.kappa_sat + self.kappa_sat2 + self.kappa_sat3)) - self.v_nq[0]
        self.v_sym2_1_no_NEP = - self.t_sat * (2 ** (2/3) * (2 + 5 * self.kappa_NM + 8 * self.kappa_NM2 + 11 * self.kappa_NM3) - (2 + 5 * self.kappa_sat + 8 * self.kappa_sat2 + 11 * self.kappa_sat3)) - self.v_nq[1]
        self.v_sym2_2_no_NEP = - 2 * self.t_sat * (2 ** (2/3) * (-1 + 5 * self.kappa_NM + 20 * self.kappa_NM2 + 44 * self.kappa_NM3) - (-1 + 5 * self.kappa_sat + 20 * self.kappa_sat2 + 44 * self.kappa_sat3)) - self.v_nq[2]
        self.v_sym2_3_no_NEP = - 2 * self.t_sat * (2 ** (2/3) * (4 - 5 * self.kappa_NM + 40 * self.kappa_NM2 + 220 * self.kappa_NM3) - ( 4 - 5 * self.kappa_sat + 40 * self.kappa_sat2 + 220 * self.kappa_sat3)) - self.v_nq[3]
        self.v_sym2_4_no_NEP = - 8 * self.t_sat * (2 ** (2/3) * (-7 + 5 * self.kappa_NM - 10 * self.kappa_NM2 + 110 * self.kappa_NM3) - (-7 + 5 * self.kappa_sat - 10 * self.kappa_sat2 + 110 * self.kappa_sat3)) - self.v_nq[4]
        
        # Load and preprocess the crust
        ns_crust, ps_crust, es_crust = load_crust(crust_name)
        max_n_crust = max_n_crust_nsat * nsat
        mask = ns_crust <= max_n_crust
        self.ns_crust, self.ps_crust, self.es_crust = ns_crust[mask], ps_crust[mask], es_crust[mask]
        
        self.mu_lowest = (es_crust[0] + ps_crust[0]) / ns_crust[0]
        self.cs2_crust = jnp.gradient(ps_crust, es_crust)
        
        # Make sure the metamodel starts above the crust
        self.max_n_crust = ns_crust[-1]
        
        # Create density arrays
        self.nmax = nmax_nsat * self.nsat
        self.ndat = ndat
        self.nmin_MM = self.nmin_MM_nsat * self.nsat
        self.n_metamodel = jnp.linspace(self.nmin_MM, self.nmax, self.ndat, endpoint = False)
        self.ns_spline = jnp.append(self.ns_crust, self.n_metamodel)
        self.n_connection = jnp.linspace(self.max_n_crust + 1e-5, self.nmin_MM, self.ndat_spline, endpoint = False)
        
    def construct_eos(self, 
                      NEP_dict: dict):
        
        E_sat = NEP_dict.get("E_sat", -16.0)
        K_sat = NEP_dict.get("K_sat", 0.0)
        Q_sat = NEP_dict.get("Q_sat", 0.0)
        Z_sat = NEP_dict.get("Z_sat", 0.0)

        E_sym = NEP_dict.get("E_sym", 0.0)
        L_sym = NEP_dict.get("L_sym", 0.0)
        K_sym = NEP_dict.get("K_sym", 0.0)
        Q_sym = NEP_dict.get("Q_sym", 0.0)
        Z_sym = NEP_dict.get("Z_sym", 0.0)

        # TODO: clean up, not used so much?
        # Add the first derivative coefficient in Esat to make it work with jax.numpy.polyval
        coefficient_sat = jnp.array([E_sat,   0.0, K_sat, Q_sat, Z_sat])
        coefficient_sym = jnp.array([E_sym, L_sym, K_sym, Q_sym, Z_sym])
        
        # Get the coefficents index array and get coefficients
        index_sat = jnp.arange(len(coefficient_sat))
        index_sym = jnp.arange(len(coefficient_sym))

        coefficient_sat = coefficient_sat / factorial(index_sat)
        coefficient_sym = coefficient_sym / factorial(index_sym)
        
        # Potential energy 
        # v_sat is defined in equations (22) - (26) in the Margueron et al. paper
        # TODO: there are more terms here, perhaps check the other reference that Rahul shared?
        v_sat = jnp.array([E_sat + self.v_sat_0_no_NEP, 
                           0.0   + self.v_sat_1_no_NEP,
                           K_sat + self.v_sat_2_no_NEP,
                           Q_sat + self.v_sat_3_no_NEP,
                           Z_sat + self.v_sat_4_no_NEP])
        
        # v_sym2 is defined in equations (27) to (31) in the Margueron et al. paper
        v_sym2 = jnp.array([E_sym + self.v_sym2_0_no_NEP, 
                            L_sym + self.v_sym2_1_no_NEP,
                            K_sym + self.v_sym2_2_no_NEP,
                            Q_sym + self.v_sym2_3_no_NEP,
                            Z_sym + self.v_sym2_4_no_NEP])
        
        # Auxiliaries first
        x = self.compute_x(self.n_metamodel)
        proton_fraction = self.compute_proton_fraction(coefficient_sym, self.n_metamodel)
        delta = 1 - 2 * proton_fraction
        
        f_1 = self.compute_f_1(delta)
        f_star = self.compute_f_star(delta)
        f_star2 = self.compute_f_star2(delta)
        f_star3 = self.compute_f_star3(delta)
        v = self.compute_v(v_sat, v_sym2, delta)
        b = self.compute_b(delta)
        
        # Other quantities
        p_metamodel = self.compute_pressure(x, f_1, f_star, f_star2, f_star3, b, v)
        e_metamodel = self.compute_energy(x, f_1, f_star, f_star2, f_star3, b, v)
        
        # Get cs2 for the metamodel
        cs2_metamodel = self.compute_cs2(self.n_metamodel, p_metamodel, e_metamodel, x, delta, f_1, f_star, f_star2, f_star3, b, v)
        
        # Spline for speed of sound for the connection region
        
        cs2_spline = jnp.append(self.cs2_crust, cs2_metamodel)
        
        cs2_connection = utils.cubic_spline(self.n_connection, self.ns_spline, cs2_spline)
        cs2_connection = jnp.clip(cs2_connection, 1e-5, 1.0)
        
        # Concatenate the arrays
        n = jnp.concatenate([self.ns_crust, self.n_connection, self.n_metamodel])
        cs2 = jnp.concatenate([self.cs2_crust, cs2_connection, cs2_metamodel])
        
        # Compute pressure and energy from chemical potential and initialize the parent class with it
        log_mu = utils.cumtrapz(cs2, jnp.log(n)) + jnp.log(self.mu_lowest)
        mu = jnp.exp(log_mu)
        p = utils.cumtrapz(cs2 * mu, n) + self.ps_crust[0]
        e = mu * n - p
        
        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)
        
        return ns, ps, hs, es, dloge_dlogps, mu, cs2
        
        
    #################
    ### AUXILIARY ###
    #################
        
    def u(self,
          x: Array,
          b: Array,
          alpha: Int):
        # TODO: documentation
        return 1 - ((-3 * x) ** (self.N + 1 - alpha) * jnp.exp(- b * (1 + 3 * x)))
        
    def compute_x(self,
                  n: Array):
        return (n - self.nsat) / (3 * self.nsat)
    
    def compute_b(self,
                  delta: Array):
        return self.b_sat + self.b_sym * delta ** 2
        
    def compute_f_1(self,
                    delta: Array):
        return (1 + delta) ** (5/3) + (1 - delta) ** (5/3)
    
    def compute_f_star(self,
                       delta: Array):
        return (self.kappa_sat + self.kappa_sym * delta) * (1 + delta) ** (5/3) + (self.kappa_sat - self.kappa_sym * delta) * (1 - delta) ** (5/3)
    
    def compute_f_star2(self,
                        delta: Array):
        return (self.kappa_sat2 + self.kappa_sym2 * delta) * (1 + delta) ** (5/3) + (self.kappa_sat2 - self.kappa_sym2 * delta) * (1 - delta) ** (5/3)
    
    def compute_f_star3(self,
                        delta: Array):
        return (self.kappa_sat3 + self.kappa_sym3 * delta) * (1 + delta) ** (5/3) + (self.kappa_sat3 - self.kappa_sym3 * delta) * (1 - delta) ** (5/3)
    
    def compute_v(self,
                  v_sat: Array,
                  v_sym2: Array,
                  delta: Array) -> Array:
        return jnp.array([v_sat[alpha] + v_sym2[alpha] * delta ** 2 + self.v_nq[alpha] * delta ** 4 for alpha in range(self.N + 1)])
    
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
    
    def esym(self,
             coefficient_sym: list,
             x: Array):
        # TODO: change this to be self-consistent: see Rahul's approach for that.
        return jnp.polyval(coefficient_sym[::-1], x)
    
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
                                coefficient_sym: list,
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
        Esym = self.esym(coefficient_sym, n)
        
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
        nsat: Float =  0.16,
        nmin_MM_nsat: Float = 0.12 / 0.16,
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
        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_MM_nsat = nmin_MM_nsat
        self.ndat_metamodel = ndat_metamodel
        self.metamodel_kwargs = metamodel_kwargs

    def construct_eos(self,
                      NEP_dict: dict,
                      ngrids: Float[Array, "n_grid_point"],
                      cs2grids: Float[Array, "n_grid_point"]):
        
        # Initializate the MetaModel part up to n_break
        metamodel = MetaModel_EOS_model(nsat = self.nsat,
                                        nmin_MM_nsat = self.nmin_MM_nsat,
                                        nmax_nsat = NEP_dict["nbreak"] / self.nsat,
                                        ndat = self.ndat_metamodel,
                                        **self.metamodel_kwargs
        )
        
        # Construct the metamodel part:
        mm_output = metamodel.construct_eos(NEP_dict)
        n_metamodel, p_metamodel, _, e_metamodel, _, mu_metamodel, cs2_metamodel = mm_output
        
        # Convert units back for CSE initialization
        n_metamodel = n_metamodel / utils.fm_inv3_to_geometric
        p_metamodel = p_metamodel / utils.MeV_fm_inv3_to_geometric
        e_metamodel = e_metamodel / utils.MeV_fm_inv3_to_geometric
        
        # Get values at break density
        p_break   = jnp.interp(NEP_dict["nbreak"], n_metamodel, p_metamodel)
        e_break   = jnp.interp(NEP_dict["nbreak"], n_metamodel, e_metamodel)
        mu_break  = jnp.interp(NEP_dict["nbreak"], n_metamodel, mu_metamodel)
        cs2_break = jnp.interp(NEP_dict["nbreak"], n_metamodel, cs2_metamodel)
        
        # Define the speed-of-sound interpolation of the extension portion
        ngrids = jnp.concatenate((jnp.array([NEP_dict["nbreak"]]), ngrids))
        cs2grids = jnp.concatenate((jnp.array([cs2_break]), cs2grids))
        cs2_extension_function = lambda n: jnp.interp(n, ngrids, cs2grids)
        
        # Compute n, p, e for CSE (number densities in unit of fm^-3)
        n_CSE = jnp.logspace(jnp.log10(NEP_dict["nbreak"]), jnp.log10(self.nmax), num=self.ndat_CSE)
        cs2_CSE = cs2_extension_function(n_CSE)
        
        # We add a very small number to avoid problems with duplicates below
        mu_CSE = mu_break * jnp.exp(utils.cumtrapz(cs2_CSE / n_CSE, n_CSE)) + 1e-6
        p_CSE = p_break + utils.cumtrapz(cs2_CSE * mu_CSE, n_CSE) + 1e-6
        e_CSE = e_break + utils.cumtrapz(mu_CSE, n_CSE) + 1e-6
        
        # Combine metamodel and CSE data
        n = jnp.concatenate((n_metamodel, n_CSE))
        p = jnp.concatenate((p_metamodel, p_CSE))
        e = jnp.concatenate((e_metamodel, e_CSE))
        
        # TODO: let's decide whether we want to save cs2 and mu or just use them for computation and then discard them.
        mu = jnp.concatenate((mu_metamodel, mu_CSE))
        cs2 = jnp.concatenate((cs2_metamodel, cs2_CSE))
        
        # # FIXME: this is pretty experimental, but we have duplicates which will break TOV solver but are hard to remove in a JIT-compatible manner. Note that we should perhaps do something similar in the metamodel EOS. 
        
        # for array_to_check in [n, p, e]:
        #     indices = jnp.where(jnp.diff(array_to_check) <= 0.0)[0][0]
        #     print(indices)
            
        #     print(f"n at duplicates +/- 1: {n[indices-1:indices+1] /0.16} nsat")
        # n = jnp.unique(n)
        # e = jnp.unique(e)
        # p = jnp.unique(p)

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)
        
        return ns, ps, hs, es, dloge_dlogps, mu, cs2
        
        
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
