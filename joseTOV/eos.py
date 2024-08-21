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
    """
    MetaModel_EOS_model is a class to interpolate EOS data with a meta-model.

    Args:
        Interpolate_EOS_model (object): Base class of interpolation EOS data.
    """
    def __init__(
        self,
        # Metamodel parameters
        NEP_dict: dict,
        kappas: tuple[Float, Float, Float, Float, Float, Float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        v_nq: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0],
        b_sat: Float = 17.0, # TODO: change default value
        b_sym: Float = 25.0, # TODO: change default value
        N: Int = 4, # TODO: I think in the current implementation this is fixed so pointless to give as user.
        which_ELF: str = "ELFc",
        # density parameters
        nsat=0.16,
        nmin=0.1, # in fm^-3
        nmax_nsat=12, # in numbers of nsat
        ndat=1000,
        # # proton fraction
        # fix_proton_fraction=False,
        # fix_proton_fraction_val=0.02,
        # crust
        crust = "BPS",
        max_n_crust: Float = 0.08, # in fm^-3,
        ndat_spline: Int = 10
    ):
        """
        Initialize the MetaModel_EOS_model with the provided coefficients and compute auxiliary data.
        
        TODO: add documentation
        """
        
        # TODO: make sure this is used properly everywhere
        assert which_ELF in ["ELFa", "ELFc"], "which_ELF must be either ELFa or ELFc"
        
        if which_ELF == "ELFa":
            self.u = self.u_ELFa
        else:
            self.u = self.u_ELFc
        
        # Save given attributes
        self.nsat = nsat
        # self.fix_proton_fraction = fix_proton_fraction
        # self.fix_proton_fraction_val = fix_proton_fraction_val
        self.max_n_crust = max_n_crust
        self.v_nq = jnp.array(v_nq)
        self.b_sat = b_sat
        self.b_sym = b_sym
        self.N = 4 # TODO: check if this is needed as input, but I don't think so
        
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
        mask = ns_crust <= max_n_crust
        ns_crust, ps_crust, es_crust = ns_crust[mask], ps_crust[mask], es_crust[mask]
        mu_lowest = (es_crust[0] + ps_crust[0]) / ns_crust[0]
        
        # # FIXME: remove this once we discussed about this with Rahul
        # print("WARNING OVERRIDING FOR DEBUGGING")
        # mu_lowest = 930.1193490245807
        # print("Mu lowest jose:", mu_lowest)
        
        cs2_crust = jnp.gradient(ps_crust, es_crust)
        
        print("For debugging: check the shape")
        
        print("len(ns_crust)")
        print(len(ns_crust))
        
        print("len(cs2_crust)")
        print(len(cs2_crust))

        # Make sure the metamodel starts above the crust
        max_n_crust = ns_crust[-1]
        nmin = max(nmin, max_n_crust + 1e-3)
        self.nmin = nmin
        
        # Create the density array
        self.nmax = nmax_nsat * self.nsat
        self.ndat = ndat
        
        # We first set the metamodel n array to self.n, to compute all auxiliary quantities
        self.n = jnp.linspace(nmin, self.nmax, ndat)
        
        # Auxiliaries first
        self.x = self.compute_x()
        self.proton_fraction = self.compute_proton_fraction()
        self.delta = self.compute_delta()
        
        self.f_1 = self.compute_f_1()
        self.f_star = self.compute_f_star()
        self.f_star2 = self.compute_f_star2()
        self.f_star3 = self.compute_f_star3()
        self.v = self.compute_v()
        self.b = self.compute_b()
        
        # Other quantities
        self.pressure = self.compute_pressure()
        self.energy = self.compute_energy()
        
        # Get cs2 for the metamodel
        cs2_metamodel = self.compute_cs2()
        
        # Spline for speed of sound for the connection region
        ns_spline = jnp.append(ns_crust, self.n)
        cs2_spline = jnp.append(cs2_crust, cs2_metamodel)
        
        n_connection = jnp.linspace(max_n_crust, self.nmin, ndat_spline)
        cs2_connection = utils.cubic_spline(n_connection, ns_spline, cs2_spline)
        cs2_connection = jnp.clip(cs2_connection, 1e-5, 1.0)
        
        # Concatenate the arrays
        n = jnp.concatenate([ns_crust, n_connection, self.n])
        cs2 = jnp.concatenate([cs2_crust, cs2_connection, cs2_metamodel])
        
        # Compute pressure and energy from chemical potential and initialize the parent class with it
        log_mu = utils.cumtrapz(cs2, jnp.log(n)) + jnp.log(mu_lowest)
        mu = jnp.exp(log_mu)
        p = utils.cumtrapz(cs2 * mu, n) + ps_crust[0]
        e = mu * n - p
        
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
        
    def u_ELFa(self, 
               alpha: Int):
        return 1
    
    def u_ELFc(self,
               alpha: Int):
        return 1 - ((-3 * self.x) ** (self.N + 1 - alpha) * jnp.exp(- self.b * (1 + 3 * self.x))) 
        
    def compute_x(self):
        return (self.n - self.nsat) / (3 * self.nsat)
    
    def compute_delta(self):
        return 1 - 2 * self.proton_fraction
    
    def compute_b(self):
        return self.b_sat + self.b_sym * self.delta ** 2
        
    def compute_f_1(self):
        return (1 + self.delta) ** (5/3) + (1 - self.delta) ** (5/3)
    
    def compute_f_star(self):
        return (self.kappa_sat + self.kappa_sym * self.delta) * (1 + self.delta) ** (5/3) + (self.kappa_sat - self.kappa_sym * self.delta) * (1 - self.delta) ** (5/3)
    
    def compute_f_star2(self):
        return (self.kappa_sat2 + self.kappa_sym2 * self.delta) * (1 + self.delta) ** (5/3) + (self.kappa_sat2 - self.kappa_sym2 * self.delta) * (1 - self.delta) ** (5/3)
    
    def compute_f_star3(self):
        return (self.kappa_sat3 + self.kappa_sym3 * self.delta) * (1 + self.delta) ** (5/3) + (self.kappa_sat3 - self.kappa_sym3 * self.delta) * (1 - self.delta) ** (5/3)
    
    def compute_v(self) -> Array:
        return jnp.array([self.v_sat[alpha] + self.v_sym2[alpha] * self.delta ** 2 + self.v_nq[alpha] * self.delta ** 4 for alpha in range(self.N + 1)])
    
    def compute_energy(self):
        
        prefac = self.t_sat / 2 * (1 + 3 * self.x) ** (2/3)
        linear = (1 + 3 * self.x) * self.f_star
        quadratic = (1 + 3 * self.x) ** 2 * self.f_star2
        cubic = (1 + 3 * self.x) ** 3 * self.f_star3
        
        kinetic_energy = prefac * (self.f_1 + linear + quadratic + cubic)
        
        # Potential energy
        # TODO: a bit cumbersome, find another way, jax tree map?
        potential_energy = 0
        for alpha in range(5):
            u = self.u(alpha)
            potential_energy += self.v.at[alpha].get() / (factorial(alpha)) * self.x ** alpha * u
        
        return kinetic_energy + potential_energy
    
    def esym(self):
        # TODO: change this to be self-consistent: see Rahul's approach for that.
        return jnp.polyval(self.coefficient_sym[::-1], self.x)
    
    def compute_pressure(self):
        
        # TODO: currently only for ELFc!
        p_kin = 1/3 * self.nsat * self.t_sat * (1 + 3 * self.x) ** (5/3) *\
            (self.f_1 + 5/2 * (1 + 3 * self.x) * self.f_star + 4 * (1 + 3 * self.x) ** 2 * self.f_star2 \
             + 11/2 * (1 + 3 * self.x) ** 3 * self.f_star3)
    
        # TODO: cumbersome with jnp.array, find another way
        p_pot = 0
        for alpha in range(1, 5):
            u = self.u(alpha)
            fac1 = alpha * u
            fac2 = (self.N + 1 - alpha - 3 * self.b * self.x) * (u - 1)
            p_pot += self.v.at[alpha].get() / (factorial(alpha)) * self.x ** (alpha - 1) * (fac1 + fac2)
            
        p_pot = p_pot - self.v.at[0].get() * (-3) ** (self.N + 1) * self.x ** self.N * (self.N + 1 - 3 * self.b * self.x) * jnp.exp(- self.b * (1 + 3 * self.x)) 
        p_pot = p_pot * (1/3) * self.nsat * (1 + 3 * self.x) ** 2
        
        return p_pot + p_kin
    
    def compute_incompressibility(self):
        # Kinetic part
        K_kin = self.t_sat * (1 + 3 * self.x) ** (2/3) *\
            (-self.f_1 + 5 * (1 + 3 * self.x) * self.f_star + 20 * (1 + 3 * self.x) ** 2 * self.f_star2 \
             + 44 * (1 + 3 * self.x) ** 3 * self.f_star3)

        # Potential part
        K_pot = 0
        for alpha in range(2, self.N + 1):
            u = 1 -  ( (-3 * self.x) ** (self.N + 1 - alpha) * jnp.exp(- self.b * (1 + 3 * self.x)))
            x_up = (self.N + 1 - alpha - 3 * self.b * self.x) * (u - 1)
            x2_upp = (-(self.N + 1 - alpha) * (self.N - alpha) + 6 * self.b * self.x * (self.N + 1 - alpha) - 9 * self.x ** 2 * self.b ** 2) * (1 - u)
            
            K_pot = K_pot + self.v.at[alpha].get() / (factorial(alpha)) * self.x ** (alpha - 2) \
                * (alpha * (alpha - 1) * u + 2 * alpha * x_up + x2_upp)
                
        K_pot += self.v.at[0].get() * (-(self.N + 1) * (self.N) + 6 * self.b * self.x * (self.N + 1) - 9 * self.x ** 2 * self.b ** 2) *((-3) ** (self.N + 1) * self.x ** (self.N - 1) * jnp.exp(- self.b * (1 + 3 * self.x)))
        K_pot += 2 * self.v.at[1].get() * (self.N - 3 * self.b * self.x) * (-(-3) ** (self.N) * self.x ** (self.N - 1) * jnp.exp(- self.b * (1 + 3 * self.x)) )
        K_pot += self.v.at[1].get() * (-(self.N) * (self.N - 1) + 6 * self.b * self.x * (self.N) - 9 * self.x ** 2 * self.b ** 2) * ((-3)** (self.N) * self.x ** (self.N - 1) * jnp.exp(- self.b * (1 + 3 * self.x)))
        K_pot *= (1 + 3 * self.x) ** 2
        
        K = K_kin + K_pot + 18 / self.n * self.pressure
        return K
    
    def compute_c2_s(self):
        
        K = self.compute_incompressibility()
        h = utils.m + self.energy + self.pressure / self.n
        
        return K/9 /h
    
    def energy_density_electron(self):
        
        K_Fb = (3. * jnp.pi ** 2 /2. * self.n) ** (1./3.) * utils.hbarc
        K_Fe = K_Fb * (1.- self.delta) ** (1./3.)  
        
        x = K_Fe / utils.m_e
        
        C = utils.m_e ** 4 / (8. * jnp.pi ** 2) / utils.hbarc ** 3
        f = x * (1 + 2 * x ** 2) * jnp.sqrt(1 + x ** 2) - jnp.arcsinh(x)
        
        energy_density = C*f
        
        return energy_density
    
    def electron_pressure(self):
        
        energy_density = self.energy_density_electron()
        
        K_Fb = (3. * jnp.pi ** 2 / 2. * self.n) ** (1./3.) * utils.hbarc
        K_Fe = K_Fb * (1. - self.delta) ** (1./3.)  
        C = utils.m_e ** 4 / (8. * jnp.pi ** 2) / utils.hbarc ** 3
        x = K_Fe / utils.m_e
        
        pressure = - energy_density + 8. / 3. * C * x ** 3 * jnp.sqrt(1 + x ** 2)
        
        return pressure
    
    def compute_incompressibility_electron(self):
        
        K_Fb = (3. * jnp.pi**2 /2. * self.n) ** (1./3.) * utils.hbarc
        K_Fe = K_Fb * (1. - self.delta) ** (1./3.)  
        C = utils.m_e ** 4 / (8. * jnp.pi ** 2) / utils.hbarc ** 3
        x = K_Fe / utils.m_e
        
        energy_density = self.energy_density_electron()
        pressure = self.electron_pressure()
        
        K = 8 * C / self.n * x ** 3 * (3 + 4 * x ** 2) / (jnp.sqrt(1 + x ** 2)) - 9 / self.n * (energy_density + pressure)
        return K
    
    def mu_e(self,
             n: Array):
        
        delta = self.compute_delta(n)
        
        K_Fb = (3. * jnp.pi ** 2 / 2. * n) ** (1./3.) * utils.hbarc
        K_Fe = K_Fb * (1. - delta)**(1./3.)  
    
        ans = jnp.sqrt(utils.m_e ** 2 + K_Fe ** 2)
        return ans
    
    
    def compute_cs2(self):
        K_tot = self.compute_incompressibility() + self.compute_incompressibility_electron()
        
        chi = K_tot/9. 
        
        total_energy_density = (self.energy + utils.m) * self.n + self.energy_density_electron()
        total_pressure = self.pressure + self.electron_pressure()
        h_tot =  (total_energy_density + total_pressure) / self.n
        
        cs2 = chi/h_tot
        
        return cs2
    
    def compute_proton_fraction(self) -> Float[Array, "n_points"]:
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
        Esym = self.esym()
        
        a = 8.0 * Esym
        b = jnp.zeros(shape=self.n.shape)
        c = utils.hbarc * jnp.power(3.0 * jnp.pi**2 * self.n, 1.0 / 3.0)
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
        coefficient_sat: Float[Array, "n_sat_coeff"],
        coefficient_sym: Float[Array, "n_sym_coeff"],
        nbreak: Float,
        # parameters for the CSE
        ngrids: Float[Array, "n_grid_point"],
        cs2grids: Float[Array, "n_grid_point"],
        nsat: Float=0.16,
        nmin: Float=0.1,
        nmax: Float=12 * 0.16,
        ndat_metamodel: Int=1000,
        ndat_CSE: Int=1000,
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

        # Initializate the MetaModel part up to n_break
        self.metamodel = MetaModel_EOS_model(
            coefficient_sat,
            coefficient_sym,
            nsat=nsat,
            nmin=nmin,
            nmax=nbreak,
            ndat=ndat_metamodel,
            **metamodel_kwargs
        )
        assert len(ngrids) == len(cs2grids), "ngrids and cs2grids must have the same length."
        # calculate the chemical potential at the transition point
        self.nbreak = nbreak
        
        # TODO: seems a bit cumbersome, can we simplify this?
        self.p_break = (
            self.metamodel.pressure_from_number_density_nuclear_unit(
                jnp.array(
                    [
                        self.nbreak,
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
                        self.nbreak,
                    ]
                )
            )
            .at[0]
            .get()
        )
        
        # TODO: this has to be checked!
        self.mu_break = (self.p_break + self.e_break) / self.nbreak
        self.cs2_break = (
            jnp.diff(self.metamodel.p).at[-1].get()
            / jnp.diff(self.metamodel.e).at[-1].get()
        )
        # define the speed-of-sound interpolation
        # of the extension portion
        
        self.ngrids = jnp.concatenate((jnp.array([self.nbreak]), ngrids))
        self.cs2grids = jnp.concatenate((jnp.array([self.cs2_break]), cs2grids))
        self.cs2_function = lambda n: jnp.interp(n, self.ngrids, self.cs2grids)
        
        # Compute n, p, e for CSE (number densities in unit of fm^-3)
        ns = jnp.logspace(jnp.log10(self.nbreak), jnp.log10(nmax), num=ndat_CSE)
        mus = self.mu_break * jnp.exp(utils.cumtrapz(self.cs2_function(ns) / ns, ns))
        ps = self.p_break + utils.cumtrapz(self.cs2_function(ns) * mus, ns)
        es = self.e_break + utils.cumtrapz(mus, ns)
        
        # Combine metamodel and CSE data
        # TODO: converting units back and forth might be numerically unstable if conversion factors are large?
        ns = jnp.concatenate((self.metamodel.n / utils.fm_inv3_to_geometric, ns))
        ps = jnp.concatenate((self.metamodel.p / utils.MeV_fm_inv3_to_geometric, ps))
        es = jnp.concatenate((self.metamodel.e / utils.MeV_fm_inv3_to_geometric, es))

        super().__init__(ns, ps, es)
        
    def cs2_from_number_density_nuclear_unit(self, n: Float[Array, "n_points"], cs2_min: float = 1e-3) -> Float[Array, "n_points"]:
        """
        Compute the speed of sound squared from the number density in nuclear units. Uses the metamodel for densities below nbreak and the CSE for densities above nbreak.

        Args:
            n (Float[Array, "n_points"]): Number density in fm^-3.
            cs2_min (float, optional): Minimal value to clip cs2 values computed. Defaults to 1e-3.

        Returns:
            Float[Array, "n_points"]: Speed of sound squared, clipped to be between [cs2_min, 1.0], and with the same size as the input n
        """
        cs2 = jnp.where(n < self.nbreak, self.metamodel.cs2_from_number_density_nuclear_unit(n), self.cs2_function(n))
        cs2 = jnp.clip(cs2, cs2_min, 1.0)
        return cs2


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
