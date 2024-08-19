"""
NOTE: I am starting a new file to really start from scratch here, but perhaps we could easily merge stuff
"""

# TODO: in general: remove all the f1 etc calls...

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


# TODO: perhaps get some dependence on the
class MetaModel_EOS:
    """
    See Margueron:2017eqc, especially Section III for details
    """
    
    def __init__(self,
                 coefficient_sat: Float[Array, "n_sat_coeff"],
                 coefficient_sym: Float[Array, "n_sym_coeff"],
                 kappas: tuple[Float, Float, Float, Float, Float, Float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 v_nq: list[float] = [0.0, 0.0, 0.0, 0.0],
                 b_sat: Float = 0.0, # TODO: change default value
                 b_sym: Float = 0.0, # TODO: change default value
                 N: Int = 4,
                 which_ELF: str = "ELFc",
                 nsat=0.16,
                 nmin=0.1, # in fm^-3
                 nmax=12 * 0.16, # 12 nsat
                 ndat=1000,
                 fix_proton_fraction=False,
                 fix_proton_fraction_val=0.02,
                 crust = "BPS",
                 max_n_crust: Float = 0.08, # in fm^-3
                 use_empty_crust: bool = False,
                 use_spline: bool = False,
                 ndat_spline: int = 50):
        """
        TODO: documentation

        Args:
            coefficient_sat (Float[Array, &quot;n_sat_coeff&quot;]): _description_
            coefficient_sym (Float[Array, &quot;n_sym_coeff&quot;]): _description_
            nsat (float, optional): _description_. Defaults to 0.16.
            nmin (float, optional): _description_. Defaults to 0.1.
            fix_proton_fraction (bool, optional): _description_. Defaults to False.
            fix_proton_fraction_val (float, optional): _description_. Defaults to 0.02.
            crust (str, optional): _description_. Defaults to "BPS".
            max_n_crust (Float, optional): _description_. Defaults to 0.08.
            use_spline (bool, optional): _description_. Defaults to False.
            ndat_spline (int, optional): _description_. Defaults to 50.
        """
        
        assert which_ELF in ["ELFa", "ELFc"], "which_ELF must be either ELFa or ELFc"
        
        if which_ELF == "ELFa":
            self.u = self.u_ELFa
        else:
            self.u = self.u_ELFc
        
        # Save given attributes
        self.nsat = nsat
        self.fix_proton_fraction = fix_proton_fraction
        self.fix_proton_fraction_val = fix_proton_fraction_val
        self.max_n_crust = max_n_crust
        self.v_nq = jnp.array(v_nq)
        self.b_sat = b_sat
        self.b_sym = b_sym
        self.N = N
        
        # Preprocess the coefficients: make sure the length is fixed and if needed pad with zeros
        print(f"coefficient_sat: {coefficient_sat}")
        coefficient_sat = jnp.pad(coefficient_sat, (0, 4 - len(coefficient_sat)), 'constant', constant_values=0)
        print(f"coefficient_sat after pad: {coefficient_sat}")
        
        print(f"coefficient_sym: {coefficient_sym}")
        coefficient_sym = jnp.pad(coefficient_sym, (0, 5 - len(coefficient_sym)), 'constant', constant_values=0)
        print(f"coefficient_sym after pad: {coefficient_sym}")
        
        self.E_sat,             self.K_sat, self.Q_sat, self.Z_sat = coefficient_sat
        self.E_sym, self.L_sym, self.K_sym, self.Q_sym, self.Z_sym = coefficient_sym
        
        # TODO: clean up, not used so much?
        # Add the first derivative coefficient in Esat to make it work with jax.numpy.polyval
        coefficient_sat = jnp.insert(coefficient_sat, 1, 0.0)
        
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
        
        # Crust:
        if use_empty_crust:
            ns_crust, ps_crust, es_crust = jnp.array([]), jnp.array([]), jnp.array([])
        else:
            ns_crust, ps_crust, es_crust = load_crust(crust)
            
            mask = ns_crust <= max_n_crust
            ns_crust, ps_crust, es_crust = ns_crust[mask], ps_crust[mask], es_crust[mask]
        nmin = max(nmin, ns_crust[-1] + 1e-3)
        
        # Kinetic energy: (t_sat is also called TFGsat in the margueron paper)
        self.t_sat = 3*utils.hbar**2/(10*utils.m) * (3*jnp.pi**2 *self.nsat/2)**(2/3)
        
        # Potential energy 
        # v_sat is defined in equations (22) - (26) in the Margueron et al. paper
        # TODO: there are more terms here, perhaps check the other reference that Rahul shared?
        v_sat_0 = self.E_sat -     self.t_sat * ( 1 +   self.kappa_sat +   self.kappa_sat2 +     self.kappa_sat3)
        v_sat_1 =       -     self.t_sat * ( 2 + 5*self.kappa_sat + 8* self.kappa_sat2 + 11* self.kappa_sat3)
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
        
    #################
    ### AUXILIARY ###
    #################
        
    def x(self, density):
        return (density - self.nsat)/(3*self.nsat)
    
    def u_ELFa(self, 
               x: Array[Float], 
               delta: Array[Float],
               b: Float,
               alpha: Int):
        return 1
    
    def u_ELFc(self, 
               x: Array[Float], 
               delta: Array[Float],
               b: Float,
               alpha: Int):
        return 1 - ((-3 * x) ** (self.N + 1 - alpha) * jnp.exp(- b * (1 + 3 * x))) 
        
    @staticmethod
    def f_1(delta: Array[Float]):
        return (1+delta)**(5/3) + (1-delta)**(5/3)
    
    def f_star(self, delta: Array[Float]):
        return (self.kappa_sat + self.kappa_sym*delta ) * (1+delta)**(5/3) + (self.kappa_sat - self.kappa_sym*delta ) * (1-delta)**(5/3)
    
    def f_star2(self, delta: Array[Float]):
        return (self.kappa_sat2 + self.kappa_sym2*delta ) * (1+delta)**(5/3) + (self.kappa_sat2 - self.kappa_sym2*delta ) * (1-delta)**(5/3)
    
    def f_star3(self, delta: Array[Float]):
        return (self.kappa_sat3 + self.kappa_sym3*delta ) * (1+delta)**(5/3) + (self.kappa_sat3 - self.kappa_sym3*delta ) * (1-delta)**(5/3)
    
    def v(self, 
          delta: Array[Float]) -> Array[Float]:
        # TODO: 5 is hardcoded?
        # TODO: Improve for jnp array
        return jnp.array([self.v_sat[alpha] + self.v_sym2[alpha] * delta ** 2 + self.v_nq[alpha] * delta ** 4 for alpha in range(5)])
    
    ################
    ### ENERGIES ###
    ################
    
    def kinetic_energy(self, 
                       density: Array[Float],
                       delta: Array[Float]):
        
        x = self.x(density)
        f_1 = self.f_1(delta)
        f_star = self.f_star(delta)
        f_star2 = self.f_star2(delta)
        f_star3 = self.f_star3(delta)
        
        prefac = self.t_sat / 2 * (1 + 3 * x) ** (2/3)
        linear = (1 + 3 * x) * f_star
        quadratic = (1 + 3 * x) ** 2 * f_star2
        cubic = (1 + 3 * x) ** 3 * f_star3
        
        return prefac * (f_1 + linear + quadratic + cubic)
    
    def potential_energy(self,
                         density: Array[Float],
                         delta: Array[Float]):
        """
        Potential energy. This is using ELFc as defined by Margueron et al
        """
        
        # TODO: 5 is hardcoded, is this what we want?
        x = self.x(density)
        v = self.v(delta)
        b = self.b_sat + self.b_sym * delta**2
        
        answer = 0
        
        # TODO: this won't work with jnp.array, find another way
        for alpha in range(5):
            u = self.u(x, delta, b, alpha)
            answer += v[alpha] / (factorial(alpha)) * x ** alpha * u
        return answer
    
    def energy(self,
               density: Array[Float],
               delta: Array[Float]):
        
        return self.kinetic_energy(density, delta) + self.potential_energy(density, delta)
    
    def esym(self, n: Float[Array, "n_points"]):
        x = self.x(n)
        return jnp.polyval(self.coefficient_sym[::-1], x)
    
    def pressure(self,
                 density: Array[Float],
                 delta: Array[Float]):
        
        # TODO: currently only for ELFc!
        
        x = self.x(density)
        f_1 = self.f_1(delta)
        f_star = self.f_star(delta)
        f_star2 = self.f_star2(delta)
        f_star3 = self.f_star3(delta)
    
        p_kin = 1/3 * self.nsat * self.t_sat * (1 + 3 * x) ** (5/3) *\
            (f_1 + 5/2 * (1 + 3 * x) * f_star + 4 * (1 + 3 * x) ** 2 * f_star2 \
             + 11/2 * (1 + 3 * x) ** 3 * f_star3)
    
        v = self.v(delta)
        b = self.b_sat + self.b_sym * delta ** 2
        
        # TODO: this won't work with jnp.array, find another way
        for alpha in range(1, 5):
            u = self.u(x, delta, b, alpha)
            fac1 = alpha * u
            fac2 = (self.N + 1 - alpha - 3 * b * x) * (u - 1)
            p_pot += v[alpha] / (factorial(alpha)) * x ** (alpha - 1) * (fac1 + fac2)
            
        p_pot = p_pot - v[0] * (-3) ** (self.N + 1) * x ** self.N * (self.N + 1 - 3 * b * x) * jnp.exp(- b * (1 + 3 * x)) 
        p_pot = p_pot * (1/3) * self.nsat * (1 + 3 * x) ** 2
        
        return p_pot + p_kin
    
    def incompressibility(self,
                          density: Array[Float],
                          delta: Array[Float]):

        x = self.x(density)
        f_1 = self.f_1(delta)
        f_star = self.f_star(delta)
        f_star2 = self.f_star2(delta)
        f_star3 = self.f_star3(delta)
    
    
        K_kin = self.t_sat * (1+3*x)**(2/3) *\
            (-f_1 + 5 * (1+3*x) * f_star + 20 * (1+3*x)**2 * f_star2 \
             + 44 * (1+3*x)**3 * f_star3)

        v = self.v(delta)
            
        K_pot = 0
        # TODO: must also become a function/attribute
        b = self.b_sat + self.b_sym * delta ** 2
        
        for alpha in range(2, 5):
            u = 1 -  ( (-3*x)**(self.N+1-alpha) * jnp.exp(-b*(1+3*x)) ) 
            x_up = (self.N+1-alpha-3*b*x)*(u-1)
            x2_upp = (-(self.N+1-alpha)*(self.N-alpha)+6*b*x*(self.N+1-alpha)-9*x**2*b**2) *(1-u)
            
            K_pot = K_pot + v[alpha]/(factorial(alpha)) * x**(alpha-2) \
                * (alpha*(alpha-1)*u+2*alpha*x_up+x2_upp)
                
        K_pot += v[0] * (-(self.N+1)*(self.N)+6*b*x*(self.N+1)-9*x**2*b**2) *((-3)**(self.N+1) * x**(self.N-1) * jnp.exp(-b*(1+3*x)))
        K_pot += 2*v[1]*(self.N-3*b*x)*(-(-3)**(self.N) *x**(self.N-1) * jnp.exp(-b*(1+3*x)) )
        K_pot += v[1] *(-(self.N)*(self.N-1)+6*b*x*(self.N)-9*x**2*b**2) *((-3)**(self.N) * x**(self.N-1) * jnp.exp(-b*(1+3*x)))
        K_pot *= (1+3*x)**2
        
        K = K_kin + K_pot + 18/density * self.pressure(density, delta)
        
        return K
    
    def c2_s(self, 
             density: Array[Float],
             delta: Array[Float]):
        
        K = self.incompressibility(density,delta)
        h = utils.m + self.energy(density, delta) + self.pressure(density, delta)/density
        
        return K/9 /h
    
    def energy_density_electron(self,
                                density: Array[Float],
                                delta: Array[Float]):
        
        K_Fb = (3. * jnp.pi ** 2 /2. * density) ** (1./3.) * utils.hbarc
        K_Fe = K_Fb * (1.- delta)**(1./3.)  
        
        x = K_Fe/utils.m_e
        
        C = utils.m_e**4/(8.*jnp.pi**2) / utils.hbarc**3
        f = x*(1+2*x**2) * jnp.sqrt(1+x**2) - jnp.arcsinh(x)
        
        energy_density = C*f
        
        return energy_density
    
    def electron_pressure(self,
                          density: Array[Float],
                          delta: Array[Float]):
        energy_density = self.energy_density_electron(density,delta)
        
        K_Fb = (3. * jnp.pi**2 /2. * density)**(1./3.) * utils.hbarc
        K_Fe = K_Fb * (1.- delta)**(1./3.)  
        C = utils.m_e**4/(8.* jnp.pi ** 2) / utils.hbarc**3
        x = K_Fe/utils.m_e
        
        pressure = -energy_density + 8./3. * C * x**3 * jnp.sqrt(1+x**2)
        
        return pressure
    
    
    def incompressibility_electron(self,density,delta):
        K_Fb = (3. * jnp.pi**2 /2. * density)**(1./3.) * utils.hbarc
        K_Fe = K_Fb * (1.- delta)**(1./3.)  
        C = utils.m_e**4/(8.*jnp.pi**2) /utils.hbarc**3
        x = K_Fe/utils.m_e
        
        energy_density = self.energy_density_electron(density,delta)
        pressure = self.electron_pressure(density,delta)
        
        K = 8*C/density * x**3*(3+4*x**2)/(jnp.sqrt(1+x**2)) - 9/density * (energy_density+pressure)
        return K
    
    def mu_e(self,
             density,
             delta):
        K_Fb = (3. * jnp.pi**2 /2. * density)**(1./3.) * utils.hbarc
        K_Fe = K_Fb * (1.- delta)**(1./3.)  
    
        ans = jnp.sqrt(utils.m_e**2 + K_Fe**2)
        
        return ans
    
    
    def c2_s_tot(self,
                 density):
        
        # Compute the value of delta for the given density
        proton_fraction = self.proton_fraction(density)
        delta = 1 - 2 * proton_fraction
        
        K_tot = self.incompressibility(density, delta) + self.incompressibility_electron(density, delta)
        
        chi = K_tot/9. 
        
        total_energy_density = (self.energy(density,delta) + utils.m) * density + self.energy_density_electron(density,delta)
        total_pressure = self.pressure(density, delta) + self.electron_pressure(density, delta)
        h_tot =  ( total_energy_density + total_pressure ) / density
        
        c2s = chi/h_tot
        
        return c2s
    
    def proton_fraction(self, 
                        n: Float[Array, "n_points"]) -> Float[Array, "n_points"]:
        """
        Get the proton fraction for a given number density. If proton fraction is fixed, return the fixed value.

        Args:
            n (Float[Array, "n_points"]): Number density in fm^-3.

        Returns:
            Float[Array, "n_points"]: Proton fraction as a function of the number density, either computed or the fixed value.
        """
        return jax.lax.cond(
                self.fix_proton_fraction,
                lambda x: self.fix_proton_fraction_val * jnp.ones(n.shape),
                self.compute_proton_fraction,
                n
            )
        
    def compute_proton_fraction(self, n: Float[Array, "n_points"]) -> Float[Array, "n_points"]:
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