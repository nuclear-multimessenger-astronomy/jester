import os
import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
from jaxtyping import Array, Float, Int

from . import utils, tov, ptov, STtov

##############
### CRUSTS ###
##############

DEFAULT_DIR = os.path.join(os.path.dirname(__file__))
CRUST_DIR = f"{DEFAULT_DIR}/crust"


def load_crust(name: str) -> tuple[Array, Array, Array]:
    r"""
    Load neutron star crust equation of state data from the default directory.

    This function loads pre-computed crust EOS data from tabulated files, supporting
    both built-in crust models (BPS, DH) and custom files.

    Args:
        name (str): Name of the crust model to load (e.g., 'BPS', 'DH') or a filename
                   with .npz extension for custom files.

    Returns:
        tuple[Array, Array, Array]: A tuple containing:
            - Number densities :math:`n` [:math:`\mathrm{fm}^{-3}`]
            - Pressures :math:`p` [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]
            - Energy densities :math:`\varepsilon` [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]

    Raises:
        ValueError: If the specified crust model is not found in the default directory.

    Note:
        Available built-in crust models can be found in the crust/ subdirectory.
    """

    # Get the available crust names
    available_crust_names = [
        f.split(".")[0] for f in os.listdir(CRUST_DIR) if f.endswith(".npz")
    ]

    # If a name is given, but it is not a filename, load the crust from the jose directory
    if not name.endswith(".npz"):
        if name in available_crust_names:
            name = os.path.join(CRUST_DIR, f"{name}.npz")
        else:
            raise ValueError(
                f"Crust {name} not found in {CRUST_DIR}. Available crusts are {available_crust_names}"
            )

    # Once the correct file is identified, load it
    crust = jnp.load(name)
    n, p, e = crust["n"], crust["p"], crust["e"]
    return n, p, e


class Interpolate_EOS_model(object):
    r"""
    Base class for interpolating equation of state (EOS) data.

    This class provides the fundamental interpolation framework for converting
    tabulated EOS data (density, pressure, energy) into the auxiliary quantities
    needed for neutron star structure calculations using the TOV equations.
    """

    def __init__(self):
        pass

    def interpolate_eos(
        self,
        n: Float[Array, "n_points"],
        p: Float[Array, "n_points"],
        e: Float[Array, "n_points"],
    ):
        r"""
        Convert physical EOS quantities to geometric units and compute auxiliary quantities.

        This method transforms the input EOS data from nuclear physics units to geometric
        units used in general relativity calculations, and computes derived quantities
        needed for the TOV equations.

        Args:
            n (Float[Array, n_points]): Number densities [:math:`\mathrm{fm}^{-3}`]
            p (Float[Array, n_points]): Pressure values [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]
            e (Float[Array, n_points]): Energy densities [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]

        Returns:
            tuple: A tuple containing (all in geometric units):

                - ns: Number densities
                - ps: Pressures
                - hs: Specific enthalpy :math:`h = \int \frac{dp}{\varepsilon + p}`
                - es: Energy densities
                - dloge_dlogps: Logarithmic derivative :math:`\frac{d\ln\varepsilon}{d\ln p}`
        """

        # Save the provided data as attributes, make conversions
        ns = jnp.array(n * utils.fm_inv3_to_geometric)
        ps = jnp.array(p * utils.MeV_fm_inv3_to_geometric)
        es = jnp.array(e * utils.MeV_fm_inv3_to_geometric)

        # rhos = utils.calculate_rest_mass_density(es, ps)

        hs = utils.cumtrapz(ps / (es + ps), jnp.log(ps))  # enthalpy
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


class MetaModel_EOS_model(Interpolate_EOS_model):
    r"""
    Meta-model equation of state for nuclear matter.

    This class implements the meta-modeling approach for nuclear equation of state
    as described in Margueron et al. (Phys. Rev. C 103, 045803, 2021). The EOS
    is constructed by combining a realistic crust model with a meta-model for
    core densities based on nuclear empirical parameters (NEPs).

    The meta-model uses a kinetic + potential energy decomposition:

    .. math::
        \varepsilon(n, \delta) = \varepsilon_{\mathrm{kin}}(n, \delta) + \varepsilon_{\mathrm{pot}}(n, \delta)

    where :math:`\delta = (n_n - n_p)/n` is the isospin asymmetry parameter.

    The kinetic part is based on a Thomas-Fermi gas with relativistic corrections,
    while the potential part uses a polynomial expansion around saturation density :math:`n_0`.
    """

    def __init__(
        self,
        kappas: tuple[Float, Float, Float, Float, Float, Float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        v_nq: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0],
        b_sat: Float = 17.0,
        b_sym: Float = 25.0,
        # density parameters
        nsat: Float = 0.16,
        nmin_MM_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        ndat: Int = 200,
        # crust parameters
        crust_name: str = "DH",
        max_n_crust_nsat: Float = 0.5,
        ndat_spline: Int = 10,
        # proton fraction
        proton_fraction: bool | float | None = None,
    ):
        r"""
        Initialize the meta-model EOS with nuclear empirical parameters.

        The meta-model approach parameterizes nuclear matter using empirical parameters
        measured from finite nuclei and infinite nuclear matter calculations. This
        implementation combines a realistic crust model with a meta-model description
        of the core using nuclear empirical parameters (NEPs).

        **Reference:** Margueron et al., Phys. Rev. C 103, 045803 (2021)

        **Physical Framework:**
        The meta-model decomposes the energy density as:

        .. math::
            \varepsilon(n, \delta) = \varepsilon_{\mathrm{kin}}(n, \delta) + \varepsilon_{\mathrm{pot}}(n, \delta)

        where :math:`\delta = (n_n - n_p)/n` is the isospin asymmetry parameter.

        Args:
            kappas (tuple[Float, Float, Float, Float, Float, Float], optional):
                Meta-model expansion coefficients :math:`(\kappa_{\mathrm{sat}}, \kappa_{\mathrm{sat2}}, \kappa_{\mathrm{sat3}}, \kappa_{\mathrm{NM}}, \kappa_{\mathrm{NM2}}, \kappa_{\mathrm{NM3}})`.
                Controls the density dependence of kinetic energy corrections in the Thomas-Fermi gas approximation.
                These parameters modify the kinetic energy beyond the non-relativistic limit.
                Defaults to (0.0, 0.0, 0.0, 0.0, 0.0, 0.0).
            v_nq (list[float], optional):
                Quartic isospin coefficients :math:`v_{\mathrm{nq},\alpha}` for symmetry energy expansion up to :math:`\delta^4` terms.
                These control the high-order isospin dependence of the potential energy contribution.
                Defaults to [0.0, 0.0, 0.0, 0.0, 0.0].
            b_sat (Float, optional):
                Saturation parameter :math:`b_{\mathrm{sat}}` controlling potential energy cutoff function.
                Higher values lead to sharper exponential cutoffs at high density. Defaults to 17.0.
            b_sym (Float, optional):
                Symmetry parameter :math:`b_{\mathrm{sym}}` for isospin-dependent cutoff corrections.
                Modifies the cutoff parameter as :math:`b = b_{\mathrm{sat}} + b_{\mathrm{sym}} \delta^2`. Defaults to 25.0.
            nsat (Float, optional):
                Nuclear saturation density :math:`n_0` [:math:`\mathrm{fm}^{-3}`]. The density at which
                symmetric nuclear matter reaches its minimum energy per nucleon. Defaults to 0.16.
            nmin_MM_nsat (Float, optional):
                Starting density for meta-model region as fraction of :math:`n_0`.
                Must be above the crust-core transition density. Defaults to 0.75 (= 0.12/0.16).
            nmax_nsat (Float, optional):
                Maximum density for EOS construction in units of :math:`n_0`.
                Determines the high-density reach of the neutron star model. Defaults to 12.
            ndat (Int, optional):
                Number of density points for meta-model region discretization.
                Higher values provide smoother interpolation at computational cost. Defaults to 200.
            crust_name (str, optional):
                Crust model name (e.g., 'DH', 'BPS') or path to custom .npz file containing crust EOS data.
                The crust provides low-density EOS data below nuclear saturation. Defaults to 'DH'.
            max_n_crust_nsat (Float, optional):
                Maximum crust density as fraction of :math:`n_0`. Defines the crust-core
                transition region where spline matching occurs. Defaults to 0.5.
            ndat_spline (Int, optional):
                Number of points for smooth spline interpolation across crust-core transition.
                Ensures thermodynamic consistency and causality preservation. Defaults to 10.
            proton_fraction (bool | float | None, optional):
                Proton fraction treatment strategy:

                - None: Calculate :math:`\beta`-equilibrium (charge neutrality + weak equilibrium)
                - float: Use fixed proton fraction value throughout the star
                - bool: Use simplified uniform composition model

                :math:`\beta`-equilibrium is the physical condition for neutron star matter. Defaults to None.

        Note:
            The meta-model uses a Thomas-Fermi kinetic energy approximation with relativistic
            corrections controlled by the :math:`\kappa` parameters, combined with a potential
            energy expansion around saturation density with an exponential cutoff at high densities.
            This approach provides a flexible framework for exploring nuclear physics uncertainties
            in neutron star structure calculations.
        """

        # Save as attributes
        self.nsat = nsat
        self.v_nq = jnp.array(v_nq)
        self.b_sat = b_sat
        self.b_sym = b_sym
        self.N = 4  # TODO: this is fixed in the metamodeling papers, but we might want to extend this in the future

        self.nmin_MM_nsat = nmin_MM_nsat
        self.nmax_nsat = nmax_nsat
        self.ndat = ndat
        self.max_n_crust_nsat = max_n_crust_nsat
        self.ndat_spline = ndat_spline

        if isinstance(proton_fraction, float):
            self.proton_fraction_val = proton_fraction
            self.proton_fraction = lambda x, y: self.proton_fraction_val
            print(f"Proton fraction fixed to {self.proton_fraction_val}")
        else:
            self.proton_fraction = lambda x, y: self.compute_proton_fraction(x, y)

        # Constructions
        assert (
            len(kappas) == 6
        ), "kappas must be a tuple of 6 values: kappa_sat, kappa_sat2, kappa_sat3, kappa_NM, kappa_NM2, kappa_NM3"
        (
            self.kappa_sat,
            self.kappa_sat2,
            self.kappa_sat3,
            self.kappa_NM,
            self.kappa_NM2,
            self.kappa_NM3,
        ) = kappas
        self.kappa_sym = self.kappa_NM - self.kappa_sat
        self.kappa_sym2 = self.kappa_NM2 - self.kappa_sat2
        self.kappa_sym3 = self.kappa_NM3 - self.kappa_sat3

        # t_sat or TFGsat is the kinetic energy per nucleons in SM and at saturation, see just after eq (13) in the Margueron paper
        self.t_sat = (
            3
            * utils.hbar**2
            / (10 * utils.m)
            * (3 * jnp.pi**2 * self.nsat / 2) ** (2 / 3)
        )

        # v_sat is defined in equations (22) - (26) in the Margueron et al. paper
        self.v_sat_0_no_NEP = -self.t_sat * (
            1 + self.kappa_sat + self.kappa_sat2 + self.kappa_sat3
        )
        self.v_sat_1_no_NEP = -self.t_sat * (
            2 + 5 * self.kappa_sat + 8 * self.kappa_sat2 + 11 * self.kappa_sat3
        )
        self.v_sat_2_no_NEP = (
            -2
            * self.t_sat
            * (-1 + 5 * self.kappa_sat + 20 * self.kappa_sat2 + 44 * self.kappa_sat3)
        )
        self.v_sat_3_no_NEP = (
            -2
            * self.t_sat
            * (4 - 5 * self.kappa_sat + 40 * self.kappa_sat2 + 220 * self.kappa_sat3)
        )
        self.v_sat_4_no_NEP = (
            -8
            * self.t_sat
            * (-7 + 5 * self.kappa_sat - 10 * self.kappa_sat2 + 110 * self.kappa_sat3)
        )

        self.v_sym2_0_no_NEP = (
            -self.t_sat
            * (
                2 ** (2 / 3) * (1 + self.kappa_NM + self.kappa_NM2 + self.kappa_NM3)
                - (1 + self.kappa_sat + self.kappa_sat2 + self.kappa_sat3)
            )
            - self.v_nq[0]
        )
        self.v_sym2_1_no_NEP = (
            -self.t_sat
            * (
                2 ** (2 / 3)
                * (2 + 5 * self.kappa_NM + 8 * self.kappa_NM2 + 11 * self.kappa_NM3)
                - (2 + 5 * self.kappa_sat + 8 * self.kappa_sat2 + 11 * self.kappa_sat3)
            )
            - self.v_nq[1]
        )
        self.v_sym2_2_no_NEP = (
            -2
            * self.t_sat
            * (
                2 ** (2 / 3)
                * (-1 + 5 * self.kappa_NM + 20 * self.kappa_NM2 + 44 * self.kappa_NM3)
                - (
                    -1
                    + 5 * self.kappa_sat
                    + 20 * self.kappa_sat2
                    + 44 * self.kappa_sat3
                )
            )
            - self.v_nq[2]
        )
        self.v_sym2_3_no_NEP = (
            -2
            * self.t_sat
            * (
                2 ** (2 / 3)
                * (4 - 5 * self.kappa_NM + 40 * self.kappa_NM2 + 220 * self.kappa_NM3)
                - (
                    4
                    - 5 * self.kappa_sat
                    + 40 * self.kappa_sat2
                    + 220 * self.kappa_sat3
                )
            )
            - self.v_nq[3]
        )
        self.v_sym2_4_no_NEP = (
            -8
            * self.t_sat
            * (
                2 ** (2 / 3)
                * (-7 + 5 * self.kappa_NM - 10 * self.kappa_NM2 + 110 * self.kappa_NM3)
                - (
                    -7
                    + 5 * self.kappa_sat
                    - 10 * self.kappa_sat2
                    + 110 * self.kappa_sat3
                )
            )
            - self.v_nq[4]
        )

        # Load and preprocess the crust
        ns_crust, ps_crust, es_crust = load_crust(crust_name)
        max_n_crust = max_n_crust_nsat * nsat
        mask = ns_crust <= max_n_crust
        self.ns_crust, self.ps_crust, self.es_crust = (
            ns_crust[mask],
            ps_crust[mask],
            es_crust[mask],
        )

        self.mu_lowest = (es_crust[0] + ps_crust[0]) / ns_crust[0]
        self.cs2_crust = jnp.gradient(self.ps_crust, self.es_crust)

        # Make sure the metamodel starts above the crust
        self.max_n_crust = self.ns_crust[-1]

        # Create density arrays
        self.nmax = nmax_nsat * self.nsat
        self.ndat = ndat
        self.nmin_MM = self.nmin_MM_nsat * self.nsat
        self.n_metamodel = jnp.linspace(
            self.nmin_MM, self.nmax, self.ndat, endpoint=False
        )
        self.ns_spline = jnp.append(self.ns_crust, self.n_metamodel)
        self.n_connection = jnp.linspace(
            self.max_n_crust + 1e-5, self.nmin_MM, self.ndat_spline, endpoint=False
        )

    def construct_eos(self, NEP_dict: dict) -> tuple:
        r"""
        Construct the complete equation of state from nuclear empirical parameters.

        This method builds the full EOS by combining the crust model with the
        meta-model core, ensuring thermodynamic consistency and causality.

        Args:
            NEP_dict (dict): Nuclear empirical parameters including:

                - **E_sat**: Saturation energy per nucleon [:math:`\mathrm{MeV}`] (default: -16.0)
                - **K_sat**: Incompressibility at saturation [:math:`\mathrm{MeV}`]
                - **Q_sat, Z_sat**: Higher-order saturation parameters [:math:`\mathrm{MeV}`]
                - **E_sym**: Symmetry energy [:math:`\mathrm{MeV}`]
                - **L_sym**: Symmetry energy slope [:math:`\mathrm{MeV}`]
                - **K_sym, Q_sym, Z_sym**: Higher-order symmetry parameters [:math:`\mathrm{MeV}`]

        Returns:
            tuple: Complete EOS data containing:

                - **ns**: Number densities [geometric units]
                - **ps**: Pressures [geometric units]
                - **hs**: Specific enthalpies [geometric units]
                - **es**: Energy densities [geometric units]
                - **dloge_dlogps**: Logarithmic derivative :math:`\frac{d\ln\varepsilon}{d\ln p}`
                - **mu**: Chemical potential [geometric units]
                - **cs2**: Speed of sound squared :math:`c_s^2 = \frac{dp}{d\varepsilon}`
        """

        E_sat = NEP_dict.get(
            "E_sat", -16.0
        )  # NOTE: this is a commong default value, therefore not zero!
        K_sat = NEP_dict.get("K_sat", 0.0)
        Q_sat = NEP_dict.get("Q_sat", 0.0)
        Z_sat = NEP_dict.get("Z_sat", 0.0)

        E_sym = NEP_dict.get("E_sym", 0.0)
        L_sym = NEP_dict.get("L_sym", 0.0)
        K_sym = NEP_dict.get("K_sym", 0.0)
        Q_sym = NEP_dict.get("Q_sym", 0.0)
        Z_sym = NEP_dict.get("Z_sym", 0.0)

        # Add the first derivative coefficient in Esat to make it work with jax.numpy.polyval
        coefficient_sat = jnp.array([E_sat, 0.0, K_sat, Q_sat, Z_sat])
        coefficient_sym = jnp.array([E_sym, L_sym, K_sym, Q_sym, Z_sym])

        # Get the coefficents index array and get coefficients
        index_sat = jnp.arange(len(coefficient_sat))
        index_sym = jnp.arange(len(coefficient_sym))

        coefficient_sat = coefficient_sat / factorial(index_sat)
        coefficient_sym = coefficient_sym / factorial(index_sym)

        # Potential energy (v_sat is defined in equations (22) - (26) in the Margueron et al. paper)
        v_sat = jnp.array(
            [
                E_sat + self.v_sat_0_no_NEP,
                0.0 + self.v_sat_1_no_NEP,
                K_sat + self.v_sat_2_no_NEP,
                Q_sat + self.v_sat_3_no_NEP,
                Z_sat + self.v_sat_4_no_NEP,
            ]
        )

        # v_sym2 is defined in equations (27) to (31) in the Margueron et al. paper
        v_sym2 = jnp.array(
            [
                E_sym + self.v_sym2_0_no_NEP,
                L_sym + self.v_sym2_1_no_NEP,
                K_sym + self.v_sym2_2_no_NEP,
                Q_sym + self.v_sym2_3_no_NEP,
                Z_sym + self.v_sym2_4_no_NEP,
            ]
        )

        # Auxiliaries first
        x = self.compute_x(self.n_metamodel)
        proton_fraction = self.proton_fraction(coefficient_sym, self.n_metamodel)
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
        cs2_metamodel = self.compute_cs2(
            self.n_metamodel,
            p_metamodel,
            e_metamodel,
            x,
            delta,
            f_1,
            f_star,
            f_star2,
            f_star3,
            b,
            v,
        )

        # Spline for speed of sound for the connection region
        cs2_spline = jnp.append(jnp.array(self.cs2_crust), cs2_metamodel)

        cs2_connection = utils.cubic_spline(
            self.n_connection, self.ns_spline, cs2_spline
        )
        cs2_connection = jnp.clip(cs2_connection, 1e-5, 1.0)

        # Concatenate the arrays
        n = jnp.concatenate([self.ns_crust, self.n_connection, self.n_metamodel])
        cs2 = jnp.concatenate(
            [jnp.array(self.cs2_crust), cs2_connection, cs2_metamodel]
        )

        # Make sure the cs2 stays within the physical limits
        cs2 = jnp.clip(cs2, 1e-5, 1.0)

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

    def u(self, x: Array, b: Array, alpha: Int):
        return 1 - ((-3 * x) ** (self.N + 1 - alpha) * jnp.exp(-b * (1 + 3 * x)))

    def compute_x(self, n: Array):
        return (n - self.nsat) / (3 * self.nsat)

    def compute_b(self, delta: Array | float):
        return self.b_sat + self.b_sym * delta**2

    def compute_f_1(self, delta: Array | float):
        return (1 + delta) ** (5 / 3) + (1 - delta) ** (5 / 3)

    def compute_f_star(self, delta: Array | float):
        return (self.kappa_sat + self.kappa_sym * delta) * (1 + delta) ** (5 / 3) + (
            self.kappa_sat - self.kappa_sym * delta
        ) * (1 - delta) ** (5 / 3)

    def compute_f_star2(self, delta: Array | float):
        return (self.kappa_sat2 + self.kappa_sym2 * delta) * (1 + delta) ** (5 / 3) + (
            self.kappa_sat2 - self.kappa_sym2 * delta
        ) * (1 - delta) ** (5 / 3)

    def compute_f_star3(self, delta: Array | float):
        return (self.kappa_sat3 + self.kappa_sym3 * delta) * (1 + delta) ** (5 / 3) + (
            self.kappa_sat3 - self.kappa_sym3 * delta
        ) * (1 - delta) ** (5 / 3)

    def compute_v(self, v_sat: Array, v_sym2: Array, delta: Array | float) -> Array:
        return jnp.array(
            [
                v_sat[alpha] + v_sym2[alpha] * delta**2 + self.v_nq[alpha] * delta**4
                for alpha in range(self.N + 1)
            ]
        )

    def compute_energy(
        self,
        x: Array,
        f_1: Array,
        f_star: Array,
        f_star2: Array,
        f_star3: Array,
        b: Array,
        v: Array,
    ) -> Array:

        prefac = self.t_sat / 2 * (1 + 3 * x) ** (2 / 3)
        linear = (1 + 3 * x) * f_star
        quadratic = (1 + 3 * x) ** 2 * f_star2
        cubic = (1 + 3 * x) ** 3 * f_star3

        kinetic_energy = prefac * (f_1 + linear + quadratic + cubic)

        # Potential energy # TODO: a bit cumbersome, find another way, like jax tree map?
        potential_energy = 0
        for alpha in range(5):
            u = self.u(x, b, alpha)
            potential_energy += v.at[alpha].get() / (factorial(alpha)) * x**alpha * u

        return kinetic_energy + potential_energy

    def esym(self, coefficient_sym: list, x: Array):
        # TODO: change this to be self-consistent: see Rahul's approach for that.
        return jnp.polyval(jnp.array(coefficient_sym[::-1]), x)

    def compute_pressure(
        self,
        x: Array,
        f_1: Array,
        f_star: Array,
        f_star2: Array,
        f_star3: Array,
        b: Array,
        v: Array,
    ) -> Array:

        # TODO: currently only for ELFc!
        p_kin = (
            1
            / 3
            * self.nsat
            * self.t_sat
            * (1 + 3 * x) ** (5 / 3)
            * (
                f_1
                + 5 / 2 * (1 + 3 * x) * f_star
                + 4 * (1 + 3 * x) ** 2 * f_star2
                + 11 / 2 * (1 + 3 * x) ** 3 * f_star3
            )
        )

        # TODO: cumbersome with jnp.array, find another way
        p_pot = 0
        for alpha in range(1, 5):
            u = self.u(x, b, alpha)
            fac1 = alpha * u
            fac2 = (self.N + 1 - alpha - 3 * b * x) * (u - 1)
            p_pot += (
                v.at[alpha].get()
                / (factorial(alpha))
                * x ** (alpha - 1)
                * (fac1 + fac2)
            )

        p_pot = p_pot - v.at[0].get() * (-3) ** (self.N + 1) * x**self.N * (
            self.N + 1 - 3 * b * x
        ) * jnp.exp(-b * (1 + 3 * x))
        p_pot = p_pot * (1 / 3) * self.nsat * (1 + 3 * x) ** 2

        return p_pot + p_kin

    def compute_cs2(
        self,
        n: Array,
        p: Array,
        e: Array,
        x: Array,
        delta: Array | float,
        f_1: Array,
        f_star: Array,
        f_star2: Array,
        f_star3: Array,
        b: Array,
        v: Array,
    ):

        ### Compute incompressibility

        # Kinetic part
        K_kin = (
            self.t_sat
            * (1 + 3 * x) ** (2 / 3)
            * (
                -f_1
                + 5 * (1 + 3 * x) * f_star
                + 20 * (1 + 3 * x) ** 2 * f_star2
                + 44 * (1 + 3 * x) ** 3 * f_star3
            )
        )

        # Potential part
        K_pot = 0
        for alpha in range(2, self.N + 1):
            u = 1 - ((-3 * x) ** (self.N + 1 - alpha) * jnp.exp(-b * (1 + 3 * x)))
            x_up = (self.N + 1 - alpha - 3 * b * x) * (u - 1)
            x2_upp = (
                -(self.N + 1 - alpha) * (self.N - alpha)
                + 6 * b * x * (self.N + 1 - alpha)
                - 9 * x**2 * b**2
            ) * (1 - u)

            K_pot = K_pot + v.at[alpha].get() / (factorial(alpha)) * x ** (
                alpha - 2
            ) * (alpha * (alpha - 1) * u + 2 * alpha * x_up + x2_upp)

        K_pot += (
            v.at[0].get()
            * (-(self.N + 1) * (self.N) + 6 * b * x * (self.N + 1) - 9 * x**2 * b**2)
            * ((-3) ** (self.N + 1) * x ** (self.N - 1) * jnp.exp(-b * (1 + 3 * x)))
        )
        K_pot += (
            2
            * v.at[1].get()
            * (self.N - 3 * b * x)
            * (-((-3) ** (self.N)) * x ** (self.N - 1) * jnp.exp(-b * (1 + 3 * x)))
        )
        K_pot += (
            v.at[1].get()
            * (-(self.N) * (self.N - 1) + 6 * b * x * (self.N) - 9 * x**2 * b**2)
            * ((-3) ** (self.N) * x ** (self.N - 1) * jnp.exp(-b * (1 + 3 * x)))
        )
        K_pot *= (1 + 3 * x) ** 2

        K = K_kin + K_pot + 18 / n * p

        # For electron

        K_Fb = (3.0 * jnp.pi**2 / 2.0 * n) ** (1.0 / 3.0) * utils.hbarc
        K_Fe = K_Fb * (1.0 - delta) ** (1.0 / 3.0)
        C = utils.m_e**4 / (8.0 * jnp.pi**2) / utils.hbarc**3
        x = K_Fe / utils.m_e
        f = x * (1 + 2 * x**2) * jnp.sqrt(1 + x**2) - jnp.arcsinh(x)

        e_electron = C * f
        p_electron = -e_electron + 8.0 / 3.0 * C * x**3 * jnp.sqrt(1 + x**2)
        K_electron = 8 * C / n * x**3 * (3 + 4 * x**2) / (
            jnp.sqrt(1 + x**2)
        ) - 9 / n * (e_electron + p_electron)

        # Sum together:
        K_tot = K + K_electron

        # Finally, get cs2:
        chi = K_tot / 9.0

        total_energy_density = (e + utils.m) * n + e_electron
        total_pressure = p + p_electron
        h_tot = (total_energy_density + total_pressure) / n

        cs2 = chi / h_tot

        return cs2

    def compute_proton_fraction(
        self, coefficient_sym: list, n: Array
    ) -> Float[Array, "n_points"]:
        r"""
        Compute proton fraction from beta-equilibrium condition.

        This method solves the beta-equilibrium condition:

        .. math::
            \mu_e + \mu_p - \mu_n = 0

        where the chemical potentials are related to the EOS through:

        .. math::
            \mu_p - \mu_n = \frac{\partial \varepsilon}{\partial x_p} = -4 E_{\mathrm{sym}} (1 - 2x_p)

        and the electron chemical potential is :math:`\mu_e = \hbar c (3\pi^2 x_p n)^{1/3}`.

        Args:
            coefficient_sym (list): Symmetry energy expansion coefficients.
            n (Float[Array, "n_points"]): Number density [:math:`\mathrm{fm}^{-3}`].

        Returns:
            Float[Array, "n_points"]: Proton fraction :math:`x_p = n_p/n` as a function of density.
        """
        # TODO: the following comments should be in the doc string
        # # chemical potential of electron -- derivation
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
    r"""
    Meta-model EOS combined with piecewise speed-of-sound extensions (CSE).

    This class extends the meta-model approach by allowing for piecewise-constant
    speed-of-sound extensions at high densities. This is useful for modeling
    phase transitions or exotic matter components in neutron star cores that
    may not be captured by the meta-model polynomial expansions.

    The EOS is constructed in two regions:

    1. **Low-to-intermediate density**: Meta-model approach (crust + core)
    2. **High density**: Speed-of-sound extension scheme
    """

    def __init__(
        self,
        nsat: Float = 0.16,
        nmin_MM_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        ndat_metamodel: Int = 100,
        ndat_CSE: Int = 100,
        **metamodel_kwargs,
    ):
        r"""
        Initialize the MetaModel with CSE EOS combining meta-model and constant speed-of-sound extensions.

        This constructor sets up a hybrid EOS that uses the meta-model approach for
        low-to-intermediate densities and allows for user-defined constant speed-of-sound
        extensions at high densities. The transition occurs at a break density specified
        in the NEP dictionary during EOS construction.

        Args:
            nsat (Float, optional):
                Nuclear saturation density :math:`n_0` [:math:`\mathrm{fm}^{-3}`].
                Reference density for the meta-model construction. Defaults to 0.16.
            nmin_MM_nsat (Float, optional):
                Starting density for meta-model region as fraction of :math:`n_0`.
                Must be above crust-core transition. Defaults to 0.75 (= 0.12/0.16).
            nmax_nsat (Float, optional):
                Maximum density for EOS construction in units of :math:`n_0`.
                Defines the high-density reach including CSE region. Defaults to 12.
            ndat_metamodel (Int, optional):
                Number of density points for meta-model region discretization.
                Higher values give smoother meta-model interpolation. Defaults to 100.
            ndat_CSE (Int, optional):
                Number of density points for constant speed-of-sound extension region.
                Controls resolution of high-density exotic matter modeling. Defaults to 100.
            **metamodel_kwargs:
                Additional keyword arguments passed to the underlying MetaModel_EOS_model.
                Includes parameters like kappas, v_nq, b_sat, b_sym, crust_name, etc.
                See MetaModel_EOS_model.__init__ for complete parameter descriptions.

        See Also:
            MetaModel_EOS_model.__init__ : Base meta-model parameters
            construct_eos : Method that defines CSE parameters and break density
        """

        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_MM_nsat = nmin_MM_nsat
        self.ndat_metamodel = ndat_metamodel
        self.metamodel_kwargs = metamodel_kwargs

    def construct_eos(
        self,
        NEP_dict: dict,
        ngrids: Float[Array, "n_grid_point"],
        cs2grids: Float[Array, "n_grid_point"],
    ) -> tuple:
        r"""
        Construct the EOS

        Args:
            NEP_dict (dict): Dictionary with the NEP keys to be passed to the metamodel EOS class.
            ngrids (Float[Array, `n_grid_point`]): Density grid points of densities for the CSE part of the EOS.
            cs2grids (Float[Array, `n_grid_point`]): Speed-of-sound squared grid points of densities for the CSE part of the EOS.

        Returns:
            tuple: EOS quantities (see Interpolate_EOS_model), as well as the chemical potential and speed of sound.
        """

        # Initializate the MetaModel part up to n_break
        metamodel = MetaModel_EOS_model(
            nsat=self.nsat,
            nmin_MM_nsat=self.nmin_MM_nsat,
            nmax_nsat=NEP_dict["nbreak"] / self.nsat,
            ndat=self.ndat_metamodel,
            **self.metamodel_kwargs,
        )

        # Construct the metamodel part:
        mm_output = metamodel.construct_eos(NEP_dict)
        n_metamodel, p_metamodel, _, e_metamodel, _, mu_metamodel, cs2_metamodel = (
            mm_output
        )

        # Convert units back for CSE initialization
        n_metamodel = n_metamodel / utils.fm_inv3_to_geometric
        p_metamodel = p_metamodel / utils.MeV_fm_inv3_to_geometric
        e_metamodel = e_metamodel / utils.MeV_fm_inv3_to_geometric

        # Get values at break density
        p_break = jnp.interp(NEP_dict["nbreak"], n_metamodel, p_metamodel)
        e_break = jnp.interp(NEP_dict["nbreak"], n_metamodel, e_metamodel)
        mu_break = jnp.interp(NEP_dict["nbreak"], n_metamodel, mu_metamodel)
        cs2_break = jnp.interp(NEP_dict["nbreak"], n_metamodel, cs2_metamodel)

        # Define the speed-of-sound interpolation of the extension portion
        ngrids = jnp.concatenate((jnp.array([NEP_dict["nbreak"]]), ngrids))
        cs2grids = jnp.concatenate((jnp.array([cs2_break]), cs2grids))
        cs2_extension_function = lambda n: jnp.interp(n, ngrids, cs2grids)

        # Compute n, p, e for CSE (number densities in unit of fm^-3)
        n_CSE = jnp.logspace(
            jnp.log10(NEP_dict["nbreak"]), jnp.log10(self.nmax), num=self.ndat_CSE
        )
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

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        return ns, ps, hs, es, dloge_dlogps, mu, cs2


class MetaModel_with_peakCSE_EOS_model(Interpolate_EOS_model):
    r"""
    Meta-model EOS with Gaussian peak Constant Speed-of-sound Extensions (peakCSE).

    This class implements a sophisticated CSE parametrization based on the peakCSE model,
    which combines a Gaussian peak structure with logistic growth to model phase transitions
    while ensuring asymptotic consistency with perturbative QCD (pQCD) at the highest densities.

    **Mathematical Framework:**
    The speed of sound squared is parametrized as:

    .. math::
        c^2_s &= c^2_{s,{\rm break}} + \frac{\frac{1}{3} - c^2_{s,{\rm break}}}{1 + e^{-l_{\rm sig}(n - n_{\rm sig})}} + c^2_{s,{\rm peak}}e^{-\frac{1}{2}\left(\frac{n - n_{\rm peak}}{\sigma_{\rm peak}}\right)^2}

    **Reference:** Greif:2018njt, arXiv:1812.08188

    Note:
        The peakCSE model provides greater physical realism than simple piecewise-constant
        CSE by incorporating smooth transitions and theoretically motivated high-density behavior.
    """

    def __init__(
        self,
        nsat: Float = 0.16,
        nmin_MM_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        ndat_metamodel: Int = 100,
        ndat_CSE: Int = 100,
        **metamodel_kwargs,
    ):
        r"""
        Initialize the MetaModel with peakCSE extensions for realistic phase transition modeling.

        This constructor sets up the peakCSE model that combines meta-model physics at
        low-to-intermediate densities with sophisticated Gaussian peak + logistic growth
        extensions at high densities, designed to model phase transitions while maintaining
        consistency with perturbative QCD predictions.

        Args:
            nsat (Float, optional):
                Nuclear saturation density :math:`n_0` [:math:`\mathrm{fm}^{-3}`].
                Reference density for the meta-model construction. Defaults to 0.16.
            nmin_MM_nsat (Float, optional):
                Starting density for meta-model region as fraction of :math:`n_0`.
                Must be above crust-core transition. Defaults to 0.75 (= 0.12/0.16).
            nmax_nsat (Float, optional):
                Maximum density for EOS construction in units of :math:`n_0`.
                Should extend to densities where pQCD limit is approached. Defaults to 12.
            ndat_metamodel (Int, optional):
                Number of density points for meta-model region discretization.
                Higher values give smoother meta-model interpolation. Defaults to 100.
            ndat_CSE (Int, optional):
                Number of density points for peakCSE region discretization.
                Controls resolution of phase transition and pQCD approach modeling. Defaults to 100.
            **metamodel_kwargs:
                Additional keyword arguments passed to the underlying MetaModel_EOS_model.
                Includes parameters like kappas, v_nq, b_sat, b_sym, crust_name, etc.
                See MetaModel_EOS_model.__init__ for complete parameter descriptions.

        See Also:
            MetaModel_EOS_model.__init__ : Base meta-model parameters
            construct_eos : Method that defines peakCSE parameters and break density
        """

        # TODO: align with new metamodel code
        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_MM_nsat = nmin_MM_nsat
        self.ndat_metamodel = ndat_metamodel
        self.metamodel_kwargs = metamodel_kwargs

    def construct_eos(self, NEP_dict: dict, peakCSE_dict: dict):
        r"""
        Construct the complete EOS using meta-model + peakCSE extensions.

        This method builds the full EOS by combining the meta-model approach with
        peakCSE extensions that model phase transitions through Gaussian peaks
        and approach the pQCD conformal limit at high densities.

        Args:
            NEP_dict (dict): Nuclear empirical parameters for meta-model construction.
                Must include 'nbreak' key specifying the transition density between
                meta-model and peakCSE regions. See MetaModel_EOS_model.construct_eos
                for complete NEP parameter descriptions.
            peakCSE_dict (dict): peakCSE model parameters defining the high-density behavior:

                - **gaussian_peak** (float): Amplitude :math:`A` of the Gaussian peak
                - **gaussian_mu** (float): Peak location :math:`\mu` [:math:`\mathrm{fm}^{-3}`]
                - **gaussian_sigma** (float): Peak width :math:`\sigma` [:math:`\mathrm{fm}^{-3}`]
                - **logit_growth_rate** (float): Growth rate :math:`k` for pQCD approach
                - **logit_midpoint** (float): Midpoint density :math:`n_{\mathrm{mid}}` for logistic transition

        Returns:
            tuple: Complete EOS data containing:

                - **ns**: Number densities [geometric units]
                - **ps**: Pressures [geometric units]
                - **hs**: Specific enthalpies [geometric units]
                - **es**: Energy densities [geometric units]
                - **dloge_dlogps**: Logarithmic derivative :math:`\frac{d\ln\varepsilon}{d\ln p}`
                - **mu**: Chemical potential [geometric units]
                - **cs2**: Speed of sound squared including peakCSE structure

        Note:
            The peakCSE speed of sound follows:
            :math:`c^2_s &= c^2_{s,{\rm break}} + \frac{\frac{1}{3} - c^2_{s,{\rm break}}}{1 + e^{-l_{\rm sig}(n - n_{\rm sig})}} + c^2_{s,{\rm peak}}e^{-\frac{1}{2}\left(\frac{n - n_{\rm peak}}{\sigma_{\rm peak}}\right)^2}`

            This ensures smooth transitions, realistic phase transition modeling,
            and asymptotic consistency with the pQCD conformal limit :math:`c_s^2 = 1/3`.
        """

        # Initializate the MetaModel part up to n_break
        metamodel = MetaModel_EOS_model(
            nsat=self.nsat,
            nmin_MM_nsat=self.nmin_MM_nsat,
            nmax_nsat=NEP_dict["nbreak"] / self.nsat,
            ndat=self.ndat_metamodel,
            **self.metamodel_kwargs,
        )

        # Construct the metamodel part:
        mm_output = metamodel.construct_eos(NEP_dict)
        n_metamodel, p_metamodel, _, e_metamodel, _, mu_metamodel, cs2_metamodel = (
            mm_output
        )

        # Convert units back for CSE initialization
        n_metamodel = n_metamodel / utils.fm_inv3_to_geometric
        p_metamodel = p_metamodel / utils.MeV_fm_inv3_to_geometric
        e_metamodel = e_metamodel / utils.MeV_fm_inv3_to_geometric

        # Get values at break density
        p_break = jnp.interp(NEP_dict["nbreak"], n_metamodel, p_metamodel)
        e_break = jnp.interp(NEP_dict["nbreak"], n_metamodel, e_metamodel)
        mu_break = jnp.interp(NEP_dict["nbreak"], n_metamodel, mu_metamodel)
        cs2_break = jnp.interp(NEP_dict["nbreak"], n_metamodel, cs2_metamodel)

        # Define the speed-of-sound of the extension portion
        # the model is taken from arXiv:1812.08188
        # but instead of energy density, I am using density as the input
        cs2_extension_function = lambda x: (
            peakCSE_dict["gaussian_peak"]
            * jnp.exp(
                -0.5
                * (
                    (x - peakCSE_dict["gaussian_mu"]) ** 2
                    / peakCSE_dict["gaussian_sigma"] ** 2
                )
            )
            + cs2_break
            + (
                (1.0 / 3.0 - cs2_break)
                / (
                    1.0
                    + jnp.exp(
                        -peakCSE_dict["logit_growth_rate"]
                        * (x - peakCSE_dict["logit_midpoint"])
                    )
                )
            )
        )
        # Compute n, p, e for peakCSE (number densities in unit of fm^-3)
        n_CSE = jnp.logspace(
            jnp.log10(NEP_dict["nbreak"]), jnp.log10(self.nmax), num=self.ndat_CSE
        )
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

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        return ns, ps, hs, es, dloge_dlogps, mu, cs2


def locate_lowest_non_causal_point(cs2):
    r"""
    Find the first point where the equation of state becomes non-causal.

    The speed of sound squared :math:`c_s^2 = dp/d\varepsilon` must satisfy
    :math:`c_s^2 \leq 1` (in units where :math:`c = 1`) for causality.
    This function locates the first density where this condition is violated.

    Args:
        cs2 (Array): Speed of sound squared values.

    Returns:
        int: Index of first non-causal point, or -1 if EOS is everywhere causal.
    """
    mask = cs2 >= 1.0
    any_ones = jnp.any(mask)
    indices = jnp.arange(len(cs2))
    masked_indices = jnp.where(mask, indices, len(cs2))
    first_index = jnp.min(masked_indices)
    return jnp.where(any_ones, first_index, -1)


def construct_family(eos: tuple, ndat: Int = 50, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    Solve the TOV equations and generate mass-radius-tidal deformability relations.
    
    This function constructs a neutron star family by solving the Tolman-Oppenheimer-Volkoff (TOV)
    equations for a range of central pressures. The TOV equations describe the hydrostatic
    equilibrium of a spherically symmetric, static star:
    
    .. math::
        \frac{dm}{dr} &= 4\pi r^2 \varepsilon(r) \\
        \frac{dp}{dr} &= -\frac{[\varepsilon(r) + p(r)][m(r) + 4\pi r^3 p(r)]}{r[r - 2m(r)]}

    Args:
        eos (tuple): Tuple of the EOS data (ns, ps, hs, es, dloge_dlogps).
        ndat (int, optional): Number of datapoints used when constructing the central pressure grid. Defaults to 50.
        min_nsat (int, optional): Starting density for central pressure in numbers of :math:`n_0` 
                                 (assumed to be 0.16 :math:`\mathrm{fm}^{-3}`). Defaults to 2.

    Returns:
        tuple: A tuple containing:
        
            - :math:`\log(p_c)`: Logarithm of central pressures [geometric units]
            - :math:`M`: Gravitational masses [:math:`M_{\odot}`]
            - :math:`R`: Circumferential radii [:math:`\mathrm{km}`]
            - :math:`\Lambda`: Dimensionless tidal deformabilities
    """
    # Construct the dictionary
    ns, ps, hs, es, dloge_dlogps = eos
    eos_dict = dict(p=ps, h=hs, e=es, dloge_dlogp=dloge_dlogps)

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    pc_max = eos_dict["p"][-1]
    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    def solve_single_pc(pc):
        """Solve for single pc value"""
        return tov.tov_solver(eos_dict, pc)

    ms, rs, ks = jax.vmap(solve_single_pc)(pcs)

    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass and the radius to km
    ms /= utils.solar_mass_in_meter
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    # Limit masses to be below MTOV
    pcs, ms, rs, lambdas = utils.limit_by_MTOV(pcs, ms, rs, lambdas)

    # Get a mass grid and interpolate, since we might have dropped provided some duplicate points
    mass_grid = jnp.linspace(jnp.min(ms), jnp.max(ms), ndat)
    rs = jnp.interp(mass_grid, ms, rs)
    lambdas = jnp.interp(mass_grid, ms, lambdas)
    pcs = jnp.interp(mass_grid, ms, pcs)

    ms = mass_grid

    return jnp.log(pcs), ms, rs, lambdas


def construct_family_nonGR(eos: tuple, ndat: Int = 50, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    Solve modified TOV equations with beyond-GR corrections.

    This function extends the standard TOV equations to include phenomenological
    modifications that parameterize deviations from General Relativity. The modified
    pressure gradient equation becomes:

    .. math::
        \frac{dp}{dr} = -\frac{[\varepsilon(r) + p(r)][m(r) + 4\pi r^3 p(r)]}{r[r - 2m(r)]} - \frac{2\sigma(r)}{r}

    where :math:`\sigma(r)` contains the non-GR corrections parameterized by
    :math:`\lambda_{\mathrm{BL}}`, :math:`\lambda_{\mathrm{DY}}`, :math:`\lambda_{\mathrm{HB}}`,
    and post-Newtonian parameters :math:`\alpha`, :math:`\beta`, :math:`\gamma`.

    Args:
        eos (tuple): Extended EOS data including GR modification parameters.
        ndat (int, optional): Number of datapoints for central pressure grid. Defaults to 50.
        min_nsat (int, optional): Starting density in units of :math:`n_0`. Defaults to 2.

    Returns:
        tuple: A tuple containing:

            - :math:`\log(p_c)`: Logarithm of central pressures [geometric units]
            - :math:`M`: Gravitational masses [:math:`M_{\odot}`]
            - :math:`R`: Circumferential radii [:math:`\mathrm{km}`]
            - :math:`\Lambda`: Dimensionless tidal deformabilities
    """
    # Construct the dictionary
    (
        ns,
        ps,
        hs,
        es,
        dloge_dlogps,
        alpha,
        beta,
        gamma,
        lambda_BL,
        lambda_DY,
        lambda_HB,
    ) = eos
    eos_dict = dict(
        p=ps,
        h=hs,
        e=es,
        dloge_dlogp=dloge_dlogps,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lambda_BL=lambda_BL,
        lambda_DY=lambda_DY,
        lambda_HB=lambda_HB,
    )

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    # end at pc at pmax at which it is causal
    cs2 = ps / es / dloge_dlogps
    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2)]

    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    def solve_single_pc(pc):
        """Solve for single pc value"""
        return ptov.tov_solver(eos_dict, pc)

    ms, rs, ks = jax.vmap(solve_single_pc)(pcs)

    ### TODO: Check the timing with respect to this implementation
    # ms, rs, ks = jnp.vectorize(
    #     tov.tov_solver,
    #     excluded=[
    #         0,
    #     ],
    # )(eos_dict, pcs)

    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass and the radius to km
    ms /= utils.solar_mass_in_meter
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    # TODO: perhaps put a boolean here to flag whether or not to do this, or do we always want to do this?
    # Limit masses to be below MTOV
    pcs, ms, rs, lambdas = utils.limit_by_MTOV(pcs, ms, rs, lambdas)

    # Get a mass grid and interpolate, since we might have dropped provided some duplicate points
    mass_grid = jnp.linspace(jnp.min(ms), jnp.max(ms), ndat)
    rs = jnp.interp(mass_grid, ms, rs)
    lambdas = jnp.interp(mass_grid, ms, lambdas)
    pcs = jnp.interp(mass_grid, ms, pcs)

    ms = mass_grid

    return jnp.log(pcs), ms, rs, lambdas

def construct_family_ST(eos: tuple, ndat: Int = 50, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    # TODO:
    (updated later)
    """

    # Construct the dictionary
    ns, ps, hs, es, dloge_dlogps, beta_STs, phi_cs, nu_cs = eos
    #Here's EoS dict names defined 
    eos_dict = dict(p=ps, h=hs, e=es, dloge_dlogp=dloge_dlogps, beta_ST = beta_STs, phi_c=phi_cs, nu_c=nu_cs)

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    pc_max = eos_dict["p"][-1]
    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)
    def solve_single_pc(pc):
        """Solve for single pc value"""
        return STtov.tov_solver(eos_dict, pc)
    ms, rs, ks = jax.vmap(solve_single_pc)(pcs)
    
    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass and the radius to km
    ms /= utils.solar_mass_in_meter
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    # Limit masses to be below MTOV
    pcs, ms, rs, lambdas = utils.limit_by_MTOV(pcs, ms, rs, lambdas)

    # Get a mass grid and interpolate, since we might have dropped provided some duplicate points
    mass_grid = jnp.linspace(jnp.min(ms), jnp.max(ms), ndat)
    rs = jnp.interp(mass_grid, ms, rs)
    lambdas = jnp.interp(mass_grid, ms, lambdas)
    pcs = jnp.interp(mass_grid, ms, pcs)

    ms = mass_grid

    return jnp.log(pcs), ms, rs, lambdas

#For diagnostic, used in example file
def construct_family_ST_sol(eos: tuple, ndat: Int = 1, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    # TODO: complete the description
    Also output stellar structure solution via sol_iter (interior) and solext (exterior)
    """

    # Construct the dictionary
    ns, ps, hs, es, dloge_dlogps, beta_STs, phi_cs, nu_cs = eos
    #Here's EoS dict names defined 
    eos_dict = dict(p=ps, h=hs, e=es, dloge_dlogp=dloge_dlogps, beta_ST = beta_STs, phi_c=phi_cs, nu_c=nu_cs)

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    pc_max = eos_dict["p"][-1]
    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)
    def solve_single_pc(pc):
        """Solve for single pc value"""
        return STtov.tov_solver_printsol(eos_dict, pc)
    ms, rs, ks, sol_iter, solext = jax.vmap(solve_single_pc)(pcs)
    
    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass and the radius to km
    ms /= utils.solar_mass_in_meter
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    # Limit masses to be below MTOV
    pcs, ms, rs, lambdas = utils.limit_by_MTOV(pcs, ms, rs, lambdas)

    # Get a mass grid and interpolate, since we might have dropped provided some duplicate points
    mass_grid = jnp.linspace(jnp.min(ms), jnp.max(ms), ndat)
    rs = jnp.interp(mass_grid, ms, rs)
    lambdas = jnp.interp(mass_grid, ms, lambdas)
    pcs = jnp.interp(mass_grid, ms, pcs)

    ms = mass_grid

    return jnp.log(pcs), ms, rs, lambdas, sol_iter, solext
