from typing import Any, Callable

import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.random import PRNGKey

from jesterTOV.logging_config import get_logger
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.utils import solar_mass_in_meter, lambda1_lambda2_to_lambda_tilde
from jesterTOV.inference.transforms import PopulationJesterTransform


logger = get_logger("jester")

class MultimessengerJesterTransform(PopulationJesterTransform):
    """
    Transform EOS parameters to 
    neutron star observables (M, R, Λ),
    samples a mass distribution for mass_1_source, mass_2_source
    from the mass distribution parameters and then
    also computes the dynamical ejecta and disk masses according to fitting formulae.

    This is the transform class that should be used for joint inference of the EOS
    and the mass population from GW+KN events.

    The transform can be created either by:
    1. Passing EOS, TOV solver, and population functions directly
    2. Using from_config() classmethod with configuration dict/object

    Parameters
    ----------
    eos : Interpolate_EOS_model
        EOS model instance (MetaModel, MetaModelCSE, Spectral, etc.)
    tov_solver : TOVSolverBase
        TOV solver instance (GRTOVSolver, PostTOVSolver, ScalarTensorTOVSolver)
    population: Callable
        jit-compatible population function that takes a random key, 
        a param dict, and a size integer and returns arrays with mass_1, mass_2.
    name_mapping : tuple[list[str], list[str]] | None
        Tuple of (input_names, output_names). If None, constructed from
        EOS and TOV required parameters.
    keep_names : list[str] | None
        Parameter names to preserve in output. If None, keeps all inputs.
    ndat_TOV : int
        Number of central pressure points for M-R-Λ curves (default: 100)
    min_nsat_TOV : float
        Minimum density for TOV integration in units of nsat (default: 0.75)
    fixed_params : dict
        Parameters that are kept fixed during the transformation and likelihood evaluation. (default: {})
    pop_random_key : PRNGKey
        fixed random key to draw the masses from the population model.
        This key has to stay the same in order to make the likelihood evaluation deterministic.
        Defaults to PRNGKey(170817).
    N_masses_evaluation : int
        Number of mass samples to draw from the population (default: 8000)
        Large values recommended - GPU parallelization makes this cheap!
    **kwargs
        Additional parameters (for compatibility)

    Attributes
    ----------
    eos : Interpolate_EOS_model
        The equation of state model
    tov_solver : TOVSolverBase
        The TOV equation solver
    population: Callable
        The population model
    eos_params : list[str]
        Parameters required by the EOS
    tov_params : list[str]
        Parameters required by the TOV solver
    keep_names : list[str]
        Parameters to preserve in output

    Examples
    --------
    >>> # Direct instantiation
    >>> from jesterTOV.eos.metamodel import MetaModel_EOS_model
    >>> from jesterTOV.tov.gr import GRTOVSolver
    >>> eos = MetaModel_EOS_model(crust_name="DH")
    >>> solver = GRTOVSolver()
    >>> transform = JesterTransform(eos=eos, tov_solver=solver)

    >>> # From configuration
    >>> from jesterTOV.inference.config.schema import MetamodelCSEEOSConfig, TOVConfig
    >>> eos_config = MetamodelCSEEOSConfig(type="metamodel_cse", nb_CSE=8)
    >>> tov_config = TOVConfig(type="gr")
    >>> transform = JesterTransform.from_config(eos_config, tov_config)

    >>> # Transform parameters to observables
    >>> params = {"E_sat": -16.0, "K_sat": 230.0, ...}
    >>> result = transform.forward(params)
    >>> print(result["masses_EOS"])  # Neutron star masses in M☉
    """

    def __init__(
        self,
        eos: Interpolate_EOS_model,
        tov_solver: TOVSolverBase,
        population: Callable,
        name_mapping: tuple[list[str], list[str]] | None = None,
        keep_names: list[str] | None = None,
        ndat_TOV: int = 100,
        min_nsat_TOV: float = 0.75,
        fixed_params: dict = {},
        pop_random_key: PRNGKey = PRNGKey(170817),
        N_masses_evaluation: int = 8000,
        **kwargs: Any,
    ) -> None:


        super().__init__(
            eos=eos,
            tov_solver=tov_solver,
            population=population,
            name_mapping=name_mapping,
            keep_names=keep_names,
            ndat_TOV=ndat_TOV,
            min_nsat_TOV=min_nsat_TOV,
            fixed_params=fixed_params,
            pop_random_key=pop_random_key,
            N_masses_evaluation=N_masses_evaluation,
            **kwargs
        )

        self.transform_func = self.eos_and_multimessenger_conversion

        logger.debug(f"Initialized Transform with multi-messenger conversion.")
  
    def eos_and_multimessenger_conversion(
        self,
        params: dict[str, Float],
    ) -> dict[str, Float | Float[Array, " n"]]:
        """Construct EOS from parameters, solve TOV equations 
            and sample population model.

        This is the transformation method that:
        1. Gets the M-R-Λ family with constraint check
        2. Samples the population model
        3. Converts the sampled population masses into ejecta and disk masses.

        Parameters
        ----------
        params : dict[str, Float]
            Input parameters (EOS + TOV parameters)

        Returns
        -------
        dict[str, Float | Float[Array, " n"]]
            Dictionary containing:
            - masses_EOS : Neutron star masses [M☉]
            - radii_EOS : Neutron star radii [km]
            - Lambdas_EOS : Tidal deformabilities
            - logpc_EOS : Log10 central pressures
            - n, p, h, e, dloge_dlogp, cs2 : EOS quantities
            - Constraint violation counts
            - masses_1_pop: Samples from the population model for the heavier component
            - masses_2_pop: Samples from the population model for the lighter component
            - log10_mej_dyn: Dynamical ejecta calculated from masses_1_pop and masses_2_pop
                             using NR fits.
            - log10_mdisk: Disk mass calculated from masses_1_pop and masses_2_pop using 
                            NR fits.
        """

        result = super().construct_eos_and_solve_tov_and_sample_population(params)
        result.update(self.multimessenger_conversion(result, params))

        return result

    def multimessenger_conversion(self,
            family: dict[str, Float[Array, " n"]],
            params: dict[str, Float]
        ) -> dict[str, Float[Array, " n"]]:
        
        masses_EOS: Float[Array, " n_points"] = family["masses_EOS"]
        radii_EOS: Float[Array, " n_points"] = family["radii_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = family["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)
        r16 = jnp.interp(1.6, masses_EOS, radii_EOS)
        m_coll = params.get("k_coll", 1.3) * mtov

        mass_1 = family["masses_1_pop"]
        mass_2 = family["masses_2_pop"]
        prompt_collapse = m_coll < (mass_1 + mass_2)

        lambda_1 = jnp.interp(mass_1, masses_EOS, Lambdas_EOS)
        lambda_2 = jnp.interp(mass_2, masses_EOS, Lambdas_EOS)
        radii_1 = jnp.interp(mass_1, masses_EOS, radii_EOS)
        radii_2 = jnp.interp(mass_2, masses_EOS, radii_EOS)
        compactness_1 = mass_1 / radii_1 * solar_mass_in_meter * 1e-3
        compactness_2 = mass_2 / radii_2 * solar_mass_in_meter * 1e-3

        mej_dyn = jnp.where(
            prompt_collapse, 
            self.dynamic_mass_fitting_prompt_collapse(mass_1, mass_2, lambda_1, lambda_2), 
            self.dynamic_mass_fitting(mass_1, mass_2, compactness_1, compactness_2)
        )
        log10_mej_dyn = jnp.log10(mej_dyn)
       

        log10_mdisk = jnp.where(
            prompt_collapse,
            self.log10_disk_mass_fitting_prompt_collapse(mass_1, mass_2, lambda_1, lambda_2),
            self.log10_disk_mass_fitting(mass_1+mass_2, mass_1/mass_2, mtov, r16)
        )
       
        return {"log10_mej_dyn": log10_mej_dyn, "log10_mdisk": log10_mdisk}
    

    @staticmethod
    def dynamic_mass_fitting_prompt_collapse(
            mass_1,
            mass_2,
            lambda_1,
            lambda_2,
            a=1.25e-4,
            b=9.82e-1,
            c=-2.44,
    ):
        """
        See https://arxiv.org/pdf/2411.02342, Eq. (9)
        """
        q = mass_2 / mass_1
        lambda_tilde = lambda1_lambda2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
        mdyn = a*lambda_tilde*(q**(-1) -b) * jnp.exp(c/q) # this is always positive

        mdyn = jnp.maximum(1e-5, mdyn)

        return mdyn
    
    @staticmethod
    def dynamic_mass_fitting(
        mass_1,
        mass_2,
        compactness_1,
        compactness_2,
        a=-9.3335,
        b=114.17,
        c=-337.56,
        n=1.5465,
    ):
        """
        See https://arxiv.org/pdf/2002.07728.pdf
        """

        mdyn = mass_1 * (
            a / compactness_1 + b * jnp.power(mass_2 / mass_1, n) + c * compactness_1
        )
        mdyn += mass_2 * (
            a / compactness_2 + b * jnp.power(mass_1 / mass_2, n) + c * compactness_2
        )
        mdyn *= 1e-3

        mdyn = jnp.maximum(1e-5, mdyn)

        return mdyn
    
    @staticmethod
    def log10_disk_mass_fitting(
        total_mass,
        mass_ratio,
        MTOV,
        R16,
        a0=-1.725,
        delta_a=-2.337,
        b0=-0.564,
        delta_b=-0.437,
        c=0.958,
        d=0.057,
        beta=5.879,
        q_trans=0.886,
    ):
        """
        See https://arxiv.org/pdf/2205.08513 Eq. (22)
        The coefficients a0, delta_a etc. have been updated since then,
        the ones here are the correct ones.
        The threshold mass is from https://arxiv.org/pdf/1908.05442.pdf.
        """
        k = -3.606 * MTOV / R16 + 2.38
        threshold_mass = k * MTOV

        xi = 0.5 * jnp.tanh(beta * (mass_ratio - q_trans))

        a = a0 + delta_a * xi
        b = b0 + delta_b * xi

        log10_mdisk = a * (1 + b * jnp.tanh((c - total_mass / threshold_mass) / d))
        log10_mdisk = jnp.maximum(-3.0, log10_mdisk)

        return log10_mdisk
    

    @staticmethod
    def log10_disk_mass_fitting_prompt_collapse(
            mass_1,
            mass_2,
            lambda_1,
            lambda_2,
            a=7.70,
            b=-13.4,
            c=8.16e-3):
        """
        See https://arxiv.org/pdf/2411.02342, Eq. (11)
        Typo for b, b=-13.4 confirmed through author correspondence
        """
        q = mass_2 / mass_1
        lambda_tilde = lambda1_lambda2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
        log10_mdisk = a + b * q + c * lambda_tilde * q**2

        log10_mdisk = jnp.minimum(log10_mdisk, -1)

        return log10_mdisk