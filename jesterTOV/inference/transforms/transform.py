r"""Unified transform for EOS parameters to neutron star observables."""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.metamodel import (
    MetaModel_EOS_model,
    MetaModel_with_CSE_EOS_model,
)
from jesterTOV.eos.spectral import SpectralDecomposition_EOS_model
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.gr import GRTOVSolver
from jesterTOV.inference.base import NtoMTransform
from jesterTOV.inference.config.schema import (
    BaseEOSConfig,
    MetamodelEOSConfig,
    MetamodelCSEEOSConfig,
    SpectralEOSConfig,
    BaseTOVConfig,
    GRTOVConfig,
    AnisotropyTOVConfig,
)
from jesterTOV.inference.likelihoods.constraints import check_all_constraints
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class JesterTransform(NtoMTransform):
    """Transform EOS parameters to neutron star observables (M, R, Λ).

    This is the main transform class that combines an equation of state (EOS)
    model with a TOV solver to produce neutron star observables from microscopic
    EOS parameters.

    The transform can be created either by:
    1. Passing EOS and TOV solver instances directly
    2. Using from_config() classmethod with configuration dict/object

    Parameters
    ----------
    eos : Interpolate_EOS_model
        EOS model instance (MetaModel, MetaModelCSE, Spectral, etc.)
    tov_solver : TOVSolverBase
        TOV solver instance (GRTOVSolver, AnisotropyTOVSolver, ScalarTensorTOVSolver)
    name_mapping : tuple[list[str], list[str]] | None
        Tuple of (input_names, output_names). If None, constructed from
        EOS and TOV required parameters.
    keep_names : list[str] | None
        Parameter names to preserve in output. If None, keeps all inputs.
    ndat_TOV : int
        Number of central pressure points for M-R-Λ curves (default: 100)
    min_nsat_TOV : float
        Minimum density for TOV integration in units of nsat (default: 0.75)
    **kwargs
        Additional parameters (for compatibility)

    Attributes
    ----------
    eos : Interpolate_EOS_model
        The equation of state model
    tov_solver : TOVSolverBase
        The TOV equation solver
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
        name_mapping: tuple[list[str], list[str]] | None = None,
        keep_names: list[str] | None = None,
        ndat_TOV: int = 100,
        min_nsat_TOV: float = 0.75,
        **kwargs: Any,
    ) -> None:
        self.eos = eos
        self.tov_solver = tov_solver
        self.ndat_TOV = ndat_TOV
        self.min_nsat_TOV = min_nsat_TOV

        # Get required parameters from EOS and TOV solver
        self.eos_params = eos.get_required_parameters()
        self.tov_params = tov_solver.get_required_parameters()

        # Construct name mapping if not provided
        if name_mapping is None:
            input_names = self.eos_params + self.tov_params
            output_names = ["logpc_EOS", "masses_EOS", "radii_EOS", "Lambdas_EOS"]
            name_mapping = (input_names, output_names)

        # Set keep_names (default: all input names)
        if keep_names is None:
            keep_names = name_mapping[0]
        self.keep_names = keep_names

        # Initialize parent NtoMTransform
        super().__init__(name_mapping)

        # Set transform_func for parent class compatibility
        self.transform_func = self.construct_eos_and_solve_tov

        logger.info(
            f"Initialized JesterTransform: EOS={repr(eos)}, TOV={repr(tov_solver)}"
        )
        logger.debug(f"  EOS parameters ({len(self.eos_params)}): {self.eos_params}")
        logger.debug(f"  TOV parameters ({len(self.tov_params)}): {self.tov_params}")

    @classmethod
    def from_config(
        cls,
        eos_config: BaseEOSConfig,
        tov_config: BaseTOVConfig,
        keep_names: list[str] | None = None,
        max_nbreak_nsat: float | None = None,
    ) -> "JesterTransform":
        """Create transform from configuration objects.

        This factory method instantiates the appropriate EOS and TOV solver
        based on the separate configurations, then creates the transform.

        Parameters
        ----------
        eos_config : EOSConfig
            EOS configuration (MetamodelEOSConfig, MetamodelCSEEOSConfig, or SpectralEOSConfig)
        tov_config : TOVConfig
            TOV solver configuration
        keep_names : list[str] | None
            Parameters to preserve in output
        max_nbreak_nsat : float | None
            Maximum nbreak value (for MetaModelCSE optimization)

        Returns
        -------
        JesterTransform
            Configured transform instance

        Raises
        ------
        ValueError
            If EOS or TOV type is unknown
        """
        # Instantiate EOS based on eos_config.type
        # If max_nbreak_nsat is not passed, fall back to the value from the config
        effective_max = (
            max_nbreak_nsat
            if max_nbreak_nsat is not None
            else getattr(eos_config, "max_nbreak_nsat", None)
        )
        eos = cls._create_eos(eos_config, effective_max)

        # Instantiate TOV solver based on tov_config
        tov_solver = cls._create_tov_solver(tov_config)

        # Create transform
        return cls(
            eos=eos,
            tov_solver=tov_solver,
            keep_names=keep_names,
            ndat_TOV=tov_config.ndat_TOV,
            min_nsat_TOV=tov_config.min_nsat_TOV,
        )

    @staticmethod
    def _create_eos(
        config: BaseEOSConfig, max_nbreak_nsat: float | None = None
    ) -> Interpolate_EOS_model:
        """Create EOS instance from config.

        Parameters
        ----------
        config : EOSConfig
            EOS configuration object (discriminated union)
        max_nbreak_nsat : float | None
            Maximum nbreak value for MetaModelCSE

        Returns
        -------
        Interpolate_EOS_model
            EOS instance

        Raises
        ------
        ValueError
            If config.type is not recognized
        """
        if isinstance(config, MetamodelEOSConfig):
            return MetaModel_EOS_model(
                nsat=0.16,
                nmin_MM_nsat=config.nmin_MM_nsat,
                nmax_nsat=config.nmax_nsat,
                ndat=config.ndat_metamodel,
                crust_name=config.crust_name,
            )

        elif isinstance(config, MetamodelCSEEOSConfig):
            return MetaModel_with_CSE_EOS_model(
                nsat=0.16,
                nmin_MM_nsat=config.nmin_MM_nsat,
                nmax_nsat=config.nmax_nsat,
                max_nbreak_nsat=max_nbreak_nsat,
                ndat_metamodel=config.ndat_metamodel,
                ndat_CSE=config.ndat_CSE,
                nb_CSE=config.nb_CSE,
                crust_name=config.crust_name,
            )

        elif isinstance(config, SpectralEOSConfig):
            return SpectralDecomposition_EOS_model(
                crust_name=config.crust_name,
                n_points_high=config.n_points_high,
                reparametrized=config.reparametrized,
                sigma_scale=config.sigma_scale,
            )

        else:
            raise ValueError(f"Unknown EOS config type: {type(config).__name__}")

    @staticmethod
    def _create_tov_solver(config: BaseTOVConfig) -> TOVSolverBase:
        """Create TOV solver instance from config.

        Parameters
        ----------
        config : TOVConfig
            TOV configuration object

        Returns
        -------
        TOVSolverBase
            TOV solver instance

        Raises
        ------
        ValueError
            If TOV solver type is not recognized
        NotImplementedError
            If TOV solver config class is not implemented yet
        """
        if isinstance(config, GRTOVConfig):
            return GRTOVSolver()

        elif isinstance(config, AnisotropyTOVConfig):
            from jesterTOV.tov.anisotropy import AnisotropyTOVSolver

            return AnisotropyTOVSolver()

        else:
            raise ValueError(f"Unknown TOV solver type: {type(config).__name__}")

    def get_eos_type(self) -> str:
        """Return EOS type identifier.

        Returns
        -------
        str
            EOS class name (e.g., 'MetaModel_EOS_model')
        """
        return repr(self.eos)

    def get_parameter_names(self) -> list[str]:
        """Return combined list of EOS and TOV parameters.

        Returns
        -------
        list[str]
            All parameter names required by this transform
        """
        return self.eos_params + self.tov_params

    def construct_eos_and_solve_tov(
        self,
        params: dict[str, Float],
    ) -> dict[str, Float | Float[Array, " n"]]:
        """Construct EOS from parameters and solve TOV equations.

        This is the core transformation method that:
        1. Constructs EOS from parameters
        2. Solves TOV equations for M-R-Λ family
        3. Returns observables with constraint checking

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
        """
        # Construct EOS from parameters
        # EOS handles all parameter preprocessing (e.g., CSE conversion)
        eos_data = self.eos.construct_eos(params)

        # Pass all sampled parameters that are not EOS parameters to the TOV solver.
        # This supports solvers whose parameters are all optional (e.g. AnisotropyTOVSolver),
        # where only a subset may appear in the prior.
        eos_param_set = set(self.eos_params)
        tov_kwargs = {key: params[key] for key in params if key not in eos_param_set}

        # Solve TOV equations to get M-R-Λ family
        family_data = self.tov_solver.construct_family(
            eos_data,
            ndat=self.ndat_TOV,
            min_nsat=self.min_nsat_TOV,
            **tov_kwargs,
        )

        # Create standardized return dictionary with constraint checking
        result = self._create_return_dict(
            logpc_EOS=family_data.log10pcs,
            masses_EOS=family_data.masses,
            radii_EOS=family_data.radii,
            Lambdas_EOS=family_data.lambdas,
            ns=eos_data.ns,
            ps=eos_data.ps,
            hs=eos_data.hs,
            es=eos_data.es,
            dloge_dlogps=eos_data.dloge_dlogps,
            cs2=eos_data.cs2,
            extra_constraints=eos_data.extra_constraints,
        )

        return result

    def _create_return_dict(
        self,
        logpc_EOS: Float[Array, " n"],
        masses_EOS: Float[Array, " n"],
        radii_EOS: Float[Array, " n"],
        Lambdas_EOS: Float[Array, " n"],
        ns: Float[Array, " n"],
        ps: Float[Array, " n"],
        hs: Float[Array, " n"],
        es: Float[Array, " n"],
        dloge_dlogps: Float[Array, " n"],
        cs2: Float[Array, " n"],
        extra_constraints: dict[str, Float | Float[Array, " n"]] | None = None,
    ) -> dict[str, Float | Float[Array, " n"]]:
        """Create standardized return dictionary with constraint checking.

        This method checks for physical constraint violations (NaN, causality, etc.)
        and adds violation counts to the output. It also cleans NaN values to prevent
        propagation through the likelihood evaluation.

        Parameters
        ----------
        logpc_EOS : Float[Array, " n"]
            Log10 of central pressures
        masses_EOS : Float[Array, " n"]
            Neutron star masses
        radii_EOS : Float[Array, " n"]
            Neutron star radii
        Lambdas_EOS : Float[Array, " n"]
            Tidal deformabilities
        ns : Float[Array, " n"]
            Number densities
        ps : Float[Array, " n"]
            Pressures
        hs : Float[Array, " n"]
            Enthalpies
        es : Float[Array, " n"]
            Energy densities
        dloge_dlogps : Float[Array, " n"]
            Logarithmic derivative d(ln ε)/d(ln p)
        cs2 : Float[Array, " n"]
            Sound speeds squared
        extra_constraints : dict | None
            Additional constraint violations from EOS

        Returns
        -------
        dict[str, Float | Float[Array, " n"]]
            Complete output dictionary with cleaned values and violation counts
        """
        # Check all constraints BEFORE cleaning NaN
        constraints = check_all_constraints(masses_EOS, radii_EOS, Lambdas_EOS, cs2, ps)

        # Clean NaN values to prevent propagation
        masses_EOS_clean = jnp.nan_to_num(masses_EOS, nan=0.0, posinf=0.0, neginf=0.0)
        radii_EOS_clean = jnp.nan_to_num(radii_EOS, nan=0.0, posinf=0.0, neginf=0.0)
        Lambdas_EOS_clean = jnp.nan_to_num(Lambdas_EOS, nan=0.0, posinf=0.0, neginf=0.0)
        logpc_EOS_clean = jnp.nan_to_num(logpc_EOS, nan=0.0, posinf=0.0, neginf=0.0)

        result = {
            # TOV solution (cleaned)
            "logpc_EOS": logpc_EOS_clean,
            "masses_EOS": masses_EOS_clean,
            "radii_EOS": radii_EOS_clean,
            "Lambdas_EOS": Lambdas_EOS_clean,
            # EOS quantities
            "n": ns,
            "p": ps,
            "h": hs,
            "e": es,
            "dloge_dlogp": dloge_dlogps,
            "cs2": cs2,
            # Constraint violation counts (scalars for JAX compatibility)
            "n_tov_failures": constraints["n_tov_failures"],
            "n_causality_violations": constraints["n_causality_violations"],
            "n_stability_violations": constraints["n_stability_violations"],
            "n_pressure_violations": constraints["n_pressure_violations"],
        }

        # Add any extra constraint violations from EOS
        if extra_constraints is not None:
            result.update(extra_constraints)

        return result

    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
        """Transform parameters and preserve keep_names.

        This overrides NtoMTransform.forward() to preserve parameters
        specified in self.keep_names.

        Parameters
        ----------
        x : dict[str, Float]
            Input parameter dictionary

        Returns
        -------
        dict[str, Float]
            Transformed parameters with keep_names preserved
        """
        # Save parameters that should be kept
        kept_params = {name: x[name] for name in self.keep_names if name in x}

        # Call parent forward() for standard transformation
        result = super().forward(x)

        # Add back the kept parameters
        result.update(kept_params)

        return result
