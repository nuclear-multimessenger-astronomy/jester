r"""Unified transform for EOS parameters to neutron star observables."""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.eos import (
    MetaModel_EOS_model,
    MetaModel_with_CSE_EOS_model,
    SpectralDecomposition_EOS_model,
)
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.tov import GRTOVSolver, PostTOVSolver, ScalarTensorTOVSolver
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.inference.base import NtoMTransform
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
        TOV solver instance (GRTOVSolver, PostTOVSolver, ScalarTensorTOVSolver)
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
    >>> from jesterTOV.eos import MetaModel_EOS_model
    >>> from jesterTOV.tov import GRTOVSolver
    >>> eos = MetaModel_EOS_model(crust_name="DH")
    >>> solver = GRTOVSolver()
    >>> transform = JesterTransform(eos=eos, tov_solver=solver)

    >>> # From configuration
    >>> config = TransformConfig(type="metamodel_cse", nb_CSE=8)
    >>> transform = JesterTransform.from_config(config)

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
        config: Any,
        keep_names: list[str] | None = None,
        max_nbreak_nsat: float | None = None,
    ) -> "JesterTransform":
        """Create transform from configuration object.

        This factory method instantiates the appropriate EOS and TOV solver
        based on the configuration, then creates the transform.

        Parameters
        ----------
        config : TransformConfig
            Configuration object with type, parameters, etc.
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
        # Instantiate EOS based on config.type
        eos = cls._create_eos(config, max_nbreak_nsat)

        # Instantiate TOV solver based on config (default: GR)
        tov_solver = cls._create_tov_solver(config)

        # Create transform
        return cls(
            eos=eos,
            tov_solver=tov_solver,
            keep_names=keep_names,
            ndat_TOV=getattr(config, "ndat_TOV", 100),
            min_nsat_TOV=getattr(config, "min_nsat_TOV", 0.75),
        )

    @staticmethod
    def _create_eos(
        config: Any, max_nbreak_nsat: float | None = None
    ) -> Interpolate_EOS_model:
        """Create EOS instance from config.

        Parameters
        ----------
        config : TransformConfig
            Configuration object
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
        eos_type = config.type

        if eos_type == "metamodel":
            return MetaModel_EOS_model(
                nsat=0.16,
                nmin_MM_nsat=getattr(config, "min_nsat_TOV", 0.75),
                nmax_nsat=getattr(config, "nmax_nsat", 25.0),
                ndat=getattr(config, "ndat_metamodel", 100),
                crust_name=getattr(config, "crust_name", "DH"),
            )

        elif eos_type == "metamodel_cse":
            return MetaModel_with_CSE_EOS_model(
                nsat=0.16,
                nmin_MM_nsat=getattr(config, "min_nsat_TOV", 0.75),
                nmax_nsat=getattr(config, "nmax_nsat", 25.0),
                max_nbreak_nsat=max_nbreak_nsat,
                ndat_metamodel=getattr(config, "ndat_metamodel", 100),
                ndat_CSE=100,
                nb_CSE=getattr(config, "nb_CSE", 8),
                crust_name=getattr(config, "crust_name", "DH"),
            )

        elif eos_type == "spectral":
            return SpectralDecomposition_EOS_model(
                crust_name=getattr(config, "crust_name", "DH"),
                n_points_high=getattr(config, "n_points_high", 500),
            )

        else:
            raise ValueError(f"Unknown EOS type: {eos_type}")

    @staticmethod
    def _create_tov_solver(config: Any) -> TOVSolverBase:
        """Create TOV solver instance from config.

        Parameters
        ----------
        config : TransformConfig
            Configuration object

        Returns
        -------
        TOVSolverBase
            TOV solver instance

        Raises
        ------
        ValueError
            If TOV solver type is not recognized
        """
        # Check if config specifies TOV solver type (future feature)
        tov_type = getattr(config, "tov_solver", "gr")

        if tov_type == "gr":
            return GRTOVSolver()
        elif tov_type == "post":
            return PostTOVSolver()
        elif tov_type == "scalar_tensor":
            return ScalarTensorTOVSolver()
        else:
            raise ValueError(f"Unknown TOV solver type: {tov_type}")

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

        # Extract TOV-specific parameters if any
        tov_kwargs = {key: params[key] for key in self.tov_params if key in params}

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
