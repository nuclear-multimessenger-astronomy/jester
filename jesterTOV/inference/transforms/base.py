"""Base class for jesterTOV EOS transforms."""

from abc import ABC, abstractmethod
from typing import Any, Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

# Following the Jim/jimgw architecture - base class copied to remove dependency
from jesterTOV.eos import construct_family
from jesterTOV.inference.base import NtoMTransform
from jesterTOV.inference.likelihoods.constraints import check_all_constraints


class JesterTransformBase(NtoMTransform, ABC):
    """Base class for all jester EOS transforms.

    Provides common interface for converting EOS parameters
    (microscopic) to observables (macroscopic: M, R, Lambda).

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
        Tuple of (input_names, output_names) for the transform
    keep_names : list[str] | None, optional
        Names to keep in the output (default: all input names)
    ndat_metamodel : int
        Number of data points for MetaModel EOS (default: 100)
    nmax_nsat : float
        Maximum density in units of saturation density (default: 25.0)
    min_nsat_TOV : float
        Minimum density for TOV integration in units of nsat (default: 0.75)
    ndat_TOV : int
        Number of data points for TOV integration (default: 100)
    nb_masses : int
        Number of masses to sample (default: 100)
    crust_name : str
        Name of crust model to use: "DH", "BPS", or "DH_fixed" (default: "DH")

    Attributes
    ----------
    keep_names : list[str]
        Names to keep in the output
    ndat_metamodel : int
        Number of data points for MetaModel EOS
    nmax_nsat : float
        Maximum density in units of saturation density
    nmax : float
        Maximum density in physical units (fm^-3)
    construct_family_lambda : Callable
        Lambda function for solving TOV equations
    """

    keep_names: list[str]
    ndat_metamodel: int
    nmax_nsat: float
    nmax: float
    min_nsat_TOV: float
    ndat_TOV: int
    nb_masses: int
    crust_name: str
    construct_family_lambda: Callable[[Any], Any]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        keep_names: list[str] | None = None,
        ndat_metamodel: int = 100,
        nmax_nsat: float = 25.0,
        min_nsat_TOV: float = 0.75,
        ndat_TOV: int = 100,
        nb_masses: int = 100,
        crust_name: str = "DH",
    ) -> None:
        # By default, keep all input names
        if keep_names is None:
            keep_names = name_mapping[0]

        super().__init__(name_mapping)

        # Store keep_names for use in forward()
        self.keep_names = keep_names

        # Validate crust_name
        if crust_name not in ["DH", "BPS", "DH_fixed"]:
            raise ValueError(
                f"crust_name must be 'DH', 'BPS', or 'DH_fixed', got {crust_name}"
            )

        # Save configuration
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nmax = nmax_nsat * 0.16  # Convert to physical units (fm^-3)
        self.min_nsat_TOV = min_nsat_TOV
        self.ndat_TOV = ndat_TOV
        self.nb_masses = nb_masses
        self.crust_name = crust_name

        print(f"Transform initialized with crust: {crust_name}")

        # Construct lambda for solving TOV equations
        self.construct_family_lambda = lambda x: construct_family(
            x, ndat=self.ndat_TOV, min_nsat=self.min_nsat_TOV
        )

        # Fixed parameters (currently empty, but available for future use)
        self.fixed_params = {}

    @abstractmethod
    def get_eos_type(self) -> str:
        """Return EOS type identifier (e.g., 'MM', 'MM_CSE').

        Returns
        -------
        str
            EOS type identifier
        """
        pass

    @abstractmethod
    def get_parameter_names(self) -> list[str]:
        """Return list of expected parameter names.

        Returns
        -------
        list[str]
            List of parameter names this transform expects
        """
        pass

    def _solve_tov(
        self, eos_tuple: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Float[Array, " n"], Float[Array, " n"], Float[Array, " n"], Float[Array, " n"]]:
        """Solve TOV equations for a given EOS.

        Parameters
        ----------
        eos_tuple : tuple[Array, Array, Array, Array, Array]
            Tuple of (ns, ps, hs, es, dloge_dlogps) arrays

        Returns
        -------
        tuple[Float[Array, " n"], Float[Array, " n"], Float[Array, " n"], Float[Array, " n"]]
            (logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS)
        """
        return self.construct_family_lambda(eos_tuple)

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
    ) -> dict[str, Float | Float[Array, " n"]]:
        """Create standardized return dictionary with constraint checking.

        This method checks for physical constraint violations (NaN, causality, etc.)
        and adds violation counts to the output. It also cleans NaN values to prevent
        propagation through the likelihood evaluation.

        Parameters
        ----------
        logpc_EOS : Float
            Log of central pressure
        masses_EOS : Float
            Neutron star masses
        radii_EOS : Float
            Neutron star radii
        Lambdas_EOS : Float
            Tidal deformabilities
        ns : Float
            Baryon number densities
        ps : Float
            Pressures
        hs : Float
            Enthalpies
        es : Float
            Energy densities
        dloge_dlogps : Float
            Logarithmic derivative of energy density w.r.t. pressure
        cs2 : Float
            Sound speeds squared

        Returns
        -------
        dict[str, Float]
            Dictionary with EOS and TOV solution data, including:
            - Original EOS quantities (with NaN cleaned)
            - Constraint violation counts (scalars for JAX compatibility)
        """
        # Check all constraints BEFORE cleaning NaN
        # This gives us violation counts as scalars (JAX-compatible)
        constraints = check_all_constraints(masses_EOS, radii_EOS, Lambdas_EOS, cs2, ps)

        # Clean NaN values to prevent propagation through likelihood evaluation
        # Use 0.0 as sentinel (will be caught by constraint likelihood)
        masses_EOS_clean = jnp.nan_to_num(masses_EOS, nan=0.0, posinf=0.0, neginf=0.0)
        radii_EOS_clean = jnp.nan_to_num(radii_EOS, nan=0.0, posinf=0.0, neginf=0.0)
        Lambdas_EOS_clean = jnp.nan_to_num(Lambdas_EOS, nan=0.0, posinf=0.0, neginf=0.0)
        logpc_EOS_clean = jnp.nan_to_num(logpc_EOS, nan=0.0, posinf=0.0, neginf=0.0)

        return {
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
            "n_tov_failures": constraints['n_tov_failures'],
            "n_causality_violations": constraints['n_causality_violations'],
            "n_stability_violations": constraints['n_stability_violations'],
            "n_pressure_violations": constraints['n_pressure_violations'],
        }

    def set_keep_names(self, keep_names: list[str] | None) -> None:
        """Set or update the list of parameter names to keep in transform output.

        This allows updating keep_names after transform construction, which is
        useful when the set of required parameters depends on which likelihoods
        are enabled.

        Parameters
        ----------
        keep_names : list[str] | None
            Parameter names to preserve in output. If None, keeps all input parameters.
        """
        if keep_names is None:
            keep_names = self.name_mapping[0]
        self.keep_names = keep_names

    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
        """
        Override NtoMTransform.forward() to preserve keep_names parameters.

        This ensures that parameters in self.keep_names are preserved in the
        output even though they're consumed by the transform.

        Parameters
        ----------
        x : dict[str, Float]
            Input parameter dictionary

        Returns
        -------
        dict[str, Float]
            Transformed parameter dictionary with keep_names preserved
        """
        import jax

        # Save parameters that should be kept
        kept_params = {name: x[name] for name in self.keep_names if name in x}

        # Call parent forward() which does the standard transformation
        result = super().forward(x)

        # Add back the kept parameters
        result.update(kept_params)

        return result
