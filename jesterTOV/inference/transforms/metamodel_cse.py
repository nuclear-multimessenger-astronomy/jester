r"""
MetaModel equation of state transform with Constant Speed Extension.

This module implements the transform from nuclear empirical parameters (NEP)
plus high-density extension parameters to neutron star observables. The CSE
(Constant Speed Extension) extends the MetaModel beyond the breaking density
by specifying the sound speed on a grid, allowing exploration of stiffer EOSs
that can support massive neutron stars with central densities up to 6+ n_sat.
"""

import jax.numpy as jnp
from jaxtyping import Float

from .base import JesterTransformBase
from jesterTOV.eos import MetaModel_with_CSE_EOS_model
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class MetaModelCSETransform(JesterTransformBase):
    """Transform from NEP+CSE parameters to neutron star observables.

    This transform extends the standard MetaModel by adding a Constant Speed
    Extension (CSE) that controls the high-density behavior beyond a breaking
    density. The CSE parametrizes the sound speed on a grid of density points,
    enabling exploration of stiff EOSs needed to explain massive pulsars like
    PSR J0740+6620 (M ~ 2.1 solar masses) without violating causality.

    The full parameter space consists of:
    - 8 NEP parameters (K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym)
    - 1 breaking density (nbreak) where CSE extension begins
    - 2*nb_CSE + 1 CSE parameters defining cs²(n) on a grid

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
        Tuple of (input_names, output_names) defining the parameter transform.
        Input names should include the 8 NEP parameters, nbreak, and CSE grid
        parameters; output names typically include mass, radius, and tidal
        deformability arrays.
    keep_names : list[str], optional
        Additional parameter names to preserve in the output dictionary alongside
        the transformed quantities. Default is to keep all input parameters.
    nb_CSE : int, optional
        Number of CSE grid points for parametrizing the high-density region.
        More points allow finer control but increase parameter dimensionality.
        Default is 8, which provides sufficient flexibility for typical applications.
    ndat_CSE : int, optional
        Number of interpolation points used when constructing the EOS in the
        CSE region. Default is 100.
    **kwargs
        Additional arguments passed to JesterTransformBase, including:
        - ndat_metamodel : int
            Number of density grid points for low-density MetaModel region
        - nmax_nsat : float
            Maximum density in units of saturation density
        - crust_name : str
            Which crust model to use (e.g., "BPS", "DH")

    Attributes
    ----------
    nb_CSE : int
        Number of CSE grid points
    ndat_CSE : int
        Number of data points for CSE interpolation
    eos : MetaModel_with_CSE_EOS_model
        The MetaModel+CSE EOS instance used for constructing p(n) curves

    Notes
    -----
    The CSE grid parameters are specified as:
    - n_CSE_i_u : Normalized density position (0 to 1) for grid point i
    - cs2_CSE_i : Sound speed squared at grid point i

    The normalized positions are converted to physical densities via:
    n_CSE_i = nbreak + n_CSE_i_u * (nmax - nbreak)

    This ensures the CSE grid spans from nbreak to nmax with proper ordering.

    See Also
    --------
    MetaModelTransform : Basic version without CSE extension

    Examples
    --------
    Create a transform with 8 CSE grid points:

    >>> from jesterTOV.inference.transforms import MetaModelCSETransform
    >>> transform = MetaModelCSETransform(
    ...     name_mapping=(
    ...         ["K_sat", "Q_sat", "Z_sat", "E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym",
    ...          "nbreak", "n_CSE_0_u", "cs2_CSE_0", ..., "cs2_CSE_8"],
    ...         ["masses_EOS", "radii_EOS", "Lambdas_EOS"]
    ...     ),
    ...     nb_CSE=8,
    ...     ndat_metamodel=100,
    ...     ndat_CSE=100
    ... )
    >>> params = {
    ...     "K_sat": 230.0, "Q_sat": 0.0, "Z_sat": 0.0,
    ...     "E_sym": 32.0, "L_sym": 90.0, "K_sym": 0.0, "Q_sym": 0.0, "Z_sym": 0.0,
    ...     "nbreak": 0.20,  # Breaking density at 0.2 fm^-3
    ...     "n_CSE_0_u": 0.1, "cs2_CSE_0": 0.5,  # First grid point
    ...     # ... more CSE parameters
    ... }
    >>> result = transform.forward(params)
    >>> print(result["masses_EOS"])  # Neutron star masses
    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        keep_names: list[str] | None = None,
        nb_CSE: int = 8,
        ndat_CSE: int = 100,
        **kwargs,
    ):
        super().__init__(name_mapping, keep_names, **kwargs)

        # Save CSE configuration
        self.nb_CSE = nb_CSE
        self.ndat_CSE = ndat_CSE

        # Create MetaModel+CSE EOS
        self.eos = MetaModel_with_CSE_EOS_model(
            nmax_nsat=self.nmax_nsat,
            ndat_metamodel=self.ndat_metamodel,
            ndat_CSE=self.ndat_CSE,
            crust_name=self.crust_name,
        )

        # Set transform function
        self.transform_func = self.transform_func_MM_CSE

        # NOTE: Cannot log here - transforms may be instantiated inside JAX-traced code

    def get_eos_type(self) -> str:
        """Return the EOS parametrization identifier.

        Returns
        -------
        str
            The string "MM_CSE" (MetaModel with Constant Speed Extension)
            identifying this EOS parametrization. This is used for logging
            and output file organization.
        """
        return "MM_CSE"

    def get_parameter_names(self) -> list[str]:
        """Return the complete list of EOS parameter names.

        Returns
        -------
        list[str]
            The full parameter list consisting of:
            - 8 NEP parameters: K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym
            - 1 breaking density: nbreak
            - 2*nb_CSE CSE grid parameters: n_CSE_i_u and cs2_CSE_i for i=0..nb_CSE-1
            - 1 final sound speed: cs2_CSE_{nb_CSE}

            Total: 8 + 1 + 2*nb_CSE + 1 = 10 + 2*nb_CSE parameters
        """
        nep_params = [
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]
        cse_params = ["nbreak"]
        for i in range(self.nb_CSE):
            cse_params.extend([f"n_CSE_{i}_u", f"cs2_CSE_{i}"])
        return nep_params + cse_params

    def transform_func_MM_CSE(self, params: dict[str, Float]) -> dict[str, Float]:
        """Compute neutron star observables from NEP and CSE parameters.

        This method constructs a two-region equation of state: the MetaModel
        below nbreak and the CSE extension above nbreak. The CSE grid points
        are sorted and converted from normalized positions (0 to 1) to physical
        densities, then the full EOS is constructed and TOV equations solved.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary containing:
            - 8 NEP parameters (K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym)
            - Breaking density (nbreak)
            - CSE grid parameters (n_CSE_i_u, cs2_CSE_i for i=0..nb_CSE)
            - Any additional parameters specified in `keep_names`

        Returns
        -------
        dict[str, Float]
            Dictionary containing EOS and TOV solution quantities:

            - logpc_EOS : Log10 of central pressures (geometric units)
            - masses_EOS : Neutron star masses at each central pressure (solar masses)
            - radii_EOS : Neutron star radii at each central pressure (km)
            - Lambdas_EOS : Dimensionless tidal deformabilities
            - n : Baryon number density grid (geometric units)
            - p : Pressure values on density grid (geometric units)
            - h : Specific enthalpy values (geometric units)
            - e : Energy density values (geometric units)
            - dloge_dlogp : Logarithmic derivative d(log e)/d(log p)
            - cs2 : Sound speed squared (dimensionless, cs²/c²)

        Notes
        -----
        The CSE grid is constructed as follows:
        1. Extract normalized positions n_CSE_i_u ∈ [0,1] and sort them
        2. Convert to physical densities: n_CSE_i = nbreak + n_CSE_i_u*(nmax - nbreak)
        3. Append final grid point at nmax with corresponding cs² value
        4. Construct EOS by joining MetaModel (n < nbreak) and CSE (n ≥ nbreak)

        Unlike the standard MetaModel, this version does not enforce causality
        automatically since the CSE parametrization directly specifies cs²(n).
        """
        # Update with fixed parameters (currently empty)
        params.update(self.fixed_params)

        # Separate the MetaModel and CSE parameters
        NEP = {
            key: value
            for key, value in params.items()
            if "_sat" in key or "_sym" in key
        }
        NEP["nbreak"] = params["nbreak"]

        # Extract CSE grid parameters
        ngrids_u = jnp.array([params[f"n_CSE_{i}_u"] for i in range(self.nb_CSE)])
        ngrids_u = jnp.sort(ngrids_u)  # Sort to ensure monotonic grid
        cs2grids = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE)])

        # Convert from "quantiles" (values between 0 and 1) to physical densities
        # between nbreak and nmax
        width = self.nmax - params["nbreak"]
        ngrids = params["nbreak"] + ngrids_u * width

        # Append the final grid point at nmax with its corresponding cs2 value
        # Note: We use cs2_CSE_{nb_CSE} as the final parameter (one extra cs2 beyond the loop)
        ngrids = jnp.append(ngrids, jnp.array([self.nmax]))
        cs2grids = jnp.append(cs2grids, jnp.array([params[f"cs2_CSE_{self.nb_CSE}"]]))

        # Create the EOS
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(
            NEP, ngrids, cs2grids
        )
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)

        # Solve the TOV equations
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self._solve_tov(eos_tuple)

        # Create and return standardized output dictionary
        return self._create_return_dict(
            logpc_EOS,
            masses_EOS,
            radii_EOS,
            Lambdas_EOS,
            ns,
            ps,
            hs,
            es,
            dloge_dlogps,
            cs2,
        )
