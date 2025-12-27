r"""
MetaModel equation of state transform for neutron star inference.

This module implements the transform from nuclear empirical parameters (NEP)
to observable neutron star properties by solving the TOV equations with the
MetaModel EOS parametrization. The MetaModel provides a physics-informed
interpolation between known nuclear physics constraints at saturation density
and high-density behavior, without an explicit extension beyond ~2 n_sat.
"""

import jax.numpy as jnp
from jaxtyping import Float

from .base import JesterTransformBase
from jesterTOV.eos import MetaModel_EOS_model
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class MetaModelTransform(JesterTransformBase):
    """Transform from nuclear empirical parameters to neutron star observables.

    This transform maps the 8-parameter nuclear empirical parameter (NEP) space
    to neutron star mass-radius-tidal deformability curves by constructing an
    equation of state and solving the Tolman-Oppenheimer-Volkoff equations. The
    MetaModel parametrization interpolates smoothly between saturation density
    constraints and high-density behavior up to approximately 2 n_sat, making
    it suitable for modeling typical neutron stars without extremely high central
    densities.

    The transform enforces causality by truncating the EOS at the first point
    where the sound speed would exceed the speed of light (cs² ≥ 1).

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
        Tuple of (input_names, output_names) defining the parameter transform.
        Input names should include the 8 NEP parameters; output names typically
        include mass, radius, and tidal deformability arrays.
    keep_names : list[str], optional
        Additional parameter names to preserve in the output dictionary alongside
        the transformed quantities. Default is to keep all input parameters.
    **kwargs
        Additional arguments passed to JesterTransformBase, including:
        - ndat_metamodel : int
            Number of density grid points for EOS construction (default: 100)
        - nmax_nsat : float
            Maximum density in units of saturation density (default: 25.0)

    Attributes
    ----------
    nb_CSE : int
        Number of CSE parameters (always 0 for MetaModel without extension)
    eos : MetaModel_EOS_model
        The MetaModel EOS instance used for constructing p(n) curves

    Notes
    -----
    The 8 NEP parameters are:
    - K_sat, Q_sat, Z_sat : Symmetric matter expansion coefficients
    - E_sym, L_sym, K_sym, Q_sym, Z_sym : Symmetry energy expansion coefficients

    See Also
    --------
    MetaModelCSETransform : Extended version with high-density CSE parametrization

    Examples
    --------
    Create a transform and apply it to a set of NEP parameters:

    >>> from jesterTOV.inference.transforms import MetaModelTransform
    >>> transform = MetaModelTransform(
    ...     name_mapping=(
    ...         ["K_sat", "Q_sat", "Z_sat", "E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym"],
    ...         ["masses_EOS", "radii_EOS", "Lambdas_EOS"]
    ...     ),
    ...     ndat_metamodel=100,
    ...     nmax_nsat=25.0
    ... )
    >>> params = {
    ...     "K_sat": 230.0, "Q_sat": 0.0, "Z_sat": 0.0,
    ...     "E_sym": 32.0, "L_sym": 60.0, "K_sym": 0.0, "Q_sym": 0.0, "Z_sym": 0.0
    ... }
    >>> result = transform.forward(params)
    >>> print(result["masses_EOS"])  # Array of NS masses at different central pressures
    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        keep_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(name_mapping, keep_names, **kwargs)

        # Create MetaModel EOS (no CSE)
        self.nb_CSE = 0
        self.eos = MetaModel_EOS_model(
            nmax_nsat=self.nmax_nsat, ndat=self.ndat_metamodel
        )

        # Set transform function
        self.transform_func = self.transform_func_MM

        # NOTE: Cannot log here - transforms may be instantiated inside JAX-traced code

    def get_eos_type(self) -> str:
        """Return the EOS parametrization identifier.

        Returns
        -------
        str
            The string "MM" (MetaModel) identifying this EOS parametrization.
            This is used for logging and output file organization.
        """
        return "MM"

    def get_parameter_names(self) -> list[str]:
        """Return the list of nuclear empirical parameter names.

        Returns
        -------
        list[str]
            The 8 NEP parameter names in canonical order:
            ["K_sat", "Q_sat", "Z_sat", "E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym"].
            These define the Taylor expansion of symmetric matter energy and
            symmetry energy around saturation density.
        """
        return [
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]

    def transform_func_MM(self, params: dict[str, Float]) -> dict[str, Float]:
        """Compute neutron star observables from nuclear empirical parameters.

        This method constructs the equation of state from the NEP parameters,
        enforces causality by truncating where cs² ≥ 1, and solves the TOV
        equations to obtain mass-radius-tidal deformability curves.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary containing the 8 NEP parameters (K_sat, Q_sat, Z_sat,
            E_sym, L_sym, K_sym, Q_sym, Z_sym) and any additional parameters
            specified in `keep_names`.

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
        The EOS is automatically truncated at the first density where the sound
        speed would become superluminal. The remaining EOS quantities are
        re-interpolated onto a regular density grid spanning from the crust
        to this causality limit.
        """
        # Update with fixed parameters (currently empty)
        params.update(self.fixed_params)

        # Extract NEP parameters
        NEP = {
            key: value
            for key, value in params.items()
            if "_sat" in key or "_sym" in key
        }

        # Create the EOS
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP)

        # Limit cs2 so that it is causal (cs2 < 1)
        # Find first index where cs2 >= 1
        idx = jnp.argmax(cs2 >= 1.0)
        final_n = ns.at[idx].get()
        first_n = ns.at[0].get()

        # Re-interpolate to ensure causal EOS
        ns_interp = jnp.linspace(first_n, final_n, len(ns))
        ps_interp = jnp.interp(ns_interp, ns, ps)
        hs_interp = jnp.interp(ns_interp, ns, hs)
        es_interp = jnp.interp(ns_interp, ns, es)
        dloge_dlogps_interp = jnp.interp(ns_interp, ns, dloge_dlogps)
        cs2_interp = jnp.interp(ns_interp, ns, cs2)

        # Solve the TOV equations
        eos_tuple = (ns_interp, ps_interp, hs_interp, es_interp, dloge_dlogps_interp)
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self._solve_tov(eos_tuple)

        # Create and return standardized output dictionary
        return self._create_return_dict(
            logpc_EOS,
            masses_EOS,
            radii_EOS,
            Lambdas_EOS,
            ns_interp,
            ps_interp,
            hs_interp,
            es_interp,
            dloge_dlogps_interp,
            cs2_interp,
        )
