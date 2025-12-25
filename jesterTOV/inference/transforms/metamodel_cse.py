"""MetaModel EOS transform with CSE (Constant Speed Extension)."""

import jax.numpy as jnp
from jaxtyping import Float

from .base import JesterTransformBase
from jesterTOV.eos import MetaModel_with_CSE_EOS_model


class MetaModelCSETransform(JesterTransformBase):
    """Transform NEP+CSE parameters to M-R-Lambda using MetaModel+CSE EOS.

    This transform uses the MetaModel parametrization with the Constant
    Speed Extension (CSE) for modeling the high-density region beyond
    the breaking density nbreak.

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
        Tuple of (input_names, output_names) for the transform
    keep_names : list[str], optional
        Names to keep in the output (default: all input names)
    nb_CSE : int
        Number of CSE grid points (default: 8)
    ndat_CSE : int
        Number of data points for CSE region (default: 100)
    **kwargs
        Additional arguments passed to JesterTransformBase

    Examples
    --------
    >>> transform = MetaModelCSETransform(
    ...     name_mapping=(["K_sat", ..., "nbreak", ...], ["masses_EOS", ...]),
    ...     nb_CSE=8,
    ...     ndat_metamodel=100
    ... )
    >>> result = transform.forward({"K_sat": 230.0, ..., "nbreak": 0.2, ...})
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

        print(f"MetaModel+CSE initialized with {nb_CSE} CSE grid points")

    def get_eos_type(self) -> str:
        """Return EOS type identifier.

        Returns
        -------
        str
            "MM_CSE" for MetaModel with CSE
        """
        return "MM_CSE"

    def get_parameter_names(self) -> list[str]:
        """Return list of expected parameter names.

        Returns
        -------
        list[str]
            List of 8 NEP + 1 nbreak + 2*nb_CSE CSE grid parameters
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
        """Core transformation: NEP+CSE → M-R-Lambda.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary containing NEP and CSE parameters

        Returns
        -------
        dict[str, Float]
            Dictionary with:
            - logpc_EOS: Log of central pressure
            - masses_EOS: Neutron star masses
            - radii_EOS: Neutron star radii
            - Lambdas_EOS: Tidal deformabilities
            - n: Baryon number densities
            - p: Pressures
            - h: Enthalpies
            - e: Energy densities
            - dloge_dlogp: Log derivative of e w.r.t. p
            - cs2: Sound speeds squared
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
