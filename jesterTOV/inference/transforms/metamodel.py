"""MetaModel EOS transform (without CSE extension)."""

import jax.numpy as jnp
from jaxtyping import Float

from .base import JesterTransformBase
from jesterTOV.eos import MetaModel_EOS_model


class MetaModelTransform(JesterTransformBase):
    """Transform NEP parameters to M-R-Lambda using MetaModel EOS.

    This transform uses the MetaModel parametrization without the
    Constant Speed Extension (CSE) for the high-density region.

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
        Tuple of (input_names, output_names) for the transform
    keep_names : list[str], optional
        Names to keep in the output (default: all input names)
    **kwargs
        Additional arguments passed to JesterTransformBase

    Examples
    --------
    >>> transform = MetaModelTransform(
    ...     name_mapping=(["K_sat", "Q_sat", ...], ["masses_EOS", "radii_EOS", ...]),
    ...     ndat_metamodel=100,
    ...     nmax_nsat=25.0
    ... )
    >>> result = transform.forward({"K_sat": 230.0, ...})
    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        keep_names: list[str] = None,
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

        print("WARNING: This is a MetaModel run with no CSE parameters!")

    def get_eos_type(self) -> str:
        """Return EOS type identifier.

        Returns
        -------
        str
            "MM" for MetaModel only
        """
        return "MM"

    def get_parameter_names(self) -> list[str]:
        """Return list of expected parameter names.

        Returns
        -------
        list[str]
            List of 8 NEP parameter names
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
        """Core transformation: NEP → M-R-Lambda.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary containing NEP parameters

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
