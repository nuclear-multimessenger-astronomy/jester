r"""Meta-model EOS with piecewise constant speed-of-sound extensions (CSE)."""

import jax.numpy as jnp
from jaxtyping import Float, Int

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.metamodel.base import MetaModel_EOS_model
from jesterTOV.tov.data_classes import EOSData


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

    The CSE grid is specified via individual parameters (n_CSE_i_u, cs2_CSE_i)
    that are converted internally to density and sound speed arrays.
    """

    def __init__(
        self,
        nsat: Float = 0.16,
        nmin_MM_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        max_nbreak_nsat: Float | None = None,
        ndat_metamodel: Int = 100,
        ndat_CSE: Int = 100,
        nb_CSE: Int = 8,
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
            max_nbreak_nsat (Float | None, optional):
                Maximum value of nbreak prior in units of :math:`n_0`.
                Used to set the upper limit for the meta-model region to avoid
                unnecessary computation. If None, defaults to nmax_nsat.
                Should be set to the maximum value from the nbreak prior distribution.
                Defaults to None.
            ndat_metamodel (Int, optional):
                Number of density points for meta-model region discretization.
                Higher values give smoother meta-model interpolation. Defaults to 100.
            ndat_CSE (Int, optional):
                Number of density points for constant speed-of-sound extension region.
                Controls resolution of high-density exotic matter modeling. Defaults to 100.
            nb_CSE (Int, optional):
                Number of CSE grid points. This determines how many individual parameters
                are expected: n_CSE_0_u, cs2_CSE_0, ..., n_CSE_{nb_CSE-1}_u, cs2_CSE_{nb_CSE-1},
                plus cs2_CSE_{nb_CSE} (final value at nmax). Defaults to 8.
            **metamodel_kwargs:
                Additional keyword arguments passed to the underlying MetaModel_EOS_model.
                Includes parameters like kappas, v_nq, b_sat, b_sym, crust_name, etc.
                See MetaModel_EOS_model.__init__ for complete parameter descriptions.

        See Also:
            MetaModel_EOS_model.__init__ : Base meta-model parameters
            construct_eos : Method that defines CSE parameters and break density

        Note:
            The metamodel is created once in __init__ with max_nbreak_nsat as the maximum
            density to avoid re-instantiating the metamodel class on each construct_eos call.
            During construct_eos, the metamodel output is interpolated to the actual nbreak
            value (which varies with each sample) while maintaining fixed array sizes for JAX.
        """

        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_MM_nsat = nmin_MM_nsat
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nb_CSE = nb_CSE

        # Use max_nbreak_nsat if provided, otherwise default to nmax_nsat
        # This allows optimization when the nbreak prior has a tighter upper bound
        metamodel_max_nsat = (
            max_nbreak_nsat if max_nbreak_nsat is not None else nmax_nsat
        )

        # Create the metamodel instance once with max density from nbreak prior
        # This will be reused in construct_eos and interpolated to actual nbreak
        self.metamodel = MetaModel_EOS_model(
            nsat=nsat,
            nmin_MM_nsat=nmin_MM_nsat,
            nmax_nsat=metamodel_max_nsat,
            ndat=ndat_metamodel,
            **metamodel_kwargs,
        )

    def construct_eos(
        self,
        params: dict[str, float],
    ) -> EOSData:
        r"""
        Construct the EOS by combining metamodel and CSE regions.

        This method constructs the full EOS by:
        1. Extracting and processing CSE grid point parameters
        2. Building the metamodel EOS up to the full nmax range
        3. Interpolating the metamodel to a fixed-size grid up to nbreak
        4. Stitching the CSE extension on top from nbreak to nmax

        Args:
            params (dict[str, float]): Dictionary containing:
                - NEP parameters: E_sat, K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym
                - Break density: nbreak
                - CSE grid points: n_CSE_0_u, cs2_CSE_0, ..., n_CSE_{nb_CSE-1}_u, cs2_CSE_{nb_CSE-1}
                - Final cs2 value: cs2_CSE_{nb_CSE}

        Returns:
            EOSData: Complete EOS with all required arrays in geometric units

        Note:
            The metamodel instance is reused from __init__ (not re-instantiated) and its output
            is interpolated to the actual nbreak value to maintain JAX compatibility with
            fixed array sizes.

            CSE parameters are converted from individual normalized positions (n_CSE_i_u ∈ [0,1])
            to physical density arrays (ngrids) via:
            n_CSE_i = nbreak + n_CSE_i_u * (nmax - nbreak)
        """

        # Extract break density
        nbreak = params["nbreak"]

        # Convert individual CSE grid point parameters to arrays
        ngrids_u = jnp.array([params[f"n_CSE_{i}_u"] for i in range(self.nb_CSE)])
        ngrids_u = jnp.sort(ngrids_u)  # Sort to ensure monotonic grid
        cs2grids = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE)])

        # Convert from normalized positions [0,1] to physical densities [nbreak, nmax]
        width = self.nmax - nbreak
        ngrids = nbreak + ngrids_u * width

        # Append final grid point at nmax
        ngrids = jnp.append(ngrids, jnp.array([self.nmax]))
        cs2grids = jnp.append(cs2grids, jnp.array([params[f"cs2_CSE_{self.nb_CSE}"]]))

        # Construct the metamodel part using the pre-instantiated metamodel
        # This gives us the full range up to nmax
        mm_output = self.metamodel.construct_eos(params)

        # Convert units back for interpolation
        n_metamodel_full = mm_output.ns / utils.fm_inv3_to_geometric
        p_metamodel_full = mm_output.ps / utils.MeV_fm_inv3_to_geometric
        e_metamodel_full = mm_output.es / utils.MeV_fm_inv3_to_geometric
        # MetaModel guarantees mu is populated
        mu_metamodel_full: Float[Array, "n_points"] = mm_output.mu  # type: ignore[assignment]
        cs2_metamodel_full = mm_output.cs2

        # Re-interpolate to a fixed-size array up to nbreak
        # This maintains JAX compatibility while allowing variable nbreak
        n_metamodel = jnp.linspace(
            n_metamodel_full[0], nbreak, self.ndat_metamodel, endpoint=True
        )
        p_metamodel = jnp.interp(n_metamodel, n_metamodel_full, p_metamodel_full)
        e_metamodel = jnp.interp(n_metamodel, n_metamodel_full, e_metamodel_full)
        mu_metamodel = jnp.interp(n_metamodel, n_metamodel_full, mu_metamodel_full)
        cs2_metamodel = jnp.interp(n_metamodel, n_metamodel_full, cs2_metamodel_full)

        # Get values at break density
        p_break = jnp.interp(nbreak, n_metamodel, p_metamodel)
        e_break = jnp.interp(nbreak, n_metamodel, e_metamodel)
        mu_break = jnp.interp(nbreak, n_metamodel, mu_metamodel)
        cs2_break = jnp.interp(nbreak, n_metamodel, cs2_metamodel)

        # Define the speed-of-sound interpolation of the extension portion
        ngrids = jnp.concatenate((jnp.array([nbreak]), ngrids))
        cs2grids = jnp.concatenate((jnp.array([cs2_break]), cs2grids))
        cs2_extension_function = lambda n: jnp.interp(n, ngrids, cs2grids)

        # Compute n, p, e for CSE (number densities in unit of fm^-3)
        n_CSE = jnp.logspace(jnp.log10(nbreak), jnp.log10(self.nmax), num=self.ndat_CSE)
        cs2_CSE = cs2_extension_function(n_CSE)

        # We add a very small number to avoid problems with duplicates below
        mu_CSE = mu_break * jnp.exp(utils.cumtrapz(cs2_CSE / n_CSE, n_CSE)) + 1e-6
        p_CSE = p_break + utils.cumtrapz(cs2_CSE * mu_CSE, n_CSE) + 1e-6
        e_CSE = e_break + utils.cumtrapz(mu_CSE, n_CSE) + 1e-6

        # Combine metamodel and CSE data
        n = jnp.concatenate((n_metamodel, n_CSE))
        p = jnp.concatenate((p_metamodel, p_CSE))
        e = jnp.concatenate((e_metamodel, e_CSE))

        mu = jnp.concatenate((mu_metamodel, mu_CSE))
        cs2 = jnp.concatenate((cs2_metamodel, cs2_CSE))

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        return EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=cs2,
            mu=mu,
            extra_constraints=None,
        )

    def get_required_parameters(self) -> list[str]:
        """
        Return list of parameters required by MetaModel with CSE.

        Returns:
            list[str]: NEP parameters + nbreak + individual CSE grid point parameters
                ["E_sat", "K_sat", "Q_sat", "Z_sat", "E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym",
                 "nbreak", "n_CSE_0_u", "cs2_CSE_0", ..., "n_CSE_{nb_CSE-1}_u", "cs2_CSE_{nb_CSE-1}",
                 "cs2_CSE_{nb_CSE}"]

        Note:
            The individual grid point parameters (n_CSE_i_u, cs2_CSE_i) are converted
            internally by construct_eos() to arrays (ngrids, cs2grids).
        """
        # NEP parameters
        params = [
            "E_sat",
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
            "nbreak",
        ]
        # CSE grid point parameters
        for i in range(self.nb_CSE):
            params.extend([f"n_CSE_{i}_u", f"cs2_CSE_{i}"])
        # Final cs2 value at nmax
        params.append(f"cs2_CSE_{self.nb_CSE}")

        return params
