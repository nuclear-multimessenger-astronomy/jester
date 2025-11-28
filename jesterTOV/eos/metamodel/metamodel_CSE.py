r"""Meta-model EOS with piecewise constant speed-of-sound extensions (CSE)."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ... import utils
from ..base import Interpolate_EOS_model
from .metamodel import MetaModel_EOS_model


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
