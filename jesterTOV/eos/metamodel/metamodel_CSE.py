r"""Meta-model EOS with piecewise constant speed-of-sound extensions (CSE)."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.metamodel.base import MetaModel_EOS_model


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
        max_nbreak_nsat: Float | None = None,
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
        NEP_dict: dict,
        ngrids: Float[Array, "n_grid_point"],
        cs2grids: Float[Array, "n_grid_point"],
    ) -> tuple:
        r"""
        Construct the EOS by combining metamodel and CSE regions.

        This method constructs the full EOS by:
        1. Building the metamodel EOS up to the full nmax range
        2. Interpolating the metamodel to a fixed-size grid up to nbreak
        3. Stitching the CSE extension on top from nbreak to nmax

        Args:
            NEP_dict (dict): Dictionary with the NEP keys to be passed to the metamodel EOS class.
                Must include 'nbreak' specifying the transition density between metamodel and CSE.
            ngrids (Float[Array, `n_grid_point`]): Density grid points for the CSE part of the EOS.
            cs2grids (Float[Array, `n_grid_point`]): Speed-of-sound squared grid points for the CSE part.

        Returns:
            tuple: EOS quantities (see Interpolate_EOS_model), as well as the chemical potential and speed of sound.

        Note:
            The metamodel instance is reused from __init__ (not re-instantiated) and its output
            is interpolated to the actual nbreak value to maintain JAX compatibility with
            fixed array sizes.
        """

        # Construct the metamodel part using the pre-instantiated metamodel
        # This gives us the full range up to nmax
        mm_output = self.metamodel.construct_eos(NEP_dict)
        (
            n_metamodel_full,
            p_metamodel_full,
            _,
            e_metamodel_full,
            _,
            mu_metamodel_full,
            cs2_metamodel_full,
        ) = mm_output

        # Convert units back for interpolation
        n_metamodel_full = n_metamodel_full / utils.fm_inv3_to_geometric
        p_metamodel_full = p_metamodel_full / utils.MeV_fm_inv3_to_geometric
        e_metamodel_full = e_metamodel_full / utils.MeV_fm_inv3_to_geometric

        # Quick variable definition to nbreak to avoid repeated dictionary lookups
        nbreak = NEP_dict["nbreak"]

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

        # TODO: let's decide whether we want to save cs2 and mu or just use them for computation and then discard them.
        mu = jnp.concatenate((mu_metamodel, mu_CSE))
        cs2 = jnp.concatenate((cs2_metamodel, cs2_CSE))

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        return ns, ps, hs, es, dloge_dlogps, mu, cs2
