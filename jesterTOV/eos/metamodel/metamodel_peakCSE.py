r"""Meta-model EOS with Gaussian peak Constant Speed-of-sound Extensions (peakCSE)."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ... import utils
from ..base import Interpolate_EOS_model
from .metamodel import MetaModel_EOS_model


class MetaModel_with_peakCSE_EOS_model(Interpolate_EOS_model):
    r"""
    Meta-model EOS with Gaussian peak Constant Speed-of-sound Extensions (peakCSE).

    This class implements a sophisticated CSE parametrization based on the peakCSE model,
    which combines a Gaussian peak structure with logistic growth to model phase transitions
    while ensuring asymptotic consistency with perturbative QCD (pQCD) at the highest densities.

    **Mathematical Framework:**
    The speed of sound squared is parametrized as:

    .. math::
        c^2_s &= c^2_{s,{\rm break}} + \frac{\frac{1}{3} - c^2_{s,{\rm break}}}{1 + e^{-l_{\rm sig}(n - n_{\rm sig})}} + c^2_{s,{\rm peak}}e^{-\frac{1}{2}\left(\frac{n - n_{\rm peak}}{\sigma_{\rm peak}}\right)^2}

    **Reference:** Greif:2018njt, arXiv:1812.08188

    Note:
        The peakCSE model provides greater physical realism than simple piecewise-constant
        CSE by incorporating smooth transitions and theoretically motivated high-density behavior.
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
        Initialize the MetaModel with peakCSE extensions for realistic phase transition modeling.

        This constructor sets up the peakCSE model that combines meta-model physics at
        low-to-intermediate densities with sophisticated Gaussian peak + logistic growth
        extensions at high densities, designed to model phase transitions while maintaining
        consistency with perturbative QCD predictions.

        Args:
            nsat (Float, optional):
                Nuclear saturation density :math:`n_0` [:math:`\mathrm{fm}^{-3}`].
                Reference density for the meta-model construction. Defaults to 0.16.
            nmin_MM_nsat (Float, optional):
                Starting density for meta-model region as fraction of :math:`n_0`.
                Must be above crust-core transition. Defaults to 0.75 (= 0.12/0.16).
            nmax_nsat (Float, optional):
                Maximum density for EOS construction in units of :math:`n_0`.
                Should extend to densities where pQCD limit is approached. Defaults to 12.
            ndat_metamodel (Int, optional):
                Number of density points for meta-model region discretization.
                Higher values give smoother meta-model interpolation. Defaults to 100.
            ndat_CSE (Int, optional):
                Number of density points for peakCSE region discretization.
                Controls resolution of phase transition and pQCD approach modeling. Defaults to 100.
            **metamodel_kwargs:
                Additional keyword arguments passed to the underlying MetaModel_EOS_model.
                Includes parameters like kappas, v_nq, b_sat, b_sym, crust_name, etc.
                See MetaModel_EOS_model.__init__ for complete parameter descriptions.

        See Also:
            MetaModel_EOS_model.__init__ : Base meta-model parameters
            construct_eos : Method that defines peakCSE parameters and break density
        """

        # TODO: align with new metamodel code
        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_MM_nsat = nmin_MM_nsat
        self.ndat_metamodel = ndat_metamodel
        self.metamodel_kwargs = metamodel_kwargs

    def construct_eos(self, NEP_dict: dict, peakCSE_dict: dict):
        r"""
        Construct the complete EOS using meta-model + peakCSE extensions.

        This method builds the full EOS by combining the meta-model approach with
        peakCSE extensions that model phase transitions through Gaussian peaks
        and approach the pQCD conformal limit at high densities.

        Args:
            NEP_dict (dict): Nuclear empirical parameters for meta-model construction.
                Must include 'nbreak' key specifying the transition density between
                meta-model and peakCSE regions. See MetaModel_EOS_model.construct_eos
                for complete NEP parameter descriptions.
            peakCSE_dict (dict): peakCSE model parameters defining the high-density behavior:

                - **gaussian_peak** (float): Amplitude :math:`A` of the Gaussian peak
                - **gaussian_mu** (float): Peak location :math:`\mu` [:math:`\mathrm{fm}^{-3}`]
                - **gaussian_sigma** (float): Peak width :math:`\sigma` [:math:`\mathrm{fm}^{-3}`]
                - **logit_growth_rate** (float): Growth rate :math:`k` for pQCD approach
                - **logit_midpoint** (float): Midpoint density :math:`n_{\mathrm{mid}}` for logistic transition

        Returns:
            tuple: Complete EOS data containing:

                - **ns**: Number densities [geometric units]
                - **ps**: Pressures [geometric units]
                - **hs**: Specific enthalpies [geometric units]
                - **es**: Energy densities [geometric units]
                - **dloge_dlogps**: Logarithmic derivative :math:`\frac{d\ln\varepsilon}{d\ln p}`
                - **mu**: Chemical potential [geometric units]
                - **cs2**: Speed of sound squared including peakCSE structure

        Note:
            The peakCSE speed of sound follows:
            :math:`c^2_s &= c^2_{s,{\rm break}} + \frac{\frac{1}{3} - c^2_{s,{\rm break}}}{1 + e^{-l_{\rm sig}(n - n_{\rm sig})}} + c^2_{s,{\rm peak}}e^{-\frac{1}{2}\left(\frac{n - n_{\rm peak}}{\sigma_{\rm peak}}\right)^2}`

            This ensures smooth transitions, realistic phase transition modeling,
            and asymptotic consistency with the pQCD conformal limit :math:`c_s^2 = 1/3`.
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

        # Define the speed-of-sound of the extension portion
        # the model is taken from arXiv:1812.08188
        # but instead of energy density, I am using density as the input
        offset = self.offset_calc(NEP_dict["nbreak"], cs2_break, peakCSE_dict)
        cs2_extension_function = lambda x: (
            peakCSE_dict["gaussian_peak"]
            * jnp.exp(
                -0.5
                * (
                    (x - peakCSE_dict["gaussian_mu"]) ** 2
                    / peakCSE_dict["gaussian_sigma"] ** 2
                )
            )
            + offset
            + (
                (1.0 / 3.0 - offset)
                / (
                    1.0
                    + jnp.exp(
                        -peakCSE_dict["logit_growth_rate"]
                        * (x - peakCSE_dict["logit_midpoint"])
                    )
                )
            )
        )
        # Compute n, p, e for peakCSE (number densities in unit of fm^-3)
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

    def offset_calc(self, nbreak, cs2_break, peakCSE_dict):
        gaussian_part = peakCSE_dict["gaussian_peak"] * jnp.exp(
            -0.5
            * (nbreak - peakCSE_dict["gaussian_mu"]) ** 2
            / peakCSE_dict["gaussian_sigma"] ** 2
        )
        exp_part = jnp.exp(
            -peakCSE_dict["logit_growth_rate"]
            * (nbreak - peakCSE_dict["logit_midpoint"])
        )
        offset = ((1.0 + exp_part) * (cs2_break - gaussian_part) - 1.0 / 3.0) / exp_part
        return offset
