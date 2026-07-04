r"""MIT bag model equation of state for self-bound strange quark matter.

This implements the simplest phenomenological quark-matter equation of state,
commonly used for strange quark stars: unpaired, massless, non-interacting quarks
confined by a constant vacuum energy density (the "bag constant" :math:`B`), giving
the linear relation

.. math::
    p = \frac{\varepsilon - 4B}{3}.

See Chodos, Jaffe, Johnson, Thorn and Weisskopf, *"New extended model of hadrons"*,
Phys. Rev. D 9, 3471 (1974).

This is a JAX port of ``CompactObject/EOSgenerators/MITbag_EOS.py``'s
``MITbag_compute_EOS``. Unlike the RMF or metamodel EOS, this model has no crust:
strange quark matter is self-bound (:math:`p=0` at :math:`\varepsilon=4B`, the stellar
surface), so there is no separate crust to stitch on -- the whole star, all the way
to the surface, is described by this one relation.

Note on scope: the reference implementation has no independent baryon-number-density
variable -- it only ever tabulates :math:`(\varepsilon, p)`. jester's ``EOSData``
interface needs a number-density array to serve as the interpolation grid (used e.g.
for the ``min_nsat_TOV`` central-density starting point, see ``tov/base.py``) and a
pseudo-enthalpy array (used by the TOV solvers directly, see e.g. ``tov/gr.py``).
Both are reconstructed here from the :math:`T=0` thermodynamic identities

.. math::
    dh = \frac{dp}{\varepsilon+p}, \qquad n \propto (\varepsilon+p)\,e^{-h},

which integrate in closed form for this EOS's linear :math:`p(\varepsilon)` relation
to :math:`h(\varepsilon) = \tfrac{1}{4}\ln[(\varepsilon-B)/(\varepsilon_0-B)]` and
:math:`n \propto (\varepsilon-B)^{3/4}` (see the derivation in
``rmf_eos/FINDINGS.md`` in the companion validation repository). The closed form is
used instead of numerically integrating :func:`jesterTOV.utils.cumtrapz` against
:math:`\log(p)` (as e.g. ``spectral_decomposition.py`` does for its high-density
branch) because :math:`p\to 0` at this model's self-bound surface, and the resulting
huge, unevenly spaced swings in :math:`\log(p)` make that numerical integral
inaccurate right where it matters most. The overall normalization of :math:`n` is
fixed by the bookkeeping convention :math:`n = (\varepsilon+p)/m_N` (average nucleon
mass) at the surface; this is not a physical baryon count for deconfined quark
matter, just a well-behaved, monotonic grid variable.
"""

import jax.numpy as jnp
from jaxtyping import Float

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.tov.data_classes import EOSData


class MITBag_EOS_model(Interpolate_EOS_model):
    r"""
    MIT bag model equation of state for self-bound strange quark matter.

    Massless, non-interacting quark matter confined by a constant vacuum energy
    density (bag constant) :math:`B`, giving the linear barotropic relation
    :math:`p = (\varepsilon - 4B)/3`. The stellar surface is at :math:`p=0`,
    i.e. :math:`\varepsilon = 4B` -- there is no crust, since strange quark matter
    is self-bound rather than requiring gravity to hold it together at low density.

    **Reference:**

    Chodos, A., Jaffe, R.L., Johnson, K., Thorn, C.B. & Weisskopf, V.F.,
    *"New extended model of hadrons"*, Phys. Rev. D 9, 3471 (1974).
    """

    def __init__(
        self,
        n_points: int = 1000,
        e_max_over_b: Float = 10.0,
    ):
        r"""
        Args:
            n_points: Number of energy-density grid points spanning
                :math:`[4B, e_{\max\_over\_b}\,B]` (matches the reference
                implementation's 1000).
            e_max_over_b: Upper end of the energy-density grid, in units of the bag
                constant :math:`B` (matches the reference implementation's 10).
        """
        self.n_points = n_points
        self.e_max_over_b = e_max_over_b

    def get_required_parameters(self) -> list[str]:
        r"""
        Return the single MIT bag model parameter.

        Returns:
            list[str]: ``["B"]`` -- the bag constant [:math:`\mathrm{MeV}\,\mathrm{fm}^{-3}`].
        """
        return ["B"]

    def construct_eos(self, params: dict[str, float]) -> EOSData:
        r"""
        Construct the full EOS from the bag constant.

        Args:
            params: Dictionary with key ``B`` (bag constant, :math:`\mathrm{MeV}\,\mathrm{fm}^{-3}`).

        Returns:
            EOSData: Complete EOS with all required arrays in geometric units.
        """
        B = params["B"]

        # Offset the starting point a hair above the exact surface (e=4B, p=0):
        # downstream TOV code (e.g. tov/base.py's pc_min lookup) interpolates in
        # log(p), which is singular at p=0 even though the closed-form n/hs below
        # are perfectly well-behaved there.
        e_min = 4.0 * B * (1.0 + 1e-8)
        e = jnp.linspace(e_min, self.e_max_over_b * B, self.n_points)
        p = (e - 4.0 * B) / 3.0

        # Closed-form pseudo-enthalpy and rest-mass-density grid (see module
        # docstring). e - B is always strictly positive on this grid (e >= 4B > B),
        # so this is singularity-free, unlike the generic log(p)-based integral.
        n_min = (e_min + p[0]) / utils.m
        # jnp.maximum (not a "+1e-30" additive offset) is essential here: the ratio
        # at i=0 is exactly 1 in exact arithmetic (log=0), but XLA can fuse/reorder
        # the (e-B) and (e_min-B) subtractions differently under jit than eager
        # execution, so the computed ratio can land a few ULPs below 1 and give
        # log(ratio) a small *negative* value (~-1e-16) that would silently swamp a
        # tiny additive epsilon -- jester's `hs > 0` validity check would then poison
        # the entire EOS table with NaN for a compilation-path-dependent fraction of
        # parameter draws (see rmf_eos/FINDINGS.md for the full diagnosis).
        hs = jnp.maximum(1e-30, 0.25 * jnp.log((e - B) / (e_min - B)))
        n = n_min * ((e - B) / (e_min - B)) ** 0.75

        # hs from interpolate_eos is discarded: its generic cumtrapz-against-log(p)
        # integral is inaccurate right at this model's p=0 surface (see module
        # docstring); the closed-form hs above replaces it.
        ns, ps, _, es, dloge_dlogps = self.interpolate_eos(n, p, e)
        cs2 = jnp.full_like(ps, 1.0 / 3.0)

        return EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=cs2,
            mu=None,
            extra_constraints=None,
        )
