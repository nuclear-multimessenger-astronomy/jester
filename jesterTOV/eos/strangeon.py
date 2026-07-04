r"""Strangeon matter equation of state for self-bound strangeon stars.

This implements the strangeon-matter equation of state used to describe compact
stars made of "strangeons" -- clusters of :math:`N_q` (anti-)up, down and strange
quarks bound by a Lennard-Jones-like interatomic potential, treated as a solid/
quantum-crystal phase of cold quark matter rather than a quark-lepton Fermi gas
(unlike, e.g., the MIT bag model in ``jesterTOV/eos/mit_bag.py``).

This is a JAX port of ``CompactObject/EOSgenerators/Strangeon_EOS.py``'s
``compute_EOS``.

**Reference:**

Lai, X.Y., Xu, R.X., *"A note on strangeon star"*, and the strangeon equation of
state as parametrized in CompactObject's own examples
(``Test_Case/test_Bayesian_inference_Strangeon_EOS.ipynb``), which cites Lai & Xu's
strangeon-matter framework fit with a 12-6 Lennard-Jones-type potential
(:math:`A_{12}, A_6` below).

Note on scope (units and variable convention, matching the reference implementation
exactly):

- The independent variable ``n`` in :func:`compute_EOS` is the number density of
  *strangeons* (clusters), not baryons -- each strangeon carries baryon number
  :math:`N_q/3` (3 quarks per baryon). ``ns``, one of the three free parameters, is
  by contrast a *baryon* number density (the surface baryon density), following the
  reference implementation's own docstring. This mismatch is inherited as-is from
  CompactObject; :meth:`StrangeonEOS_model.construct_eos` converts the strangeon
  density grid to baryon density (multiplying by :math:`N_q/3`) before handing it to
  jester's ``EOSData.ns``, since that field is used elsewhere (e.g. ``min_nsat_TOV``)
  as a physical baryon density.
- :math:`N_q` (quarks per strangeon) is an integer model-selection choice, not a
  continuously-sampled Bayesian parameter, matching the reference implementation and
  its own inference notebooks (which fix ``Nq=18`` and only sample ``epsilon``,
  ``ns``). It is therefore a constructor argument here, not a ``construct_eos``
  parameter.
"""

import jax.numpy as jnp

from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.tov.data_classes import EOSData

# Fixed Lennard-Jones-type potential coefficients and quark mass, matching
# CompactObject/EOSgenerators/Strangeon_EOS.py exactly.
_A12 = 6.2
_A6 = 8.4
_MQ = 300.0  # Quark mass [MeV]


class StrangeonEOS_model(Interpolate_EOS_model):
    r"""
    Strangeon matter equation of state for self-bound strangeon stars.

    Clusters of :math:`N_q` quarks ("strangeons") interacting via a 12-6
    Lennard-Jones-type potential, in the zero-temperature, zero-pressure-surface
    limit (self-bound matter, like the MIT bag model -- no crust).

    .. math::
        \varepsilon(n) &= 2\epsilon\left(A_{12}\sigma^4 n^5 - A_6\sigma^2 n^3\right)
            + n N_q m_q \\
        p(n) &= 4\epsilon\left(2A_{12}\sigma^4 n^5 - A_6\sigma^2 n^3\right)

    where :math:`\sigma = \sqrt{A_6/(2A_{12})}\,(N_q/(3n_s))`, :math:`n` is the
    strangeon number density, :math:`\epsilon` is the potential well depth, and
    :math:`n_s` is the baryon number density at the star's surface.

    **Reference:** Lai, X.Y. & Xu, R.X., strangeon star equation of state, as used
    in CompactObject's ``EOSgenerators/Strangeon_EOS.py``.
    """

    def __init__(self, Nq: int = 18, n_points: int = 2000):
        r"""
        Args:
            Nq: Number of quarks per strangeon (fixed model-selection choice, not
                sampled -- matches CompactObject's own inference examples, which
                default to ``Nq=18``).
            n_points: Number of strangeon-density grid points (matches the
                reference implementation's 2000).
        """
        self.Nq = Nq
        self.n_points = n_points

    def get_required_parameters(self) -> list[str]:
        r"""
        Return the two sampled strangeon EOS parameters.

        Returns:
            list[str]: ``["epsilon", "ns"]`` -- the potential well depth
            [:math:`\mathrm{MeV}`] and the baryon number density at the star's
            surface [:math:`\mathrm{fm}^{-3}`], respectively. ``Nq`` (quarks per
            strangeon) is a fixed constructor argument, not sampled.
        """
        return ["epsilon", "ns"]

    def construct_eos(self, params: dict[str, float]) -> EOSData:
        r"""
        Construct the full EOS from the strangeon model parameters.

        Args:
            params: Dictionary with keys ``epsilon`` (potential well depth,
                :math:`\mathrm{MeV}`) and ``ns`` (surface baryon number density,
                :math:`\mathrm{fm}^{-3}`); see :meth:`get_required_parameters`.

        Returns:
            EOSData: Complete EOS with all required arrays in geometric units.
        """
        epsilon = params["epsilon"]
        ns = params["ns"]
        Nq = self.Nq

        sigma = jnp.sqrt(_A6 / (2.0 * _A12)) * (Nq / (3.0 * ns))

        # p(n) is the difference of two O(sigma^4 n^5)/O(sigma^2 n^3) terms that are
        # designed to cancel exactly to zero at n=n_min (that is the definition of the
        # surface density). Evaluating exactly at that root is subject to catastrophic
        # cancellation: the residual floating-point noise can land on either side of
        # zero (compilation-path-dependent), and a spuriously negative p[0] poisons
        # this class's log(p)-based derivatives (dloge_dlogps, hence cs2) with NaN.
        # Starting a hair above n_min keeps p[0] comfortably positive and far above the
        # roundoff floor (see rmf_eos/FINDINGS.md for the diagnosis of the analogous
        # MIT bag issue this mirrors).
        n_min = 3.0 * ns / Nq * (1.0 + 1e-6)
        n_max = 0.16 * 8.0 * 3.0 / Nq
        n = jnp.linspace(n_min, n_max, self.n_points)

        e = (
            2.0 * epsilon * (_A12 * sigma**4 * n**5 - _A6 * sigma**2 * n**3)
            + n * Nq * _MQ
        )
        p = 4.0 * epsilon * (2.0 * _A12 * sigma**4 * n**5 - _A6 * sigma**2 * n**3)

        # Strangeon number density -> baryon number density (see module docstring).
        n_baryon = n * (Nq / 3.0)

        ns_out, ps, hs, es, dloge_dlogps = self.interpolate_eos(n_baryon, p, e)
        cs2 = jnp.clip(ps / (es * dloge_dlogps), 1e-6, 1.0)

        return EOSData(
            ns=ns_out,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=cs2,
            mu=None,
            extra_constraints=None,
        )
