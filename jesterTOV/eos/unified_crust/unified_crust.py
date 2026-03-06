r"""Unified neutron star crust equation of state.

This module implements a thermodynamically consistent unified EOS where both the
crust and core are derived from the same nuclear empirical parameters (NEPs). It
extends the existing :class:`~jesterTOV.eos.metamodel.MetaModel_EOS_model` to lower
densities, replacing the fixed pre-tabulated crust (DH/BPS) with:

- **Outer crust** (n < neutron drip ≈ 2.6×10⁻⁴ fm⁻³): BSk22 or BSk24 Pearson
  analytical fit (fixed — outer crust changes negligibly with NEPs).
- **Inner crust + core** (n > neutron drip): metamodel Taylor expansion around
  saturation density, using the same NEPs as the core. A short spline transition
  connects the BSk outer crust to the metamodel.

This approach approximates the Compressible Liquid Drop Model (CLDM) used by the
CUTER tool. It is fully JAX-traceable and JIT-compilable, enabling efficient
Bayesian inference.

**Physics approximation:** In the inner crust, the bulk metamodel (homogeneous
nuclear matter) is used instead of the full CLDM (Wigner–Seitz cells with surface
and Coulomb corrections). This slightly overestimates the pressure in the inner
crust but has a negligible effect on neutron star radii (expected bias < 0.1 km).

**Reference:** Margueron et al., Phys. Rev. C 103, 045803 (2021)

**CUTER cross-check:** Use :class:`UnifiedCrustEOS_CUTER` (in ``cuter_wrapper.py``)
for validation against the full CUTER calculation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jesterTOV.eos.metamodel.base import MetaModel_EOS_model
from jesterTOV.eos.unified_crust.bsk_outer_crust import (
    build_outer_crust_table,
    get_neutron_drip_density,
)
from jesterTOV.tov.data_classes import EOSData
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

# Inner-crust start: nmin_MM is set just above the BSk outer crust limit, with a
# small buffer so the spline connection region has room to interpolate.
_INNER_CRUST_BUFFER_FM3 = 5e-4  # fm^-3; distance above n_drip where metamodel begins


class UnifiedCrustEOS_MetaModel(MetaModel_EOS_model):
    r"""Unified neutron star EOS with self-consistent crust from nuclear parameters.

    Extends :class:`~jesterTOV.eos.metamodel.MetaModel_EOS_model` by replacing
    the fixed pre-tabulated crust (DH/BPS) with:

    - A BSk22 or BSk24 Pearson analytical fit for the outer crust (n < n_drip)
    - The same metamodel Taylor expansion for the inner crust (n > n_drip) and core

    This provides a thermodynamically consistent EOS where the crust properties are
    determined by the same NEPs that describe the dense core, rather than fixed tables
    from a specific nuclear model.

    All computations are JAX-traceable and JIT-compilable. The BSk outer crust table
    is evaluated once at initialization and stored as frozen JAX arrays.

    Parameters
    ----------
    bsk_outer_crust : int
        BSk model number for the outer crust Pearson fit (22 or 24, default 24).
    ndat_outer : int
        Number of density points in the BSk outer crust table (default 50).
    nsat : float
        Nuclear saturation density [fm⁻³] (default 0.16).
    nmax_nsat : float
        Maximum density in units of n_sat (default 12).
    ndat_core : int
        Number of metamodel density grid points for the inner crust + core (default 200).
    ndat_spline : int
        Number of spline interpolation points connecting outer crust to metamodel (default 10).
    kappas : tuple
        Meta-model kinetic energy correction coefficients (default all zero).
    v_nq : list[float]
        Quartic isospin coefficients (default all zero).
    b_sat : float
        Saturation potential cutoff parameter (default 17.0).
    b_sym : float
        Symmetry potential cutoff parameter (default 25.0).

    Examples
    --------
    >>> eos = UnifiedCrustEOS_MetaModel(bsk_outer_crust=24, ndat_outer=50, ndat_core=200)
    >>> params = {
    ...     "E_sat": -16.05, "K_sat": 245.5, "Q_sat": -400.0, "Z_sat": 0.0,
    ...     "E_sym": 30.0, "L_sym": 46.4, "K_sym": -145.0, "Q_sym": 0.0, "Z_sym": 0.0,
    ... }
    >>> eos_data = eos.construct_eos(params)
    >>> print(eos_data.cs2.shape)

    Notes
    -----
    The outer crust table (BSk24) ends at the neutron drip density:
    - BSk24: 2.56×10⁻⁴ fm⁻³
    - BSk22: 2.69×10⁻⁴ fm⁻³

    The metamodel begins at n_drip + buffer (≈ 7.6×10⁻⁴ fm⁻³ for BSk24), with
    a 10-point cubic spline smoothly connecting the two regions. The spline uses the
    cs² values from both boundaries to ensure a smooth transition.

    For cross-validation against the full CLDM calculation, use
    :class:`~jesterTOV.eos.unified_crust.cuter_wrapper.UnifiedCrustEOS_CUTER`.
    """

    def __init__(
        self,
        bsk_outer_crust: int = 24,
        ndat_outer: Int = 50,
        # Core/inner crust metamodel parameters
        nsat: Float = 0.16,
        nmax_nsat: Float = 12.0,
        ndat_core: Int = 200,
        ndat_spline: Int = 10,
        kappas: tuple[Float, Float, Float, Float, Float, Float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        v_nq: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0],
        b_sat: Float = 17.0,
        b_sym: Float = 25.0,
    ) -> None:
        """Initialize unified crust EOS.

        Builds the BSk outer crust table (numpy, evaluated once at init) and
        configures the metamodel to cover the inner crust + core density range.

        Parameters
        ----------
        bsk_outer_crust : int
            BSk model number (22 or 24)
        ndat_outer : int
            Points in BSk outer crust table
        nsat : float
            Saturation density [fm⁻³]
        nmax_nsat : float
            Max density in units of nsat
        ndat_core : int
            Metamodel grid points (inner crust + core)
        ndat_spline : int
            Spline connection points between outer crust and metamodel
        kappas : tuple
            Metamodel kinetic correction coefficients
        v_nq : list[float]
            Quartic isospin coefficients
        b_sat : float
            Saturation cutoff parameter
        b_sym : float
            Symmetry cutoff parameter
        """
        if bsk_outer_crust not in (22, 24):
            raise ValueError(f"bsk_outer_crust must be 22 or 24, got {bsk_outer_crust}")

        self.bsk_outer_crust = bsk_outer_crust
        self.ndat_outer = ndat_outer

        # Neutron drip density for chosen BSk model [fm^-3]
        n_drip = get_neutron_drip_density(bsk_outer_crust)

        # The metamodel starts just above n_drip, leaving room for spline connection
        nmin_MM = n_drip + _INNER_CRUST_BUFFER_FM3
        nmin_MM_nsat = nmin_MM / nsat

        # Call parent __init__ with nmin_MM just above n_drip and a temporary crust.
        # We use max_n_crust_nsat just below nmin_MM so the initial connection region
        # is valid (increasing from max_n_crust to nmin_MM). The crust data will be
        # overridden below with BSk values.
        max_n_crust_nsat_temp = (n_drip * 1.01) / nsat  # just above n_drip

        super().__init__(
            kappas=kappas,
            v_nq=v_nq,
            b_sat=b_sat,
            b_sym=b_sym,
            nsat=nsat,
            nmin_MM_nsat=nmin_MM_nsat,
            nmax_nsat=nmax_nsat,
            ndat=ndat_core,
            crust_name="BPS",  # loaded then overridden; BPS is the lightest table
            max_n_crust_nsat=max_n_crust_nsat_temp,
            min_n_crust_nsat=2e-13,
            ndat_spline=ndat_spline,
        )

        # Build BSk outer crust table (numpy, once, static)
        n_outer_min = 1e-11  # fm^-3 (deep outer crust, near Fe-56 ground state)
        n_B, p_bsk, e_bsk, cs2_bsk, mub_bsk = build_outer_crust_table(
            nbmin=n_outer_min,
            nbmax=n_drip,
            ndim=ndat_outer,
            bsk=bsk_outer_crust,
        )

        # Override the crust attributes loaded by the parent with BSk data
        self.ns_crust: Float[Array, "n_crust"] = jnp.array(n_B)
        self.ps_crust: Float[Array, "n_crust"] = jnp.array(p_bsk)
        self.es_crust: Float[Array, "n_crust"] = jnp.array(e_bsk)
        self.cs2_crust: Float[Array, "n_crust"] = jnp.array(cs2_bsk)
        self.mu_lowest: Float = float(
            mub_bsk[0]
        )  # baryon chem. potential at lowest density
        self.max_n_crust: Float = float(n_B[-1])  # = n_drip

        # Recompute spline connection arrays with the correct BSk boundary
        # n_connection runs from just above n_drip to nmin_MM (= n_drip + buffer)
        self.ns_spline: Float[Array, "n_spline"] = jnp.append(
            self.ns_crust, self.n_metamodel
        )
        self.n_connection: Float[Array, "n_conn"] = jnp.linspace(
            self.max_n_crust + 1e-5,
            self.nmin_MM,
            self.ndat_spline,
            endpoint=False,
        )

        logger.debug(
            f"UnifiedCrustEOS_MetaModel initialized: "
            f"BSk{bsk_outer_crust} outer crust [{n_outer_min:.1e}, {n_drip:.2e}] fm^-3, "
            f"metamodel [{nmin_MM:.2e}, {nmax_nsat * nsat:.2f}] fm^-3, "
            f"mu_lowest={self.mu_lowest:.3f} MeV"
        )

    def construct_eos(self, params: dict[str, float]) -> EOSData:
        r"""Construct the unified crust EOS with correct pressure integration.

        The parent's :meth:`~MetaModel_EOS_model.construct_eos` uses
        :func:`~jesterTOV.utils.cumtrapz` which prepends ``1e-30`` as the
        first element (to avoid ``log(0)``). For the BSk outer crust, all
        pressures lie in the range ``[10^{-57}, 3 \times 10^{-37}]``
        MeV fm⁻³ — far below ``1e-30`` — so the prepend would make
        ``p[0] \approx 1e{-30} \gg p[1]``, breaking strict monotonicity.

        This override calls the parent to obtain the correctly computed
        :math:`c_s^2` array, then re-integrates pressure and energy from the
        BSk outer-crust boundary (n_drip) using a standard trapezoidal sum
        (first element exactly zero, not ``1e-30``).

        Parameters
        ----------
        params : dict[str, float]
            Nuclear empirical parameters (same 9 NEPs as the standard
            metamodel).

        Returns
        -------
        EOSData
            Complete EOS with monotone pressure, BSk outer crust energies,
            and metamodel inner crust + core.
        """
        from jesterTOV.tov.data_classes import EOSData

        # ── Step 1: call parent to get the correctly computed cs2 ────────────
        # The parent already uses the overridden BSk crust attributes
        # (ns_crust, cs2_crust, n_connection, n_metamodel, …). Its pressure
        # array is wrong (1e-30 prepend issue), but cs2 is correct.
        parent_eos = super().construct_eos(params)

        nout = len(self.ns_crust)

        # cs2 for connection + metamodel (in order)
        cs2_inner = parent_eos.cs2[nout:]  # type: ignore[index]

        # ── Step 2: outer crust — use BSk table directly ─────────────────────
        p_outer = self.ps_crust  # [MeV fm⁻³]
        e_outer = self.es_crust  # [MeV fm⁻³]
        mu_outer = (p_outer + e_outer) / self.ns_crust  # [MeV]

        # ── Step 3: inner region (connection + core) ─────────────────────────
        # Integrate starting from the last BSk point (n_drip) so that there
        # is no flat or non-monotone pressure step at the join.
        n_inner = jnp.concatenate([self.n_connection, self.n_metamodel])

        # Extend cs2 and n to include n_drip as the left boundary
        n_ext = jnp.concatenate([self.ns_crust[-1:], n_inner])  # len = 1 + n_inner
        cs2_ext = jnp.concatenate([self.cs2_crust[-1:], cs2_inner])  # same length

        mu_start = mu_outer[-1]  # baryon chem. potential at n_drip [MeV]
        p_start = p_outer[-1]  # pressure at n_drip [MeV fm⁻³]

        # Integrate d(log mu) = cs2 * d(log n) using the trapezoid rule.
        # Result has length len(n_inner) (excludes the left boundary point).
        log_n = jnp.log(n_ext)
        dlog_mu = jnp.diff(log_n) * (cs2_ext[:-1] + cs2_ext[1:]) / 2.0
        log_mu_inner = jnp.log(mu_start) + jnp.cumsum(dlog_mu)
        mu_inner = jnp.exp(log_mu_inner)

        # Integrate dp = cs2 * mu * dn using the trapezoid rule.
        # mu_ext[0] = mu_start ensures a proper half-step from n_drip.
        mu_ext = jnp.concatenate([jnp.array([mu_start]), mu_inner])
        dn = jnp.diff(n_ext)
        dp = dn * (cs2_ext[:-1] * mu_ext[:-1] + cs2_ext[1:] * mu_ext[1:]) / 2.0
        p_inner = p_start + jnp.cumsum(dp)  # strictly increasing
        e_inner = mu_inner * n_inner - p_inner

        # ── Step 4: assemble full arrays and convert units ────────────────────
        n_full = jnp.concatenate([self.ns_crust, n_inner])
        p_full = jnp.concatenate([p_outer, p_inner])
        e_full = jnp.concatenate([e_outer, e_inner])
        mu_full = jnp.concatenate([mu_outer, mu_inner])

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n_full, p_full, e_full)

        return EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=parent_eos.cs2,  # reuse the correctly computed cs2
            mu=mu_full,
            extra_constraints=None,
        )

    def get_required_parameters(self) -> list[str]:
        """Return the nuclear empirical parameters required by this EOS.

        Returns the same 9 NEPs as the standard
        :class:`~jesterTOV.eos.metamodel.MetaModel_EOS_model`.

        Returns
        -------
        list[str]
            ``["E_sat", "K_sat", "Q_sat", "Z_sat", "E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym"]``
        """
        return [
            "E_sat",
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]
