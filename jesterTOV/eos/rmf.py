r"""Relativistic mean-field (RMF) equation of state for nucleonic neutron star matter.

This implements a non-linear Walecka-type (:math:`\sigma`-:math:`\omega`-:math:`\rho`
meson-exchange, mean-field approximation) relativistic mean-field model for
:math:`npe\mu` (neutron-proton-electron-muon) matter in :math:`\beta`-equilibrium, with
the non-linear self-couplings introduced by Todd-Rutel and Piekarewicz (Phys. Rev. Lett.
95, 122501, 2005, arXiv:nucl-th/0504034), refit as the FSU2 parametrization by Tolos,
Centelles and Ramos (Astrophys. J. 834, 3, 2017, arXiv:1610.00919), and used as a free
10-parameter Bayesian nuclear physics model by Huang, Raaijmakers, Watts, Tolos and
Providência (Mon. Not. Roy. Astron. Soc. 529, 4650, 2024, arXiv:2303.17518).

This is a JAX/``optimistix`` port of the beta-equilibrium branch of
``CompactObject/EOSgenerators/RMF_EOS.py``'s ``compute_EOS`` (the public reference
implementation this model is ported from). See ``rmf_eos/Notes/`` in the companion
validation repository for the full physics derivation and literature trail, and
``rmf_eos/FINDINGS.md`` for the numerical validation against the reference implementation.

Note on scope (deliberate deviations from the reference implementation):

- Equal proton and neutron rest mass is used (matching the reference implementation's
  hardcoded ``m_n = m_p``), rather than jesterTOV.utils's distinct proton/neutron masses,
  so that this can be validated point-for-point against the reference table first.
- The reference implementation's ``compute_EOS`` has an off-by-one bug: it evaluates the
  energy density/pressure using the *previous* density's converged field solution rather
  than the current one. This port intentionally does NOT reproduce that bug -- it evaluates
  thermodynamics at each density's own converged field solution.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float
import optimistix as optx

from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.crust import Crust
from jesterTOV.tov.data_classes import EOSData

# Matches CompactObject/EOSgenerators/RMF_EOS.py exactly (equal-mass convention, see module
# docstring). Values are given in fm^-1.
_M_E = 2.5896041e-3
_M_MU = 0.5354479981
_M_N = 4.7583690772  # equal to proton mass in the reference implementation
_ONEOVERFM_MEV = 197.327053

_PARAM_NAMES = [
    "m_sig",
    "m_w",
    "m_rho",
    "g_sigma",
    "g_omega",
    "g_rho",
    "kappa",
    "lambda_0",
    "zeta",
    "Lambda_w",
]


def _safe_sqrt(x: Array) -> Array:
    r"""``sqrt`` with a NaN-safe gradient at :math:`x \leq 0`.

    :math:`d\sqrt{x}/dx` diverges at :math:`x=0`, which poisons the Newton/Levenberg-
    Marquardt Jacobian even behind a ``jnp.where`` guard, since ``jnp.where`` still
    differentiates both branches. Standard JAX double-``where`` trick: substitute a dummy
    positive value before the ``sqrt`` so the unselected branch's gradient stays finite.
    """
    safe_x = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, jnp.sqrt(safe_x), 0.0)


def _initial_values(rho: Array, theta: Array) -> Array:
    """Analytic leading-order-in-density seed for the field equations, used only to
    initialize the sequential density continuation at its lowest density point."""
    m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, _kappa, _lambda_0, _zeta, Lambda_w = (
        theta
    )
    sigma = g_sigma * rho / (m_sig**2)
    rho_03 = -g_rho * rho / (2.0 * (m_rho**2))
    omega = rho * (
        ((m_w**2) / g_omega + 2.0 * Lambda_w * (g_rho * rho_03) ** 2 * g_omega)
        ** (-1.0)
    )
    m_eff = _M_N - g_sigma * sigma
    mu_n = m_eff + g_omega * omega + g_rho * rho_03 * (-0.5)
    mu_e = 0.12 * _M_E * (rho / 0.1505) ** (2.0 / 3.0)
    return jnp.array([sigma, omega, rho_03, mu_n, mu_e])


def _field_residuals(x: Array, args: tuple[Array, Array]) -> Array:
    r"""Residuals of the mean-field equations of motion for :math:`\sigma,\omega,\rho_{03}`
    plus the baryon-density and charge-neutrality constraints, for :math:`npe\mu` matter in
    :math:`\beta`-equilibrium. Root of this function is the physical solution at a given
    total baryon density ``rho`` for RMF parameters ``theta``. Matches
    ``RMF_EOS.py``'s ``functie`` (which instead squares the residuals for least-squares
    solving); here the raw (unsquared) residuals are used directly with a least-squares
    root finder.
    """
    rho, theta = args
    m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, kappa, lambda_0, zeta, Lambda_w = theta
    sigma, omega, rho_03, mu_n, mu_e = x

    m_eff = _M_N - g_sigma * sigma

    def baryon(mu_b, isospin3):
        E_fb = mu_b - g_omega * omega - g_rho * rho_03 * isospin3
        raw = E_fb**2 - m_eff**2
        E_fb_eff = jnp.where(raw < 0, m_eff, E_fb)
        k_fb = _safe_sqrt(raw)
        rho_B = k_fb**3 / (3.0 * jnp.pi**2)
        rho_SB = (m_eff / (2.0 * jnp.pi**2)) * (
            E_fb_eff * k_fb - m_eff**2 * jnp.log((E_fb_eff + k_fb) / m_eff)
        )
        return rho_B, rho_SB

    mu_p = mu_n - mu_e
    rho_Bp, rho_SBp = baryon(mu_p, 0.5)
    rho_Bn, rho_SBn = baryon(mu_n, -0.5)

    k_fe = _safe_sqrt(mu_e**2 - _M_E**2)
    k_fmu = _safe_sqrt(mu_e**2 - _M_MU**2)

    sum_rho_SB = rho_SBp + rho_SBn
    sum_rho_B = rho_Bp + rho_Bn
    sum3 = 0.5 * rho_Bp - 0.5 * rho_Bn
    charge = rho_Bp - (k_fe**3 / (3.0 * jnp.pi**2)) - (k_fmu**3 / (3.0 * jnp.pi**2))

    r0 = (
        sigma * m_sig**2 / g_sigma
        - sum_rho_SB
        + kappa * (g_sigma * sigma) ** 2 / 2.0
        + lambda_0 * (g_sigma * sigma) ** 3 / 6.0
    )
    r1 = (
        omega * m_w**2 / g_omega
        - sum_rho_B
        + zeta * (g_omega * omega) ** 3 / 6.0
        + 2.0 * Lambda_w * g_omega * omega * (rho_03 * g_rho) ** 2
    )
    r2 = (
        rho_03 * m_rho**2 / g_rho
        - sum3
        + 2.0 * Lambda_w * g_rho * rho_03 * (omega * g_omega) ** 2
    )
    r3 = rho - sum_rho_B
    r4 = charge

    return jnp.array([r0, r1, r2, r3, r4])


def _energy_density_pressure(
    x: Array, rho: Array, theta: Array
) -> tuple[Array, Array, Array]:
    """Energy density, pressure [natural units] and proton fraction at a converged field
    solution ``x``. Matches ``RMF_EOS.py``'s ``Energy_density_Pressure``, evaluated at the
    field solution converged for this same ``rho`` (see module docstring re: the reference
    implementation's off-by-one lag, not reproduced here)."""
    m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, kappa, lambda_0, zeta, Lambda_w = theta
    sigma, omega, rho_03, mu_n, mu_e = x
    m_eff = _M_N - g_sigma * sigma

    def baryon_energy(mu_b, isospin3):
        E_fb = mu_b - g_omega * omega - g_rho * rho_03 * isospin3
        raw = E_fb**2 - m_eff**2
        E_fb_eff = jnp.where(raw < 0, m_eff, E_fb)
        k_fb = _safe_sqrt(raw)
        rho_B = k_fb**3 / (3.0 * jnp.pi**2)
        energy = (1.0 / (8.0 * jnp.pi**2)) * (
            k_fb * E_fb_eff**3
            + k_fb**3 * E_fb_eff
            - jnp.log((k_fb + E_fb_eff) / m_eff) * m_eff**4
        )
        return rho_B, energy

    mu_p = mu_n - mu_e
    rho_Bp, energy_p = baryon_energy(mu_p, 0.5)
    rho_Bn, energy_n = baryon_energy(mu_n, -0.5)

    def lepton_energy(m_l):
        mu_l = mu_e
        k_fl = _safe_sqrt(mu_l**2 - m_l**2)
        rho_l = k_fl**3 / (3.0 * jnp.pi**2)
        energy = (1.0 / (8.0 * jnp.pi**2)) * (
            k_fl * mu_l**3 + mu_l * k_fl**3 - m_l**4 * jnp.log((k_fl + mu_l) / m_l)
        )
        return rho_l, energy

    rho_le, energy_le = lepton_energy(_M_E)
    rho_lmu, energy_lmu = lepton_energy(_M_MU)

    multi = mu_p * rho_Bp + mu_n * rho_Bn + mu_e * rho_le + mu_e * rho_lmu

    sigma_terms = (
        0.5 * (sigma * m_sig) ** 2
        + kappa * (g_sigma * sigma) ** 3 / 6.0
        + lambda_0 * (g_sigma * sigma) ** 4 / 24.0
    )
    omega_terms = 0.5 * (omega * m_w) ** 2 + zeta * (g_omega * omega) ** 4 / 8.0
    rho_terms = (
        0.5 * (rho_03 * m_rho) ** 2
        + 3.0 * Lambda_w * (g_rho * rho_03 * g_omega * omega) ** 2
    )

    energy_density = (
        energy_p
        + energy_n
        + energy_le
        + energy_lmu
        + sigma_terms
        + omega_terms
        + rho_terms
    )
    pressure = multi - energy_density
    proton_fraction = rho_Bp / rho

    return energy_density, pressure, proton_fraction


class RMF_EOS_model(Interpolate_EOS_model):
    r"""
    Non-linear relativistic mean-field (RMF) equation of state for nucleonic
    :math:`\beta`-equilibrium neutron star matter.

    Field-theoretic model of :math:`npe\mu` matter interacting via :math:`\sigma`
    (scalar), :math:`\omega` (vector) and :math:`\rho` (isovector-vector) meson
    exchange in the mean-field approximation, with cubic/quartic scalar self-coupling
    (:math:`\kappa,\lambda_0`), quartic vector self-coupling (:math:`\zeta`), and an
    :math:`\omega`-:math:`\rho` cross-coupling (:math:`\Lambda_w`) that controls the
    density dependence of the symmetry energy independently of the other saturation
    properties.

    **References:**

    - Todd-Rutel, B.G. & Piekarewicz, J., *"Neutron-Rich Nuclei and Neutron Stars: A New
      Accurately Calibrated Interaction for the Study of Neutron-Rich Matter"*,
      Phys. Rev. Lett. 95, 122501 (2005), arXiv:nucl-th/0504034. Introduces the
      :math:`\zeta` and :math:`\Lambda_w` non-linear couplings used here.
    - Tolos, L., Centelles, M. & Ramos, A., *"Equation of State for Nucleonic and
      Hyperonic Neutron Stars with Mass and Radius Constraints"*, Astrophys. J. 834, 3
      (2017), arXiv:1610.00919. FSU2 refit of this Lagrangian for 2-solar-mass neutron
      stars.
    - Huang, C., Raaijmakers, G., Watts, A.L., Tolos, L. & Providência, C.,
      *"Constraining a relativistic mean field model using neutron star mass-radius
      measurements I: nucleonic models"*, Mon. Not. Roy. Astron. Soc. 529, 4650 (2024),
      arXiv:2303.17518. Uses this Lagrangian as a free 10-parameter Bayesian nuclear
      physics model; this class is a JAX port of the reference implementation
      accompanying that inference framework (``CompactObject/EOSgenerators/RMF_EOS.py``).

    The field equations are solved self-consistently at each density via a sequential,
    warm-started Levenberg-Marquardt continuation (:func:`jax.lax.scan` over the density
    grid, each step initialized from the previous density's converged solution) --
    independent per-density solves were found not to converge reliably (see
    ``rmf_eos/FINDINGS.md`` in the companion validation repository), so this mirrors the
    reference implementation's own sequential density-stepping approach, translated to be
    JAX-traceable end to end.
    """

    def __init__(
        self,
        crust_name: str = "Tolos",
        nsat: Float = 0.16,
        max_n_crust_nsat: Float = 0.5,
        min_n_crust_nsat: Float = 2e-13,
        rho_0: Float = 0.1505,
        dt: Float = 0.05,
        ndat_full: int = 124,
        min_core_nsat_rmf: Float = 1.0,
        newton_max_steps: int = 200,
        root_rtol: Float = 1e-10,
        root_atol: Float = 1e-10,
    ):
        r"""
        Args:
            crust_name: Crust model name (e.g. 'Tolos', 'DH', 'BPS') or path to a custom
                .npz file. Defaults to 'Tolos' (Tolos, Centelles & Ramos 2017 outer-crust
                table, see ``jesterTOV/crust_files/convert_crust.py::convert_tolos_crust``)
                rather than jester's usual 'DH' default, because this RMF parametrization
                was fit and validated by CompactObject against the Tolos crust -- joining
                it to 'DH' instead produces a real (non-numerical) pressure discontinuity
                at the crust-core junction, see ``rmf_eos/FINDINGS.md`` in the companion
                validation repository.

                Note: the bundled 'Tolos' crust only covers CompactObject's raw outer-crust
                table (up to :math:`n \approx 2.6\times 10^{-4}\,\mathrm{fm}^{-3}`).
                CompactObject itself bridges the remaining gap up to the RMF core's
                starting density with an analytic polytrope expressed directly in
                (energy density, pressure) space with no associated number density, which
                cannot be ported into jester's ``(n, p, e)`` triplet format as-is -- so
                that bridge region is NOT reproduced here, and there is a real, wide,
                unbridged density gap between the 'Tolos' crust ceiling and
                ``min_core_nsat_rmf * rho_0`` (see that argument below). See
                ``rmf_eos/FINDINGS.md`` for the full discussion and follow-up plan.
            nsat: Nuclear saturation density convention used for the crust ceiling
                [:math:`\mathrm{fm}^{-3}`]. Note this is jester's own crust convention and
                is independent of the RMF model's intrinsic saturation density ``rho_0``.
            max_n_crust_nsat: Crust ceiling as a fraction of ``nsat``.
            min_n_crust_nsat: Crust floor as a fraction of ``nsat``.
            rho_0: RMF model's intrinsic nuclear saturation density
                [:math:`\mathrm{fm}^{-3}`], sets the density grid spacing (matches the
                reference implementation's ``rho_0 = 0.1505``).
            dt: Density grid step size, in units of ``rho_0`` (matches reference
                implementation's ``dt = 0.05``).
            ndat_full: Number of raw density grid points used for the warm-started field
                continuation (matches reference implementation's 124), spanning
                ``dt*rho_0`` to ``ndat_full*dt*rho_0``. Only the points above both the
                crust ceiling and ``min_core_nsat_rmf * rho_0`` are kept in the final EOS
                table -- the earlier ones exist purely to seed the sequential continuation
                robustly from low density.
            min_core_nsat_rmf: Minimum density the RMF core table is allowed to start at,
                as a fraction of ``rho_0``. This nucleonic RMF model produces unphysical
                negative pressure well below saturation density (it is not meant to
                describe sub-saturation matter) -- the reference implementation guards
                against this with its own hardcoded ``i > 18`` index floor (roughly
                ``0.9 * rho_0``) in addition to its crust-ceiling comparison. Defaults to
                ``1.0`` (start the core at the RMF model's own saturation density) as a
                conservative choice, independent of whichever crust is used.
            newton_max_steps: Maximum Levenberg-Marquardt iterations per density point.
            root_rtol: Relative tolerance for the field-equation root solve.
            root_atol: Absolute tolerance for the field-equation root solve.
        """
        self.rho_0 = rho_0
        self.newton_max_steps = newton_max_steps
        self.root_rtol = root_rtol
        self.root_atol = root_atol

        crust = Crust(
            crust_name,
            min_density=min_n_crust_nsat * nsat,
            max_density=max_n_crust_nsat * nsat,
        )
        self.n_crust: Float[Array, "n_crust"] = crust.n
        self.p_crust: Float[Array, "n_crust"] = crust.p
        self.e_crust: Float[Array, "n_crust"] = crust.e

        n_raw = jnp.arange(1, ndat_full + 1) * dt * rho_0
        core_floor = max(float(crust.max_density), min_core_nsat_rmf * rho_0)
        # Static (non-traced) mask: the raw grid, crust ceiling, and core floor are all
        # fixed at __init__ time, independent of the sampled RMF parameters, so this index
        # selection is baked in as a constant rather than needing jnp.where at trace time.
        keep_mask = [bool(v) for v in (n_raw > core_floor)]
        self.keep_idx = jnp.array([i for i, keep in enumerate(keep_mask) if keep])
        self.n_raw: Float[Array, "ndat_full"] = n_raw
        self.n_core: Float[Array, "n_core"] = n_raw[self.keep_idx]

    def get_required_parameters(self) -> list[str]:
        r"""
        Return the 10 RMF Lagrangian parameters.

        Returns:
            list[str]: ``["m_sig", "m_w", "m_rho", "g_sigma", "g_omega", "g_rho",
            "kappa", "lambda_0", "zeta", "Lambda_w"]`` -- meson masses, nucleon-meson
            couplings, scalar self-coupling, vector self-coupling, and the
            :math:`\omega`-:math:`\rho` cross-coupling, respectively.
        """
        return list(_PARAM_NAMES)

    def _solve_one(
        self, x0: Array, rho: Array, theta: Array
    ) -> tuple[Array, optx.RESULTS]:
        solver = optx.LevenbergMarquardt(rtol=self.root_rtol, atol=self.root_atol)
        sol = optx.least_squares(
            _field_residuals,
            solver,
            x0,
            args=(rho, theta),
            max_steps=self.newton_max_steps,
            throw=False,
        )
        return sol.value, sol.result

    def construct_eos(self, params: dict[str, float]) -> EOSData:
        r"""
        Construct the full EOS from the 10 RMF Lagrangian parameters.

        Args:
            params: Dictionary with keys ``m_sig, m_w, m_rho, g_sigma, g_omega, g_rho,
                kappa, lambda_0, zeta, Lambda_w`` (see :meth:`get_required_parameters`).

        Returns:
            EOSData: Complete EOS with all required arrays in geometric units.
                ``extra_constraints["n_root_solver_failures"]`` counts density points
                where the field-equation solve did not converge (diagnostic only).
        """
        theta = jnp.array([params[name] for name in _PARAM_NAMES])
        x0 = _initial_values(0.1 * self.rho_0, theta)

        def scan_fn(x_prev: Array, rho: Array):
            x_new, result = self._solve_one(x_prev, rho, theta)
            return x_new, (x_new, result)

        _, (xs_all, results_all) = lax.scan(scan_fn, x0, self.n_raw)

        n_root_solver_failures = jnp.sum(results_all != optx.RESULTS.successful)

        xs_core = xs_all[self.keep_idx]
        n_core = self.n_core

        e_core, p_core, _ = jax.vmap(_energy_density_pressure, in_axes=(0, 0, None))(
            xs_core, n_core, theta
        )
        e_core = e_core * _ONEOVERFM_MEV
        p_core = p_core * _ONEOVERFM_MEV

        n_full = jnp.concatenate([self.n_crust, n_core])
        p_full = jnp.concatenate([self.p_crust, p_core])
        e_full = jnp.concatenate([self.e_crust, e_core])

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n_full, p_full, e_full)
        # Concatenating jester's crust table directly with the RMF core (no smoothing
        # spline) leaves a few points of small negative cs2 right at the junction, from
        # the log-log finite-difference derivative straddling the composition discontinuity.
        # Clipped here (matches MetaModel_EOS_model's connection-region clipping) since cs2
        # is a diagnostic/causality-check field, not used by the TOV solver's ODE integration
        # itself (which interpolates on ps/es/hs).
        cs2 = jnp.clip(ps / (es * dloge_dlogps), 1e-6, 1.0)

        extra_constraints = {"n_root_solver_failures": n_root_solver_failures}

        return EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=cs2,
            mu=None,
            extra_constraints=extra_constraints,
        )
