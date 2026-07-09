r"""Meta-model EOS driven directly by chiral EFT low-energy constants (LECs).

TODO: docstring.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
from jaxtyping import Array, Float, Int

from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.metamodel.metamodel_CSE import MetaModel_with_CSE_EOS_model
from jesterTOV.tov.data_classes import EOSData

# Reference densities [fm^-3] at which the PNM energy per particle is known
# as a (quadratic) function of the LECs c1 and c3, via the eigenvalues of a
# 2x2 matrix. See _pnm_energy below.
_PMM_DENSITIES: Float[Array, "6"] = jnp.array([0.12, 0.16, 0.24, 0.32, 0.40, 0.48])

# Hardcoded matrix coefficients for each of the 6 reference densities above.
# Each row is [x0, x1, x2, x3, x4, x5, x6, x7] such that
#   M(c1, c3) = [[x0, 0], [0, x1]] + c1 * [[x2, x3], [x3, x4]] + c3 * [[x5, x6], [x6, x7]]
# and the PNM energy per particle at that density is the lowest eigenvalue of M.
_PMM_COEFFS: Float[Array, "6 8"] = jnp.array(
    [
        [
            12.201939,
            568.76472071,
            -0.24248967,
            1.41775278,
            114.34516393,
            -0.32284581,
            -2.69212688,
            -10.01734106,
        ],
        [
            1.505819145200185538e01,
            2.137022252640809938e01,
            -3.428992926362893767e-01,
            2.123106763017263143e-01,
            1.029956254509725611e00,
            -5.572724242961462071e-01,
            -3.701343802029128938e-01,
            -3.034584754963404185e-01,
        ],
        [
            2.193473081639075417e01,
            4.236274177616851375e03,
            -6.963525817981026655e-01,
            -4.530711351270891640e00,
            -3.694404000448697900e01,
            -1.250048532018936065e00,
            -1.673697607467108384e01,
            1.307926964511519685e01,
        ],
        [
            3.124851089398970316e01,
            8.508484699954726693e01,
            -1.095955111893218259e00,
            -2.650812234358737030e-01,
            2.493710211537390364e00,
            -2.057916935729835473e00,
            -2.174923149776333275e00,
            5.316320260816258525e-01,
        ],
        [
            1.124395411481622205e02,
            4.198885974788384345e01,
            -1.089870782432588570e01,
            -3.115727300784213227e00,
            -1.886825553771974917e00,
            1.840923217820445323e00,
            -3.138566400421714420e00,
            -3.157914597527116563e00,
        ],
        [
            3.767789415470389258e02,
            5.493152693078990723e01,
            8.890565407304581669e00,
            -2.184433582989460554e00,
            -1.974581259973218206e00,
            1.019234513535115383e01,
            -8.617672605447211254e00,
            -4.312863013486001762e00,
        ],
    ]
)


def _pnm_energy(coeffs: Float[Array, "8"], c1: Float, c3: Float) -> Float:
    """PNM energy per particle at a single density, as the lowest eigenvalue of the 2x2 LEC matrix."""
    x0, x1, x2, x3, x4, x5, x6, x7 = coeffs
    matrix = (
        jnp.array([[x0, 0.0], [0.0, x1]])
        + c1 * jnp.array([[x2, x3], [x3, x4]])
        + c3 * jnp.array([[x5, x6], [x6, x7]])
    )
    return jnp.min(jnp.linalg.eigvalsh(matrix))


def get_PNM_points(c1: Float, c3: Float) -> Float[Array, "6"]:
    """PNM energies per particle at the 6 reference densities, as a function of c1 and c3."""
    return jax.vmap(_pnm_energy, in_axes=(0, None, None))(_PMM_COEFFS, c1, c3)


class LEC_MetaModel_with_CSE_EOS_model(Interpolate_EOS_model):
    r"""
    Meta-model EOS with CSE extension, parametrized directly by chiral EFT LECs.

    TODO: docstring.
    """

    # TODO: Check if we want to sample E_sat, K_sat, Q_sat, Z_sat as well, or just fix them in the source code already
    def __init__(
        self,
        E_sat: Float = -16.0,
        K_sat: Float = 230.0,
        Q_sat: Float = 0.0,
        Z_sat: Float = 0.0,
        nsat: Float = 0.16,
        nmin_MM_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        max_nbreak_nsat: Float | None = None,
        ndat_metamodel: Int = 100,
        ndat_CSE: Int = 100,
        nb_CSE: Int = 8,
        **metamodel_kwargs,
    ):
        r"""TODO: docstring."""

        assert nsat == 0.16, (
            "LEC_MetaModel_with_CSE_EOS_model assumes nsat=0.16, matching the "
            "densities at which the LEC -> PNM energy matrices were derived."
        )

        self.E_sat = E_sat
        self.K_sat = K_sat
        self.Q_sat = Q_sat
        self.Z_sat = Z_sat
        self.nb_CSE = nb_CSE

        self.mm_cse = MetaModel_with_CSE_EOS_model(
            nsat=nsat,
            nmin_MM_nsat=nmin_MM_nsat,
            nmax_nsat=nmax_nsat,
            max_nbreak_nsat=max_nbreak_nsat,
            ndat_metamodel=ndat_metamodel,
            ndat_CSE=ndat_CSE,
            nb_CSE=nb_CSE,
            **metamodel_kwargs,
        )
        # Reuse the underlying MetaModel_EOS_model instance to build the
        # LEC -> symmetry NEP fit, so the potential-energy formulas are not
        # duplicated here.
        self.metamodel = self.mm_cse.metamodel

        # Precompute the fixed (theta-independent) part of the PNM energy at
        # the fit densities, and the design matrix mapping the free symmetry
        # NEPs (L_sym, K_sym, Q_sym, Z_sym) linearly onto the PNM energy.
        mm = self.metamodel
        x = mm.compute_x(_PMM_DENSITIES)
        b_val = mm.compute_b(1.0)
        u = mm.compute_u(x, b_val)  # shape (N+1, 6)

        f_1 = mm.compute_f_1(1.0)
        f_star = mm.compute_f_star(1.0)
        f_star2 = mm.compute_f_star2(1.0)
        f_star3 = mm.compute_f_star3(1.0)
        kinetic = (
            mm.t_sat
            / 2
            * (1 + 3 * x) ** (2 / 3)
            * (
                f_1
                + (1 + 3 * x) * f_star
                + (1 + 3 * x) ** 2 * f_star2
                + (1 + 3 * x) ** 3 * f_star3
            )
        )

        v_sat = jnp.array(
            [
                self.E_sat + mm.v_sat_0_no_NEP,
                0.0 + mm.v_sat_1_no_NEP,
                self.K_sat + mm.v_sat_2_no_NEP,
                self.Q_sat + mm.v_sat_3_no_NEP,
                self.Z_sat + mm.v_sat_4_no_NEP,
            ]
        )
        sym_offsets = jnp.array(
            [
                mm.v_sym2_0_no_NEP,
                mm.v_sym2_1_no_NEP,
                mm.v_sym2_2_no_NEP,
                mm.v_sym2_3_no_NEP,
                mm.v_sym2_4_no_NEP,
            ]
        )

        alphas = jnp.arange(mm.N + 1)
        # basis[alpha, n] is the coefficient of the symmetry NEP theta[alpha]
        # in e_tot(n, delta=1), i.e. d(e_tot)/d(theta[alpha]).
        self._basis = u * x[None, :] ** alphas[:, None] / factorial(alphas)[:, None]

        v_theta0 = v_sat + sym_offsets
        self._baseline = kinetic + jnp.sum(v_theta0[:, None] * self._basis, axis=0)

    def construct_eos(self, params: dict[str, float]) -> EOSData:
        r"""TODO: docstring."""
        c1 = params["c1"]
        c3 = params["c3"]

        pnm_energies = get_PNM_points(c1, c3)  # shape (6,)

        # E_sym is fixed exactly by the PNM energy at n = nsat.
        E_sym = pnm_energies[1] - self.E_sat

        # Solve the remaining (L_sym, K_sym, Q_sym, Z_sym) via linear least
        # squares, since e_tot(n, delta=1) is affine in the symmetry NEPs.
        target = pnm_energies - self._baseline - E_sym * self._basis[0]
        design = self._basis[1:].T  # shape (6, 4)
        theta_free = jnp.linalg.solve(design.T @ design, design.T @ target)
        L_sym, K_sym, Q_sym, Z_sym = theta_free

        params_full = dict(params)
        params_full["E_sat"] = self.E_sat
        params_full["K_sat"] = self.K_sat
        params_full["Q_sat"] = self.Q_sat
        params_full["Z_sat"] = self.Z_sat
        params_full["E_sym"] = E_sym
        params_full["L_sym"] = L_sym
        params_full["K_sym"] = K_sym
        params_full["Q_sym"] = Q_sym
        params_full["Z_sym"] = Z_sym

        return self.mm_cse.construct_eos(params_full)

    def get_required_parameters(self) -> list[str]:
        r"""TODO: docstring."""
        params = ["c1", "c3", "nbreak"]
        for i in range(self.nb_CSE):
            params.extend([f"n_CSE_{i}_u", f"cs2_CSE_{i}"])
        params.append(f"cs2_CSE_{self.nb_CSE}")
        return params
