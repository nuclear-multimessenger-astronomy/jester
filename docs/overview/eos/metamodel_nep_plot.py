"""
Nuclear Empirical Parameters (NEPs) illustrated via the meta-model energy per nucleon.

The energy per nucleon E/A is shown for symmetric nuclear matter (delta=0, red)
and pure neutron matter (delta=1, blue) as a function of baryon number density n.
Four key NEPs are annotated at saturation density n_sat:

* E_sat -- binding energy per nucleon at saturation (the minimum of the
  symmetric matter curve)
* K_sat -- incompressibility, proportional to the curvature at saturation
* E_sym -- symmetry energy, the difference between the two curves at n_sat
* L_sym -- slope of the symmetry energy, proportional to the derivative of
  the neutron matter curve at n_sat

.. developer note::
   Use LaTeX rendering (``text.usetex = True``) in all docs plot scripts for
   publication-quality typography.  Requires a working LaTeX installation with
   the ``texlive-latex-base``, ``texlive-latex-extra``, ``dvipng``, and
   ``cm-super`` packages, which are available in the CI environment.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Use LaTeX rendering for publication-quality text
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }
)

from jesterTOV.eos.metamodel.base import MetaModel_EOS_model

# ── Fiducial nuclear empirical parameters ──────────────────────────────────
E_sat, K_sat, Q_sat, Z_sat = -16.0, 230.0, -300.0, 0.0
E_sym, L_sym, K_sym, Q_sym, Z_sym = 32.0, 60.0, -100.0, 0.0, 0.0

nsat = 0.16  # nuclear saturation density [fm^-3]

model = MetaModel_EOS_model(nsat=nsat)

# Build potential-energy coefficient arrays (mirroring construct_eos)
v_sat = jnp.array(
    [
        E_sat + model.v_sat_0_no_NEP,
        0.0 + model.v_sat_1_no_NEP,
        K_sat + model.v_sat_2_no_NEP,
        Q_sat + model.v_sat_3_no_NEP,
        Z_sat + model.v_sat_4_no_NEP,
    ]
)
v_sym2 = jnp.array(
    [
        E_sym + model.v_sym2_0_no_NEP,
        L_sym + model.v_sym2_1_no_NEP,
        K_sym + model.v_sym2_2_no_NEP,
        Q_sym + model.v_sym2_3_no_NEP,
        Z_sym + model.v_sym2_4_no_NEP,
    ]
)

# ── Density array ───────────────────────────────────────────────────────────
# Start at 1e-5 so both curves visually pass through the origin (E/A → 0 as n→0)
n_vals = jnp.array(
    np.concatenate(
        [
            np.logspace(-5, np.log10(0.005), 100),
            np.linspace(0.005, 0.32, 500),
        ]
    )
)
x = model.compute_x(n_vals)


def energy_per_nucleon(delta_const: float) -> np.ndarray:
    """Return E/A [MeV] for constant isospin asymmetry delta."""
    delta = jnp.full_like(n_vals, delta_const)
    f_1 = model.compute_f_1(delta)
    f_star = model.compute_f_star(delta)
    f_star2 = model.compute_f_star2(delta)
    f_star3 = model.compute_f_star3(delta)
    b = model.compute_b(delta)
    v = model.compute_v(v_sat, v_sym2, delta)
    return np.asarray(model.compute_energy(x, f_1, f_star, f_star2, f_star3, b, v))


n = np.asarray(n_vals)
EA_sym = energy_per_nucleon(0.0)  # symmetric nuclear matter
EA_neut = energy_per_nucleon(1.0)  # pure neutron matter

# Values at saturation density n_sat
idx_n0 = int(np.argmin(np.abs(n - nsat)))
EA_sym_n0 = float(EA_sym[idx_n0])  # ≈ E_sat
EA_neut_n0 = float(EA_neut[idx_n0])  # ≈ E_sat + E_sym

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5.5))

COLOR_SM = "#c0392b"  # red -- symmetric matter
COLOR_NM = "#2980b9"  # blue -- neutron matter
COLOR_ARROW = "black"
COLOR_WEDGE = "gray"
ALPHA_WEDGE = 0.45
FS = 13  # base font size for annotations
FS_TICK = 12

ax.plot(n, EA_sym, color=COLOR_SM, lw=2.2, label="Symmetric matter")
ax.plot(n, EA_neut, color=COLOR_NM, lw=2.2, label="Neutron matter")

# ── Axis limits ──────────────────────────────────────────────────────────────
ymin, ymax = -21, 31
ax.set_ylim(ymin, ymax)
ax.set_xlim(0.0, 0.31)

# ── Saturation density: tick mark + label inside plot (no dashed line) ───────
ax.plot(nsat, ymin, marker="|", ms=9, mew=1.8, color=COLOR_SM, zorder=6, clip_on=False)
ax.text(
    nsat,
    ymin + 1.8,
    r"$n_\mathrm{sat}$",
    color=COLOR_SM,
    ha="center",
    va="bottom",
    fontsize=FS,
)

# ── E_sat: text label only on the left (no horizontal line) ──────────────────
ax.text(
    -0.007,
    EA_sym_n0,
    r"$E_\mathrm{sat}$",
    color=COLOR_SM,
    ha="right",
    va="center",
    fontsize=FS,
    clip_on=False,
)

# ── E_sym: double-headed arrow between curves at n_sat ────────────────────────
ax.annotate(
    "",
    xy=(nsat, EA_sym_n0),
    xytext=(nsat, EA_neut_n0),
    arrowprops=dict(arrowstyle="<->", color=COLOR_ARROW, lw=1.8),
)
ax.text(
    nsat + 0.014,
    0.5 * (EA_sym_n0 + EA_neut_n0),
    r"$E_\mathrm{sym}$",
    ha="left",
    va="center",
    fontsize=FS,
)

# ── K_sat: arrow pointing to the symmetric matter minimum ────────────────────
# K_sat arrow offset to the right so it does not overlap with the E_sym arrow
k_x = nsat + 0.022
ax.annotate(
    "",
    xy=(k_x, EA_sym_n0 + 0.3),
    xytext=(k_x, EA_sym_n0 + 5.5),
    arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=1.6),
)
ax.text(
    k_x + 0.007,
    EA_sym_n0 + 3.5,
    r"$K_\mathrm{sat}$",
    ha="left",
    va="center",
    fontsize=FS,
)

# ── L_sym: slope arrow centred on n_sat, label directly above the NM curve there ──
dn_l = 0.040
x_tail = nsat - dn_l * 0.5
x_tip = nsat + dn_l * 0.5
dEdN_neut = L_sym / (3.0 * nsat)  # first derivative of E/A for NM at n_sat [MeV fm^3]
y_tail = EA_neut_n0 + dEdN_neut * (x_tail - nsat)
y_tip = EA_neut_n0 + dEdN_neut * (x_tip - nsat)

ax.annotate(
    "",
    xy=(x_tip, y_tip),
    xytext=(x_tail, y_tail),
    arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=1.8),
)
# Label sits directly above the neutron matter curve at n_sat
ax.text(
    nsat,
    EA_neut_n0 + 2.5,
    r"$L_\mathrm{sym}$",
    ha="center",
    va="bottom",
    fontsize=FS,
)

# ── Axes labels and legend ────────────────────────────────────────────────────
ax.set_xlabel(r"$n\ [\mathrm{fm}^{-3}]$", fontsize=FS + 1)
ax.set_ylabel(r"$E/A\ [\mathrm{MeV}]$", fontsize=FS + 1)
ax.tick_params(labelsize=FS_TICK)

ax.legend(loc="upper left", fontsize=FS - 1, framealpha=0.85)

fig.tight_layout()
