"""
Metamodel + CSE: example EOS construction.

The figure shows the pressure and squared speed of sound as a function of baryon
number density for a single representative EOS built with fiducial nuclear empirical
parameters and a simple four-node CSE extension above the break density.
"""

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }
)

from jesterTOV import utils
from jesterTOV.eos.metamodel.metamodel_CSE import MetaModel_with_CSE_EOS_model

# ── Model setup ────────────────────────────────────────────────────────────────
nsat = 0.16  # fm^-3
nbreak = 0.32  # fm^-3 (= 2 n_sat)
nmax_nsat = 10.0
nmax = nmax_nsat * nsat  # fm^-3
nb_CSE = 4

model = MetaModel_with_CSE_EOS_model(
    nsat=nsat,
    nmax_nsat=nmax_nsat,
    nb_CSE=nb_CSE,
    ndat_metamodel=120,
    ndat_CSE=80,
    crust_name="DH",
)

# ── Fiducial parameters ────────────────────────────────────────────────────────
params: dict[str, float] = {
    "E_sat": -16.0,
    "K_sat": 230.0,
    "Q_sat": -300.0,
    "Z_sat": 0.0,
    "E_sym": 32.0,
    "L_sym": 60.0,
    "K_sym": -100.0,
    "Q_sym": 0.0,
    "Z_sym": 0.0,
    "nbreak": nbreak,
    "n_CSE_0_u": 0.15,
    "n_CSE_1_u": 0.35,
    "n_CSE_2_u": 0.60,
    "n_CSE_3_u": 0.80,
    "cs2_CSE_0": 0.25,
    "cs2_CSE_1": 0.45,
    "cs2_CSE_2": 0.55,
    "cs2_CSE_3": 0.40,
    "cs2_CSE_4": 0.30,
}

eos = model.construct_eos(params)

# Convert from geometric units to nuclear units
n_full = np.asarray(eos.ns) / utils.fm_inv3_to_geometric
p_full = np.asarray(eos.ps) / utils.MeV_fm_inv3_to_geometric
cs2_full = np.asarray(eos.cs2)

# ── Compute CSE grid-point positions for the markers ──────────────────────────
# Internal nodes: n_i = nbreak + u_i * (nmax - nbreak)
u_vals = np.array([params[f"n_CSE_{i}_u"] for i in range(nb_CSE)])
n_nodes = np.concatenate([[nbreak], nbreak + np.sort(u_vals) * (nmax - nbreak), [nmax]])
cs2_nodes = np.array(
    [np.interp(nbreak, n_full, cs2_full)]  # continuity at nbreak from metamodel
    + [params[f"cs2_CSE_{i}"] for i in range(nb_CSE + 1)]
)

# ── Constants ─────────────────────────────────────────────────────────────────
n_crust_max = 0.075  # DH crust ends around here [fm^-3]
COLOR = "#2c7bb6"
ALPHA_SPAN = 0.10
FS = 16

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_p, ax_cs) = plt.subplots(2, 1, figsize=(5.5, 6.5), sharex=True)
fig.subplots_adjust(hspace=0.08)

# ─ Pressure panel ─────────────────────────────────────────────────────────────
ax_p.semilogy(n_full, p_full, color=COLOR, lw=2.0)
ax_p.axvspan(0, n_crust_max, alpha=ALPHA_SPAN, color="gray", label="Crust")
ax_p.axvspan(n_crust_max, nbreak, alpha=ALPHA_SPAN, color="#e67e22", label="Metamodel")
ax_p.axvspan(nbreak, nmax, alpha=ALPHA_SPAN, color="#27ae60", label="CSE")
ax_p.axvline(nbreak, color="black", lw=0.9, ls="--")
ax_p.text(
    nbreak + 0.04,
    1e2,
    r"$n_\mathrm{break}$",
    ha="left",
    va="center",
    fontsize=FS - 1,
)
ax_p.set_ylabel(r"$P\ [\mathrm{MeV\,fm}^{-3}]$", fontsize=FS)
ax_p.set_ylim(1e-3, 3e3)
ax_p.set_xlim(0, nmax)
# Legend in the pressure panel, bottom-right
ax_p.legend(loc="lower right", fontsize=FS - 2, framealpha=0.85)

# ─ Speed-of-sound panel ───────────────────────────────────────────────────────
ax_cs.plot(n_full, cs2_full, color=COLOR, lw=2.0)
ax_cs.axvspan(0, n_crust_max, alpha=ALPHA_SPAN, color="gray")
ax_cs.axvspan(n_crust_max, nbreak, alpha=ALPHA_SPAN, color="#e67e22")
ax_cs.axvspan(nbreak, nmax, alpha=ALPHA_SPAN, color="#27ae60")
ax_cs.axvline(nbreak, color="black", lw=0.9, ls="--")
ax_cs.axhline(1.0, color="black", lw=0.9, ls="--")

# CSE grid-point markers (filled circles); skip the nbreak point (fixed by metamodel)
ax_cs.scatter(
    n_nodes[1:],
    cs2_nodes[1:],
    s=55,
    zorder=5,
    color="#d7191c",
    label="CSE nodes",
    clip_on=False,
)

ax_cs.set_xlabel(r"$n\ [\mathrm{fm}^{-3}]$", fontsize=FS)
ax_cs.set_ylabel(r"$c_s^2 / c^2$", fontsize=FS)
ax_cs.set_ylim(0, 1.05)

for ax in (ax_p, ax_cs):
    ax.tick_params(labelsize=FS - 1)

fig.tight_layout()
