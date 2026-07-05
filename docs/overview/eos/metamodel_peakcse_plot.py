"""
Metamodel + peakCSE: example EOS construction.

The figure shows the pressure and squared speed of sound as a function of baryon
number density for a single representative EOS built with fiducial nuclear empirical
parameters and a peakCSE extension above the break density.
Annotations highlight the key peakCSE parameters (Gaussian peak amplitude, centre,
and width) and the conformal limit approached at high densities.
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
from jesterTOV.eos.metamodel.metamodel_peakCSE import MetaModel_with_peakCSE_EOS_model

# ══════════════════════════════════════════════════════════════════════════════
# TUNABLE LAYOUT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure ────────────────────────────────────────────────────────────────────
FIG_WIDTH = 6.0
FIG_HEIGHT = 7.0
HSPACE = 0.08

# ── Font sizes ────────────────────────────────────────────────────────────────
FS = 16  # Base font size
FS_LABEL = FS - 1  # Axis labels (tick-level annotations)
FS_ANNOT = FS - 2  # In-plot annotations

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_EOS = "#2c7bb6"
COLOR_LOGISTIC = "#888888"
COLOR_CONFORMAL = "#c0392b"
COLOR_PEAK = "#d35400"
COLOR_SIGMA = "#8e44ad"
ALPHA_SPAN = 0.10

# ── Pressure panel – n_break label ────────────────────────────────────────────
NBREAK_LABEL_DX = 0.03  # horizontal offset from the n_break line [fm^-3]
NBREAK_LABEL_Y = 5e2  # y position in pressure panel [MeV fm^-3]

# ── cs² panel – conformal-limit label ─────────────────────────────────────────
CONFORMAL_LABEL_X_FRAC = 0.97  # fraction of nmax for x position
CONFORMAL_LABEL_DY = 0.025  # vertical offset above the 1/3 line

# ── cs² panel – n_peak vertical-line label ────────────────────────────────────
NPEAK_LABEL_DX = 0.015  # horizontal offset from n_peak line [fm^-3]
NPEAK_LABEL_Y = 0.06  # y position (in cs² units)

# ── cs² panel – peak-amplitude arrow (c_{s,peak}^2) ──────────────────────────
AMP_ARROW_DX = -0.055  # horizontal position relative to n_peak [fm^-3]
AMP_LABEL_DX = -0.06  # label x relative to n_peak [fm^-3]
AMP_LABEL_Y = 0.65  # label y position [cs² units]; None → midpoint of arrow

# ── cs² panel – sigma_peak arrow (horizontal double-headed) ──────────────────
SIGMA_ARROW_DY = 0.0  # vertical offset of arrow from half-height level
SIGMA_LABEL_DY = 0.03  # label vertical offset above the arrow

# ══════════════════════════════════════════════════════════════════════════════

# ── Model setup ────────────────────────────────────────────────────────────────
nsat = 0.16  # fm^-3
nbreak = 0.32  # fm^-3 (= 2 n_sat)
nmax_nsat = 10.0
nmax = nmax_nsat * nsat  # fm^-3

model = MetaModel_with_peakCSE_EOS_model(
    nsat=nsat,
    nmax_nsat=nmax_nsat,
    ndat_metamodel=120,
    ndat_CSE=200,
    crust_name="DH",
)

# ── Fiducial parameters ────────────────────────────────────────────────────────
gaussian_peak = 0.45
gaussian_mu = 0.60
gaussian_sigma = 0.12
logit_growth_rate = 6.0
logit_midpoint = 1.0

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
    "gaussian_peak": gaussian_peak,
    "gaussian_mu": gaussian_mu,
    "gaussian_sigma": gaussian_sigma,
    "logit_growth_rate": logit_growth_rate,
    "logit_midpoint": logit_midpoint,
}

eos = model.construct_eos(params)

# Convert from geometric units to nuclear units
n_full = np.asarray(eos.ns) / utils.fm_inv3_to_geometric
p_full = np.asarray(eos.ps) / utils.MeV_fm_inv3_to_geometric
cs2_full = np.asarray(eos.cs2)

# ── Reconstruct the logistic baseline (without Gaussian) for annotation ───────
cs2_break = float(np.interp(nbreak, n_full, cs2_full))
gaussian_at_break = gaussian_peak * np.exp(
    -0.5 * (nbreak - gaussian_mu) ** 2 / gaussian_sigma**2
)
exp_part = np.exp(-logit_growth_rate * (nbreak - logit_midpoint))
offset = ((1.0 + exp_part) * (cs2_break - gaussian_at_break) - 1.0 / 3.0) / exp_part

n_cse = n_full[n_full >= nbreak]
logistic_baseline = offset + (1.0 / 3.0 - offset) / (
    1.0 + np.exp(-logit_growth_rate * (n_cse - logit_midpoint))
)

# ── Derived annotation quantities ─────────────────────────────────────────────
cs2_at_peak = float(np.interp(gaussian_mu, n_full, cs2_full))
logistic_at_peak = float(
    offset
    + (1.0 / 3.0 - offset)
    / (1.0 + np.exp(-logit_growth_rate * (gaussian_mu - logit_midpoint)))
)
half_height = logistic_at_peak + 0.5 * (cs2_at_peak - logistic_at_peak)

n_crust_max = 0.075  # DH crust ends around here [fm^-3]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_p, ax_cs) = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
fig.subplots_adjust(hspace=HSPACE)

# ─ Pressure panel ─────────────────────────────────────────────────────────────
ax_p.semilogy(n_full, p_full, color=COLOR_EOS, lw=2.0)
ax_p.axvspan(0, n_crust_max, alpha=ALPHA_SPAN, color="gray", label="Crust")
ax_p.axvspan(n_crust_max, nbreak, alpha=ALPHA_SPAN, color="#e67e22", label="Metamodel")
ax_p.axvspan(nbreak, nmax, alpha=ALPHA_SPAN, color="#27ae60", label="peakCSE")
ax_p.axvline(nbreak, color="black", lw=0.9, ls="--")
ax_p.text(
    nbreak + NBREAK_LABEL_DX,
    NBREAK_LABEL_Y,
    r"$n_\mathrm{break}$",
    ha="left",
    va="center",
    fontsize=FS_LABEL,
)
ax_p.set_ylabel(r"$P\ [\mathrm{MeV\,fm}^{-3}]$", fontsize=FS)
ax_p.set_ylim(1e-3, 3e3)
ax_p.set_xlim(0, nmax)
ax_p.legend(loc="lower right", fontsize=FS - 2, framealpha=0.85)

# ─ Speed-of-sound panel ───────────────────────────────────────────────────────
ax_cs.plot(n_full, cs2_full, color=COLOR_EOS, lw=2.0)

# Logistic baseline
ax_cs.plot(
    n_cse,
    logistic_baseline,
    color=COLOR_LOGISTIC,
    lw=1.4,
    ls="--",
    label="Logistic baseline",
)

# Conformal limit
ax_cs.axhline(1.0 / 3.0, color=COLOR_CONFORMAL, lw=1.2, ls=":", zorder=2)
ax_cs.text(
    CONFORMAL_LABEL_X_FRAC * nmax,
    1.0 / 3.0 + CONFORMAL_LABEL_DY,
    r"$c_s^2 = 1/3$",
    ha="right",
    va="bottom",
    fontsize=FS_LABEL,
    color=COLOR_CONFORMAL,
)

# Shaded regions
ax_cs.axvspan(0, n_crust_max, alpha=ALPHA_SPAN, color="gray")
ax_cs.axvspan(n_crust_max, nbreak, alpha=ALPHA_SPAN, color="#e67e22")
ax_cs.axvspan(nbreak, nmax, alpha=ALPHA_SPAN, color="#27ae60")
ax_cs.axvline(nbreak, color="black", lw=0.9, ls="--")

# Causality limit
ax_cs.axhline(1.0, color="black", lw=0.9, ls="--")

# ── Annotation: n_peak vertical line ──────────────────────────────────────────
ax_cs.axvline(gaussian_mu, color=COLOR_PEAK, lw=1.0, ls=":", alpha=0.8)
ax_cs.text(
    gaussian_mu + NPEAK_LABEL_DX,
    NPEAK_LABEL_Y,
    r"$n_\mathrm{peak}$",
    ha="left",
    va="bottom",
    fontsize=FS_ANNOT,
    color=COLOR_PEAK,
)

# ── Annotation: peak amplitude c_{s,peak}^2 (vertical double-headed arrow) ───
ax_cs.annotate(
    "",
    xy=(gaussian_mu + AMP_ARROW_DX, cs2_at_peak),
    xytext=(gaussian_mu + AMP_ARROW_DX, logistic_at_peak),
    arrowprops=dict(arrowstyle="<->", color=COLOR_PEAK, lw=1.3),
)
_amp_label_y = (
    AMP_LABEL_Y if AMP_LABEL_Y is not None else 0.5 * (cs2_at_peak + logistic_at_peak)
)
ax_cs.text(
    gaussian_mu + AMP_LABEL_DX,
    _amp_label_y,
    r"$c_{s,\mathrm{peak}}^2$",
    ha="right",
    va="center",
    fontsize=FS_ANNOT,
    color=COLOR_PEAK,
)

# ── Annotation: sigma_peak (horizontal double-headed arrow at half-height) ────
arrow_y = half_height + SIGMA_ARROW_DY
ax_cs.annotate(
    "",
    xy=(gaussian_mu + gaussian_sigma, arrow_y),
    xytext=(gaussian_mu - gaussian_sigma, arrow_y),
    arrowprops=dict(arrowstyle="<->", color=COLOR_SIGMA, lw=1.3),
)
ax_cs.text(
    gaussian_mu,
    arrow_y + SIGMA_LABEL_DY,
    r"$2\,\sigma_\mathrm{peak}$",
    ha="center",
    va="bottom",
    fontsize=FS_ANNOT,
    color=COLOR_SIGMA,
)

ax_cs.set_xlabel(r"$n\ [\mathrm{fm}^{-3}]$", fontsize=FS)
ax_cs.set_ylabel(r"$c_s^2 / c^2$", fontsize=FS)
ax_cs.set_ylim(0, 1.05)
ax_cs.legend(loc="upper right", fontsize=FS - 2, framealpha=0.85)

for ax in (ax_p, ax_cs):
    ax.tick_params(labelsize=FS_LABEL)

# fig.savefig("metamodel_peakcse_plot.pdf", bbox_inches="tight") # for local testing, disable afterwards
fig.tight_layout()
