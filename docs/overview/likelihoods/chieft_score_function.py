"""
Plot the chiEFT score function f(p, n) as a 2-D heatmap.

The score function encodes how well a candidate pressure p at density n
agrees with the chiral EFT band [p_-, p_+].  Inside the band f = 1;
outside it decays exponentially with slope beta = 6 / (p_+ - p_-):

    f(p, n) = exp(-6 (p - p_+)/(p_+ - p_-))   if p > p_+
    f(p, n) = 1                                  if p_- <= p <= p_+
    f(p, n) = exp(-6 (p_- - p)/(p_+ - p_-))   if p < p_-

Pressure bands are taken from Koehn et al. (2025), arXiv:2402.04172.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }
)

import jesterTOV

DATA_DIR = (
    Path(jesterTOV.__file__).parent / "inference" / "data" / "chiEFT" / "2402.04172"
)

NSAT = 0.16  # saturation density in fm^-3
BETA = 6.0  # penalty slope (Koehn et al. 2025)

# Load pressure bands (columns: density [fm^-3], pressure [MeV fm^-3], energy density)
low_data = np.loadtxt(DATA_DIR / "low.dat")
high_data = np.loadtxt(DATA_DIR / "high.dat")

n_low = low_data[:, 0] / NSAT  # n_sat units
p_low = low_data[:, 1]  # MeV fm^-3

n_high = high_data[:, 0] / NSAT
p_high = high_data[:, 1]

# ── Grid ──────────────────────────────────────────────────────────────────────
n_grid = np.linspace(0.75, 2.0, 400)
p_grid = np.linspace(0.0, 30.0, 400)

N, P = np.meshgrid(n_grid, p_grid)

# Interpolate band boundaries on the density grid
p_minus = np.interp(n_grid, n_low, p_low)  # (400,)
p_plus = np.interp(n_grid, n_high, p_high)  # (400,)

# Broadcast to the 2-D grid (pressure axis = rows, density axis = cols)
p_minus_2d = p_minus[np.newaxis, :]  # (1, 400)
p_plus_2d = p_plus[np.newaxis, :]

width = p_plus_2d - p_minus_2d  # (p_+ - p_-) > 0 everywhere

# ── Score function ─────────────────────────────────────────────────────────────
log_f = np.where(
    P >= p_plus_2d,
    -BETA * (P - p_plus_2d) / width,
    np.where(P <= p_minus_2d, -BETA * (p_minus_2d - P) / width, 0.0),
)
f = np.clip(np.exp(log_f), 1e-10, 1.0)

# ── Colormap: warm cream → deep blue ──────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    "chieft",
    ["#F9F5E7", "#1A4D8F"],
    N=512,
)
norm = mcolors.LogNorm(vmin=1e-4, vmax=1.0)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 5.5))

im = ax.pcolormesh(N, P, f, cmap=cmap, norm=norm, shading="auto", rasterized=True)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$f(p,\, n)$", fontsize=12)

# ── Band boundaries ────────────────────────────────────────────────────────────
mask = (n_high >= 0.75) & (n_high <= 2.0)
ax.plot(n_high[mask], p_high[mask], "k--", linewidth=2.0)

mask = (n_low >= 0.75) & (n_low <= 2.0)
ax.plot(n_low[mask], p_low[mask], "k--", linewidth=2.0)

# ── Labels placed near the right end of each curve ────────────────────────────
# p_+ label — just above the upper curve, near the right edge
n_plus_end = n_high[mask][-1]
p_plus_end = p_high[mask][-1]

delta_n = 0.03
ax.text(
    n_plus_end - delta_n,
    p_plus_end + 1.0,
    r"$p_+$",
    fontsize=13,
    ha="right",
)

# p_- label — just below the lower curve, near the right edge
n_minus_end = n_low[mask][-1]
p_minus_end = p_low[mask][-1]
ax.text(
    n_minus_end - delta_n,
    p_minus_end - 1.6,
    r"$p_-$",
    fontsize=13,
    ha="right",
)

ax.set_xlabel(r"$n\;[n_{\mathrm{sat}}]$", fontsize=12)
ax.set_ylabel(r"$p\;[\mathrm{MeV\,fm}^{-3}]$", fontsize=12)
ax.set_xlim(0.75, 2.0)
ax.set_ylim(0.0, 30.0)
ax.tick_params(labelsize=11)

# plt.savefig("./output.png", bbox_inches="tight") # for local testing

fig.tight_layout()
