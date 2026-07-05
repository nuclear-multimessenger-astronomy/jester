"""
Spectral EOS: adiabatic index and pressure-density relation.

The figure shows the adiabatic index Gamma(x) (top) and the pressure-density
relation (bottom) for three representative sets of spectral coefficients: the
radio-timing posterior mean and two perturbations that illustrate the range
of physically reasonable EOS shapes within the parametrization.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
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
from jesterTOV.eos.spectral.spectral_decomposition import (
    SpectralDecomposition_EOS_model,
)

# ── Three representative parameter sets ───────────────────────────────────────
# Radio posterior mean (fiducial), and two physically plausible perturbations.
GAMMA_SETS = [
    {
        "label": "Radio posterior mean",
        "color": "#2c7bb6",
        "ls": "-",
        "params": {
            "gamma_0": 0.922,
            "gamma_1": 0.342,
            "gamma_2": -0.083,
            "gamma_3": 0.004,
        },
    },
    {
        "label": "Stiffer",
        "color": "#d7191c",
        "ls": "--",
        "params": {
            "gamma_0": 1.30,
            "gamma_1": 0.20,
            "gamma_2": -0.05,
            "gamma_3": 0.002,
        },
    },
    {
        "label": "Softer",
        "color": "#1a9641",
        "ls": "-.",
        "params": {
            "gamma_0": 0.60,
            "gamma_1": 0.50,
            "gamma_2": -0.10,
            "gamma_3": 0.005,
        },
    },
]

model = SpectralDecomposition_EOS_model(crust_name="SLy", n_points_high=300)

# x-grid for the adiabatic index plot
x_vals = np.linspace(0.0, model.xmax, 300)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_gamma, ax_p) = plt.subplots(2, 1, figsize=(5.5, 6.5))
fig.subplots_adjust(hspace=0.35)
FS = 16

for entry in GAMMA_SETS:
    gamma = jnp.array(list(entry["params"].values()))

    # Adiabatic index curve
    Gamma_vals = np.asarray([float(model._adiabatic_index(x, gamma)) for x in x_vals])
    ax_gamma.plot(
        x_vals,
        Gamma_vals,
        color=entry["color"],
        ls=entry["ls"],
        lw=2.0,
        label=entry["label"],
    )

    # Pressure-density from construct_eos
    eos = model.construct_eos(entry["params"])
    n_vals = np.asarray(eos.ns) / utils.fm_inv3_to_geometric
    p_vals = np.asarray(eos.ps) / utils.MeV_fm_inv3_to_geometric

    # Only plot the spectral (high-density) region, above ~0.08 fm^-3
    N_MIN = 0.08
    mask = n_vals > N_MIN
    ax_p.semilogy(
        n_vals[mask],
        p_vals[mask],
        color=entry["color"],
        ls=entry["ls"],
        lw=2.0,
        label=entry["label"],
    )

# ── Gamma validity band ────────────────────────────────────────────────────────
ax_gamma.axhspan(0.6, 4.5, alpha=0.07, color="gray", label=r"Valid range $[0.6,\,4.5]$")
ax_gamma.axhline(0.6, color="gray", lw=0.8, ls=":")
ax_gamma.axhline(4.5, color="gray", lw=0.8, ls=":")
ax_gamma.set_xlabel(r"$x = \log(p/p_0)$", fontsize=FS)
ax_gamma.set_ylabel(r"$\Gamma(x)$", fontsize=FS)
ax_gamma.set_xlim(0, model.xmax)
ax_gamma.set_ylim(0, 5.0)
# ax_gamma.legend(fontsize=FS - 2, loc="upper right", framealpha=0.85)
ax_gamma.tick_params(labelsize=FS - 1)

ax_p.set_xlabel(r"$n\ [\mathrm{fm}^{-3}]$", fontsize=FS)
ax_p.set_ylabel(r"$P\ [\mathrm{MeV\,fm}^{-3}]$", fontsize=FS)
ax_p.set_xlim(N_MIN, 1.2)
ax_p.set_ylim(1e-1, 3e3)
ax_p.legend(
    fontsize=FS - 2, loc="lower right", framealpha=0.85
)  # only put the legend here
ax_p.tick_params(labelsize=FS - 1)

fig.tight_layout()
