"""
Crust EOS: pressure and energy density vs. baryon number density.

The figure shows the pressure (top) and energy density (bottom) as a function
of baryon number density for the three crust models available in ``jester``:
BPS, DH, and SLy.  Only the crust region is shown; no high-density core model
is appended.
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

from jesterTOV.eos.crust import Crust

# ── Styling ──────────────────────────────────────────────────────────────────
NAMES = ["BPS", "DH", "SLy"]
COLORS = ["#2c7bb6", "#d7191c", "#1a9641"]
STYLES = ["-", "--", "-."]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_p, ax_e) = plt.subplots(2, 1, figsize=(5.5, 6.5))
fig.subplots_adjust(hspace=0.38)
FS = 16

NSAT = 0.16  # nuclear saturation density [fm^-3]

for name, color, ls in zip(NAMES, COLORS, STYLES):
    crust = Crust(name, filter_zero_pressure=True)
    n = np.asarray(crust.n) / NSAT
    p = np.asarray(crust.p)
    e = np.asarray(crust.e)
    kw: dict[str, object] = dict(color=color, ls=ls, lw=2.0, label=name)
    ax_p.loglog(n, p, **kw)
    ax_e.loglog(n, e, **kw)

for ax, ylabel in [
    (ax_p, r"$P\ [\mathrm{MeV\,fm}^{-3}]$"),
    (ax_e, r"$\varepsilon\ [\mathrm{MeV\,fm}^{-3}]$"),
]:
    ax.set_xlabel(r"$n$ $[n_\mathrm{sat}]$", fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.tick_params(labelsize=FS - 1)
    ax.legend(fontsize=FS - 2, loc="upper left", framealpha=0.85)

# fig.savefig("crust.png", bbox_inches="tight") # for local testing
fig.tight_layout()
