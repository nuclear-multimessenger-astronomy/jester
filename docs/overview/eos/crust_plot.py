"""
Crust EOS: pressure and energy density vs. baryon number density.

The figure shows the pressure (top) and energy density (bottom) as a function
of baryon number density for all crust models available in ``jester``: the
original BPS, DH, and SLy models, plus the GMRS and MVCD family of models
converted from ``nucleardatapy`` (see
:mod:`jesterTOV.crust_files.convert_nucleardatapy_crusts`). Only the crust
region is shown; no high-density core model is appended.
"""

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }
)

from jesterTOV.eos.crust import Crust

# ── Styling ──────────────────────────────────────────────────────────────────
ORIGINAL_NAMES = ["BPS", "DH", "SLy"]
ORIGINAL_COLORS = ["#2c7bb6", "#d7191c", "#1a9641"]
ORIGINAL_STYLES = ["-", "--", "-."]

GMRS_NAMES = [
    name
    for name in Crust.list_available()
    if name.startswith("GMRS_") or name.startswith("MVCD_")
]
GMRS_COLORS = plt.get_cmap("viridis")(np.linspace(0, 1, len(GMRS_NAMES)))

NAMES = ORIGINAL_NAMES + GMRS_NAMES
COLORS = list(ORIGINAL_COLORS) + list(GMRS_COLORS)
STYLES = list(ORIGINAL_STYLES) + [":"] * len(GMRS_NAMES)
# Only the original three get individual legend entries; the nucleardatapy-derived
# GMRS/MVCD family is numerous and gets a single grouped entry instead.
LABELS = ORIGINAL_NAMES + [None] * len(GMRS_NAMES)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_p, ax_e) = plt.subplots(2, 1, figsize=(5.5, 6.5))
fig.subplots_adjust(hspace=0.38)
FS = 16

NSAT = 0.16  # nuclear saturation density [fm^-3]

for name, color, ls, label in zip(NAMES, COLORS, STYLES, LABELS):
    crust = Crust(name, filter_zero_pressure=True)
    n = np.asarray(crust.n) / NSAT
    p = np.asarray(crust.p)
    e = np.asarray(crust.e)
    kw: dict[str, object] = dict(color=color, ls=ls, lw=2.0, label=label)
    ax_p.loglog(n, p, **kw)
    ax_e.loglog(n, e, **kw)

# Proxy artist for the grouped GMRS/MVCD legend entry.
group_proxy = Line2D(
    [],
    [],
    color=GMRS_COLORS[len(GMRS_COLORS) // 2],
    ls=":",
    lw=2.0,
    label="GMRS / MVCD (nucleardatapy)",
)

for ax, ylabel in [
    (ax_p, r"$P\ [\mathrm{MeV\,fm}^{-3}]$"),
    (ax_e, r"$\varepsilon\ [\mathrm{MeV\,fm}^{-3}]$"),
]:
    ax.set_xlabel(r"$n$ $[n_\mathrm{sat}]$", fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.tick_params(labelsize=FS - 1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [group_proxy],
        labels + [group_proxy.get_label()],
        fontsize=FS - 2,
        loc="upper left",
        framealpha=0.85,
    )

# fig.savefig("crust.png", bbox_inches="tight") # for local testing
fig.tight_layout()
