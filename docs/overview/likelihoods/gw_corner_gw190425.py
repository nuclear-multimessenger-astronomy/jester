"""
Corner plot for GW190425 posterior: data samples vs normalizing-flow samples.

Shows the joint and marginal distributions of the four parameters used by the
GW likelihood: ``mass_1_source``, ``mass_2_source``, ``lambda_1``, ``lambda_2``.

Blue shows the original posterior samples (IMRPhenomPNRT, low-spin prior).  Red
shows samples drawn from the trained normalizing flow, demonstrating how
faithfully the flow reproduces the posterior.

.. developer note::
   Use LaTeX rendering (``text.usetex = True``) in all docs plot scripts for
   publication-quality typography.
"""

from pathlib import Path

import corner
import jax
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

import jesterTOV
from jesterTOV.inference.flows.flow import Flow

# ── paths ────────────────────────────────────────────────────────────────────
_PKG = Path(jesterTOV.__file__).parent
_FLOWS_BASE = _PKG / "inference" / "flows" / "models" / "gw_maf"
_DATA_BASE = _PKG / "inference" / "data"

FLOW_DIR = _FLOWS_BASE / "gw190425" / "gw190425_phenompnrt_ls"
DATA_FILE = _DATA_BASE / "gw190425" / "gw190425_phenompnrt-ls_posterior.npz"

# ── font sizes ───────────────────────────────────────────────────────────────
LABEL_FONTSIZE = 20  # axis parameter labels
TICK_FONTSIZE = 16  # tick labels on all axes
LEGEND_FONTSIZE = 24  # legend entries
TITLE_FONTSIZE = 20  # suptitle

# ── config ───────────────────────────────────────────────────────────────────
N_FLOW_SAMPLES = 20_000
PARAMS = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]
LABELS = [
    r"$m_1^{\rm source}\ [M_\odot]$",
    r"$m_2^{\rm source}\ [M_\odot]$",
    r"$\Lambda_1$",
    r"$\Lambda_2$",
]
DATA_COLOR = "#1f77b4"
FLOW_COLOR = "#d62728"
LEVELS = (0.68, 0.90)

# ── load samples ─────────────────────────────────────────────────────────────
raw = np.load(DATA_FILE)
data_samples = np.stack([raw[p] for p in PARAMS], axis=1)  # (N, 4)

flow = Flow.from_directory(str(FLOW_DIR))
flow_samples = np.asarray(flow.sample(jax.random.key(0), (N_FLOW_SAMPLES,)))  # (N, 4)

# ── axis ranges: 99% credible interval from data ────────────────────────────
_ranges = [
    (
        float(np.percentile(data_samples[:, i], 0.5)),
        float(np.percentile(data_samples[:, i], 99.5)),
    )
    for i in range(data_samples.shape[1])
]

# ── shared corner kwargs ──────────────────────────────────────────────────────
_corner_kw = dict(
    labels=LABELS,
    label_kwargs={"fontsize": LABEL_FONTSIZE},
    range=_ranges,
    smooth=1.0,
    levels=LEVELS,
    plot_datapoints=False,
    plot_density=False,
    fill_contours=True,
    no_fill_contours=False,
    bins=30,
)

# ── draw data first, then overlay flow ───────────────────────────────────────
fig = corner.corner(
    data_samples,
    color=DATA_COLOR,
    hist_kwargs={"density": True, "alpha": 0.6},
    contourf_kwargs={"alpha": [0.0, 0.25, 0.50]},
    contour_kwargs={"linewidths": 1.4},
    **_corner_kw,
)

corner.corner(
    flow_samples,
    fig=fig,
    color=FLOW_COLOR,
    hist_kwargs={"density": True, "alpha": 0.6},
    contourf_kwargs={"alpha": [0.0, 0.25, 0.50]},
    contour_kwargs={"linewidths": 1.4},
    **_corner_kw,
)

# ── apply tick font sizes ─────────────────────────────────────────────────────
for ax in fig.axes:
    ax.tick_params(labelsize=TICK_FONTSIZE)

# ── legend ───────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=DATA_COLOR, linewidth=2, label="Data"),
    Line2D([0], [0], color=FLOW_COLOR, linewidth=2, label="Flow"),
]
fig.legend(
    handles=legend_handles,
    loc="upper right",
    fontsize=LEGEND_FONTSIZE,
    framealpha=0.8,
    bbox_to_anchor=(0.98, 0.98),
)

fig.suptitle("GW190425 (IMRPhenomPNRT, low spin)", fontsize=TITLE_FONTSIZE)
