"""
Plot NICER mass-radius posteriors as 68% and 90% credible interval contours.

Samples are drawn from the pre-trained normalizing flows (one per PSR/group),
not from the raw posterior npz files.  This mirrors how NICERLikelihood uses
the flows at inference time.

Each pulsar is assigned a single colour.  Where both the Amsterdam (X-PSI) and
Maryland groups have published an independent analysis, both are shown for
that pulsar: Amsterdam as a solid contour, Maryland as a dashed contour. The
preferred/headline analysis is used for each group:
  - PSR J0030+0451  — Amsterdam ST+PST (Riley et al. 2019);
    Maryland 3-spot, full prior (Miller et al. 2019)
  - PSR J0437-4715  — Amsterdam CST+PDT (Choudhury et al. 2024);
    Maryland 3-spot+GPL (Miller, Dittmann, Holt et al. 2026)
  - PSR J0614-3329  — Amsterdam ST+PDT (Mauviard et al. 2025); no Maryland analysis
  - PSR J0740+6620  — Amsterdam gamma, NICER+XMM (Salmi et al. 2024);
    Maryland, NICER+XMM, full prior (Miller et al. 2021)

.. developer note::
   Use LaTeX rendering (``text.usetex = True``) in all docs plot scripts for
   publication-quality typography.  Requires a working LaTeX installation with
   the ``texlive-latex-base``, ``texlive-latex-extra``, ``dvipng``, and
   ``cm-super`` packages, which are available in the CI environment.
"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Use LaTeX rendering for publication-quality text
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }
)

# Locate the trained flow models relative to the installed package
import jesterTOV
from jesterTOV.inference.flows.flow import Flow

FLOWS_DIR = (
    Path(jesterTOV.__file__).parent / "inference" / "flows" / "models" / "nicer_maf"
)

# One entry per pulsar, each assigned a single colour. Within a pulsar, the
# "groups" list holds the preferred Amsterdam (solid) and Maryland (dashed)
# analyses, where available. Column order in flow samples is
# [mass (Msun), radius (km)] — matches the parameter_names field in each
# flow's metadata.json.
PSR_CONFIGS = [
    {
        "label": "PSR J0030+0451",
        "color": "#17becf",
        "groups": [
            {
                "name": "Amsterdam",
                "flow_dir": FLOWS_DIR / "J00300451" / "amsterdam_st_pst",
                "linestyle": "-",
            },
            {
                "name": "Maryland",
                "flow_dir": FLOWS_DIR / "J00300451" / "maryland_3spot_full",
                "linestyle": "--",
            },
        ],
    },
    {
        "label": "PSR J0437-4715",
        "color": "#d62728",
        "groups": [
            {
                "name": "Amsterdam",
                "flow_dir": FLOWS_DIR / "J04374715" / "amsterdam_cst_pdt",
                "linestyle": "-",
            },
            {
                "name": "Maryland",
                "flow_dir": FLOWS_DIR / "J04374715" / "maryland",
                "linestyle": "--",
            },
        ],
    },
    {
        "label": "PSR J0614-3329",
        "color": "#1f77b4",
        "groups": [
            {
                "name": "Amsterdam",
                "flow_dir": FLOWS_DIR / "J06143329" / "amsterdam_st_pdt",
                "linestyle": "-",
            },
        ],
    },
    {
        "label": "PSR J0740+6620",
        "color": "#9467bd",
        "groups": [
            {
                "name": "Amsterdam",
                "flow_dir": FLOWS_DIR / "J07406620" / "amsterdam_gamma_nicerxmm",
                "linestyle": "-",
            },
            {
                "name": "Maryland",
                "flow_dir": FLOWS_DIR / "J07406620" / "maryland_unknown_nicerxmm_full",
                "linestyle": "--",
            },
        ],
    },
]

# Number of samples to draw from each flow for the KDE
N_FLOW_SAMPLES = 50_000

# KDE grid resolution
GRID_SIZE = 150


def kde_credible_levels(
    mass: np.ndarray,
    radius: np.ndarray,
    cis: tuple[float, ...] = (0.68, 0.90),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Return a 2-D KDE on a grid plus the density thresholds for each CI.

    Parameters
    ----------
    mass:
        Neutron-star mass samples [solar masses].
    radius:
        Neutron-star radius samples [km].
    cis:
        Credible intervals to compute, e.g. ``(0.68, 0.90)``.

    Returns
    -------
    R_grid, M_grid : (GRID_SIZE, GRID_SIZE) arrays
        Meshgrid coordinates.
    Z : (GRID_SIZE, GRID_SIZE) array
        KDE density values on the grid.
    levels : list of float
        Density thresholds corresponding to *cis*, sorted ascending (outer first).
    """
    kde = gaussian_kde(np.vstack([radius, mass]))

    r_pad = (radius.max() - radius.min()) * 0.05
    m_pad = (mass.max() - mass.min()) * 0.05
    R_grid, M_grid = np.mgrid[
        radius.min() - r_pad : radius.max() + r_pad : GRID_SIZE * 1j,
        mass.min() - m_pad : mass.max() + m_pad : GRID_SIZE * 1j,
    ]
    Z = kde(np.vstack([R_grid.ravel(), M_grid.ravel()])).reshape(R_grid.shape)

    z_flat = Z.ravel()
    z_desc = np.sort(z_flat)[::-1]
    z_cumsum = np.cumsum(z_desc) / z_desc.sum()

    levels: list[float] = []
    for ci in sorted(cis):  # ascending: 68% then 90%
        idx_ci = int(np.searchsorted(z_cumsum, ci))
        idx_ci = min(idx_ci, len(z_desc) - 1)
        levels.append(float(z_desc[idx_ci]))

    # Return ascending: [L90, L68] — L90 < L68 (lower density = larger region)
    return R_grid, M_grid, Z, sorted(levels)


fig, ax = plt.subplots(figsize=(6.5, 5.5))

sample_counter = 0
groups_seen: set[str] = set()

for cfg in PSR_CONFIGS:
    color: str = cfg["color"]

    for group in cfg["groups"]:
        flow_dir = group["flow_dir"]
        if not flow_dir.exists():
            continue  # skip gracefully if model is not present

        flow = Flow.from_directory(str(flow_dir))
        key = jax.random.key(sample_counter)
        sample_counter += 1
        samples = np.asarray(flow.sample(key, (N_FLOW_SAMPLES,)))
        # Flow samples have shape (N, 2) with columns [mass, radius]
        mass: np.ndarray = samples[:, 0]
        radius: np.ndarray = samples[:, 1]
        linestyle: str = group["linestyle"]
        groups_seen.add(group["name"])

        R_grid, M_grid, Z, levels = kde_credible_levels(mass, radius)

        # Fill 90% CI (outer, lighter) and 68% CI (inner, darker)
        ax.contourf(
            R_grid, M_grid, Z, levels=[levels[0], Z.max()], colors=[color], alpha=0.25
        )
        ax.contourf(
            R_grid, M_grid, Z, levels=[levels[1], Z.max()], colors=[color], alpha=0.50
        )
        # Contour lines, solid for Amsterdam and dashed for Maryland
        ax.contour(
            R_grid,
            M_grid,
            Z,
            levels=levels,
            colors=[color],
            linewidths=1.5,
            linestyles=linestyle,
        )

ax.set_xlabel(r"Radius [km]", fontsize=12)
ax.set_ylabel(r"Mass [$M_\odot$]", fontsize=12)
ax.set_xlim(8.0, 17.5)
ax.set_ylim(1.0, 2.5)
ax.tick_params(labelsize=11)

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# CI shading legend (top-left)
ci_handles = [
    Patch(facecolor="grey", alpha=0.50, label="68\% CI"),
    Patch(facecolor="grey", alpha=0.25, label="90\% CI"),
]
legend_ci = ax.legend(handles=ci_handles, loc="upper left", framealpha=0.8, fontsize=16)
ax.add_artist(legend_ci)

# PSR legend (top-right): one filled square per pulsar
psr_handles = [
    Patch(facecolor=cfg["color"], alpha=0.6, label=cfg["label"]) for cfg in PSR_CONFIGS
]
legend_psr = ax.legend(
    handles=psr_handles, loc="upper right", framealpha=0.8, fontsize=10
)
ax.add_artist(legend_psr)

# Group legend (lower-right): black solid/dashed lines for Amsterdam/Maryland,
# only shown if both groups actually appear in the plot
if {"Amsterdam", "Maryland"} <= groups_seen:
    group_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="Amsterdam"),
        Line2D([0], [0], color="black", linestyle="--", label="Maryland"),
    ]
    ax.legend(handles=group_handles, loc="lower right", framealpha=0.8, fontsize=10)

fig.tight_layout()
