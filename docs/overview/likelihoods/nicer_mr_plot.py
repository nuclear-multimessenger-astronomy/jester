"""
Plot NICER mass-radius posteriors as 68% and 90% credible interval contours.

Samples are drawn from the pre-trained normalizing flows (one per PSR/group),
not from the raw posterior npz files.  This mirrors how NICERLikelihood uses
the flows at inference time.

One representative analysis is chosen per pulsar:
  - PSR J0030+0451  — Amsterdam ST+PST, Riley et al. 2019
  - PSR J0437-4715  — Amsterdam CST+PDT, Choudhury et al. 2024
  - PSR J0614-3329  — Amsterdam ST+PDT, Dittmann et al. 2025
  - PSR J0740+6620  — Amsterdam gamma NICER+XMM, Salmi et al. 2024
"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Locate the trained flow models relative to the installed package
import jesterTOV
from jesterTOV.inference.flows.flow import Flow

FLOWS_DIR = (
    Path(jesterTOV.__file__).parent / "inference" / "flows" / "models" / "nicer_maf"
)

# One trained flow directory per pulsar with display name and colour.
# Column order in flow samples: [mass (Msun), radius (km)] — matches
# the parameter_names field in each flow's metadata.json.
PSR_CONFIGS = [
    {
        "label": "PSR J0030+0451",
        "flow_dir": FLOWS_DIR / "J00300451" / "amsterdam_st_pst",
        "color": "#17becf",
    },
    {
        "label": "PSR J0437-4715",
        "flow_dir": FLOWS_DIR / "J04374715" / "amsterdam_cst_pdt",
        "color": "#d62728",
    },
    {
        "label": "PSR J0614-3329",
        "flow_dir": FLOWS_DIR / "J06143329" / "amsterdam_st_pdt",
        "color": "#1f77b4",
    },
    {
        "label": "PSR J0740+6620",
        "flow_dir": FLOWS_DIR / "J07406620" / "amsterdam_gamma_nicerxmm",
        "color": "#9467bd",
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

for i, cfg in enumerate(PSR_CONFIGS):
    flow_dir = cfg["flow_dir"]
    if not flow_dir.exists():
        continue  # skip gracefully if model is not present

    flow = Flow.from_directory(str(flow_dir))
    key = jax.random.key(i)
    samples = np.asarray(flow.sample(key, (N_FLOW_SAMPLES,)))
    # Flow samples have shape (N, 2) with columns [mass, radius]
    mass: np.ndarray = samples[:, 0]
    radius: np.ndarray = samples[:, 1]
    color: str = cfg["color"]

    R_grid, M_grid, Z, levels = kde_credible_levels(mass, radius)

    # Fill 90% CI (outer, lighter)
    ax.contourf(
        R_grid, M_grid, Z, levels=[levels[0], Z.max()], colors=[color], alpha=0.25
    )
    # Fill 68% CI (inner, darker)
    ax.contourf(
        R_grid, M_grid, Z, levels=[levels[1], Z.max()], colors=[color], alpha=0.50
    )
    # Draw both contour lines
    ax.contour(R_grid, M_grid, Z, levels=levels, colors=[color], linewidths=1.5)

    # Invisible proxy artist for the pulsar legend
    ax.fill_between([], [], color=color, alpha=0.6, label=cfg["label"])

ax.set_xlabel("Radius [km]", fontsize=12)
ax.set_ylabel(r"Mass [$M_\odot$]", fontsize=12)
ax.set_xlim(8.0, 16.0)
ax.set_ylim(1.0, 2.5)
ax.tick_params(labelsize=11)

# CI shading legend (top-left)
from matplotlib.patches import Patch  # noqa: E402

ci_handles = [
    Patch(facecolor="grey", alpha=0.50, label="68% CI"),
    Patch(facecolor="grey", alpha=0.25, label="90% CI"),
]
legend_ci = ax.legend(handles=ci_handles, loc="upper left", framealpha=0.8, fontsize=10)
ax.add_artist(legend_ci)

# PSR legend (top-right)
ax.legend(loc="upper right", framealpha=0.8, fontsize=10)

fig.tight_layout()
