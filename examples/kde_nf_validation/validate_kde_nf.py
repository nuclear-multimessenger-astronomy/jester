"""
Validation script for KDE and NF methods against original samples.

This script compares:
1. GW likelihoods: Original posterior samples vs Normalizing Flow (NF) samples
2. NICER likelihoods: Original posterior samples vs Kernel Density Estimate (KDE) samples

For each dataset, it generates corner plots with density overlays to visualize
the comparison between truth and approximation methods.

"""

import os
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from jax.scipy.stats import gaussian_kde

# Import JESTER modules
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jesterTOV.inference.flows.flow import Flow


# ============================================================================
# Configuration
# ============================================================================

# Base paths
JESTER_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = JESTER_ROOT / "jesterTOV" / "inference" / "data"
MODELS_DIR = JESTER_ROOT / "jesterTOV" / "inference" / "flows" / "models" / "gw_maf"
OUTPUT_DIR = Path(__file__).parent / "figures"

# GW events configuration
GW_EVENTS = {
    "GW170817": {
        "data_file": DATA_DIR / "gw170817" / "gw170817_gwtc1_lowspin_posterior.npz",
        "model_dir": MODELS_DIR / "gw170817" / "gw170817_gwtc1_lowspin_posterior",
        "parameters": ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
        "labels": [
            r"$m_1$ [$M_\odot$]",
            r"$m_2$ [$M_\odot$]",
            r"$\Lambda_1$",
            r"$\Lambda_2$",
        ],
        "n_samples": 10000,
    },
    "GW190425": {
        "data_file": DATA_DIR / "gw190425" / "gw190425_phenompnrt-ls_posterior.npz",
        "model_dir": MODELS_DIR / "gw190425" / "gw190425_phenompnrt-ls_posterior",
        "parameters": ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
        "labels": [
            r"$m_1$ [$M_\odot$]",
            r"$m_2$ [$M_\odot$]",
            r"$\Lambda_1$",
            r"$\Lambda_2$",
        ],
        "n_samples": 10000,
    },
}

# NICER pulsars configuration
NICER_PULSARS = {
    "J0030": {
        "amsterdam": DATA_DIR
        / "NICER"
        / "J00300451_amsterdam_ST_U_NICER_only_Riley2019.npz",
        "maryland": DATA_DIR / "NICER" / "J00300451_maryland_3spot_NICER_only_RM.npz",
        "parameters": ["mass", "radius"],
        "labels": [r"$M$ [$M_\odot$]", r"$R$ [km]"],
        "n_samples": 10000,
    },
    "J0740": {
        "amsterdam": DATA_DIR
        / "NICER"
        / "J07406620_amsterdam_gamma_NICERXMM_equal_weights_recent.npz",
        "maryland": DATA_DIR / "NICER" / "J07406620_maryland_unknown_NICER_only_RM.npz",
        "parameters": ["mass", "radius"],
        "labels": [r"$M$ [$M_\odot$]", r"$R$ [km]"],
        "n_samples": 10000,
    },
}

# Random seed
SEED = 42


# ============================================================================
# GW Validation Functions
# ============================================================================


def load_gw_original_samples(data_file: Path, parameters: list) -> np.ndarray:
    """
    Load original GW posterior samples.

    Parameters
    ----------
    data_file : Path
        Path to .npz file with posterior samples
    parameters : list
        List of parameter names to load

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, n_params)
    """
    data = np.load(data_file, allow_pickle=True)
    samples = np.column_stack([data[param] for param in parameters])
    return samples


def sample_from_nf(model_dir: Path, n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Sample from trained normalizing flow model.

    Parameters
    ----------
    model_dir : Path
        Directory containing flow_weights.eqx, metadata.json, flow_kwargs.json
    n_samples : int
        Number of samples to draw
    seed : int
        Random seed

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, n_params)
    """
    # Load flow model
    flow = Flow.from_directory(str(model_dir))

    # Sample from flow
    key = jax.random.key(seed)
    samples_jax = flow.sample(key, (n_samples,))

    # Convert to numpy
    samples = np.array(samples_jax)

    return samples


def validate_gw_event(event_name: str, config: Dict) -> None:
    """
    Validate NF approximation for a GW event.

    Parameters
    ----------
    event_name : str
        Name of GW event (e.g., "GW170817")
    config : dict
        Configuration dictionary with data_file, model_dir, parameters, labels, n_samples
    """
    print(f"\n{'='*70}")
    print(f"Validating {event_name} - Normalizing Flow vs Original Samples")
    print(f"{'='*70}")

    # Load original samples
    print(f"Loading original samples from {config['data_file']}...")
    original_samples = load_gw_original_samples(
        config["data_file"], config["parameters"]
    )
    print(
        f"  Loaded {original_samples.shape[0]} samples with {original_samples.shape[1]} parameters"
    )

    # Sample from NF
    print(f"Sampling from NF model at {config['model_dir']}...")
    nf_samples = sample_from_nf(config["model_dir"], config["n_samples"], SEED)
    print(f"  Generated {nf_samples.shape[0]} NF samples")

    # Create corner plot comparison
    print(f"Creating corner plot...")
    fig = plot_corner_comparison(
        original_samples,
        nf_samples,
        config["labels"],
        title=f"{event_name}: Original vs NF",
        truth_label="Original",
        approx_label="NF",
    )

    # Save figure in subdirectory for the event
    model_name = config["model_dir"].name  # e.g., "gw170817_gwtc1_lowspin_posterior"
    output_subdir = OUTPUT_DIR / event_name.lower()
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_file = output_subdir / f"{model_name}_nf_validation.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved figure to {output_file}")
    plt.close(fig)


# ============================================================================
# NICER Validation Functions
# ============================================================================


def load_nicer_original_samples(data_file: Path, parameters: list) -> np.ndarray:
    """
    Load original NICER posterior samples.

    Parameters
    ----------
    data_file : Path
        Path to .npz file with posterior samples
    parameters : list
        List of parameter names to load (e.g., ["mass", "radius"])

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, n_params)
    """
    data = np.load(data_file, allow_pickle=True)
    samples = np.column_stack([data[param] for param in parameters])
    return samples


def sample_from_kde(kde: gaussian_kde, n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Sample from JAX KDE.

    Parameters
    ----------
    kde : gaussian_kde
        Fitted JAX KDE object
    n_samples : int
        Number of samples to draw
    seed : int
        Random seed

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, n_params), transposed to match original format
    """
    key = jax.random.key(seed)
    # JAX KDE.resample returns shape (n_dims, n_samples), transpose to (n_samples, n_dims)
    samples_jax = kde.resample(key, shape=(n_samples,))
    samples = np.array(samples_jax).T  # Transpose to (n_samples, n_params)
    return samples


def validate_nicer_pulsar(pulsar_name: str, config: Dict) -> None:
    """
    Validate KDE approximation for a NICER pulsar.

    Combines Amsterdam and Maryland posteriors (equal weights).

    Parameters
    ----------
    pulsar_name : str
        Name of pulsar (e.g., "J0030")
    config : dict
        Configuration dictionary with amsterdam, maryland, parameters, labels, n_samples
    """
    print(f"\n{'='*70}")
    print(f"Validating NICER {pulsar_name} - KDE vs Original Samples")
    print(f"{'='*70}")

    # Load original samples from both groups
    print(f"Loading Amsterdam samples from {config['amsterdam']}...")
    amsterdam_samples = load_nicer_original_samples(
        config["amsterdam"], config["parameters"]
    )
    print(f"  Loaded {amsterdam_samples.shape[0]} Amsterdam samples")

    print(f"Loading Maryland samples from {config['maryland']}...")
    maryland_samples = load_nicer_original_samples(
        config["maryland"], config["parameters"]
    )
    print(f"  Loaded {maryland_samples.shape[0]} Maryland samples")

    # Construct JAX KDEs (same as in NICERLikelihood)
    print(f"Constructing JAX KDEs...")
    # Convert to JAX arrays and transpose to (n_dims, n_samples) for KDE
    amsterdam_kde = gaussian_kde(jnp.array(amsterdam_samples.T))
    maryland_kde = gaussian_kde(jnp.array(maryland_samples.T))
    print(f"  JAX KDEs constructed successfully")

    # Sample from KDEs (equal weights from each group)
    n_samples_per_group = config["n_samples"] // 2
    print(f"Sampling from KDEs ({n_samples_per_group} from each group)...")
    amsterdam_kde_samples = sample_from_kde(amsterdam_kde, n_samples_per_group, SEED)
    maryland_kde_samples = sample_from_kde(maryland_kde, n_samples_per_group, SEED + 1)
    kde_samples = np.vstack([amsterdam_kde_samples, maryland_kde_samples])
    print(f"  Generated {kde_samples.shape[0]} KDE samples")

    # Combine original samples (equal weights)
    n_original_per_group = (
        min(amsterdam_samples.shape[0], maryland_samples.shape[0]) // 2
    )
    np.random.seed(SEED + 2)
    amsterdam_idx = np.random.choice(
        amsterdam_samples.shape[0], n_original_per_group, replace=False
    )
    maryland_idx = np.random.choice(
        maryland_samples.shape[0], n_original_per_group, replace=False
    )
    original_samples = np.vstack(
        [amsterdam_samples[amsterdam_idx], maryland_samples[maryland_idx]]
    )
    print(f"  Subsampled {original_samples.shape[0]} original samples for comparison")

    # Create corner plot comparison
    print(f"Creating corner plot...")
    fig = plot_corner_comparison(
        original_samples,
        kde_samples,
        config["labels"],
        title=f"NICER {pulsar_name}: Original vs KDE",
        truth_label="Original",
        approx_label="KDE",
    )

    # Save figure in subdirectory named after the Amsterdam dataset
    amsterdam_name = config[
        "amsterdam"
    ].stem  # e.g., "J00300451_amsterdam_ST_U_NICER_only_Riley2019"
    maryland_name = config["maryland"].stem
    output_subdir = OUTPUT_DIR / "nicer" / pulsar_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    # Use both group names in filename for clarity
    output_file = output_subdir / f"{pulsar_name}_amsterdam-maryland_kde_validation.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved figure to {output_file}")
    plt.close(fig)


# ============================================================================
# Plotting Functions
# ============================================================================


def compute_plot_ranges(
    original_samples: np.ndarray,
    approx_samples: np.ndarray,
    percentile: float = 99.9,
) -> list:
    """
    Compute plot ranges for each parameter based on percentiles.

    Takes the widest bounds across both sample sets to ensure full visibility.

    Parameters
    ----------
    original_samples : np.ndarray
        Original posterior samples, shape (n_samples, n_params)
    approx_samples : np.ndarray
        Approximation samples (NF or KDE), shape (n_samples, n_params)
    percentile : float
        Percentile to use for range (default: 99.9 means 0.05% to 99.95%)

    Returns
    -------
    ranges : list
        List of (min, max) tuples for each parameter
    """
    n_params = original_samples.shape[1]
    ranges = []

    lower_percentile = (100 - percentile) / 2.0
    upper_percentile = 100 - lower_percentile

    for i in range(n_params):
        # Compute percentiles for original samples
        orig_lower = np.percentile(original_samples[:, i], lower_percentile)
        orig_upper = np.percentile(original_samples[:, i], upper_percentile)

        # Compute percentiles for approximation samples
        approx_lower = np.percentile(approx_samples[:, i], lower_percentile)
        approx_upper = np.percentile(approx_samples[:, i], upper_percentile)

        # Take widest bounds
        param_min = min(orig_lower, approx_lower)
        param_max = max(orig_upper, approx_upper)

        ranges.append((param_min, param_max))

    return ranges


def plot_corner_comparison(
    original_samples: np.ndarray,
    approx_samples: np.ndarray,
    labels: list,
    title: str,
    truth_label: str = "Truth",
    approx_label: str = "Approximation",
) -> plt.Figure:
    """
    Create corner plot comparing original samples and approximation.

    Parameters
    ----------
    original_samples : np.ndarray
        Original posterior samples, shape (n_samples, n_params)
    approx_samples : np.ndarray
        Approximation samples (NF or KDE), shape (n_samples, n_params)
    labels : list
        Parameter labels for axes
    title : str
        Plot title
    truth_label : str
        Label for original samples in legend
    approx_label : str
        Label for approximation samples in legend

    Returns
    -------
    fig : plt.Figure
        Corner plot figure
    """
    # Compute plot ranges based on 99.9% percentiles
    ranges = compute_plot_ranges(original_samples, approx_samples, percentile=99.9)

    # Create figure with original samples (blue)
    fig = corner(
        original_samples,
        labels=labels,
        color="blue",
        alpha=0.5,
        plot_density=True,
        plot_datapoints=False,
        fill_contours=True,
        levels=[0.68, 0.95],
        smooth=1.0,
        label=truth_label,
        density=True,
        hist_kwargs={"color": "blue", "density": True},
        range=ranges,
    )

    # Overlay approximation samples (red)
    corner(
        approx_samples,
        fig=fig,
        color="red",
        alpha=0.5,
        plot_density=True,
        plot_datapoints=False,
        fill_contours=True,
        levels=[0.68, 0.95],
        smooth=1.0,
        label=approx_label,
        density=True,
        hist_kwargs={"color": "red", "density": True},
        range=ranges,
    )

    # Add title
    fig.suptitle(title, fontsize=16, y=1.0)

    # Add legend manually (corner doesn't propagate labels automatically)
    from matplotlib.patches import Patch

    axes = fig.get_axes()
    legend_elements = [
        Patch(facecolor="blue", alpha=0.5, label=truth_label),
        Patch(facecolor="red", alpha=0.5, label=approx_label),
    ]
    axes[0].legend(
        handles=legend_elements, loc="upper right", fontsize=12, framealpha=0.9
    )

    return fig


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Run validation for all GW events and NICER pulsars."""
    print("KDE and NF Validation Script")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Validate GW events
    print("\n" + "=" * 70)
    print("GRAVITATIONAL WAVE EVENTS")
    print("=" * 70)
    for event_name, config in GW_EVENTS.items():
        try:
            validate_gw_event(event_name, config)
        except Exception as e:
            print(f"ERROR: Failed to validate {event_name}")
            print(f"  {e}")
            import traceback

            traceback.print_exc()

    # Validate NICER pulsars
    print("\n" + "=" * 70)
    print("NICER PULSARS")
    print("=" * 70)
    for pulsar_name, config in NICER_PULSARS.items():
        try:
            validate_nicer_pulsar(pulsar_name, config)
        except Exception as e:
            print(f"ERROR: Failed to validate {pulsar_name}")
            print(f"  {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Validation complete!")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
