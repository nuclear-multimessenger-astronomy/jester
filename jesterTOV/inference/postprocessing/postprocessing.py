r"""Modular postprocessing script for EOS inference results.

This script provides comprehensive visualization tools for analyzing equation of state (EOS)
inference results. It generates various plots including cornerplots, mass-radius diagrams,
pressure-density relationships, and speed of sound squared vs density with posterior
probability color coding.

Usage:
    run_jester_postprocessing --outdir <path> [--make-cornerplot] [--make-massradius] [--make-pressuredensity] [--make-cs2]

Example:
    run_jester_postprocessing --outdir ./results --make-all
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import corner
import os
import argparse
from scipy.stats import gaussian_kde
from typing import Dict, Optional, Any
import warnings
import arviz as az

np.random.seed(2)
import jesterTOV.utils as utils
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


# Configure matplotlib with TeX support and fallback
def setup_matplotlib(use_tex: bool = True):
    """Configure matplotlib plotting parameters with TeX fallback.

    Parameters
    ----------
    use_tex : bool, optional
        Whether to attempt using LaTeX rendering, by default True

    Returns
    -------
    bool
        True if TeX is successfully enabled, False otherwise
    """
    tex_enabled = False

    if use_tex:
        try:
            # Try to enable TeX
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Serif"],
                }
            )
            # Test if TeX actually works
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, r"$\alpha$")
            plt.close(fig)
            tex_enabled = True
            logger.info("TeX rendering enabled")
        except Exception as e:
            warnings.warn(
                f"TeX rendering failed ({e}). Falling back to non-TeX rendering."
            )
            plt.rcParams.update(
                {
                    "text.usetex": False,
                    "font.family": "sans-serif",
                }
            )

    # Common matplotlib parameters
    mpl_params = {
        "axes.grid": False,
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16,
    }
    plt.rcParams.update(mpl_params)

    return tex_enabled


# Initialize matplotlib with TeX fallback
TEX_ENABLED = setup_matplotlib(use_tex=True)

# Default colormap
DEFAULT_COLORMAP = sns.color_palette("crest", as_cmap=True)

# Constants
COLORS_DICT = {"prior": "gray", "posterior": "blue"}
ALPHA = 0.3
figsize_vertical = (6, 8)
figsize_horizontal = (8, 6)

# Injection EOS plotting style (consistent across all plots)
INJECTION_COLOR = "black"
INJECTION_LINESTYLE = "--"
INJECTION_LINEWIDTH = 2.5
INJECTION_ALPHA = 0.8

# Credible interval probability (used in histograms and contour plots)
HDI_PROB = 0.90  # 90% highest density interval

# Default plot bounding boxes
# These are the default ranges for mass-radius plots
M_MIN = 0.75  # Minimum mass [M_sun]
M_MAX = 3.5  # Maximum mass [M_sun]
R_MIN = 6.0  # Minimum radius [km]
R_MAX = 18.0  # Maximum radius [km]

# Prior directory (for loading prior samples)
PRIOR_DIR = "./outdir/"


def load_eos_data(outdir: str) -> Dict[str, np.ndarray]:
    """Load EOS data from the specified output directory.

    Parameters
    ----------
    outdir : str
        Path to output directory containing results.h5

    Returns
    -------
    dict
        Dictionary containing EOS data arrays

    Raises
    ------
    FileNotFoundError
        If results.h5 is not found in the specified directory
    """
    from jesterTOV.inference.result import InferenceResult

    filename = os.path.join(outdir, "results.h5")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file not found: {filename}")

    # Load HDF5 results
    result = InferenceResult.load(filename)

    # Load macroscopic quantities
    m = result.posterior["masses_EOS"]
    r = result.posterior["radii_EOS"]
    l = result.posterior["Lambdas_EOS"]
    n = result.posterior["n"]
    p = result.posterior["p"]
    e = result.posterior["e"]
    cs2 = result.posterior["cs2"]

    # Convert units
    n = n / utils.fm_inv3_to_geometric / 0.16
    p = p / utils.MeV_fm_inv3_to_geometric
    e = e / utils.MeV_fm_inv3_to_geometric

    log_prob = result.posterior["log_prob"]

    # Load prior parameters directly from saved parameter names (no magic!)
    # This works for any EOS parametrization (NEP, spectral, CSE, etc.)
    prior_params = {}
    parameter_names = result.metadata.get("parameter_names", [])

    if parameter_names:
        logger.info(
            f"Found {len(parameter_names)} parameter names in metadata: {parameter_names}"
        )
        for key in parameter_names:
            if key in result.posterior:
                prior_params[key] = result.posterior[key]
    else:
        logger.warning(
            "No parameter_names found in metadata. Cornerplot may be empty. This may occur if results were saved with an older version of JESTER."
        )

    output = {
        "masses": m,
        "radii": r,
        "lambdas": l,
        "densities": n,
        "pressures": p,
        "energies": e,
        "cs2": cs2,
        "log_prob": log_prob,
        "prior_params": prior_params,  # General key for all parameters
    }

    return output


def load_prior_data(prior_dir: str = PRIOR_DIR) -> Optional[Dict[str, np.ndarray]]:
    """Load prior EOS data for comparison.

    Parameters
    ----------
    prior_dir : str, optional
        Path to prior output directory

    Returns
    -------
    dict or None
        Prior data dictionary, or None if not found
    """
    try:
        return load_eos_data(prior_dir)
    except FileNotFoundError:
        logger.warning(f"Prior data not found at {prior_dir}")
        return None


def load_injection_eos(
    injection_path: Optional[str],
) -> Optional[Dict[str, np.ndarray]]:
    r"""Load injection EOS data from NPZ file.

    Parameters
    ----------
    injection_path : str or None
        Path to NPZ file containing injection EOS data

    Returns
    -------
    dict or None
        Dictionary containing injection EOS data arrays, or None if loading fails.
        Expected keys: masses_EOS, radii_EOS, Lambda_EOS, n, p, e, cs2

    Notes
    -----
    **Units:** The injection file should contain data in **geometric units**:
    - masses_EOS: Solar masses :math:`M_{\odot}`
    - radii_EOS: :math:`\mathrm{km}`
    - Lambda_EOS: dimensionless tidal deformability
    - n: geometric units :math:`m^{-2}`, will be converted to n_sat
    - p: geometric units :math:`m^{-2}`, will be converted to :math:`\mathrm{MeV\ fm^{-3}}`
    - e: geometric units :math:`m^{-2}`, will be converted to :math:`\mathrm{MeV\ fm^{-3}}`
    - cs2: dimensionless (speed of sound squared)

    This matches the format used by:
    - LALSuite EOS tables (extracted with lalsimulation)
    - JESTER HDF5 output files (results.h5)

    Missing keys are handled gracefully. If the file doesn't exist or
    can't be loaded, a warning is logged and None is returned. If the file loads
    but is missing expected keys, those keys are omitted from the output.
    """
    if injection_path is None:
        return None

    try:
        # Load NPZ file
        with np.load(injection_path) as data:
            logger.info(f"Loaded injection EOS from {injection_path}")
            logger.info(f"Available keys: {list(data.keys())}")

            # Expected keys that match jester output format
            expected_keys = [
                "masses_EOS",
                "radii_EOS",
                "Lambda_EOS",
                "n",
                "p",
                "e",
                "cs2",
            ]

            # Build output dictionary with available keys
            output = {}
            missing_keys = []

            for key in expected_keys:
                if key in data:
                    # Handle both single curves and multiple samples
                    arr = data[key]
                    if arr.ndim == 1:
                        # Single curve - wrap in extra dimension for consistency
                        output[key] = arr[np.newaxis, :]
                    else:
                        output[key] = arr
                else:
                    missing_keys.append(key)

        # Apply unit conversions for density, pressure, and energy (same as load_eos_data)
        if "n" in output:
            output["n"] = output["n"] / utils.fm_inv3_to_geometric / 0.16
        if "p" in output:
            output["p"] = output["p"] / utils.MeV_fm_inv3_to_geometric
        if "e" in output:
            output["e"] = output["e"] / utils.MeV_fm_inv3_to_geometric

        if missing_keys:
            logger.warning(
                f"Injection EOS file missing some keys: {missing_keys}. "
                f"Available keys: {list(output.keys())}"
            )

        if not output:
            logger.error(
                f"Injection EOS file contains none of the expected keys: {expected_keys}"
            )
            return None

        logger.info(f"Loaded injection EOS with keys: {list(output.keys())}")
        return output

    except FileNotFoundError:
        logger.warning(f"Injection EOS file not found: {injection_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load injection EOS from {injection_path}: {e}")
        return None


def report_credible_interval(
    values: np.ndarray, hdi_prob: float = HDI_PROB, verbose: bool = False
) -> tuple:
    """Calculate credible intervals for given values.

    Parameters
    ----------
    values : np.ndarray
        Array of parameter values
    hdi_prob : float, optional
        Highest density interval probability, by default 0.90
    verbose : bool, optional
        Whether to print results, by default False

    Returns
    -------
    tuple
        (low, median, high) credible interval bounds
    """
    med = np.median(values)
    low_percentile = (1 - hdi_prob) / 2 * 100
    high_percentile = (1 + hdi_prob) / 2 * 100

    low = np.percentile(values, low_percentile)
    high = np.percentile(values, high_percentile)

    low_err = med - low
    high_err = high - med

    if verbose:
        logger.info(
            f"{med:.2f} -{low_err:.2f} +{high_err:.2f} (at {hdi_prob} HDI prob)"
        )

    return low_err, med, high_err


def make_cornerplot(
    data: Dict[str, Any], outdir: str, max_params: Optional[int] = None
):
    """Create cornerplot for EOS parameters.

    Parameters
    ----------
    data : dict
        EOS data dictionary from load_eos_data
    outdir : str
        Output directory for saving the plot
    max_params : int, optional
        Maximum number of parameters to include. If None, includes all parameters.
    """
    logger.info("Creating cornerplot...")

    # Collect parameters for cornerplot from prior_params (works for any EOS)
    samples_dict = {}
    labels = []

    prior_params = data.get("prior_params", {})
    for key in prior_params.keys():
        samples_dict[key] = prior_params[key]

        # Format labels based on parameter name
        if TEX_ENABLED:
            # Handle common formatting patterns
            if "_" in key:
                # Format: "K_sat" -> "$K_{sat}$", "gamma_0" -> "$\gamma_0$"
                base = key.split("_")[0]
                sub = "_".join(key.split("_")[1:])

                # Escape underscores in subscript to avoid double subscript errors
                # e.g., "CSE_0_u" -> "CSE\_0\_u" for LaTeX
                sub_escaped = sub.replace("_", r"\_")

                # Greek letters
                if base == "gamma":
                    labels.append(f"$\\gamma_{{{sub_escaped}}}$")
                elif base == "nbreak":
                    labels.append(r"$n_{\rm{break}}$")
                else:
                    labels.append(f"${base}_{{{sub_escaped}}}$")
            else:
                labels.append(f"${key}$")
        else:
            labels.append(key)

    # Limit number of parameters if specified
    if max_params is not None and len(samples_dict) > max_params:
        logger.info(f"Limiting cornerplot to first {max_params} parameters")
        samples_dict = dict(list(samples_dict.items())[:max_params])
        labels = labels[:max_params]

    if len(samples_dict) == 0:
        logger.warning("No parameters found for cornerplot")
        return

    logger.info(f"Creating cornerplot with {len(samples_dict)} parameters")

    # Convert to array
    samples = np.column_stack([samples_dict[key] for key in samples_dict.keys()])

    # Create cornerplot
    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        color=COLORS_DICT["posterior"],
        plot_datapoints=True,
        fill_contours=True,
        levels=(0.68, 0.95),
        smooth=1.0,
    )

    # Save figure
    save_name = os.path.join(outdir, "cornerplot.pdf")
    fig.savefig(save_name, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Cornerplot saved to {save_name}")


def make_mass_radius_plot(
    data: Dict[str, Any],
    prior_data: Optional[Dict[str, Any]],
    outdir: str,
    use_crest_cmap: bool = True,
    injection_data: Optional[Dict[str, Any]] = None,
):
    """Create mass-radius plot with posterior probability coloring.

    Parameters
    ----------
    data : dict
        EOS data dictionary
    prior_data : dict or None
        Prior EOS data for comparison
    outdir : str
        Output directory
    use_crest_cmap : bool, optional
        Whether to use seaborn crest colormap, by default True
    injection_data : dict or None, optional
        Injection EOS data for plotting true values, by default None
    """
    logger.info("Creating mass-radius plot...")

    plt.figure(figsize=(10, 8))
    m_min, m_max = M_MIN, M_MAX
    r_min, r_max = R_MIN, R_MAX

    # Plot prior first (background)
    if prior_data is not None:
        m_prior, r_prior = prior_data["masses"], prior_data["radii"]
        for i in range(len(m_prior)):
            plt.plot(
                r_prior[i],
                m_prior[i],
                color=COLORS_DICT["prior"],
                alpha=0.1,
                rasterized=True,
                zorder=1,
            )

    # Plot posterior with probability coloring
    m, r, l = data["masses"], data["radii"], data["lambdas"]
    log_prob = data["log_prob"]
    nb_samples = np.shape(m)[0]
    logger.info(f"Number of samples: {nb_samples}")

    # Verify log_prob matches EOS sample count
    if len(log_prob) != nb_samples:
        raise ValueError(
            f"Mismatch between log_prob ({len(log_prob)}) and EOS samples ({nb_samples}). "
            "This indicates a bug in the EOS sample generation code."
        )

    # Normalize probabilities for coloring
    prob = np.exp(log_prob - np.max(log_prob))  # Normalize to avoid overflow
    norm = Normalize(vmin=np.min(prob), vmax=np.max(prob))
    cmap = DEFAULT_COLORMAP if use_crest_cmap else plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # First pass: identify valid samples and find maximum MTOV
    valid_indices = []
    max_mtov = 0.0
    for i in range(len(prob)):
        # Skip invalid samples (same checks as plotting loop)
        if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
            continue
        if any(l[i] < 0):
            continue
        if any((m[i] > M_MIN) * (r[i] > R_MAX)):
            continue

        # This is a valid sample
        valid_indices.append(i)
        mtov = np.max(m[i])
        if mtov > max_mtov:
            max_mtov = mtov

    # Dynamically widen m_max if needed
    if max_mtov > m_max:
        m_max = max_mtov + 0.25
        logger.info(
            f"Widening mass axis to {m_max:.2f} M_sun (max MTOV: {max_mtov:.2f})"
        )

    bad_counter = nb_samples - len(valid_indices)
    logger.info(
        f"Plotting {len(valid_indices)} M-R curves (excluded {bad_counter} invalid samples)..."
    )

    # Second pass: plot only valid samples
    for i in valid_indices:
        # Get color based on probability
        normalized_value = norm(prob[i])
        color = cmap(normalized_value)

        plt.plot(
            r[i],
            m[i],
            color=color,
            alpha=1.0,
            rasterized=True,
            zorder=1e10 + normalized_value,
        )

    # Plot injection EOS if provided (on top of everything else)
    if injection_data is not None:
        if "masses_EOS" in injection_data and "radii_EOS" in injection_data:
            m_inj = injection_data["masses_EOS"]
            r_inj = injection_data["radii_EOS"]
            logger.info(f"Plotting injection EOS with {len(m_inj)} curves")
            for i in range(len(m_inj)):
                plt.plot(
                    r_inj[i],
                    m_inj[i],
                    color=INJECTION_COLOR,
                    alpha=INJECTION_ALPHA,
                    linewidth=INJECTION_LINEWIDTH,
                    linestyle=INJECTION_LINESTYLE,
                    zorder=1e11,  # Plot on top of everything
                    label="Injection" if i == 0 else "",
                )

    # Styling
    xlabel = r"$R$ [km]" if TEX_ENABLED else "R [km]"
    ylabel = r"$M$ [$M_{\odot}$]" if TEX_ENABLED else "M [M_sun]"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(r_min, r_max)
    plt.ylim(m_min, m_max)

    # Add colorbar
    fig = plt.gcf()
    sm.set_array([])
    cbar_ax = fig.add_axes((0.15, 0.94, 0.7, 0.03))  # tuple for type checker
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalized posterior probability", fontsize=16)
    cbar.set_ticks([])
    cbar.ax.xaxis.labelpad = 5
    cbar.ax.tick_params(labelsize=0, length=0)
    cbar.ax.xaxis.set_label_position("top")

    # Add legend for prior and/or injection
    if prior_data is not None or injection_data is not None:
        from matplotlib.lines import Line2D

        legend_elements = []
        if prior_data is not None:
            legend_elements.append(
                Line2D(
                    [0], [0], color=COLORS_DICT["prior"], lw=2, alpha=0.7, label="Prior"
                )
            )
        if injection_data is not None and "masses_EOS" in injection_data:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=INJECTION_COLOR,
                    lw=INJECTION_LINEWIDTH,
                    linestyle=INJECTION_LINESTYLE,
                    alpha=INJECTION_ALPHA,
                    label="Injection",
                )
            )
        plt.legend(handles=legend_elements, loc="upper right")

    # Save figure
    save_name = os.path.join(outdir, "mass_radius_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Mass-radius plot saved to {save_name}")


def make_mass_lambda_plot(
    data: Dict[str, Any],
    prior_data: Optional[Dict[str, Any]],
    outdir: str,
    use_crest_cmap: bool = True,
    injection_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Create mass-Lambda plot with posterior probability coloring.

    Parameters
    ----------
    data : dict
        EOS data dictionary
    prior_data : dict or None
        Prior EOS data for comparison
    outdir : str
        Output directory
    use_crest_cmap : bool, optional
        Whether to use seaborn crest colormap, by default True
    injection_data : dict or None, optional
        Injection EOS data for plotting true values, by default None
    """
    logger.info("Creating mass-Lambda plot...")

    plt.figure(figsize=(10, 8))
    m_min, m_max = M_MIN, M_MAX

    # Plot prior first (background)
    if prior_data is not None:
        m_prior, l_prior = prior_data["masses"], prior_data["lambdas"]
        for i in range(len(m_prior)):
            plt.plot(
                m_prior[i],
                l_prior[i],
                color=COLORS_DICT["prior"],
                alpha=0.1,
                rasterized=True,
                zorder=1,
            )

    # Plot posterior with probability coloring
    m, r, l = data["masses"], data["radii"], data["lambdas"]
    log_prob = data["log_prob"]
    nb_samples = np.shape(m)[0]
    logger.info(f"Number of samples: {nb_samples}")

    # Verify log_prob matches EOS sample count
    if len(log_prob) != nb_samples:
        raise ValueError(
            f"Mismatch between log_prob ({len(log_prob)}) and EOS samples ({nb_samples}). "
            "This indicates a bug in the EOS sample generation code."
        )

    # Normalize probabilities for coloring
    prob = np.exp(log_prob - np.max(log_prob))  # Normalize to avoid overflow
    norm = Normalize(vmin=np.min(prob), vmax=np.max(prob))
    cmap = DEFAULT_COLORMAP if use_crest_cmap else plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # First pass: identify valid samples and find maximum MTOV
    valid_indices = []
    max_mtov = 0.0
    for i in range(len(prob)):
        # Skip invalid samples (same checks as plotting loop)
        if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
            continue
        if any(l[i] < 0):
            continue
        if any((m[i] > M_MIN) * (r[i] > R_MAX)):
            continue

        # This is a valid sample
        valid_indices.append(i)
        mtov = np.max(m[i])
        if mtov > max_mtov:
            max_mtov = mtov

    # Dynamically widen m_max if needed
    if max_mtov > m_max:
        m_max = max_mtov + 0.25
        logger.info(
            f"Widening mass axis to {m_max:.2f} M_sun (max MTOV: {max_mtov:.2f})"
        )

    bad_counter = nb_samples - len(valid_indices)
    logger.info(
        f"Plotting {len(valid_indices)} M-Lambda curves (excluded {bad_counter} invalid samples)..."
    )

    # Second pass: plot only valid samples
    for i in valid_indices:
        # Get color based on probability
        normalized_value = norm(prob[i])
        color = cmap(normalized_value)

        plt.plot(
            m[i],
            l[i],
            color=color,
            alpha=1.0,
            rasterized=True,
            zorder=1e10 + normalized_value,
        )

    # Plot injection EOS if provided (on top of everything else)
    if injection_data is not None:
        if "masses_EOS" in injection_data and "Lambda_EOS" in injection_data:
            m_inj = injection_data["masses_EOS"]
            l_inj = injection_data["Lambda_EOS"]
            logger.info(f"Plotting injection EOS with {len(m_inj)} curves")
            for i in range(len(m_inj)):
                plt.plot(
                    m_inj[i],
                    l_inj[i],
                    color=INJECTION_COLOR,
                    alpha=INJECTION_ALPHA,
                    linewidth=INJECTION_LINEWIDTH,
                    linestyle=INJECTION_LINESTYLE,
                    zorder=1e11,  # Plot on top of everything
                    label="Injection" if i == 0 else "",
                )

    # Styling
    xlabel = r"$M$ [$M_{\odot}$]" if TEX_ENABLED else "M [M_sun]"
    ylabel = r"$\Lambda$" if TEX_ENABLED else "Lambda"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(m_min, m_max)
    plt.yscale("log")

    # Add colorbar
    fig = plt.gcf()
    sm.set_array([])
    cbar_ax = fig.add_axes((0.15, 0.94, 0.7, 0.03))  # tuple for type checker
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalized posterior probability", fontsize=16)
    cbar.set_ticks([])
    cbar.ax.xaxis.labelpad = 5
    cbar.ax.tick_params(labelsize=0, length=0)
    cbar.ax.xaxis.set_label_position("top")

    # Add legend for prior and/or injection
    if prior_data is not None or injection_data is not None:
        from matplotlib.lines import Line2D

        legend_elements = []
        if prior_data is not None:
            legend_elements.append(
                Line2D(
                    [0], [0], color=COLORS_DICT["prior"], lw=2, alpha=0.7, label="Prior"
                )
            )
        if injection_data is not None and "masses_EOS" in injection_data:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=INJECTION_COLOR,
                    lw=INJECTION_LINEWIDTH,
                    linestyle=INJECTION_LINESTYLE,
                    alpha=INJECTION_ALPHA,
                    label="Injection",
                )
            )
        plt.legend(handles=legend_elements, loc="upper right")

    # Save figure
    save_name = os.path.join(outdir, "mass_lambda_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Mass-Lambda plot saved to {save_name}")


def make_pressure_density_plot(
    data: Dict[str, Any],
    prior_data: Optional[Dict[str, Any]],
    outdir: str,
    use_crest_cmap: bool = True,
    injection_data: Optional[Dict[str, Any]] = None,
):
    """Create equation of state plot (pressure vs density) with EOS color coding.

    Parameters
    ----------
    data : dict
        EOS data dictionary
    prior_data : dict or None
        Prior EOS data for comparison
    outdir : str
        Output directory
    use_crest_cmap : bool, optional
        Whether to use seaborn crest colormap, by default True
    injection_data : dict or None, optional
        Injection EOS data for plotting true values, by default None
    """
    logger.info("Creating pressure-density plot...")

    plt.figure(figsize=(11, 6))

    # Plot prior first (background)
    if prior_data is not None:
        n_prior, p_prior = prior_data["densities"], prior_data["pressures"]
        for i in range(len(n_prior)):
            mask = (n_prior[i] > 0.5) * (n_prior[i] < 6.0)
            plt.plot(
                n_prior[i][mask],
                p_prior[i][mask],
                color=COLORS_DICT["prior"],
                alpha=0.1,
                rasterized=True,
                zorder=1,
            )

    # Plot posterior with probability coloring
    m, r, l = data["masses"], data["radii"], data["lambdas"]
    n, p = data["densities"], data["pressures"]
    log_prob = data["log_prob"]
    nb_samples = np.shape(m)[0]

    # Verify log_prob matches EOS sample count
    if len(log_prob) != nb_samples:
        raise ValueError(
            f"Mismatch between log_prob ({len(log_prob)}) and EOS samples ({nb_samples}). "
            "This indicates a bug in the EOS sample generation code."
        )

    # Normalize probabilities for coloring
    prob = np.exp(log_prob - np.max(log_prob))
    norm = Normalize(vmin=np.min(prob), vmax=np.max(prob))
    cmap = (
        DEFAULT_COLORMAP
        if use_crest_cmap
        else sns.color_palette("viridis", as_cmap=True)
    )

    bad_counter = 0
    logger.info(f"Plotting {len(prob)} p-n curves...")
    for i in range(len(prob)):
        # Skip invalid samples
        if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
            bad_counter += 1
            continue

        if any(l[i] < 0):
            bad_counter += 1
            continue

        # Exclude samples with R > R_MAX for M > M_MIN
        # This can sometimes happen due to numerical issues in the TOV solver,
        # but we know physically this should not be possible for realistic neutron stars
        if any((m[i] > M_MIN) * (r[i] > R_MAX)):
            bad_counter += 1
            continue

        # Get color and plot
        normalized_value = norm(prob[i])
        color = cmap(normalized_value)

        mask = (n[i] > 0.5) * (n[i] < 6.0)
        plt.plot(
            n[i][mask],
            p[i][mask],
            color=color,
            alpha=1.0,
            rasterized=True,
            zorder=1e10 + normalized_value,
        )

    logger.info(f"Excluded {bad_counter} invalid samples")

    # Plot injection EOS if provided (on top of everything else)
    if injection_data is not None:
        if "n" in injection_data and "p" in injection_data:
            n_inj = injection_data["n"]
            p_inj = injection_data["p"]
            logger.info(f"Plotting injection EOS with {len(n_inj)} curves")
            for i in range(len(n_inj)):
                mask = (n_inj[i] > 0.5) * (n_inj[i] < 6.0)
                plt.plot(
                    n_inj[i][mask],
                    p_inj[i][mask],
                    color=INJECTION_COLOR,
                    alpha=INJECTION_ALPHA,
                    linewidth=INJECTION_LINEWIDTH,
                    linestyle=INJECTION_LINESTYLE,
                    zorder=1e11,  # Plot on top of everything
                    label="Injection" if i == 0 else "",
                )

    xlabel = r"$n$ [$n_{\rm{sat}}$]" if TEX_ENABLED else "n [n_sat]"
    ylabel = r"$p$ [MeV fm$^{-3}$]" if TEX_ENABLED else "p [MeV fm^-3]"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.xlim(0.5, 6.0)

    # Add legend for prior and/or injection
    if prior_data is not None or injection_data is not None:
        from matplotlib.lines import Line2D

        legend_elements = []
        if prior_data is not None:
            legend_elements.append(
                Line2D(
                    [0], [0], color=COLORS_DICT["prior"], lw=2, alpha=0.7, label="Prior"
                )
            )
        if injection_data is not None and "n" in injection_data:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=INJECTION_COLOR,
                    lw=INJECTION_LINEWIDTH,
                    linestyle=INJECTION_LINESTYLE,
                    alpha=INJECTION_ALPHA,
                    label="Injection",
                )
            )
        plt.legend(handles=legend_elements, loc="upper left")

    # Save figure
    save_name = os.path.join(outdir, "pressure_density_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Pressure-density plot saved to {save_name}")


# TODO: Fill histograms between the credible interval edges and not fill outside of it
def make_cs2_plot(
    data: Dict[str, Any],
    prior_data: Optional[Dict[str, Any]],
    outdir: str,
    use_crest_cmap: bool = True,
    injection_data: Optional[Dict[str, Any]] = None,
):
    """Create plot of speed of sound squared vs density.

    Parameters
    ----------
    data : dict
        EOS data dictionary
    prior_data : dict or None
        Prior EOS data for comparison
    outdir : str
        Output directory
    use_crest_cmap : bool, optional
        Whether to use seaborn crest colormap, by default True
    injection_data : dict or None, optional
        Injection EOS data for plotting true values, by default None
    """
    logger.info("Creating cs2-density plot...")

    plt.figure(figsize=(11, 6))

    # Plot prior first (background)
    if prior_data is not None:
        n_prior, cs2_prior = prior_data["densities"], prior_data["cs2"]
        for i in range(len(n_prior)):
            mask = (n_prior[i] > 0.5) * (n_prior[i] < 6.0)
            plt.plot(
                n_prior[i][mask],
                cs2_prior[i][mask],
                color=COLORS_DICT["prior"],
                alpha=0.1,
                rasterized=True,
                zorder=1,
            )

    # Plot posterior with probability coloring
    m, r, l = data["masses"], data["radii"], data["lambdas"]
    n, cs2 = data["densities"], data["cs2"]
    log_prob = data["log_prob"]
    nb_samples = np.shape(m)[0]

    # Verify log_prob matches EOS sample count
    if len(log_prob) != nb_samples:
        raise ValueError(
            f"Mismatch between log_prob ({len(log_prob)}) and EOS samples ({nb_samples}). "
            "This indicates a bug in the EOS sample generation code."
        )

    # Normalize probabilities for coloring
    prob = np.exp(log_prob - np.max(log_prob))
    norm = Normalize(vmin=np.min(prob), vmax=np.max(prob))
    cmap = (
        DEFAULT_COLORMAP
        if use_crest_cmap
        else sns.color_palette("viridis", as_cmap=True)
    )

    bad_counter = 0
    logger.info(f"Plotting {len(prob)} cs2-n curves...")
    for i in range(len(prob)):
        # Skip invalid samples
        if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
            bad_counter += 1
            continue

        if any(l[i] < 0):
            bad_counter += 1
            continue

        # Exclude samples with R > R_MAX for M > M_MIN
        # This can sometimes happen due to numerical issues in the TOV solver,
        # but we know physically this should not be possible for realistic neutron stars
        if any((m[i] > M_MIN) * (r[i] > R_MAX)):
            bad_counter += 1
            continue

        # Get color and plot
        normalized_value = norm(prob[i])
        color = cmap(normalized_value)

        mask = (n[i] > 0.5) * (n[i] < 6.0)
        plt.plot(
            n[i][mask],
            cs2[i][mask],
            color=color,
            alpha=1.0,
            rasterized=True,
            zorder=1e10 + normalized_value,
        )

    logger.info(f"Excluded {bad_counter} invalid samples")

    # Plot injection EOS if provided (on top of everything else)
    if injection_data is not None:
        if "n" in injection_data and "cs2" in injection_data:
            n_inj = injection_data["n"]
            cs2_inj = injection_data["cs2"]
            logger.info(f"Plotting injection EOS with {len(n_inj)} curves")
            for i in range(len(n_inj)):
                mask = (n_inj[i] > 0.5) * (n_inj[i] < 6.0)
                plt.plot(
                    n_inj[i][mask],
                    cs2_inj[i][mask],
                    color=INJECTION_COLOR,
                    alpha=INJECTION_ALPHA,
                    linewidth=INJECTION_LINEWIDTH,
                    linestyle=INJECTION_LINESTYLE,
                    zorder=1e11,  # Plot on top of everything
                    label="Injection" if i == 0 else "",
                )

    xlabel = r"$n$ [$n_{\rm{sat}}$]" if TEX_ENABLED else "n [n_sat]"
    ylabel = r"$c_s^2$" if TEX_ENABLED else "cs2"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0.5, 6.0)
    plt.ylim(0.0, 1.2)  # Speed of sound squared should be between 0 and 1 (c=1)

    # Add legend for prior and/or injection
    if prior_data is not None or injection_data is not None:
        from matplotlib.lines import Line2D

        legend_elements = []
        if prior_data is not None:
            legend_elements.append(
                Line2D(
                    [0], [0], color=COLORS_DICT["prior"], lw=2, alpha=0.7, label="Prior"
                )
            )
        if injection_data is not None and "cs2" in injection_data:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=INJECTION_COLOR,
                    lw=INJECTION_LINEWIDTH,
                    linestyle=INJECTION_LINESTYLE,
                    alpha=INJECTION_ALPHA,
                    label="Injection",
                )
            )
        plt.legend(handles=legend_elements, loc="upper left")

    # Save figure
    save_name = os.path.join(outdir, "cs2_density_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"cs2-density plot saved to {save_name}")


# TODO: Fill histograms between the credible interval edges and not fill outside of it
def make_parameter_histograms(
    data: Dict[str, Any],
    prior_data: Optional[Dict[str, Any]],
    outdir: str,
    injection_data: Optional[Dict[str, Any]] = None,
):
    """Create histograms for key EOS parameters.

    Parameters
    ----------
    data : dict
        EOS data dictionary
    prior_data : dict or None
        Prior EOS data for comparison
    outdir : str
        Output directory
    injection_data : dict or None, optional
        Injection EOS data for plotting true values, by default None
    """
    logger.info("Creating parameter histograms...")

    m, r, l = data["masses"], data["radii"], data["lambdas"]
    n, p = data["densities"], data["pressures"]

    # Calculate derived parameters
    MTOV_list = np.array([np.max(mass) for mass in m])
    R14_list = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(m, r)])
    Lambda14_list = np.array(
        [np.interp(1.4, mass, lambda_arr) for mass, lambda_arr in zip(m, l)]
    )
    p3nsat_list = np.array([np.interp(3.0, dens, press) for dens, press in zip(n, p)])

    # Calculate prior parameters if available
    prior_params = {}
    if prior_data is not None:
        m_prior, r_prior, l_prior = (
            prior_data["masses"],
            prior_data["radii"],
            prior_data["lambdas"],
        )
        n_prior, p_prior = prior_data["densities"], prior_data["pressures"]

        prior_params["MTOV"] = np.array([np.max(mass) for mass in m_prior])
        prior_params["R14"] = np.array(
            [np.interp(1.4, mass, radius) for mass, radius in zip(m_prior, r_prior)]
        )
        prior_params["Lambda14"] = np.array(
            [
                np.interp(1.4, mass, lambda_arr)
                for mass, lambda_arr in zip(m_prior, l_prior)
            ]
        )
        prior_params["p3nsat"] = np.array(
            [np.interp(3.0, dens, press) for dens, press in zip(n_prior, p_prior)]
        )

    # Calculate injection parameters if available
    injection_params = {}
    if injection_data is not None:
        if "masses_EOS" in injection_data and "radii_EOS" in injection_data:
            m_inj, r_inj = injection_data["masses_EOS"], injection_data["radii_EOS"]
            # Take first curve (or average if multiple curves)
            injection_params["MTOV"] = np.max(m_inj[0])
            injection_params["R14"] = np.interp(1.4, m_inj[0], r_inj[0])

        if "masses_EOS" in injection_data and "Lambda_EOS" in injection_data:
            m_inj, l_inj = injection_data["masses_EOS"], injection_data["Lambda_EOS"]
            injection_params["Lambda14"] = np.interp(1.4, m_inj[0], l_inj[0])

        if "n" in injection_data and "p" in injection_data:
            n_inj, p_inj = injection_data["n"], injection_data["p"]
            injection_params["p3nsat"] = np.interp(3.0, n_inj[0], p_inj[0])

    # Define parameters to plot (without fixed ranges)
    if TEX_ENABLED:
        parameters = {
            "MTOV": {"values": MTOV_list, "xlabel": r"$M_{\rm{TOV}}$ [$M_{\odot}$]"},
            "R14": {"values": R14_list, "xlabel": r"$R_{1.4}$ [km]"},
            "Lambda14": {"values": Lambda14_list, "xlabel": r"$\Lambda_{1.4}$"},
            "p3nsat": {
                "values": p3nsat_list,
                "xlabel": r"$p(3n_{\rm{sat}})$ [MeV fm$^{-3}$]",
            },
        }
    else:
        parameters = {
            "MTOV": {"values": MTOV_list, "xlabel": "M_TOV [M_sun]"},
            "R14": {"values": R14_list, "xlabel": "R_1.4 [km]"},
            "Lambda14": {"values": Lambda14_list, "xlabel": "Lambda_1.4"},
            "p3nsat": {"values": p3nsat_list, "xlabel": "p(3n_sat) [MeV fm^-3]"},
        }

    for param_name, param_data in parameters.items():
        plt.figure(figsize=figsize_horizontal)

        # Calculate HDI using the global HDI_PROB constant
        hdi = az.hdi(param_data["values"], hdi_prob=HDI_PROB)
        hdi_low, hdi_high = hdi
        median = np.median(param_data["values"])

        # Calculate errors for title formatting
        low_err = median - hdi_low
        high_err = hdi_high - median

        # Auto-zoom: 25% wider than HDI
        hdi_width = hdi_high - hdi_low
        x_min = hdi_low - 0.25 * hdi_width
        x_max = hdi_high + 0.25 * hdi_width

        # Plot prior histogram if available
        if prior_data is not None and param_name in prior_params:
            kde_prior = gaussian_kde(prior_params[param_name])
            x = np.linspace(x_min, x_max, 1000)
            y_prior = kde_prior(x)
            plt.fill_between(
                x, y_prior, alpha=ALPHA, color=COLORS_DICT["prior"], label="Prior"
            )

        # Create posterior KDE
        kde = gaussian_kde(param_data["values"])
        x = np.linspace(x_min, x_max, 1000)
        y = kde(x)

        plt.plot(x, y, color=COLORS_DICT["posterior"], lw=3.0, label="Posterior")
        plt.fill_between(x, y, alpha=0.3, color=COLORS_DICT["posterior"])

        # Plot injection value if available
        if injection_data is not None and param_name in injection_params:
            inj_value = injection_params[param_name]
            plt.axvline(
                inj_value,
                color=INJECTION_COLOR,
                linestyle=INJECTION_LINESTYLE,
                linewidth=INJECTION_LINEWIDTH,
                alpha=INJECTION_ALPHA,
                label="Injection",
            )
            logger.info(f"Injection {param_name}: {inj_value:.2f}")

        plt.xlabel(param_data["xlabel"])
        plt.ylabel("Density")
        plt.xlim(x_min, x_max)
        plt.ylim(bottom=0.0)
        plt.legend()

        # Format title using x-axis label + credible interval + credibility percentage
        # Extract the parameter symbol from xlabel (e.g., "$M_{\rm{TOV}}$" from full label)
        xlabel = param_data["xlabel"]
        credibility_pct = int(HDI_PROB * 100)

        if TEX_ENABLED:
            plt.title(
                f"{xlabel}: ${median:.2f}_{{-{low_err:.2f}}}^{{+{high_err:.2f}}}$ ({credibility_pct}\\% credibility)"
            )
        else:
            plt.title(
                f"{xlabel}: {median:.2f} -{low_err:.2f} +{high_err:.2f} ({credibility_pct}% credibility)"
            )

        save_name = os.path.join(outdir, f"{param_name}_histogram.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        logger.info(f"{param_name} histogram saved to {save_name}")


def make_contour_radii_plot(
    data: Dict[str, Any],
    prior_data: Optional[Dict[str, Any]],
    outdir: str,
    m_min: float = 0.6,
    m_max: float = 2.1,
):
    """Create contour plot of radii vs mass.

    Parameters
    ----------
    data : dict
        EOS data dictionary
    prior_data : dict or None
        Prior EOS data for comparison
    outdir : str
        Output directory
    m_min : float, optional
        Minimum mass, by default 0.6
    m_max : float, optional
        Maximum mass, by default 2.1
    """
    logger.info("Creating radii contour plot...")

    plt.figure(figsize=figsize_vertical)

    masses_array = np.linspace(m_min, m_max, 100)

    # Plot prior contours if available
    if prior_data is not None:
        m_prior, r_prior = prior_data["masses"], prior_data["radii"]
        radii_low_prior = np.empty_like(masses_array)
        radii_high_prior = np.empty_like(masses_array)

        for i, mass_point in enumerate(masses_array):
            radii_at_mass = []
            for mass, radius in zip(m_prior, r_prior):
                radii_at_mass.append(np.interp(mass_point, mass, radius))
            radii_at_mass = np.array(radii_at_mass)

            low, med, high = report_credible_interval(radii_at_mass, hdi_prob=HDI_PROB)
            radii_low_prior[i] = med - low
            radii_high_prior[i] = med + high

        plt.fill_betweenx(
            masses_array,
            radii_low_prior,
            radii_high_prior,
            alpha=ALPHA,
            color=COLORS_DICT["prior"],
            label="Prior",
        )

    # Plot posterior contours
    m, r = data["masses"], data["radii"]
    radii_low = np.empty_like(masses_array)
    radii_high = np.empty_like(masses_array)

    logger.info(f"Computing radii contours for {len(masses_array)} mass points...")
    for i, mass_point in enumerate(masses_array):
        # Gather all radii at this mass point
        radii_at_mass = []
        for mass, radius in zip(m, r):
            radii_at_mass.append(np.interp(mass_point, mass, radius))
        radii_at_mass = np.array(radii_at_mass)

        # Construct credible interval using global HDI_PROB
        low, med, high = report_credible_interval(radii_at_mass, hdi_prob=HDI_PROB)

        radii_low[i] = med - low
        radii_high[i] = med + high

    # Plot posterior contours
    plt.fill_betweenx(
        masses_array,
        radii_low,
        radii_high,
        alpha=0.5,
        color=COLORS_DICT["posterior"],
        label="Posterior",
    )
    plt.plot(radii_low, masses_array, lw=2.0, color=COLORS_DICT["posterior"])
    plt.plot(radii_high, masses_array, lw=2.0, color=COLORS_DICT["posterior"])

    xlabel = r"$R$ [km]" if TEX_ENABLED else "R [km]"
    ylabel = r"$M$ [$M_{\odot}$]" if TEX_ENABLED else "M [M_sun]"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(8.0, 16.0)
    plt.ylim(m_min, m_max)
    plt.legend()

    save_name = os.path.join(outdir, "radii_contour_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Radii contour plot saved to {save_name}")


def make_contour_pressures_plot(
    data: Dict[str, Any], outdir: str, n_min: float = 0.5, n_max: float = 6.0
):
    """Create contour plot of pressures vs density.

    Parameters
    ----------
    data : dict
        EOS data dictionary
    outdir : str
        Output directory
    n_min : float, optional
        Minimum density, by default 0.5
    n_max : float, optional
        Maximum density, by default 6.0
    """
    logger.info("Creating pressures contour plot...")

    n, p = data["densities"], data["pressures"]

    plt.figure(figsize=figsize_horizontal)

    dens_array = np.linspace(n_min, n_max, 100)
    press_low = np.empty_like(dens_array)
    press_high = np.empty_like(dens_array)

    logger.info(f"Computing pressure contours for {len(dens_array)} density points...")
    for i, dens in enumerate(dens_array):
        # Gather all pressures at this density point
        press_at_dens = []
        for density, pressure in zip(n, p):
            press_at_dens.append(np.interp(dens, density, pressure))
        press_at_dens = np.array(press_at_dens)

        # Construct credible interval using global HDI_PROB
        low, med, high = report_credible_interval(press_at_dens, hdi_prob=HDI_PROB)

        press_low[i] = med - low
        press_high[i] = med + high

    # Plot contours
    plt.fill_between(
        dens_array, press_low, press_high, alpha=0.5, color=COLORS_DICT["posterior"]
    )
    plt.plot(dens_array, press_low, lw=2.0, color=COLORS_DICT["posterior"])
    plt.plot(dens_array, press_high, lw=2.0, color=COLORS_DICT["posterior"])

    xlabel = r"$n$ [$n_{\rm{sat}}$]" if TEX_ENABLED else "n [n_sat]"
    ylabel = r"$p$ [MeV fm$^{-3}$]" if TEX_ENABLED else "p [MeV fm^-3]"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(n_min, n_max)
    plt.yscale("log")
    plt.legend()

    save_name = os.path.join(outdir, "pressures_contour_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Pressures contour plot saved to {save_name}")


def generate_all_plots(
    outdir: str,
    prior_dir: Optional[str] = None,
    make_cornerplot_flag: bool = True,
    make_massradius_flag: bool = True,
    make_masslambda_flag: bool = True,
    make_pressuredensity_flag: bool = True,
    make_histograms_flag: bool = True,
    make_cs2_flag: bool = True,
    injection_eos_path: Optional[str] = None,
):
    """Generate selected plots for the specified output directory.

    Parameters
    ----------
    outdir : str
        Output directory containing eos_samples.npz
    prior_dir : str, optional
        Directory containing prior samples for comparison
    make_cornerplot_flag : bool, optional
        Whether to generate cornerplot, by default True
    make_massradius_flag : bool, optional
        Whether to generate mass-radius plot, by default True
    make_masslambda_flag : bool, optional
        Whether to generate mass-Lambda plot, by default True
    make_pressuredensity_flag : bool, optional
        Whether to generate pressure-density plot, by default True
    make_histograms_flag : bool, optional
        Whether to generate parameter histograms, by default True
    make_cs2_flag : bool, optional
        Whether to generate cs2-density plot, by default True
    injection_eos_path : str, optional
        Path to NPZ file containing injection EOS data, by default None
    """
    logger.info(f"Generating plots for directory: {outdir}")

    # Create figures subdirectory
    figures_dir = os.path.join(outdir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    logger.info(f"Saving plots to: {figures_dir}")

    # Load data
    try:
        data = load_eos_data(outdir)
        logger.info("Data loaded successfully!")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return

    # Load prior data
    prior_data = None
    if prior_dir is not None:
        prior_data = load_prior_data(prior_dir)
        if prior_data is not None:
            logger.info("Prior data loaded successfully!")

    # Load injection EOS data
    injection_data = None
    if injection_eos_path is not None:
        injection_data = load_injection_eos(injection_eos_path)
        if injection_data is not None:
            logger.info("Injection EOS data loaded successfully!")

    # Create plots based on flags (pass figures_dir instead of outdir)
    if make_cornerplot_flag:
        try:
            make_cornerplot(data, figures_dir)
        except Exception as e:
            logger.error(f"Failed to create cornerplot: {e}")
            logger.warning("Continuing with other plots...")

    if make_massradius_flag:
        make_mass_radius_plot(
            data, prior_data, figures_dir, injection_data=injection_data
        )

    if make_masslambda_flag:
        make_mass_lambda_plot(
            data, prior_data, figures_dir, injection_data=injection_data
        )

    if make_pressuredensity_flag:
        make_pressure_density_plot(
            data, prior_data, figures_dir, injection_data=injection_data
        )

    if make_histograms_flag:
        make_parameter_histograms(
            data, prior_data, figures_dir, injection_data=injection_data
        )

    if make_cs2_flag:
        make_cs2_plot(data, prior_data, figures_dir, injection_data=injection_data)

    # TODO: Decide whether to keep mass-radius and pressure-density contour plots
    # These are currently commented out pending decision on their utility
    # if make_contours_flag:
    #     make_contour_radii_plot(data, prior_data, figures_dir)
    #     make_contour_pressures_plot(data, figures_dir)

    logger.info(f"All plots generated and saved to {figures_dir}")


def run_from_config(config_path: str) -> None:
    """Run postprocessing from a YAML config file.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    """
    from jesterTOV.inference.config.parser import load_config

    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Check if postprocessing is enabled
    if not config.postprocessing.enabled:
        logger.warning(
            "Postprocessing is disabled in config. Set postprocessing.enabled: true to run."
        )
        return

    # Get output directory from sampler config
    outdir = config.sampler.output_dir

    logger.info("=" * 60)
    logger.info("Running postprocessing from config...")
    logger.info("=" * 60)
    logger.info(f"Output directory: {outdir}")
    logger.info(f"Prior directory: {config.postprocessing.prior_dir}")
    logger.info(f"Injection EOS: {config.postprocessing.injection_eos_path}")
    logger.info(f"Make cornerplot: {config.postprocessing.make_cornerplot}")
    logger.info(f"Make mass-radius: {config.postprocessing.make_massradius}")
    logger.info(f"Make mass-lambda: {config.postprocessing.make_masslambda}")
    logger.info(f"Make pressure-density: {config.postprocessing.make_pressuredensity}")
    logger.info(f"Make histograms: {config.postprocessing.make_histograms}")
    logger.info(f"Make cs2: {config.postprocessing.make_cs2}")
    logger.info("=" * 60)

    # Run postprocessing with config settings
    generate_all_plots(
        outdir=outdir,
        prior_dir=config.postprocessing.prior_dir,
        make_cornerplot_flag=config.postprocessing.make_cornerplot,
        make_massradius_flag=config.postprocessing.make_massradius,
        make_masslambda_flag=config.postprocessing.make_masslambda,
        make_pressuredensity_flag=config.postprocessing.make_pressuredensity,
        make_histograms_flag=config.postprocessing.make_histograms,
        make_cs2_flag=config.postprocessing.make_cs2,
        injection_eos_path=config.postprocessing.injection_eos_path,
    )

    logger.info(
        f"\nPostprocessing complete! Plots saved to {os.path.join(outdir, 'figures')}"
    )


def main():
    """Main function to parse arguments and generate plots."""
    # Check if first argument is a YAML config file
    import sys

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        run_from_config(sys.argv[1])
        return

    parser = argparse.ArgumentParser(
        description="Generate EOS postprocessing plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from config file (recommended)
  run_jester_postprocessing config.yaml

  # Generate all plots using command-line arguments
  run_jester_postprocessing --outdir ./results --make-all

  # Generate only cornerplot and mass-radius plot
  run_jester_postprocessing --outdir ./results --make-cornerplot --make-massradius

  # Generate with prior comparison
  run_jester_postprocessing --outdir ./results --prior-dir ./prior --make-all

  # Generate with injection EOS comparison
  run_jester_postprocessing --outdir ./results --injection-eos ./injection.npz --make-all
        """,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory containing eos_samples.npz",
    )
    parser.add_argument(
        "--prior-dir",
        type=str,
        default=None,
        help="Directory containing prior samples for comparison",
    )

    # Plot selection flags
    parser.add_argument(
        "--make-all",
        action="store_true",
        help="Generate all plots (default if no specific flags given)",
    )
    parser.add_argument(
        "--make-cornerplot",
        action="store_true",
        help="Generate cornerplot of EOS parameters",
    )
    parser.add_argument(
        "--make-massradius", action="store_true", help="Generate mass-radius plot"
    )
    parser.add_argument(
        "--make-masslambda", action="store_true", help="Generate mass-Lambda plot"
    )
    parser.add_argument(
        "--make-pressuredensity",
        action="store_true",
        help="Generate pressure-density plot",
    )
    parser.add_argument(
        "--make-histograms", action="store_true", help="Generate parameter histograms"
    )
    parser.add_argument(
        "--make-cs2", action="store_true", help="Generate cs2-density plot"
    )

    # Additional options
    parser.add_argument(
        "--injection-eos",
        type=str,
        default=None,
        help="Path to NPZ file containing injection EOS data",
    )
    parser.add_argument(
        "--m-min", type=float, default=0.6, help="Minimum mass for contour plots"
    )
    parser.add_argument(
        "--m-max", type=float, default=2.5, help="Maximum mass for contour plots"
    )
    parser.add_argument(
        "--n-min",
        type=float,
        default=0.5,
        help="Minimum density for pressure contour plots",
    )
    parser.add_argument(
        "--n-max",
        type=float,
        default=6.0,
        help="Maximum density for pressure contour plots",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.outdir):
        logger.error(f"Error: Output directory {args.outdir} does not exist")
        return

    # Determine which plots to make
    # If --make-all or no specific flags given, make all plots
    make_all = args.make_all or not any(
        [
            args.make_cornerplot,
            args.make_massradius,
            args.make_masslambda,
            args.make_pressuredensity,
            args.make_histograms,
            args.make_cs2,
        ]
    )

    # Generate plots
    generate_all_plots(
        args.outdir,
        prior_dir=args.prior_dir,
        make_cornerplot_flag=make_all or args.make_cornerplot,
        make_massradius_flag=make_all or args.make_massradius,
        make_masslambda_flag=make_all or args.make_masslambda,
        make_pressuredensity_flag=make_all or args.make_pressuredensity,
        make_histograms_flag=make_all or args.make_histograms,
        make_cs2_flag=make_all or args.make_cs2,
        injection_eos_path=args.injection_eos,
    )


if __name__ == "__main__":
    main()
