r"""Modular postprocessing script for EOS inference results.

Provides visualization tools for analysing equation-of-state inference results:
cornerplots, mass-radius diagrams, pressure-density, speed-of-sound, and parameter
histograms.

Usage::

    run_jester_postprocessing config.yaml
"""

import re
import sys
import os
import warnings
from typing import Any, Dict, Optional

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

np.random.seed(2)
import jesterTOV.utils as utils
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


# ─── Matplotlib setup ─────────────────────────────────────────────────────────


def setup_matplotlib(use_tex: bool = True) -> bool:
    """Configure matplotlib with TeX rendering and sensible defaults.

    Parameters
    ----------
    use_tex : bool, optional
        Whether to attempt LaTeX rendering, by default True.

    Returns
    -------
    bool
        ``True`` if TeX is successfully enabled, ``False`` otherwise.
    """
    tex_enabled = False
    if use_tex:
        try:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Serif"],
                }
            )
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, r"$\alpha$")
            plt.close(fig)
            tex_enabled = True
            logger.info("TeX rendering enabled")
        except Exception as e:
            warnings.warn(
                f"TeX rendering failed ({e}). Falling back to non-TeX rendering."
            )
            plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

    plt.rcParams.update(
        {
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
    )
    return tex_enabled


TEX_ENABLED = setup_matplotlib(use_tex=True)
DEFAULT_COLORMAP = sns.color_palette("crest", as_cmap=True)

# ─── Constants ────────────────────────────────────────────────────────────────

COLORS_DICT = {"prior": "gray", "posterior": "blue"}
ALPHA = 0.3
figsize_vertical = (6, 8)
figsize_horizontal = (8, 6)

INJECTION_COLOR = "black"
INJECTION_LINESTYLE = "--"
INJECTION_LINEWIDTH = 2.5
INJECTION_ALPHA = 0.8

HDI_PROB = 0.90  # 90% highest density interval

M_MIN = 0.75  # [M_sun]
M_MAX = 3.5  # [M_sun]
R_MIN = 6.0  # [km]
R_MAX = 18.0  # [km]

PRIOR_DIR = "./outdir/"


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_eos_data(outdir: str) -> Dict[str, Any]:
    """Load EOS data from the specified output directory.

    Parameters
    ----------
    outdir : str
        Path to output directory containing ``results.h5``.

    Returns
    -------
    dict
        Dictionary containing EOS data arrays.

    Raises
    ------
    FileNotFoundError
        If ``results.h5`` is not found in the directory.
    """
    from jesterTOV.inference.result import InferenceResult

    filename = os.path.join(outdir, "results.h5")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file not found: {filename}")

    result = InferenceResult.load(filename)

    m = result.posterior["masses_EOS"]
    r = result.posterior["radii_EOS"]
    l = result.posterior["Lambdas_EOS"]
    n = result.posterior["n"]
    p = result.posterior["p"]
    e = result.posterior["e"]
    cs2 = result.posterior["cs2"]

    n = n / utils.fm_inv3_to_geometric / 0.16
    p = p / utils.MeV_fm_inv3_to_geometric
    e = e / utils.MeV_fm_inv3_to_geometric

    n_TOV = result.posterior.get("n_TOV", None)
    if n_TOV is not None:
        n_TOV = n_TOV / utils.fm_inv3_to_geometric / 0.16

    log_prob = result.posterior["log_prob"]

    prior_params: Dict[str, Any] = {}
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
            "No parameter_names found in metadata. Cornerplot may be empty. "
            "This may occur if results were saved with an older version of JESTER."
        )

    output: Dict[str, Any] = {
        "masses": m,
        "radii": r,
        "lambdas": l,
        "densities": n,
        "pressures": p,
        "energies": e,
        "cs2": cs2,
        "log_prob": log_prob,
        "prior_params": prior_params,
    }
    if n_TOV is not None:
        output["n_TOV"] = n_TOV
    return output


def load_prior_data(prior_dir: str = PRIOR_DIR) -> Optional[Dict[str, Any]]:
    """Load prior EOS data for comparison.

    Parameters
    ----------
    prior_dir : str, optional
        Path to prior output directory.

    Returns
    -------
    dict or None
        Prior data dictionary, or ``None`` if not found.
    """
    try:
        return load_eos_data(prior_dir)
    except FileNotFoundError:
        logger.warning(f"Prior data not found at {prior_dir}")
        return None


def load_injection_eos(
    injection_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    r"""Load injection EOS data from an NPZ file.

    Parameters
    ----------
    injection_path : str or None
        Path to NPZ file containing injection EOS data.

    Returns
    -------
    dict or None
        Dictionary containing injection EOS data, or ``None`` if loading fails.

    Notes
    -----
    **Units:** The injection file should contain data in geometric units:

    - ``masses_EOS``: solar masses :math:`M_{\odot}`
    - ``radii_EOS``: :math:`\mathrm{km}`
    - ``Lambda_EOS``: dimensionless tidal deformability
    - ``n``, ``p``, ``e``: geometric units :math:`m^{-2}`
    - ``cs2``: dimensionless

    Missing keys are handled gracefully.
    """
    if injection_path is None:
        return None

    try:
        with np.load(injection_path) as data:
            logger.info(f"Loaded injection EOS from {injection_path}")
            logger.info(f"Available keys: {list(data.keys())}")

            expected_keys = [
                "masses_EOS",
                "radii_EOS",
                "Lambda_EOS",
                "n",
                "p",
                "e",
                "cs2",
            ]
            output: Dict[str, Any] = {}
            missing_keys = []

            for key in expected_keys:
                if key in data:
                    arr = data[key]
                    output[key] = arr[np.newaxis, :] if arr.ndim == 1 else arr
                else:
                    missing_keys.append(key)

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


# ─── Data analysis helpers ────────────────────────────────────────────────────


def _split_into_monotone_branches(
    masses: np.ndarray, lambdas: np.ndarray
) -> list[tuple[int, int]]:
    """Split a mass-Lambda curve into monotone-decreasing segments.

    A branch break is detected wherever Lambda increases as mass increases,
    signalling a fold-back in M(pc) (twin-star / third-family scenario).

    Parameters
    ----------
    masses : np.ndarray
        Mass array (monotone increasing, uniform grid).
    lambdas : np.ndarray
        Tidal deformability array on the same grid.

    Returns
    -------
    list of (start, end) index pairs
        Each pair defines a half-open slice that is monotone decreasing in
        Lambda. Always contains at least one segment.
    """
    segments: list[tuple[int, int]] = []
    start = 0
    for j in range(1, len(lambdas)):
        if lambdas[j] > lambdas[j - 1]:
            segments.append((start, j))
            start = j
    segments.append((start, len(masses)))
    return segments


def report_credible_interval(
    values: np.ndarray, hdi_prob: float = HDI_PROB, verbose: bool = False
) -> tuple[float, float, float]:
    """Compute a symmetric credible interval around the median.

    Parameters
    ----------
    values : np.ndarray
        Array of parameter values.
    hdi_prob : float, optional
        Credible interval probability, by default 0.90.
    verbose : bool, optional
        Whether to log the result, by default False.

    Returns
    -------
    tuple
        ``(low_err, median, high_err)`` where ``low_err = median - low`` and
        ``high_err = high - median``.
    """
    med = float(np.median(values))
    low_percentile = (1 - hdi_prob) / 2 * 100
    high_percentile = (1 + hdi_prob) / 2 * 100
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    low_err = med - low
    high_err = high - med
    if verbose:
        logger.info(
            f"{med:.2f} -{low_err:.2f} +{high_err:.2f} (at {hdi_prob} HDI prob)"
        )
    return low_err, med, high_err


# Regex matching MetaModel+CSE grid parameters that clutter the cornerplot.
# Covers: n_CSE_0_u, n_CSE_1_u, ..., cs2_CSE_0, cs2_CSE_1, ...
# nbreak is intentionally excluded so it still appears in the plot.
_CSE_PARAM_RE = re.compile(r"^(n_CSE_\d+_u|cs2_CSE_\d+)$")


def _is_cse_param(key: str) -> bool:
    """Return True if *key* is a MetaModel+CSE grid parameter."""
    return bool(_CSE_PARAM_RE.match(key))


# ─── Posterior preprocessing helpers ─────────────────────────────────────────


def _get_valid_indices(data: Dict[str, Any]) -> tuple[list[int], float]:
    """Return indices of valid posterior samples and the maximum TOV mass.

    A sample is considered invalid if it contains NaN masses/radii/lambdas,
    any negative lambdas, radii exceeding ``R_MAX`` for masses above ``M_MIN``,
    or a failed n_TOV computation (value ≤ 0).

    Parameters
    ----------
    data : dict
        EOS data dictionary from :func:`load_eos_data`.

    Returns
    -------
    valid_indices : list[int]
        Indices of valid samples.
    max_mtov : float
        Maximum TOV mass across valid samples.

    Raises
    ------
    ValueError
        If ``log_prob`` length does not match the number of EOS samples.
    """
    m, r, l = data["masses"], data["radii"], data["lambdas"]
    log_prob = data["log_prob"]
    n_TOV = data.get("n_TOV", None)

    nb_samples = len(m)
    if len(log_prob) != nb_samples:
        raise ValueError(
            f"Mismatch between log_prob ({len(log_prob)}) and EOS samples ({nb_samples}). "
            "This indicates a bug in the EOS sample generation code."
        )

    valid: list[int] = []
    max_mtov = 0.0
    for i in range(nb_samples):
        if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
            continue
        if any(l[i] < 0):
            continue
        if any((m[i] > M_MIN) * (r[i] > R_MAX)):
            continue
        if n_TOV is not None and n_TOV[i] <= 0.0:
            continue
        valid.append(i)
        mtov = float(np.max(m[i]))
        if mtov > max_mtov:
            max_mtov = mtov

    return valid, max_mtov


def _setup_prob_coloring(
    data: Dict[str, Any], use_crest_cmap: bool = True
) -> tuple[np.ndarray, Normalize, Any, Any]:
    """Set up probability-based colour mapping for posterior samples.

    Parameters
    ----------
    data : dict
        EOS data dictionary.
    use_crest_cmap : bool, optional
        Use the seaborn crest colourmap, by default True.

    Returns
    -------
    prob : np.ndarray
        Normalised (non-log) posterior probabilities.
    norm : Normalize
        Matplotlib normalisation object.
    cmap : Colormap
        Colourmap instance.
    sm : ScalarMappable
        Scalar mappable for the colourbar.
    """
    log_prob = data["log_prob"]
    prob = np.exp(log_prob - np.max(log_prob))
    norm = Normalize(vmin=float(np.min(prob)), vmax=float(np.max(prob)))
    cmap = DEFAULT_COLORMAP if use_crest_cmap else plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    return prob, norm, cmap, sm


# ─── Plot utilities ───────────────────────────────────────────────────────────


def _add_colorbar(fig: Any, sm: Any) -> None:
    """Add a horizontal posterior-probability colourbar at the top of the figure."""
    sm.set_array([])
    cbar_ax = fig.add_axes((0.15, 0.94, 0.7, 0.03))
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalized posterior probability", fontsize=16)
    cbar.set_ticks([])
    cbar.ax.xaxis.labelpad = 5
    cbar.ax.tick_params(labelsize=0, length=0)
    cbar.ax.xaxis.set_label_position("top")


def _add_prior_injection_legend(
    prior_data: Optional[Dict[str, Any]],
    injection_data: Optional[Dict[str, Any]],
    injection_check_key: str,
    loc: str = "upper right",
) -> None:
    """Add a legend for prior and/or injection EOS lines.

    Parameters
    ----------
    prior_data : dict or None
        Prior EOS data; a legend entry is added when not ``None``.
    injection_data : dict or None
        Injection EOS data; a legend entry is added when not ``None`` and
        *injection_check_key* is present in the dictionary.
    injection_check_key : str
        Key used to confirm the injection curve was actually plotted.
    loc : str, optional
        Legend location, by default ``"upper right"``.
    """
    if prior_data is None and injection_data is None:
        return
    from matplotlib.lines import Line2D

    elements = []
    if prior_data is not None:
        elements.append(
            Line2D([0], [0], color=COLORS_DICT["prior"], lw=2, alpha=0.7, label="Prior")
        )
    if injection_data is not None and injection_check_key in injection_data:
        elements.append(
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
    if elements:
        plt.legend(handles=elements, loc=loc)


def _plot_ns_family_curves(
    valid_indices: list[int],
    x_data: np.ndarray,
    y_data: np.ndarray,
    m_data: np.ndarray,
    l_data: np.ndarray,
    prob: np.ndarray,
    norm: Normalize,
    cmap: Any,
) -> int:
    """Plot NS family curves with multi-branch handling and probability colouring.

    Parameters
    ----------
    valid_indices : list[int]
        Indices of valid posterior samples.
    x_data : np.ndarray
        2-D array (n_samples x n_points) for the x-axis quantity.
    y_data : np.ndarray
        2-D array (n_samples x n_points) for the y-axis quantity.
    m_data : np.ndarray
        Mass arrays used for branch detection (shared with ``l_data``).
    l_data : np.ndarray
        Lambda arrays used for branch detection.
    prob : np.ndarray
        Normalised posterior probabilities (length n_samples).
    norm : Normalize
        Matplotlib normalisation object.
    cmap : Colormap
        Colourmap instance.

    Returns
    -------
    int
        Number of samples containing multiple stable branches.
    """
    n_unstable = 0
    for i in valid_indices:
        normalized_value = float(norm(prob[i]))
        color = cmap(normalized_value)
        branches = _split_into_monotone_branches(m_data[i], l_data[i])
        if len(branches) > 1:
            n_unstable += 1
        for start, end in branches:
            plt.plot(
                x_data[i][start:end],
                y_data[i][start:end],
                color=color,
                alpha=1.0,
                rasterized=True,
                zorder=1e10 + normalized_value,
            )
    return n_unstable


def _get_density_mask(n_i: np.ndarray, n_tov_i: Optional[float]) -> np.ndarray:
    """Return a boolean mask selecting the physical density range.

    Parameters
    ----------
    n_i : np.ndarray
        Density array for sample *i* in units of n_sat.
    n_tov_i : float or None
        Central density at the TOV maximum for sample *i*, or ``None`` if
        not available.

    Returns
    -------
    np.ndarray
        Boolean mask with ``True`` where ``n > 0.5 n_sat`` (and ``n <= n_TOV``
        when available).
    """
    mask = n_i > 0.5
    if n_tov_i is not None:
        mask = mask & (n_i <= n_tov_i)
    return mask


def _warn_unstable_branches(n_unstable: int, n_valid: int) -> None:
    """Emit a warning if multi-branch (unstable) NS solutions were found."""
    if n_unstable > 0:
        pct = 100.0 * n_unstable / n_valid
        logger.warning(
            f"{n_unstable}/{n_valid} ({pct:.1f}%) samples had an unstable part "
            "(multi-branch) in their NS solution. These samples are not accounted "
            "for properly during inference; the plot shows each stable branch "
            "separately. If this percentage is low, the impact on the posterior is "
            "negligible."
        )


# ─── Plot functions ───────────────────────────────────────────────────────────


def make_cornerplot(
    data: Dict[str, Any], outdir: str, max_params: Optional[int] = None
) -> None:
    """Create a cornerplot for EOS parameters.

    Parameters
    ----------
    data : dict
        EOS data dictionary from :func:`load_eos_data`.
    outdir : str
        Output directory for saving the plot.
    max_params : int, optional
        Maximum number of parameters to include. If ``None``, includes all.
    """
    logger.info("Creating cornerplot...")

    prior_params = data.get("prior_params", {})
    samples_dict: Dict[str, Any] = {}
    labels: list[str] = []

    cse_keys = [k for k in prior_params if _is_cse_param(k)]
    if cse_keys:
        logger.info(
            f"Excluding {len(cse_keys)} CSE parameters from cornerplot: {sorted(cse_keys)}"
        )

    for key in prior_params:
        if _is_cse_param(key):
            continue
        samples_dict[key] = prior_params[key]

        if TEX_ENABLED and "_" in key:
            base = key.split("_")[0]
            sub = "_".join(key.split("_")[1:])
            sub_escaped = sub.replace("_", r"\_")
            if base == "gamma" and key.endswith("_tilde"):
                idx = key.split("_")[1]
                labels.append(f"$\\tilde{{\\gamma}}_{{{idx}}}$")
            elif base == "gamma":
                labels.append(f"$\\gamma_{{{sub_escaped}}}$")
            elif base == "nbreak":
                labels.append(r"$n_{\rm{break}}$")
            else:
                labels.append(f"${base}_{{{sub_escaped}}}$")
        elif TEX_ENABLED:
            labels.append(f"${key}$")
        else:
            labels.append(key)

    if max_params is not None and len(samples_dict) > max_params:
        logger.info(f"Limiting cornerplot to first {max_params} parameters")
        samples_dict = dict(list(samples_dict.items())[:max_params])
        labels = labels[:max_params]

    if not samples_dict:
        logger.warning("No parameters found for cornerplot")
        return

    logger.info(f"Creating cornerplot with {len(samples_dict)} parameters")
    samples = np.column_stack([samples_dict[key] for key in samples_dict])

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
) -> None:
    """Create a mass-radius plot with posterior probability colouring.

    Parameters
    ----------
    data : dict
        EOS data dictionary.
    prior_data : dict or None
        Prior EOS data for background comparison.
    outdir : str
        Output directory.
    use_crest_cmap : bool, optional
        Use the seaborn crest colourmap, by default True.
    injection_data : dict or None, optional
        Injection EOS data for plotting the true values, by default None.
    """
    logger.info("Creating mass-radius plot...")

    plt.figure(figsize=(10, 8))
    m_min, m_max = M_MIN, M_MAX

    if prior_data is not None:
        for i in range(len(prior_data["masses"])):
            plt.plot(
                prior_data["radii"][i],
                prior_data["masses"][i],
                color=COLORS_DICT["prior"],
                alpha=0.1,
                rasterized=True,
                zorder=1,
            )

    m, r, l = data["masses"], data["radii"], data["lambdas"]
    valid_indices, max_mtov = _get_valid_indices(data)
    prob, norm, cmap, sm = _setup_prob_coloring(data, use_crest_cmap)

    if max_mtov > m_max:
        m_max = max_mtov + 0.25
        logger.info(
            f"Widening mass axis to {m_max:.2f} M_sun (max MTOV: {max_mtov:.2f})"
        )

    n_bad = len(m) - len(valid_indices)
    logger.info(
        f"Plotting {len(valid_indices)} M-R curves (excluded {n_bad} invalid samples)..."
    )

    n_unstable = _plot_ns_family_curves(valid_indices, r, m, m, l, prob, norm, cmap)
    _warn_unstable_branches(n_unstable, len(valid_indices))

    if (
        injection_data is not None
        and "masses_EOS" in injection_data
        and "radii_EOS" in injection_data
    ):
        for i, (m_inj, r_inj) in enumerate(
            zip(injection_data["masses_EOS"], injection_data["radii_EOS"])
        ):
            plt.plot(
                r_inj,
                m_inj,
                color=INJECTION_COLOR,
                alpha=INJECTION_ALPHA,
                linewidth=INJECTION_LINEWIDTH,
                linestyle=INJECTION_LINESTYLE,
                zorder=1e11,
                label="Injection" if i == 0 else "",
            )

    plt.xlabel(r"$R$ [km]" if TEX_ENABLED else "R [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]" if TEX_ENABLED else "M [M_sun]")
    plt.xlim(R_MIN, R_MAX)
    plt.ylim(m_min, m_max)

    _add_colorbar(plt.gcf(), sm)
    _add_prior_injection_legend(prior_data, injection_data, "masses_EOS")

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
    r"""Create a mass-Lambda plot with posterior probability colouring.

    Parameters
    ----------
    data : dict
        EOS data dictionary.
    prior_data : dict or None
        Prior EOS data for background comparison.
    outdir : str
        Output directory.
    use_crest_cmap : bool, optional
        Use the seaborn crest colourmap, by default True.
    injection_data : dict or None, optional
        Injection EOS data for plotting the true values, by default None.
    """
    logger.info("Creating mass-Lambda plot...")

    plt.figure(figsize=(10, 8))
    m_min, m_max = M_MIN, M_MAX

    if prior_data is not None:
        for i in range(len(prior_data["masses"])):
            plt.plot(
                prior_data["masses"][i],
                prior_data["lambdas"][i],
                color=COLORS_DICT["prior"],
                alpha=0.1,
                rasterized=True,
                zorder=1,
            )

    m, _, l = data["masses"], data["radii"], data["lambdas"]
    valid_indices, max_mtov = _get_valid_indices(data)
    prob, norm, cmap, sm = _setup_prob_coloring(data, use_crest_cmap)

    if max_mtov > m_max:
        m_max = max_mtov + 0.25
        logger.info(
            f"Widening mass axis to {m_max:.2f} M_sun (max MTOV: {max_mtov:.2f})"
        )

    n_bad = len(m) - len(valid_indices)
    logger.info(
        f"Plotting {len(valid_indices)} M-Lambda curves (excluded {n_bad} invalid samples)..."
    )

    n_unstable = _plot_ns_family_curves(valid_indices, m, l, m, l, prob, norm, cmap)
    _warn_unstable_branches(n_unstable, len(valid_indices))

    if (
        injection_data is not None
        and "masses_EOS" in injection_data
        and "Lambda_EOS" in injection_data
    ):
        for i, (m_inj, l_inj) in enumerate(
            zip(injection_data["masses_EOS"], injection_data["Lambda_EOS"])
        ):
            plt.plot(
                m_inj,
                l_inj,
                color=INJECTION_COLOR,
                alpha=INJECTION_ALPHA,
                linewidth=INJECTION_LINEWIDTH,
                linestyle=INJECTION_LINESTYLE,
                zorder=1e11,
                label="Injection" if i == 0 else "",
            )

    plt.xlabel(r"$M$ [$M_{\odot}$]" if TEX_ENABLED else "M [M_sun]")
    plt.ylabel(r"$\Lambda$" if TEX_ENABLED else "Lambda")
    plt.xlim(m_min, m_max)
    plt.yscale("log")

    _add_colorbar(plt.gcf(), sm)
    _add_prior_injection_legend(prior_data, injection_data, "masses_EOS")

    save_name = os.path.join(outdir, "mass_lambda_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Mass-Lambda plot saved to {save_name}")


def make_mass_lambda_ratio_plot(
    data: Dict[str, Any],
    outdir: str,
    injection_data: Dict[str, Any],
) -> None:
    r"""Create a mass-Lambda ratio plot relative to the injection.

    At each mass grid point the ratio :math:`\Lambda / \Lambda_{\rm inj}` is
    computed for every valid posterior sample by interpolating both curves onto
    a common mass grid.  The 90% credible interval (HDI) and median are shown
    as a filled band; the injection appears as a horizontal reference at unity.

    Parameters
    ----------
    data : dict
        EOS data dictionary.
    outdir : str
        Output directory.
    injection_data : dict
        Injection EOS data.  Must contain ``"masses_EOS"`` and ``"Lambda_EOS"``.
    """
    if "masses_EOS" not in injection_data or "Lambda_EOS" not in injection_data:
        logger.warning(
            "Injection data missing 'masses_EOS' or 'Lambda_EOS'; "
            "skipping mass-Lambda ratio plot."
        )
        return

    logger.info("Creating mass-Lambda ratio plot...")

    m_inj_ref = injection_data["masses_EOS"][0]
    l_inj_ref = injection_data["Lambda_EOS"][0]

    m, l = data["masses"], data["lambdas"]
    valid_indices, max_mtov = _get_valid_indices(data)
    m_min = M_MIN
    m_max = max_mtov + 0.25 if max_mtov > M_MAX else M_MAX

    logger.info(
        f"Computing Lambda ratio CI over {len(valid_indices)} valid samples "
        f"(excluded {len(m) - len(valid_indices)} invalid)..."
    )

    masses_array = np.linspace(m_min, min(m_max - 0.1, max_mtov - 0.05), 100)
    ratio_low = np.empty_like(masses_array)
    ratio_med = np.empty_like(masses_array)
    ratio_high = np.empty_like(masses_array)

    for j, mass_point in enumerate(masses_array):
        l_inj_at_mass = float(np.interp(mass_point, m_inj_ref, l_inj_ref))
        if l_inj_at_mass <= 0:
            ratio_low[j] = ratio_med[j] = ratio_high[j] = np.nan
            continue
        ratios = np.array(
            [
                float(np.interp(mass_point, m[i], l[i])) / l_inj_at_mass
                for i in valid_indices
            ]
        )
        low_err, med, high_err = report_credible_interval(ratios, hdi_prob=HDI_PROB)
        ratio_low[j] = med - low_err
        ratio_med[j] = med
        ratio_high[j] = med + high_err

    valid_mask = np.isfinite(ratio_low) & np.isfinite(ratio_high)

    plt.figure(figsize=(10, 8))
    plt.fill_between(
        masses_array[valid_mask],
        ratio_low[valid_mask],
        ratio_high[valid_mask],
        alpha=0.5,
        color=COLORS_DICT["posterior"],
        label=(
            f"Posterior ({int(HDI_PROB * 100)}\\% CI)"
            if TEX_ENABLED
            else f"Posterior ({int(HDI_PROB * 100)}% CI)"
        ),
    )
    plt.plot(
        masses_array[valid_mask],
        ratio_med[valid_mask],
        color=COLORS_DICT["posterior"],
        lw=2.0,
        label="Posterior median",
    )
    plt.plot(
        masses_array[valid_mask],
        ratio_low[valid_mask],
        color=COLORS_DICT["posterior"],
        lw=1.5,
    )
    plt.plot(
        masses_array[valid_mask],
        ratio_high[valid_mask],
        color=COLORS_DICT["posterior"],
        lw=1.5,
    )

    plt.axhline(
        1.0,
        color=INJECTION_COLOR,
        linestyle=INJECTION_LINESTYLE,
        linewidth=INJECTION_LINEWIDTH,
        alpha=INJECTION_ALPHA,
        label="Injection",
        zorder=1e11,
    )

    finite_low = ratio_low[valid_mask]
    finite_high = ratio_high[valid_mask]
    if len(finite_low) > 0:
        y_lo = float(np.percentile(finite_low, 5))
        y_hi = float(np.percentile(finite_high, 95))
        margin = 0.05 * (y_hi - y_lo)
        plt.ylim(y_lo - margin, y_hi + margin)

    plt.xlabel(r"$M$ [$M_{\odot}$]" if TEX_ENABLED else "M [M_sun]")
    plt.ylabel(
        r"$\Lambda / \Lambda_{\rm{inj}}$" if TEX_ENABLED else "Lambda / Lambda_inj"
    )
    plt.xlim(m_min, float(np.max(m_inj_ref)))
    plt.legend(loc="upper right")

    save_name = os.path.join(outdir, "mass_lambda_plot_ratio.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Mass-Lambda ratio plot saved to {save_name}")


def make_pressure_density_plot(
    data: Dict[str, Any],
    prior_data: Optional[Dict[str, Any]],
    outdir: str,
    use_crest_cmap: bool = True,
    injection_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Create an equation-of-state plot (pressure vs density).

    Parameters
    ----------
    data : dict
        EOS data dictionary.
    prior_data : dict or None
        Prior EOS data for background comparison.
    outdir : str
        Output directory.
    use_crest_cmap : bool, optional
        Use the seaborn crest colourmap, by default True.
    injection_data : dict or None, optional
        Injection EOS data for plotting the true values, by default None.
    """
    logger.info("Creating pressure-density plot...")

    plt.figure(figsize=(11, 6))

    if prior_data is not None:
        n_prior, p_prior = prior_data["densities"], prior_data["pressures"]
        n_TOV_prior = prior_data.get("n_TOV", None)
        for i in range(len(n_prior)):
            mask = _get_density_mask(
                n_prior[i], float(n_TOV_prior[i]) if n_TOV_prior is not None else None
            )
            plt.plot(
                n_prior[i][mask],
                p_prior[i][mask],
                color=COLORS_DICT["prior"],
                alpha=0.1,
                rasterized=True,
                zorder=1,
            )

    n, p = data["densities"], data["pressures"]
    n_TOV = data.get("n_TOV", None)

    valid_indices, _ = _get_valid_indices(data)
    prob, norm, cmap, _ = _setup_prob_coloring(data, use_crest_cmap)

    n_bad = len(data["masses"]) - len(valid_indices)
    logger.info(
        f"Plotting {len(valid_indices)} p-n curves (excluded {n_bad} invalid)..."
    )

    for i in valid_indices:
        normalized_value = float(norm(prob[i]))
        color = cmap(normalized_value)
        mask = _get_density_mask(n[i], float(n_TOV[i]) if n_TOV is not None else None)
        plt.plot(
            n[i][mask],
            p[i][mask],
            color=color,
            alpha=1.0,
            rasterized=True,
            zorder=1e10 + normalized_value,
        )

    if injection_data is not None and "n" in injection_data and "p" in injection_data:
        for i, (n_inj, p_inj) in enumerate(
            zip(injection_data["n"], injection_data["p"])
        ):
            mask = n_inj > 0.5
            plt.plot(
                n_inj[mask],
                p_inj[mask],
                color=INJECTION_COLOR,
                alpha=INJECTION_ALPHA,
                linewidth=INJECTION_LINEWIDTH,
                linestyle=INJECTION_LINESTYLE,
                zorder=1e11,
                label="Injection" if i == 0 else "",
            )

    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]" if TEX_ENABLED else "n [n_sat]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]" if TEX_ENABLED else "p [MeV fm^-3]")
    plt.yscale("log")
    plt.xlim(left=0.5, right=6.0)

    _add_prior_injection_legend(prior_data, injection_data, "n", loc="upper left")

    save_name = os.path.join(outdir, "pressure_density_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Pressure-density plot saved to {save_name}")


def make_cs2_plot(
    data: Dict[str, Any],
    prior_data: Optional[Dict[str, Any]],
    outdir: str,
    use_crest_cmap: bool = True,
    injection_data: Optional[Dict[str, Any]] = None,
) -> None:
    r"""Create a speed-of-sound-squared vs density plot.

    Parameters
    ----------
    data : dict
        EOS data dictionary.
    prior_data : dict or None
        Prior EOS data for background comparison.
    outdir : str
        Output directory.
    use_crest_cmap : bool, optional
        Use the seaborn crest colourmap, by default True.
    injection_data : dict or None, optional
        Injection EOS data for plotting the true values, by default None.
    """
    logger.info("Creating cs2-density plot...")

    plt.figure(figsize=(11, 6))

    if prior_data is not None:
        n_prior, cs2_prior = prior_data["densities"], prior_data["cs2"]
        n_TOV_prior = prior_data.get("n_TOV", None)
        for i in range(len(n_prior)):
            mask = _get_density_mask(
                n_prior[i], float(n_TOV_prior[i]) if n_TOV_prior is not None else None
            )
            plt.plot(
                n_prior[i][mask],
                cs2_prior[i][mask],
                color=COLORS_DICT["prior"],
                alpha=0.1,
                rasterized=True,
                zorder=1,
            )

    n, cs2 = data["densities"], data["cs2"]
    n_TOV = data.get("n_TOV", None)

    valid_indices, _ = _get_valid_indices(data)
    prob, norm, cmap, _ = _setup_prob_coloring(data, use_crest_cmap)

    n_bad = len(data["masses"]) - len(valid_indices)
    logger.info(
        f"Plotting {len(valid_indices)} cs2-n curves (excluded {n_bad} invalid)..."
    )

    for i in valid_indices:
        normalized_value = float(norm(prob[i]))
        color = cmap(normalized_value)
        mask = _get_density_mask(n[i], float(n_TOV[i]) if n_TOV is not None else None)
        plt.plot(
            n[i][mask],
            cs2[i][mask],
            color=color,
            alpha=1.0,
            rasterized=True,
            zorder=1e10 + normalized_value,
        )

    if injection_data is not None and "n" in injection_data and "cs2" in injection_data:
        for i, (n_inj, cs2_inj) in enumerate(
            zip(injection_data["n"], injection_data["cs2"])
        ):
            mask = n_inj > 0.5
            plt.plot(
                n_inj[mask],
                cs2_inj[mask],
                color=INJECTION_COLOR,
                alpha=INJECTION_ALPHA,
                linewidth=INJECTION_LINEWIDTH,
                linestyle=INJECTION_LINESTYLE,
                zorder=1e11,
                label="Injection" if i == 0 else "",
            )

    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]" if TEX_ENABLED else "n [n_sat]")
    plt.ylabel(r"$c_s^2$" if TEX_ENABLED else "cs2")
    plt.xlim(left=0.5, right=6.0)
    plt.ylim(0.0, 1.2)

    _add_prior_injection_legend(prior_data, injection_data, "cs2", loc="upper left")

    save_name = os.path.join(outdir, "cs2_density_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"cs2-density plot saved to {save_name}")


def make_parameter_histograms(
    data: Dict[str, Any],
    outdir: str,
    injection_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Create KDE histograms for key EOS-derived parameters.

    The KDE line is drawn over the full x-range; the shaded region is filled
    only within the :data:`HDI_PROB` credible interval.

    Parameters
    ----------
    data : dict
        EOS data dictionary.
    outdir : str
        Output directory.
    injection_data : dict or None, optional
        Injection EOS data for plotting true values, by default None.
    """
    logger.info("Creating parameter histograms...")

    m, r, l = data["masses"], data["radii"], data["lambdas"]
    n, p = data["densities"], data["pressures"]

    MTOV_list = np.array([np.max(mass) for mass in m])
    R14_list = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(m, r)])
    Lambda14_list = np.array([np.interp(1.4, mass, lam) for mass, lam in zip(m, l)])
    p3nsat_list = np.array([np.interp(3.0, dens, press) for dens, press in zip(n, p)])

    _n_TOV_raw = data.get("n_TOV", None)
    n_TOV_list = (
        np.array(_n_TOV_raw)[np.array(_n_TOV_raw) > 0.0]
        if _n_TOV_raw is not None
        else None
    )

    # Injection values
    injection_params: Dict[str, float] = {}
    if injection_data is not None:
        if "masses_EOS" in injection_data and "radii_EOS" in injection_data:
            m_inj, r_inj = injection_data["masses_EOS"], injection_data["radii_EOS"]
            injection_params["MTOV"] = float(np.max(m_inj[0]))
            injection_params["R14"] = float(np.interp(1.4, m_inj[0], r_inj[0]))
        if "masses_EOS" in injection_data and "Lambda_EOS" in injection_data:
            m_inj, l_inj = injection_data["masses_EOS"], injection_data["Lambda_EOS"]
            injection_params["Lambda14"] = float(np.interp(1.4, m_inj[0], l_inj[0]))
        if "n" in injection_data and "p" in injection_data:
            n_inj, p_inj = injection_data["n"], injection_data["p"]
            injection_params["p3nsat"] = float(np.interp(3.0, n_inj[0], p_inj[0]))
        if "n_TOV" in injection_data:
            injection_params["n_TOV"] = float(injection_data["n_TOV"][0])

    if TEX_ENABLED:
        parameters: Dict[str, Dict[str, Any]] = {
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

    if n_TOV_list is not None:
        label = r"$n_{\rm{TOV}}$ [$n_{\rm{sat}}$]" if TEX_ENABLED else "n_TOV [n_sat]"
        parameters["n_TOV"] = {"values": n_TOV_list, "xlabel": label}

    for param_name, param_info in parameters.items():
        values = param_info["values"]
        if values is None or len(values) == 0:
            logger.warning(f"Skipping histogram for {param_name}: no data.")
            continue

        plt.figure(figsize=figsize_horizontal)

        hdi = az.hdi(values, hdi_prob=HDI_PROB)
        hdi_low, hdi_high = float(hdi[0]), float(hdi[1])
        median = float(np.median(values))
        low_err = median - hdi_low
        high_err = hdi_high - median

        hdi_width = hdi_high - hdi_low
        x_min = hdi_low - 0.25 * hdi_width
        x_max = hdi_high + 0.25 * hdi_width

        kde_ok = len(values) > 1 and float(np.std(values)) > 0.0
        if kde_ok:
            try:
                kde = gaussian_kde(values)
                x = np.linspace(x_min, x_max, 1000)
                y = kde(x)
                plt.plot(
                    x, y, color=COLORS_DICT["posterior"], lw=3.0, label="Posterior"
                )
                hdi_mask: list[bool] = list((x >= hdi_low) & (x <= hdi_high))
                plt.fill_between(
                    x, y, where=hdi_mask, alpha=0.3, color=COLORS_DICT["posterior"]
                )
            except Exception:
                kde_ok = False
                logger.warning(
                    f"KDE failed for {param_name}; falling back to histogram."
                )
        if not kde_ok:
            logger.warning(
                f"Skipping KDE for {param_name}: insufficient or degenerate data "
                f"(n={len(values)}, std={float(np.std(values)) if len(values) > 0 else 0:.4g})."
            )
            plt.axvline(
                median,
                color=COLORS_DICT["posterior"],
                lw=3.0,
                label="Posterior (median)",
            )

        if param_name in injection_params:
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

        plt.xlabel(param_info["xlabel"])
        plt.ylabel("Density")
        plt.xlim(x_min, x_max)
        plt.ylim(bottom=0.0)
        plt.legend()

        xlabel = param_info["xlabel"]
        credibility_pct = int(HDI_PROB * 100)
        if TEX_ENABLED:
            plt.title(
                f"{xlabel}: ${median:.2f}_{{-{low_err:.2f}}}^{{+{high_err:.2f}}}$"
                f" ({credibility_pct}\\% credibility)"
            )
        else:
            plt.title(
                f"{xlabel}: {median:.2f} -{low_err:.2f} +{high_err:.2f}"
                f" ({credibility_pct}% credibility)"
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
) -> None:
    """Create a contour plot of radii vs mass.

    Parameters
    ----------
    data : dict
        EOS data dictionary.
    prior_data : dict or None
        Prior EOS data for comparison.
    outdir : str
        Output directory.
    m_min : float, optional
        Minimum mass, by default 0.6.
    m_max : float, optional
        Maximum mass, by default 2.1.
    """
    logger.info("Creating radii contour plot...")

    plt.figure(figsize=figsize_vertical)
    masses_array = np.linspace(m_min, m_max, 100)

    if prior_data is not None:
        m_prior, r_prior = prior_data["masses"], prior_data["radii"]
        radii_low_prior = np.empty_like(masses_array)
        radii_high_prior = np.empty_like(masses_array)
        for i, mass_point in enumerate(masses_array):
            radii_at_mass = np.array(
                [
                    float(np.interp(mass_point, mass, radius))
                    for mass, radius in zip(m_prior, r_prior)
                ]
            )
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

    m, r = data["masses"], data["radii"]
    radii_low = np.empty_like(masses_array)
    radii_high = np.empty_like(masses_array)

    logger.info(f"Computing radii contours for {len(masses_array)} mass points...")
    for i, mass_point in enumerate(masses_array):
        radii_at_mass = np.array(
            [float(np.interp(mass_point, mass, radius)) for mass, radius in zip(m, r)]
        )
        low, med, high = report_credible_interval(radii_at_mass, hdi_prob=HDI_PROB)
        radii_low[i] = med - low
        radii_high[i] = med + high

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

    plt.xlabel(r"$R$ [km]" if TEX_ENABLED else "R [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]" if TEX_ENABLED else "M [M_sun]")
    plt.xlim(8.0, 16.0)
    plt.ylim(m_min, m_max)
    plt.legend()

    save_name = os.path.join(outdir, "radii_contour_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Radii contour plot saved to {save_name}")


def make_contour_pressures_plot(
    data: Dict[str, Any], outdir: str, n_min: float = 0.5, n_max: float = 6.0
) -> None:
    """Create a contour plot of pressure vs density.

    Parameters
    ----------
    data : dict
        EOS data dictionary.
    outdir : str
        Output directory.
    n_min : float, optional
        Minimum density, by default 0.5.
    n_max : float, optional
        Maximum density, by default 6.0.
    """
    logger.info("Creating pressures contour plot...")

    n, p = data["densities"], data["pressures"]
    plt.figure(figsize=figsize_horizontal)

    dens_array = np.linspace(n_min, n_max, 100)
    press_low = np.empty_like(dens_array)
    press_high = np.empty_like(dens_array)

    logger.info(f"Computing pressure contours for {len(dens_array)} density points...")
    for i, dens in enumerate(dens_array):
        press_at_dens = np.array(
            [
                float(np.interp(dens, density, pressure))
                for density, pressure in zip(n, p)
            ]
        )
        low, med, high = report_credible_interval(press_at_dens, hdi_prob=HDI_PROB)
        press_low[i] = med - low
        press_high[i] = med + high

    plt.fill_between(
        dens_array, press_low, press_high, alpha=0.5, color=COLORS_DICT["posterior"]
    )
    plt.plot(dens_array, press_low, lw=2.0, color=COLORS_DICT["posterior"])
    plt.plot(dens_array, press_high, lw=2.0, color=COLORS_DICT["posterior"])

    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]" if TEX_ENABLED else "n [n_sat]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]" if TEX_ENABLED else "p [MeV fm^-3]")
    plt.xlim(n_min, n_max)
    plt.yscale("log")
    plt.legend()

    save_name = os.path.join(outdir, "pressures_contour_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Pressures contour plot saved to {save_name}")


# ─── Orchestration ────────────────────────────────────────────────────────────


def generate_all_plots(
    outdir: str,
    prior_dir: Optional[str] = None,
    make_cornerplot_flag: bool = True,
    make_massradius_flag: bool = True,
    make_masslambda_flag: bool = True,
    make_pressuredensity_flag: bool = True,
    make_histograms_flag: bool = True,
    make_cs2_flag: bool = True,
    make_contours_flag: bool = False,
    injection_eos_path: Optional[str] = None,
) -> None:
    """Generate selected plots for the specified output directory.

    Parameters
    ----------
    outdir : str
        Output directory containing ``results.h5``.
    prior_dir : str, optional
        Directory containing prior samples for comparison.
    make_cornerplot_flag : bool, optional
        Generate cornerplot of EOS parameters, by default True.
    make_massradius_flag : bool, optional
        Generate mass-radius plot, by default True.
    make_masslambda_flag : bool, optional
        Generate mass-Lambda plot, by default True.
    make_pressuredensity_flag : bool, optional
        Generate pressure-density plot, by default True.
    make_histograms_flag : bool, optional
        Generate parameter histograms, by default True.
    make_cs2_flag : bool, optional
        Generate cs2-density plot, by default True.
    make_contours_flag : bool, optional
        Generate radii and pressure credible-interval contour plots, by default False.
    injection_eos_path : str, optional
        Path to NPZ file containing injection EOS data, by default None.
    """
    logger.info(f"Generating plots for directory: {outdir}")

    figures_dir = os.path.join(outdir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    logger.info(f"Saving plots to: {figures_dir}")

    try:
        data = load_eos_data(outdir)
        logger.info("Data loaded successfully!")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return

    prior_data = None
    if prior_dir is not None:
        prior_data = load_prior_data(prior_dir)
        if prior_data is not None:
            logger.info("Prior data loaded successfully!")

    injection_data = None
    if injection_eos_path is not None:
        injection_data = load_injection_eos(injection_eos_path)
        if injection_data is not None:
            logger.info("Injection EOS data loaded successfully!")

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
        if injection_data is not None:
            make_mass_lambda_ratio_plot(data, figures_dir, injection_data)

    if make_pressuredensity_flag:
        make_pressure_density_plot(
            data, prior_data, figures_dir, injection_data=injection_data
        )

    if make_histograms_flag:
        make_parameter_histograms(data, figures_dir, injection_data=injection_data)

    if make_cs2_flag:
        make_cs2_plot(data, prior_data, figures_dir, injection_data=injection_data)

    if make_contours_flag:
        make_contour_radii_plot(data, prior_data, figures_dir)
        make_contour_pressures_plot(data, figures_dir)

    logger.info(f"All plots generated and saved to {figures_dir}")


def run_from_config(config_path: str) -> None:
    """Run postprocessing from a YAML config file.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    """
    from jesterTOV.inference.config.parser import load_config

    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    if not config.postprocessing.enabled:
        logger.warning(
            "Postprocessing is disabled in config. "
            "Set postprocessing.enabled: true to run."
        )
        return

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
    logger.info(f"Make contours: {config.postprocessing.make_contours}")
    logger.info("=" * 60)

    generate_all_plots(
        outdir=outdir,
        prior_dir=config.postprocessing.prior_dir,
        make_cornerplot_flag=config.postprocessing.make_cornerplot,
        make_massradius_flag=config.postprocessing.make_massradius,
        make_masslambda_flag=config.postprocessing.make_masslambda,
        make_pressuredensity_flag=config.postprocessing.make_pressuredensity,
        make_histograms_flag=config.postprocessing.make_histograms,
        make_cs2_flag=config.postprocessing.make_cs2,
        make_contours_flag=config.postprocessing.make_contours,
        injection_eos_path=config.postprocessing.injection_eos_path,
    )

    logger.info(
        f"\nPostprocessing complete! Plots saved to {os.path.join(outdir, 'figures')}"
    )


def main() -> None:
    """Entry point: run_jester_postprocessing <config.yaml>."""
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".yaml"):
        print(
            "Usage: run_jester_postprocessing <config.yaml>\n"
            "Example: run_jester_postprocessing my_run/config.yaml"
        )
        sys.exit(1)
    run_from_config(sys.argv[1])


if __name__ == "__main__":
    main()
