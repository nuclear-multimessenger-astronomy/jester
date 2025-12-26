r"""Modular postprocessing script for EOS inference results.

This script provides comprehensive visualization tools for analyzing equation of state (EOS)
inference results. It generates various plots including cornerplots, mass-radius diagrams,
and pressure-density relationships with posterior probability color coding.

Usage:
    run_jester_postprocessing --outdir <path> [--make-cornerplot] [--make-massradius] [--make-pressuredensity]

Example:
    run_jester_postprocessing --outdir ./results --make-all
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
import os
import argparse
from scipy.stats import gaussian_kde
from typing import Dict, Optional, Any
import warnings
import arviz as az

np.random.seed(2)
import jesterTOV.utils as jose_utils
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
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Serif"],
            })
            # Test if TeX actually works
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, r"$\alpha$")
            plt.close(fig)
            tex_enabled = True
            logger.info("TeX rendering enabled")
        except Exception as e:
            warnings.warn(f"TeX rendering failed ({e}). Falling back to non-TeX rendering.")
            plt.rcParams.update({
                "text.usetex": False,
                "font.family": "sans-serif",
            })

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
        "figure.titlesize": 16
    }
    plt.rcParams.update(mpl_params)

    return tex_enabled

# Initialize matplotlib with TeX fallback
TEX_ENABLED = setup_matplotlib(use_tex=True)

# Default colormap
DEFAULT_COLORMAP = sns.color_palette("crest", as_cmap=True)

# Constants
COLORS_DICT = {
    "prior": "gray",
    "GW170817": "orange",
    "GW231109": "teal",
    "GW231109_only": "red",
    "posterior": "blue"
}
ALPHA = 0.3
figsize_vertical = (6, 8)
figsize_horizontal = (8, 6)

# Prior directory (for loading prior samples)
PRIOR_DIR = "./outdir/"


def load_eos_data(outdir: str) -> Dict[str, np.ndarray]:
    """Load EOS data from the specified output directory.

    Parameters
    ----------
    outdir : str
        Path to output directory containing eos_samples.npz

    Returns
    -------
    dict
        Dictionary containing EOS data arrays

    Raises
    ------
    FileNotFoundError
        If eos_samples.npz is not found in the specified directory
    """
    filename = os.path.join(outdir, "eos_samples.npz")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"EOS samples file not found: {filename}")

    data = np.load(filename)

    # Load macroscopic quantities
    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]

    # Convert units
    n = n / jose_utils.fm_inv3_to_geometric / 0.16
    p = p / jose_utils.MeV_fm_inv3_to_geometric
    e = e / jose_utils.MeV_fm_inv3_to_geometric

    log_prob = data["log_prob"]

    # Load NEP parameters if available (for cornerplot)
    nep_params = {}
    nep_keys = ["K_sat", "L_sym", "Q_sat", "Q_sym", "Z_sat", "Z_sym", "E_sym", "K_sym"]
    for key in nep_keys:
        if key in data:
            nep_params[key] = data[key]

    # Load CSE parameters if available
    cse_params = {}
    if "nbreak" in data:
        cse_params["nbreak"] = data["nbreak"]
    for key in data.keys():
        if key.startswith("cs2_CSE_") or key.startswith("n_CSE_"):
            cse_params[key] = data[key]

    result = {
        'masses': m,
        'radii': r,
        'lambdas': l,
        'densities': n,
        'pressures': p,
        'energies': e,
        'cs2': cs2,
        'log_prob': log_prob,
        'nep_params': nep_params,
        'cse_params': cse_params
    }

    return result


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


def report_credible_interval(values: np.ndarray,
                             hdi_prob: float = 0.90,
                             verbose: bool = False) -> tuple:
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
        logger.info(f"{med:.2f} -{low_err:.2f} +{high_err:.2f} (at {hdi_prob} HDI prob)")

    return low_err, med, high_err


def make_cornerplot(data: Dict[str, Any], outdir: str, max_params: int = 10):
    """Create cornerplot for EOS parameters.

    Parameters
    ----------
    data : dict
        EOS data dictionary from load_eos_data
    outdir : str
        Output directory for saving the plot
    max_params : int, optional
        Maximum number of parameters to include, by default 10
    """
    logger.info("Creating cornerplot...")

    # Collect parameters for cornerplot
    samples_dict = {}
    labels = []

    # Add NEP parameters
    nep_params = data.get('nep_params', {})
    for key in ["K_sat", "L_sym", "Q_sat", "Q_sym", "Z_sat", "Z_sym", "E_sym", "K_sym"]:
        if key in nep_params:
            samples_dict[key] = nep_params[key]
            if TEX_ENABLED:
                # Format LaTeX labels
                base = key.split("_")[0]
                sub = key.split("_")[1] if "_" in key else ""
                labels.append(f"${base}_{{{sub}}}$")
            else:
                labels.append(key)

    # Add a few CSE parameters if available (limit to avoid overcrowding)
    cse_params = data.get('cse_params', {})
    if "nbreak" in cse_params:
        samples_dict["nbreak"] = cse_params["nbreak"]
        labels.append(r"$n_{\rm{break}}$" if TEX_ENABLED else "n_break")

    # Limit number of parameters
    if len(samples_dict) > max_params:
        logger.info(f"Limiting cornerplot to first {max_params} parameters")
        samples_dict = dict(list(samples_dict.items())[:max_params])
        labels = labels[:max_params]

    if len(samples_dict) == 0:
        logger.warning("No parameters found for cornerplot")
        return

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
        smooth=1.0
    )

    # Save figure
    save_name = os.path.join(outdir, "cornerplot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Cornerplot saved to {save_name}")


def make_mass_radius_plot(data: Dict[str, Any],
                          prior_data: Optional[Dict[str, Any]],
                          outdir: str,
                          use_crest_cmap: bool = True):
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
    """
    logger.info("Creating mass-radius plot...")

    plt.figure(figsize=(10, 8))
    m_min, m_max = 0.75, 3.5
    r_min, r_max = 6.0, 18.0

    # Plot prior first (background)
    if prior_data is not None:
        m_prior, r_prior = prior_data['masses'], prior_data['radii']
        for i in range(len(m_prior)):
            plt.plot(r_prior[i], m_prior[i],
                    color=COLORS_DICT['prior'],
                    alpha=0.1,
                    rasterized=True,
                    zorder=1)

    # Plot posterior with probability coloring
    m, r, l = data['masses'], data['radii'], data['lambdas']
    log_prob = data['log_prob']
    nb_samples = np.shape(m)[0]
    logger.info(f"Number of samples: {nb_samples}")

    # Normalize probabilities for coloring
    prob = np.exp(log_prob - np.max(log_prob))  # Normalize to avoid overflow
    norm = plt.Normalize(vmin=np.min(prob), vmax=np.max(prob))
    cmap = DEFAULT_COLORMAP if use_crest_cmap else plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    bad_counter = 0
    logger.info(f"Plotting {len(prob)} M-R curves...")
    for i in range(len(prob)):
        # Skip invalid samples
        if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
            bad_counter += 1
            continue

        if any(l[i] < 0):
            bad_counter += 1
            continue

        if any((m[i] > 1.0) * (r[i] > 20.0)):
            bad_counter += 1
            continue

        # Get color based on probability
        normalized_value = norm(prob[i])
        color = cmap(normalized_value)

        plt.plot(r[i], m[i],
                color=color,
                alpha=1.0,
                rasterized=True,
                zorder=1e10 + normalized_value)

    logger.info(f"Excluded {bad_counter} invalid samples")

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
    cbar_ax = fig.add_axes([0.15, 0.94, 0.7, 0.03])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Normalized posterior probability", fontsize=16)
    cbar.set_ticks([])
    cbar.ax.xaxis.labelpad = 5
    cbar.ax.tick_params(labelsize=0, length=0)
    cbar.ax.xaxis.set_label_position('top')

    # Add legend for prior
    if prior_data is not None:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=COLORS_DICT['prior'], lw=2, alpha=0.7, label='Prior')]
        plt.legend(handles=legend_elements, loc='upper right')

    # Save figure
    save_name = os.path.join(outdir, "mass_radius_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Mass-radius plot saved to {save_name}")


def make_pressure_density_plot(data: Dict[str, Any],
                               prior_data: Optional[Dict[str, Any]],
                               outdir: str,
                               use_crest_cmap: bool = True):
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
    """
    logger.info("Creating pressure-density plot...")

    plt.figure(figsize=(11, 6))

    # Plot prior first (background)
    if prior_data is not None:
        n_prior, p_prior = prior_data['densities'], prior_data['pressures']
        for i in range(len(n_prior)):
            mask = (n_prior[i] > 0.5) * (n_prior[i] < 6.0)
            plt.plot(n_prior[i][mask], p_prior[i][mask],
                    color=COLORS_DICT['prior'],
                    alpha=0.1,
                    rasterized=True,
                    zorder=1)

    # Plot posterior with probability coloring
    m, r, l = data['masses'], data['radii'], data['lambdas']
    n, p = data['densities'], data['pressures']
    log_prob = data['log_prob']

    # Normalize probabilities for coloring
    prob = np.exp(log_prob - np.max(log_prob))
    norm = plt.Normalize(vmin=np.min(prob), vmax=np.max(prob))
    cmap = DEFAULT_COLORMAP if use_crest_cmap else sns.color_palette("viridis", as_cmap=True)

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

        if any((m[i] > 1.0) * (r[i] > 20.0)):
            bad_counter += 1
            continue

        # Get color and plot
        normalized_value = norm(prob[i])
        color = cmap(normalized_value)

        mask = (n[i] > 0.5) * (n[i] < 6.0)
        plt.plot(n[i][mask], p[i][mask],
                color=color,
                alpha=1.0,
                rasterized=True,
                zorder=1e10 + normalized_value)

    logger.info(f"Excluded {bad_counter} invalid samples")

    xlabel = r"$n$ [$n_{\rm{sat}}$]" if TEX_ENABLED else "n [n_sat]"
    ylabel = r"$p$ [MeV fm$^{-3}$]" if TEX_ENABLED else "p [MeV fm^-3]"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.xlim(0.5, 6.0)

    # Add legend for prior
    if prior_data is not None:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=COLORS_DICT['prior'], lw=2, alpha=0.7, label='Prior')]
        plt.legend(handles=legend_elements, loc='upper left')

    # Save figure
    save_name = os.path.join(outdir, "pressure_density_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Pressure-density plot saved to {save_name}")


def make_parameter_histograms(data: Dict[str, Any],
                              prior_data: Optional[Dict[str, Any]],
                              outdir: str):
    """Create histograms for key EOS parameters.

    Parameters
    ----------
    data : dict
        EOS data dictionary
    prior_data : dict or None
        Prior EOS data for comparison
    outdir : str
        Output directory
    """
    logger.info("Creating parameter histograms...")

    m, r = data['masses'], data['radii']
    n, p = data['densities'], data['pressures']

    # Calculate derived parameters
    MTOV_list = np.array([np.max(mass) for mass in m])
    R14_list = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(m, r)])
    p3nsat_list = np.array([np.interp(3.0, dens, press) for dens, press in zip(n, p)])

    # Calculate prior parameters if available
    prior_params = {}
    if prior_data is not None:
        m_prior, r_prior = prior_data['masses'], prior_data['radii']
        n_prior, p_prior = prior_data['densities'], prior_data['pressures']

        prior_params['MTOV'] = np.array([np.max(mass) for mass in m_prior])
        prior_params['R14'] = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(m_prior, r_prior)])
        prior_params['p3nsat'] = np.array([np.interp(3.0, dens, press) for dens, press in zip(n_prior, p_prior)])

    # Define parameters to plot (without fixed ranges)
    if TEX_ENABLED:
        parameters = {
            'MTOV': {'values': MTOV_list, 'xlabel': r"$M_{\rm{TOV}}$ [$M_{\odot}$]"},
            'R14': {'values': R14_list, 'xlabel': r"$R_{1.4}$ [km]"},
            'p3nsat': {'values': p3nsat_list, 'xlabel': r"$p(3n_{\rm{sat}})$ [MeV fm$^{-3}$]"}
        }
    else:
        parameters = {
            'MTOV': {'values': MTOV_list, 'xlabel': "M_TOV [M_sun]"},
            'R14': {'values': R14_list, 'xlabel': "R_1.4 [km]"},
            'p3nsat': {'values': p3nsat_list, 'xlabel': "p(3n_sat) [MeV fm^-3]"}
        }

    for param_name, param_data in parameters.items():
        plt.figure(figsize=figsize_horizontal)

        # Calculate 90% HDI using arviz
        hdi = az.hdi(param_data['values'], hdi_prob=0.90)
        hdi_low, hdi_high = hdi
        median = np.median(param_data['values'])

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
            plt.fill_between(x, y_prior, alpha=ALPHA, color=COLORS_DICT['prior'], label='Prior')

        # Create posterior KDE
        kde = gaussian_kde(param_data['values'])
        x = np.linspace(x_min, x_max, 1000)
        y = kde(x)

        plt.plot(x, y, color=COLORS_DICT['posterior'], lw=3.0, label='Posterior')
        plt.fill_between(x, y, alpha=0.3, color=COLORS_DICT['posterior'])

        plt.xlabel(param_data['xlabel'])
        plt.ylabel('Density')
        plt.xlim(x_min, x_max)
        plt.ylim(bottom=0.0)
        plt.legend()

        # Format title with subscript/superscript notation
        if TEX_ENABLED:
            plt.title(f'{param_name}: ${median:.2f}_{{-{low_err:.2f}}}^{{+{high_err:.2f}}}$')
        else:
            plt.title(f'{param_name}: {median:.2f}_{{{-low_err:.2f}}}^{{{+high_err:.2f}}}')

        save_name = os.path.join(outdir, f"{param_name}_histogram.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        logger.info(f"{param_name} histogram saved to {save_name}")


def make_contour_radii_plot(data: Dict[str, Any],
                            prior_data: Optional[Dict[str, Any]],
                            outdir: str,
                            m_min: float = 0.6,
                            m_max: float = 2.1):
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
        m_prior, r_prior = prior_data['masses'], prior_data['radii']
        radii_low_prior = np.empty_like(masses_array)
        radii_high_prior = np.empty_like(masses_array)

        for i, mass_point in enumerate(masses_array):
            radii_at_mass = []
            for mass, radius in zip(m_prior, r_prior):
                radii_at_mass.append(np.interp(mass_point, mass, radius))
            radii_at_mass = np.array(radii_at_mass)

            low, med, high = report_credible_interval(radii_at_mass, hdi_prob=0.90)
            radii_low_prior[i] = med - low
            radii_high_prior[i] = med + high

        plt.fill_betweenx(masses_array, radii_low_prior, radii_high_prior,
                          alpha=ALPHA, color=COLORS_DICT['prior'], label='Prior')

    # Plot posterior contours
    m, r = data['masses'], data['radii']
    radii_low = np.empty_like(masses_array)
    radii_high = np.empty_like(masses_array)

    logger.info(f"Computing radii contours for {len(masses_array)} mass points...")
    for i, mass_point in enumerate(masses_array):
        # Gather all radii at this mass point
        radii_at_mass = []
        for mass, radius in zip(m, r):
            radii_at_mass.append(np.interp(mass_point, mass, radius))
        radii_at_mass = np.array(radii_at_mass)

        # Construct 95% credible interval
        low, med, high = report_credible_interval(radii_at_mass, hdi_prob=0.90)

        radii_low[i] = med - low
        radii_high[i] = med + high

    # Plot posterior contours
    plt.fill_betweenx(masses_array, radii_low, radii_high,
                      alpha=0.5, color=COLORS_DICT['posterior'], label='Posterior')
    plt.plot(radii_low, masses_array, lw=2.0, color=COLORS_DICT['posterior'])
    plt.plot(radii_high, masses_array, lw=2.0, color=COLORS_DICT['posterior'])

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


def make_contour_pressures_plot(data: Dict[str, Any],
                                outdir: str,
                                n_min: float = 0.5,
                                n_max: float = 6.0):
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

    n, p = data['densities'], data['pressures']

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

        # Construct 95% credible interval
        low, med, high = report_credible_interval(press_at_dens, hdi_prob=0.95)

        press_low[i] = med - low
        press_high[i] = med + high

    # Plot contours
    plt.fill_between(dens_array, press_low, press_high,
                     alpha=0.5, color=COLORS_DICT['posterior'])
    plt.plot(dens_array, press_low, lw=2.0, color=COLORS_DICT['posterior'])
    plt.plot(dens_array, press_high, lw=2.0, color=COLORS_DICT['posterior'])

    xlabel = r"$n$ [$n_{\rm{sat}}$]" if TEX_ENABLED else "n [n_sat]"
    ylabel = r"$p$ [MeV fm$^{-3}$]" if TEX_ENABLED else "p [MeV fm^-3]"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(n_min, n_max)
    plt.yscale('log')
    plt.legend()

    save_name = os.path.join(outdir, "pressures_contour_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    logger.info(f"Pressures contour plot saved to {save_name}")


def generate_all_plots(outdir: str,
                      prior_dir: Optional[str] = None,
                      make_cornerplot_flag: bool = True,
                      make_massradius_flag: bool = True,
                      make_pressuredensity_flag: bool = True,
                      make_histograms_flag: bool = True,
                      make_contours_flag: bool = True):
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
    make_pressuredensity_flag : bool, optional
        Whether to generate pressure-density plot, by default True
    make_histograms_flag : bool, optional
        Whether to generate parameter histograms, by default True
    make_contours_flag : bool, optional
        Whether to generate contour plots, by default True
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

    # Create plots based on flags (pass figures_dir instead of outdir)
    if make_cornerplot_flag:
        make_cornerplot(data, figures_dir)

    if make_massradius_flag:
        make_mass_radius_plot(data, prior_data, figures_dir)

    if make_pressuredensity_flag:
        make_pressure_density_plot(data, prior_data, figures_dir)

    if make_histograms_flag:
        make_parameter_histograms(data, prior_data, figures_dir)

    if make_contours_flag:
        make_contour_radii_plot(data, prior_data, figures_dir)
        make_contour_pressures_plot(data, figures_dir)

    logger.info(f"All plots generated and saved to {figures_dir}")


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description='Generate EOS postprocessing plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots
  run_jester_postprocessing --outdir ./results --make-all

  # Generate only cornerplot and mass-radius plot
  run_jester_postprocessing --outdir ./results --make-cornerplot --make-massradius

  # Generate with prior comparison
  run_jester_postprocessing --outdir ./results --prior-dir ./prior --make-all
        """
    )
    parser.add_argument('--outdir', type=str, required=True,
                       help='Output directory containing eos_samples.npz')
    parser.add_argument('--prior-dir', type=str, default=None,
                       help='Directory containing prior samples for comparison')

    # Plot selection flags
    parser.add_argument('--make-all', action='store_true',
                       help='Generate all plots (default if no specific flags given)')
    parser.add_argument('--make-cornerplot', action='store_true',
                       help='Generate cornerplot of EOS parameters')
    parser.add_argument('--make-massradius', action='store_true',
                       help='Generate mass-radius plot')
    parser.add_argument('--make-pressuredensity', action='store_true',
                       help='Generate pressure-density plot')
    parser.add_argument('--make-histograms', action='store_true',
                       help='Generate parameter histograms')
    parser.add_argument('--make-contours', action='store_true',
                       help='Generate contour plots')

    # Additional options
    parser.add_argument('--m-min', type=float, default=0.6,
                       help='Minimum mass for contour plots')
    parser.add_argument('--m-max', type=float, default=2.5,
                       help='Maximum mass for contour plots')
    parser.add_argument('--n-min', type=float, default=0.5,
                       help='Minimum density for pressure contour plots')
    parser.add_argument('--n-max', type=float, default=6.0,
                       help='Maximum density for pressure contour plots')

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.outdir):
        logger.error(f"Error: Output directory {args.outdir} does not exist")
        return

    # Determine which plots to make
    # If --make-all or no specific flags given, make all plots
    make_all = args.make_all or not any([
        args.make_cornerplot,
        args.make_massradius,
        args.make_pressuredensity,
        args.make_histograms,
        args.make_contours
    ])

    # Generate plots
    generate_all_plots(
        args.outdir,
        prior_dir=args.prior_dir,
        make_cornerplot_flag=make_all or args.make_cornerplot,
        make_massradius_flag=make_all or args.make_massradius,
        make_pressuredensity_flag=make_all or args.make_pressuredensity,
        make_histograms_flag=make_all or args.make_histograms,
        make_contours_flag=make_all or args.make_contours
    )


if __name__ == "__main__":
    main()
