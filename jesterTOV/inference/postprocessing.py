"""Modular postprocessing script for EOS inference results."""

import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import argparse
from scipy.stats import gaussian_kde
# import seaborn as sns # TODO: not sure if we need this
# import arviz # TODO: not sure if we need this

np.random.seed(2)
import jesterTOV.utils as jose_utils

# Matplotlib parameters
mpl_params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}
plt.rcParams.update(mpl_params)

# Constants
COLORS_DICT = {"prior": "gray",
               "GW170817": "orange",
               "GW231109": "teal",
               "GW231109_only": "red"}
ALPHA = 0.3
figsize_vertical = (6, 8)
figsize_horizontal = (8, 6)

# Prior directory
PRIOR_DIR = "./outdir/"

def load_eos_data(outdir: str):
    """Load EOS data from the specified output directory."""
    filename = os.path.join(outdir, "eos_samples.npz")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"EOS samples file not found: {filename}")
    
    data = np.load(filename)
    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
    
    # Convert units
    n = n / jose_utils.fm_inv3_to_geometric / 0.16
    p = p / jose_utils.MeV_fm_inv3_to_geometric
    e = e / jose_utils.MeV_fm_inv3_to_geometric
    
    log_prob = data["log_prob"]
    
    return {
        'masses': m,
        'radii': r,
        'lambdas': l,
        'densities': n,
        'pressures': p,
        'energies': e,
        'cs2': cs2,
        'log_prob': log_prob
    }

def load_prior_data():
    """Load prior EOS data for comparison."""
    try:
        return load_eos_data(PRIOR_DIR)
    except FileNotFoundError:
        print(f"Warning: Prior data not found at {PRIOR_DIR}")
        return None

def report_credible_interval(values: np.array, 
                             hdi_prob: float = 0.90,
                             verbose: bool = False) -> tuple:
    """Calculate credible intervals for given values."""
    med = np.median(values)
    low, high = arviz.hdi(values, hdi_prob=hdi_prob)
    
    low = med - low
    high = high - med
    
    if verbose:
        print(f"\n\n\n{med:.2f}-{low:.2f}+{high:.2f} (at {hdi_prob} HDI prob)\n\n\n")
    return low, med, high

def make_mass_radius_plot(data: dict, prior_data: dict, outdir: str):
    """Create mass-radius plot with posterior probability coloring."""
    print("Creating mass-radius plot...")
    
    plt.figure(figsize=(6, 12))
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
    print(f"Number of samples: {nb_samples}")
    
    # Normalize probabilities for coloring
    log_prob = np.exp(log_prob)
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = plt.get_cmap("viridis") # TODO: choose prettier colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    bad_counter = 0
    for i in tqdm.tqdm(range(len(log_prob))):
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
        normalized_value = norm(log_prob[i])
        color = cmap(normalized_value)
        
        plt.plot(r[i], m[i], 
                color=color, 
                alpha=1.0, 
                rasterized=True,
                zorder=1e10 + normalized_value)
        
    print(f"Excluded {bad_counter} invalid samples")
    
    # Styling
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")
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
    
    # Add Hauke's injection
    if ("ET" in outdir) or ("CE" in outdir):
        # This was an injection where we used Hauke's EOS
        data = np.loadtxt("../figures/EOS_data/hauke_macroscopic.dat")
        data = data.T
        r, m, l, _ = data
        
        plt.plot(r, m, color='black', lw=2.0, ls='--', label="Hauke's EOS")

    # Add legend for prior
    if prior_data is not None:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=COLORS_DICT['prior'], lw=2, alpha=0.7, label='Prior')]
        if ("ET" in outdir) or ("CE" in outdir):
            legend_elements.append(Line2D([0], [0], color='black', lw=2, ls='--', label="Hauke's EOS"))
        plt.legend(handles=legend_elements, loc='upper right')
        
    # Save figure (PDF only)
    save_name = os.path.join(outdir, "mass_radius_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"Mass-radius plot saved to {save_name}")

def make_eos_plot(data: dict, prior_data: dict, outdir: str):
    """Create equation of state plot (pressure vs density)."""
    print("Creating EOS plot...")
    
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
    log_prob = np.exp(log_prob)
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = sns.color_palette("crest", as_cmap=True)

    bad_counter = 0
    for i in tqdm.tqdm(range(len(log_prob))):
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
        normalized_value = norm(log_prob[i])
        color = cmap(normalized_value)
        
        mask = (n[i] > 0.5) * (n[i] < 6.0)
        plt.plot(n[i][mask], p[i][mask], 
                color=color, 
                alpha=1.0, 
                rasterized=True,
                zorder=1e10 + normalized_value)
        
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
    plt.yscale('log')
    plt.xlim(0.5, 6.0)
    
    # Add legend for prior
    if prior_data is not None:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=COLORS_DICT['prior'], lw=2, alpha=0.7, label='Prior')]
        plt.legend(handles=legend_elements, loc='upper left')

    # Save figure (PDF only)
    save_name = os.path.join(outdir, "eos_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"EOS plot saved to {save_name}")

def make_parameter_histograms(data: dict, prior_data: dict, outdir: str):
    """Create histograms for key EOS parameters."""
    print("Creating parameter histograms...")
    
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
    
    parameters = {
        'MTOV': {'values': MTOV_list, 'range': (1.75, 2.75), 'xlabel': r"$M_{\rm{TOV}}$ [$M_{\odot}$]"},
        'R14': {'values': R14_list, 'range': (10.0, 16.0), 'xlabel': r"$R_{1.4}$ [km]"},
        'p3nsat': {'values': p3nsat_list, 'range': (0.1, 200.0), 'xlabel': r"$p(3n_{\rm{sat}})$ [MeV fm$^{-3}$]"}
    }
    
    for param_name, param_data in parameters.items():
        plt.figure(figsize=figsize_horizontal)
        
        # Plot prior histogram if available
        if prior_data is not None and param_name in prior_params:
            kde_prior = gaussian_kde(prior_params[param_name])
            x = np.linspace(param_data['range'][0], param_data['range'][1], 1000)
            y_prior = kde_prior(x)
            plt.fill_between(x, y_prior, alpha=ALPHA, color=COLORS_DICT['prior'], label='Prior')
        
        # Create posterior KDE
        kde = gaussian_kde(param_data['values'])
        x = np.linspace(param_data['range'][0], param_data['range'][1], 1000)
        y = kde(x)
        
        plt.plot(x, y, color='blue', lw=3.0, label='Posterior')
        plt.fill_between(x, y, alpha=0.3, color='blue')
        
        if ("ET" in outdir) or ("CE" in outdir):
            if param_name == "R14":
                # This was an injection where we used Hauke's EOS
                data = np.loadtxt("../figures/EOS_data/hauke_macroscopic.dat")
                data = data.T
                r, m, l, _ = data
                plt.axvline(np.interp(1.4, m, r), color='black', lw=2.0, ls='--', label="Hauke's EOS")
            else:
                continue
            
        # Add credible interval information
        low, med, high = report_credible_interval(param_data['values'])
        
        plt.xlabel(param_data['xlabel'])
        plt.ylabel('Density')
        plt.xlim(param_data['range'])
        plt.ylim(bottom=0.0)
        plt.legend()
        plt.title(f'{param_name}: {med:.2f} -{low:.2f} +{high:.2f}')
        
        save_name = os.path.join(outdir, f"{param_name}_histogram.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        print(f"{param_name} histogram saved to {save_name}")

def make_contour_radii_plot(data: dict, prior_data: dict, outdir: str, m_min: float = 0.6, m_max: float = 2.1):
    """Create contour plot of radii vs mass."""
    print("Creating radii contour plot...")
    
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
    
    for i, mass_point in tqdm.tqdm(enumerate(masses_array)):
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
                      alpha=0.5, color='blue', label='Posterior')
    plt.plot(radii_low, masses_array, lw=2.0, color='blue')
    plt.plot(radii_high, masses_array, lw=2.0, color='blue')
    
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")
    plt.xlim(8.0, 16.0)
    plt.ylim(m_min, m_max)
    plt.legend()
    
    save_name = os.path.join(outdir, "radii_contour_plot.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"Radii contour plot saved to {save_name}")
    
def make_contour_pressures_plot(data: dict, outdir: str, n_min: float = 0.5, n_max: float = 6.0):
    """Create contour plot of pressures vs density."""
    print("Creating pressures contour plot...")
    
    n, p = data['densities'], data['pressures']
    
    plt.figure(figsize=figsize_horizontal)
    
    print("n_min")
    print(n_min)
    
    print("n_max")
    print(n_max)
    
    dens_array = np.linspace(n_min, n_max, 100)
    press_low = np.empty_like(dens_array)
    press_high = np.empty_like(dens_array)
    
    for i, dens in tqdm.tqdm(enumerate(dens_array)):
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
                     alpha=0.5, color='blue')
    plt.plot(dens_array, press_low, lw=2.0, color='blue')
    plt.plot(dens_array, press_high, lw=2.0, color='blue')
    
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
    plt.xlim(n_min, n_max)
    plt.yscale('log')
    plt.legend()
    
    save_name = os.path.join(outdir, "pressures_contour_plot.png")
    plt.savefig(save_name, bbox_inches="tight", dpi=300)
    plt.savefig(save_name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Pressures contour plot saved to {save_name}")

def generate_all_plots(outdir: str):
    """Generate all plots for the specified output directory."""
    print(f"Generating all plots for directory: {outdir}")
    
    # Load data
    try:
        data = load_eos_data(outdir)
        print("Data loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load prior data
    prior_data = load_prior_data()
    if prior_data is not None:
        print("Prior data loaded successfully!")
    
    # Create all plots with prior data
    make_contour_pressures_plot(data, outdir)
    make_parameter_histograms(data, prior_data, outdir)
    make_mass_radius_plot(data, prior_data, outdir)
    make_eos_plot(data, prior_data, outdir)
    make_contour_radii_plot(data, prior_data, outdir)
    
    print(f"All plots generated and saved to {outdir}")

def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(description='Generate EOS postprocessing plots')
    parser.add_argument('outdir', type=str, help='Output directory containing eos_samples.npz')
    parser.add_argument('--m_min', type=float, default=0.6, help='Minimum mass for contour plots')
    parser.add_argument('--m_max', type=float, default=2.5, help='Maximum mass for contour plots')
    parser.add_argument('--n_min', type=float, default=0.5, help='Minimum density for pressure contour plots')
    parser.add_argument('--n_max', type=float, default=6.0, help='Maximum density for pressure contour plots')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.outdir):
        print(f"Error: Output directory {args.outdir} does not exist")
        return
    
    # Generate all plots (this function handles the prior data loading internally)
    generate_all_plots(args.outdir)

if __name__ == "__main__":
    main()