"""
Cross-check between Jester and LALSuite TOV solvers using named EOSs.
Note: this assumes lalsuite is installed, but is not part of the Jester package dependencies. Run `pip install lalsuite` to install it.
"""

import matplotlib.pyplot as plt
params = {"text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"],
          "xtick.labelsize": 16,
          "ytick.labelsize": 16,
          "axes.labelsize": 16,
          "legend.fontsize": 16,
          "legend.title_fontsize": 16}
plt.rcParams.update(params)

import numpy as np
import jax.numpy as jnp
from jesterTOV.eos import construct_family
import jesterTOV.utils as utils
import lalsimulation as lalsim
import tempfile
import os

# LALSuite constants for unit conversion
M_SUN = 1.988409870698050731911960804878414216e30  # LAL_MSUN_SI
G = 6.67430e-11
c = 299792458e0

# Named EOSs from LALSuite (same as in example_lalsuite.py)
EOS_NAMES = ["HQC18", "SLY230A", "MPA1"]
COLORS = ["b", "g", "r"]

def extract_eos_table_from_lalsuite(eos_name):
    """Extract EOS table from LALSuite named EOS by accessing internal data directly.
    
    This function accesses the internal tabulated data arrays instead of sampling
    with pressure values, which avoids interpolation errors.
    
    Returns:
        tuple: (ns_geom, ps_geom, hs_geom, es_geom, dloge_dlogps) in geometric units
    """
    print(f"Extracting EOS table for {eos_name}...")
    
    # Create LALSuite EOS
    eos_lal = lalsim.SimNeutronStarEOSByName(eos_name)
    
    # Access the internal tabular data structure
    # Based on LALSimNeutronStarEOSTabular.c structure # Get pressure range from EOS object properties
       
    p_max_si = lalsim.SimNeutronStarEOSMaxPressure(eos_lal)  # This function should exist
    
    # Use a conservative range from ~10^-4 to full range of the EOS
    p_min_si = p_max_si * 1e-4  # Start from small fraction of max pressure
    n_points = 500  # Reasonable resolution
    
    pressures_si = np.logspace(np.log10(p_min_si), np.log10(p_max_si), n_points)
    
    # Extract thermodynamic quantities in SI units
    energy_densities_si = []
    rest_mass_densities_si = []
    pseudo_enthalpies_si = []
    valid_pressures = []
    
    for p_si in pressures_si:
        eps_si = lalsim.SimNeutronStarEOSEnergyDensityOfPressure(p_si, eos_lal)
        h_si = lalsim.SimNeutronStarEOSPseudoEnthalpyOfPressure(p_si, eos_lal)
        rho_si = lalsim.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(h_si, eos_lal)
        
        # Check for valid values
        if eps_si > 0 and rho_si > 0 and np.isfinite(eps_si) and np.isfinite(rho_si) and np.isfinite(h_si):
            energy_densities_si.append(eps_si)
            rest_mass_densities_si.append(rho_si)
            pseudo_enthalpies_si.append(h_si)
            valid_pressures.append(p_si)
    
    # Convert to arrays
    pressures_si = np.array(valid_pressures)
    energy_densities_si = np.array(energy_densities_si)
    rest_mass_densities_si = np.array(rest_mass_densities_si)
    pseudo_enthalpies_si = np.array(pseudo_enthalpies_si)
    
    if len(pressures_si) == 0:
        raise ValueError(f"No valid EOS points found for {eos_name}")
    
    # Convert SI to geometric units using jester's conversion factors
    # SI pressure (Pa) -> geometric
    ps_geom = pressures_si * utils.pressure_SI_to_geometric
    
    # SI energy density (J/m³) -> geometric  
    es_geom = energy_densities_si * utils.pressure_SI_to_geometric
    
    # SI rest mass density (kg/m³) -> number density in geometric units
    # First convert to number density: rho_si / m_nucleon
    m_nucleon_kg = utils.m_n * utils.MeV_to_J / utils.c**2  # nucleon mass in kg
    number_densities_si = rest_mass_densities_si / m_nucleon_kg  # particles/m³
    # Convert from particles/m³ to particles/fm³, then to geometric units
    ns_geom = number_densities_si * (utils.fm_to_m)**3 * utils.fm_inv3_to_geometric
    
    # Use LALSuite's pseudo-enthalpy directly - it's already dimensionless (geometric units)
    hs_geom = jnp.array(pseudo_enthalpies_si)
    
    # Calculate dloge/dlogp for adiabatic index
    log_es = jnp.log(es_geom)
    log_ps = jnp.log(ps_geom)
    dloge_dlogps = jnp.gradient(log_es, log_ps)
    
    return ns_geom, ps_geom, hs_geom, es_geom, dloge_dlogps

def solve_lalsuite_tov_for_eos(eos_name, ndat=200):
    """Solve TOV equations using LALSuite for a named EOS."""
    print(f"Solving TOV with LALSuite for {eos_name}...")
    
    # Create LALSuite EOS and family
    eos_lal = lalsim.SimNeutronStarEOSByName(eos_name)
    family = lalsim.CreateSimNeutronStarFamily(eos_lal)
    
    # Get maximum mass
    m_max = lalsim.SimNeutronStarMaximumMass(family)
    
    # Create mass array
    masses_si = np.linspace(0.5 * M_SUN, m_max, ndat)
    
    # Calculate properties
    radii = []
    k2_values = []
    
    for m in masses_si:
        radius = lalsim.SimNeutronStarRadius(m, family)
        k2 = lalsim.SimNeutronStarLoveNumberK2(m, family)
        radii.append(radius)
        k2_values.append(k2)
    
    radii = np.array(radii)
    k2_values = np.array(k2_values)
    valid_masses = masses_si[:len(radii)]
    
    # Calculate tidal deformability
    m_meter = valid_masses * G / c / c
    lambdas = 2. / 3. * k2_values * radii**5 / m_meter**5
    
    # Convert units
    masses_msun = valid_masses / M_SUN
    radii_km = radii / 1e3
    
    return masses_msun, radii_km, lambdas

def test_named_eos_crosscheck():
    """Cross-check jester vs LALSuite TOV solvers using named EOSs."""
    print("=== Named EOS Cross-check Test ===")
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    
    for i, (eos_name, color) in enumerate(zip(EOS_NAMES, COLORS)):
        print(f"\nProcessing {eos_name}...")
        
        # Extract EOS from LALSuite and convert to jester format
        ns_geom, ps_geom, hs_geom, es_geom, dloge_dlogps = extract_eos_table_from_lalsuite(eos_name)
        
        # Solve with jester
        eos_tuple = (ns_geom, ps_geom, hs_geom, es_geom, dloge_dlogps)
        logpc, masses_jester, radii_jester, Lambdas_jester = construct_family(
            eos_tuple, ndat=200, min_nsat=1.0
        )
        
        # Solve with LALSuite
        masses_lal, radii_lal, Lambdas_lal = solve_lalsuite_tov_for_eos(eos_name)
        
        # Filter masses for plotting
        m_min = 0.90
        mask_jester = masses_jester > m_min
        mask_lal = masses_lal > m_min
        
        # M-R plot
        plt.subplot(1, 2, 1)
        plt.plot(radii_jester[mask_jester], masses_jester[mask_jester], 
                color=color, linestyle='-', linewidth=2, label=f'{eos_name} (jester)')
        plt.plot(radii_lal[mask_lal], masses_lal[mask_lal], 
                color=color, linestyle='--', linewidth=2, label=f'{eos_name} (LAL)')
        
        # Lambda-M plot
        plt.subplot(1, 2, 2)
        plt.plot(masses_jester[mask_jester], Lambdas_jester[mask_jester], 
                color=color, linestyle='-', linewidth=2, label=f'{eos_name} (jester)')
        plt.plot(masses_lal[mask_lal], Lambdas_lal[mask_lal], 
                color=color, linestyle='--', linewidth=2, label=f'{eos_name} (LAL)')
        
        # Print comparison statistics
        print(f"\n{eos_name} Comparison:")
        print(f"  Jester: M_max = {np.max(masses_jester):.3f} M_sun")
        print(f"  LALSuite: M_max = {np.max(masses_lal):.3f} M_sun")
        
        # Find R_1.4 for both
        if len(masses_jester) > 0 and np.max(masses_jester) > 1.4:
            r_14_jester = radii_jester[np.argmin(np.abs(masses_jester - 1.4))]
        
        if len(masses_lal) > 0 and np.max(masses_lal) > 1.4:
            r_14_lal = radii_lal[np.argmin(np.abs(masses_lal - 1.4))]
        
        # Calculate max radius error in [1, 2] Msun range by interpolation
        mass_range = [1.0, 2.0]
        if (len(masses_jester) > 0 and len(masses_lal) > 0 and 
            np.max(masses_jester) >= mass_range[1] and np.max(masses_lal) >= mass_range[1]):
            
            # Create common mass grid in [1, 2] Msun range
            mass_grid = np.linspace(mass_range[0], mass_range[1], 100)
            
            # Interpolate radii and Lambdas for both solutions
            r_jester_interp = np.interp(mass_grid, masses_jester, radii_jester)
            r_lal_interp = np.interp(mass_grid, masses_lal, radii_lal)
            
            l_jester_interp = np.interp(mass_grid, masses_jester, Lambdas_jester)
            l_lal_interp = np.interp(mass_grid, masses_lal, Lambdas_lal)
            
            # Calculate absolute and relative errors
            abs_errors = np.abs(r_jester_interp - r_lal_interp)
            rel_errors = abs_errors / r_lal_interp * 100
            
            abs_errors_Lambdas = np.abs(l_jester_interp - l_lal_interp)
            rel_errors_Lambdas = abs_errors_Lambdas / l_lal_interp * 100
            
            max_abs_error = np.max(abs_errors)
            max_rel_error = np.max(rel_errors)
            
            max_abs_error_Lambdas = np.max(abs_errors_Lambdas)
            max_rel_error_Lambdas = np.max(rel_errors_Lambdas)
            
            print(f"  Max radius error in [1-2] Msun: {max_abs_error:.3f} km ({max_rel_error:.2f}%)")
            print(f"  Max Lambdas error in [1-2] Msun: {max_abs_error_Lambdas:.3f} ({max_rel_error_Lambdas:.2f}%)")
    
    # Format plots
    plt.subplot(1, 2, 1)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_\odot$]")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.xlabel(r"$M$ [$M_\odot$]")
    plt.ylabel(r"$\Lambda$")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./figures/named_eos_crosscheck.pdf", bbox_inches="tight")
    plt.close()

def test_custom_metamodel_eos():
    """Test using custom MetaModel EOS (original functionality)."""
    print("\n=== Custom MetaModel EOS Test ===")
    
    from jesterTOV.eos import MetaModel_with_CSE_EOS_model
    
    nsat = 0.16  # nuclear saturation density in fm^-3
    
    # Define the EOS object, here we focus on Metamodel with CSE
    eos = MetaModel_with_CSE_EOS_model(nmax_nsat=6.0)
    
    # Define the nuclear empirical parameters (NEPs) -- all in MeV
    NEP_dict = {"E_sat": -16.0,  # saturation parameters
                "K_sat": 200.0,
                "Q_sat": 0.0,
                "Z_sat": 0.0,
                "E_sym": 32.0,  # symmetry parameters
                "L_sym": 40.0,
                "K_sym": -100.0,
                "Q_sym": 0.0,
                "Z_sym": 0.0,
                }
    
    # Define the breakdown density -- this is usually between 1-2 nsat
    nbreak = 1.5 * nsat
    NEP_dict["nbreak"] = nbreak
    
    # Then we extend with some CSE grid points
    ngrids = jnp.array([2.0, 3.0, 4.0, 5.0]) * nsat
    cs2grids = jnp.array([0.5, 0.4, 0.3, 0.2])  # speed of sound squared at the grid points
    
    # Now create the EOS -- returns a tuple with most useful EOS quantities
    ns, ps, hs, es, dloge_dlogps, mu, cs2 = eos.construct_eos(NEP_dict, ngrids, cs2grids)
    
    # Solve TOV equations with Jester
    eos_tuple = (ns, ps, hs, es, dloge_dlogps)
    logpc, masses, radii, Lambdas = construct_family(eos_tuple, ndat=200, min_nsat=1.0)
    
    # Create LALSuite EOS from the custom table
    def create_lalsuite_eos_from_tables(ns_geom, ps_geom, es_geom):
        """Create LALSuite EOS from tabulated data in geometric units."""
        print(f"Input data points: {len(ns_geom)}")
        print(f"Input pressure range (geometric): {np.min(ps_geom):.2e} to {np.max(ps_geom):.2e}")
        print(f"Input energy range (geometric): {np.min(es_geom):.2e} to {np.max(es_geom):.2e}")
        
        # Convert from jester's geometric units back to MeV/fm³ first
        # This follows the same pattern as in CSE model (lines 900-903 in eos.py)
        p_MeV_fm3 = ps_geom / utils.MeV_fm_inv3_to_geometric  # geometric -> MeV/fm³
        eps_MeV_fm3 = es_geom / utils.MeV_fm_inv3_to_geometric  # geometric -> MeV/fm³
        
        print(f"Converted to MeV/fm³ - Pressure range: {np.min(p_MeV_fm3):.2e} to {np.max(p_MeV_fm3):.2e}")
        print(f"Converted to MeV/fm³ - Energy range: {np.min(eps_MeV_fm3):.2e} to {np.max(eps_MeV_fm3):.2e}")
        
        # Convert MeV/fm³ to CGS units expected by LALSuite
        # Using the standard nuclear physics conversion factors
        MeV_fm3_to_dyn_cm2 = utils.MeV_to_J * 1e7 / (utils.fm_to_m * 100)**3  # MeV/fm³ to dyn/cm²
        MeV_fm3_to_erg_cm3 = utils.MeV_to_J * 1e7 / (utils.fm_to_m * 100)**3  # MeV/fm³ to erg/cm³
        
        p_cgs = p_MeV_fm3 * MeV_fm3_to_dyn_cm2
        eps_cgs = eps_MeV_fm3 * MeV_fm3_to_erg_cm3
        
        print(f"Pressure range: {np.min(p_cgs):.2e} to {np.max(p_cgs):.2e} dyn/cm^2")
        print(f"Energy density range: {np.min(eps_cgs):.2e} to {np.max(eps_cgs):.2e} erg/cm^3")
        
        # LALSuite requires: pressure vs energy density in MONOTONIC order
        # Key insight: LALSuite interpolates ε(P), so P must be monotonic
        
        # Remove any invalid values
        valid_mask = (p_cgs > 0) & (eps_cgs > 0) & np.isfinite(p_cgs) & np.isfinite(eps_cgs)
        p_clean = p_cgs[valid_mask]
        eps_clean = eps_cgs[valid_mask]
        
        # Sort by pressure (required for LALSuite)
        sort_idx = np.argsort(p_clean)
        p_sorted = p_clean[sort_idx]
        eps_sorted = eps_clean[sort_idx]
        
        # Remove duplicate pressure values (LALSuite fails on duplicates)
        _, unique_idx = np.unique(p_sorted, return_index=True)
        p_final = p_sorted[unique_idx]
        eps_final = eps_sorted[unique_idx]
        
        # Ensure causality: ε should generally increase with P
        # If not monotonic, this indicates EOS issues, but let's proceed
        deps_dp = np.diff(eps_final) / np.diff(p_final)
        causal_points = np.sum(deps_dp > 0)
        print(f"Causal points: {causal_points}/{len(deps_dp)} ({100*causal_points/len(deps_dp):.1f}%)")
        
        print(f"Final data: {len(p_final)} points")
        print(f"Pressure range: {np.min(p_final):.2e} to {np.max(p_final):.2e} dyn/cm²")
        print(f"Energy range: {np.min(eps_final):.2e} to {np.max(eps_final):.2e} erg/cm³")
        
        # Create temporary file with 2-column format (p, eps)
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat')
        
        for i in range(len(p_final)):
            temp_file.write(f"{p_final[i]:.12e} {eps_final[i]:.12e}\n")
        
        temp_file.close()
        
        # Create LALSuite EOS object
        eos_lal = lalsim.SimNeutronStarEOSFromFile(temp_file.name)
        print("Successfully created LALSuite EOS object")
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        return eos_lal
    
    def solve_lalsuite_tov_family(eos_lal, ndat=200):
        """Solve TOV equations using LALSuite for a family of stars."""
        family = lalsim.CreateSimNeutronStarFamily(eos_lal)
        m_max = lalsim.SimNeutronStarMaximumMass(family)
        masses_si = np.linspace(0.5 * M_SUN, m_max, ndat)
        
        radii = []
        k2_values = []
        
        for m in masses_si:
            radius = lalsim.SimNeutronStarRadius(m, family)
            k2 = lalsim.SimNeutronStarLoveNumberK2(m, family)
            radii.append(radius)
            k2_values.append(k2)
        
        radii = np.array(radii)
        k2_values = np.array(k2_values)
        valid_masses = masses_si[:len(radii)]
        
        # Calculate tidal deformability
        m_meter = valid_masses * G / c / c
        lambdas = 2. / 3. * k2_values * radii**5 / m_meter**5
        
        # Convert units
        masses_msun = valid_masses / M_SUN
        radii_km = radii / 1e3
        
        return masses_msun, radii_km, lambdas
    
    # Solve with LALSuite
    print("Creating LALSuite EOS...")
    eos_lal = create_lalsuite_eos_from_tables(ns, ps, es)
    
    print("Solving TOV equations with LALSuite...")
    masses_lal, radii_lal, lambdas_lal = solve_lalsuite_tov_family(eos_lal)
    
    # Make comparison plot
    plt.figure(figsize=(12, 6))
    
    # Limit masses to be above certain mass to make plot prettier
    m_min = 0.5
    mask = masses > m_min
    masses = masses[mask]
    radii = radii[mask]
    Lambdas = Lambdas[mask]
    
    # M(R) plot
    plt.subplot(121)
    plt.plot(radii, masses, 'b-', label='jester', linewidth=2)
    plt.plot(radii_lal, masses_lal, 'r--', label='LALSuite', linewidth=2)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_\odot$]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lambda(M) plot
    plt.subplot(122)
    plt.plot(masses, Lambdas, 'b-', label='jester', linewidth=2)
    plt.plot(masses_lal, lambdas_lal, 'r--', label='LALSuite', linewidth=2)
    plt.xlabel(r"$M$ [$M_\odot$]")
    plt.ylabel(r"$\Lambda$")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./figures/jester_vs_lalsuite_custom.pdf", bbox_inches="tight")
    plt.show()
    plt.close()
    
    # Print comparison statistics
    print(f"\nCustom MetaModel Comparison Summary:")
    print(f"Jester: M_max = {np.max(masses):.3f} M_sun")
    print(f"LALSuite: M_max = {np.max(masses_lal):.3f} M_sun")
    
    # Find R_1.4 for both
    if len(masses) > 0 and np.max(masses) > 1.4:
        r_14_jester = radii[np.argmin(np.abs(masses - 1.4))]
        print(f"Jester: R_1.4 = {r_14_jester:.2f} km")
    
    if len(masses_lal) > 0 and np.max(masses_lal) > 1.4:
        r_14_lal = radii_lal[np.argmin(np.abs(masses_lal - 1.4))]
        print(f"LALSuite: R_1.4 = {r_14_lal:.2f} km")
    
    # Calculate max radius and Lambda errors in [1, 2] Msun range
    mass_range = [1.0, 2.0]
    if (len(masses) > 0 and len(masses_lal) > 0 and 
        np.max(masses) >= mass_range[1] and np.max(masses_lal) >= mass_range[1]):
        
        # Create common mass grid in [1, 2] Msun range
        mass_grid = np.linspace(mass_range[0], mass_range[1], 100)
        
        # Interpolate radii and Lambdas for both solutions
        r_jester_interp = np.interp(mass_grid, masses, radii)
        r_lal_interp = np.interp(mass_grid, masses_lal, radii_lal)
        
        l_jester_interp = np.interp(mass_grid, masses, Lambdas)
        l_lal_interp = np.interp(mass_grid, masses_lal, lambdas_lal)
        
        # Calculate absolute and relative errors
        abs_errors = np.abs(r_jester_interp - r_lal_interp)
        rel_errors = abs_errors / r_lal_interp * 100
        
        abs_errors_Lambdas = np.abs(l_jester_interp - l_lal_interp)
        rel_errors_Lambdas = abs_errors_Lambdas / l_lal_interp * 100
        
        max_abs_error = np.max(abs_errors)
        max_rel_error = np.max(rel_errors)
        
        max_abs_error_Lambdas = np.max(abs_errors_Lambdas)
        max_rel_error_Lambdas = np.max(rel_errors_Lambdas)
        
        print(f"Max radius error in [1-2] Msun: {max_abs_error:.3f} km ({max_rel_error:.2f}%)")
        print(f"Max Lambdas error in [1-2] Msun: {max_abs_error_Lambdas:.3f} ({max_rel_error_Lambdas:.2f}%)")

if __name__ == "__main__":
    # # Run the named EOS cross-check test
    # test_named_eos_crosscheck()
    
    # Run the custom MetaModel EOS test - Jester EOS -> LALSuite comparison
    test_custom_metamodel_eos()