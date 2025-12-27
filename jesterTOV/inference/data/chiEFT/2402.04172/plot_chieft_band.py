#!/usr/bin/env python
"""
Script to load and visualize chiEFT pressure-density uncertainty bands.

Data from: Koehn et al., "From existing and new nuclear and astrophysical
constraints to stringent limits on the equation of state of neutron-rich
dense matter", Phys.Rev.X 15 (2025) 2, 021014

The data files (low.dat and high.dat) contain the lower and upper bounds
of the chiEFT uncertainty band.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
data_dir = Path(__file__).parent
low_file = data_dir / "low.dat"
high_file = data_dir / "high.dat"

# Load data
print("Loading chiEFT data...")
print(f"  Low bound: {low_file}")
print(f"  High bound: {high_file}")

low_data = np.loadtxt(low_file)
high_data = np.loadtxt(high_file)

print(f"\nData shape: {low_data.shape}")
print(f"Number of points: {len(low_data)}")
print(f"Number of columns: {low_data.shape[1]}")

# Explore the data structure
print("\n" + "="*60)
print("DATA STRUCTURE EXPLORATION")
print("="*60)

print("\nFirst 5 rows of low.dat:")
print("  n (fm^-3)        Col1             Col2")
for i in range(5):
    print(f"  {low_data[i, 0]:.6e}  {low_data[i, 1]:.6e}  {low_data[i, 2]:.6e}")

print("\nLast 5 rows of low.dat:")
print("  n (fm^-3)        Col1             Col2")
for i in range(-5, 0):
    print(f"  {low_data[i, 0]:.6f}  {low_data[i, 1]:.6f}  {low_data[i, 2]:.6f}")

print("\nLast 5 rows of high.dat:")
print("  n (fm^-3)        Col1             Col2")
for i in range(-5, 0):
    print(f"  {high_data[i, 0]:.6f}  {high_data[i, 1]:.6f}  {high_data[i, 2]:.6f}")

# Extract columns
# Based on chieft.py code analysis:
# - Column 0: density in fm^-3
# - Column 1: pressure (used in chieft.py as f[:, 1])
# - Column 2: likely energy density (not used in chieft.py)
n_low = low_data[:, 0]  # fm^-3
p_low = low_data[:, 1]  # Pressure (MeV/fm^3 presumably)

n_high = high_data[:, 0]  # fm^-3
p_high = high_data[:, 1]  # Pressure (MeV/fm^3 presumably)

# Convert density to nsat units (nsat = 0.16 fm^-3)
nsat = 0.16  # fm^-3
n_low_nsat = n_low / nsat
n_high_nsat = n_high / nsat

print("\n" + "="*60)
print("DENSITY AND PRESSURE RANGES")
print("="*60)
print(f"\nDensity range:")
print(f"  Min: {n_low.min():.6e} fm^-3 ({n_low_nsat.min():.6f} nsat)")
print(f"  Max: {n_low.max():.6f} fm^-3 ({n_low_nsat.max():.6f} nsat)")

print(f"\nPressure range (low bound):")
print(f"  Min: {p_low.min():.6e}")
print(f"  Max: {p_low.max():.6f}")

print(f"\nPressure range (high bound):")
print(f"  Min: {p_high.min():.6e}")
print(f"  Max: {p_high.max():.6f}")

# Find pressure at nuclear saturation density
idx_sat = np.argmin(np.abs(n_low - nsat))
print(f"\nAt nuclear saturation density (n = {nsat} fm^-3):")
print(f"  Low bound pressure: {p_low[idx_sat]:.6f}")
print(f"  High bound pressure: {p_high[idx_sat]:.6f}")

# Create visualization
print("\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot: Pressure vs density in fm^-3 units
ax.fill_between(n_low, p_low, p_high, alpha=0.3, color='steelblue', label='chiEFT band')
ax.plot(n_low, p_low, 'b-', linewidth=1.5, label='Lower bound')
ax.plot(n_high, p_high, 'r-', linewidth=1.5, label='Upper bound')
ax.axvline(nsat, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'n_sat = {nsat} fm⁻³')
ax.set_xlabel('Density n [fm⁻³]', fontsize=12)
ax.set_ylabel('Pressure P [MeV/fm³]', fontsize=12)
ax.set_title('chiEFT Pressure-Density Band (Koehn et al. 2024)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, n_low.max() * 1.05)
ax.set_ylim(0, max(p_high.max(), p_low.max()) * 1.05)

plt.tight_layout()

# Save figure
output_file = data_dir / "chieft_pressure_density_band.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

print("\n" + "="*60)
print("DONE")
print("="*60)
