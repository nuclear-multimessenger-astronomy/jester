r"""
Using bilby functionalities, generate EOS files from lalsuite

NOTE: This requires bilby and lalsuite dependency, and is not meant to be executed using the base installation. Can install them in the uv venv with:
```bash
uv pip install -e ".[dev]"
```

This script generates NPZ files containing EOS data in geometric units, compatible with JESTER's injection_eos_path feature.

Output format:
- masses_EOS: Solar masses :math:`M_{\odot}` - converted from geometric
- radii_EOS: :math:`\mathrm{km}` - converted from geometric
- Lambda_EOS: dimensionless tidal deformability
- n: baryon number density in geometric units :math:`m^{-3}`
- p: pressure in geometric units :math:`m^{-2}`
- e: energy density in geometric units :math:`m^{-2}`
- cs2: speed of sound squared (dimensionless)
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from bilby.gw.eos import TabularEOS, EOSFamily  # type: ignore[import-untyped]

# Physical constants (from bilby)
from bilby.core import utils  # type: ignore[import-untyped]

C_SI = utils.speed_of_light  # m/s
G_SI = utils.gravitational_constant  # m^3 kg^-1 s^-2
MSUN_SI = utils.solar_mass  # kg

# Conversion from geometric to physical units
GEOM_TO_MSUN = (C_SI**2 / G_SI) / MSUN_SI  # Convert geometric mass (m) to solar masses
GEOM_TO_KM = 1e-3  # Convert geometric length (m) to km

# Average nucleon mass for density conversion (MeV)
m_n = 939.5654205203889  # Neutron mass in MeV
m_p = 938.2720881604904  # Proton mass in MeV
m_avg = (m_n + m_p) / 2.0

# Conversion factors (from jester utils)
hbarc = 197.3269804593025  # MeV⋅fm
fm_to_m = 1e-15
MeV_to_J = 1e6 * 1.602176634e-19
fm_inv3_to_SI = 1.0 / fm_to_m**3
MeV_fm_inv3_to_SI = MeV_to_J * fm_inv3_to_SI


def get_bilby_eos(name: str) -> tuple[TabularEOS, EOSFamily]:
    """
    Generates the bilby EOS objects for a given EOS name.

    Args:
        name (str): The name of the EOS.

    Returns:
        tuple[TabularEOS, EOSFamily]: The tabular EOS and EOS family objects.

    Raises:
        ValueError: If the EOS name is not recognized or objects cannot be created.
    """
    try:
        tabular_eos = TabularEOS(name)
        eos_family = EOSFamily(tabular_eos)
        return tabular_eos, eos_family
    except Exception as e:
        raise ValueError(f"Could not create bilby EOS objects for EOS name {name}: {e}")


def compute_density_and_cs2(
    pressure_geom: NDArray[np.floating],
    energy_density_geom: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    r"""
    Compute baryon number density and speed of sound squared from p and e.

    Args:
        pressure_geom: Pressure in geometric units :math:`m^{-2}`
        energy_density_geom: Energy density in geometric units :math:`m^{-2}`

    Returns:
        n_geom: Baryon number density in geometric units :math:`m^{-3}`
        cs2: Speed of sound squared (dimensionless)
    """
    # Speed of sound squared: cs^2 = dp/de
    # Use finite differences
    cs2 = np.gradient(pressure_geom, energy_density_geom)

    # Baryon number density from energy density
    # e = n * m_avg (approximately, in natural units where c=1)
    # Convert m_avg from MeV to geometric units

    # m_avg in MeV → m_avg in Joules → m_avg in kg → m_avg in geometric (m)
    m_avg_J = m_avg * MeV_to_J
    m_avg_kg = m_avg_J / C_SI**2
    m_avg_geom = m_avg_kg * G_SI / C_SI**2  # geometric mass (m)

    # n = e / m_avg (both in geometric units)
    n_geom = energy_density_geom / m_avg_geom

    return n_geom, cs2


def generate_eos_file(eos_name: str, output_dir: str | Path | None = None) -> Path:
    """
    Generate NPZ file for a single EOS.

    Args:
        eos_name: Name of the EOS (e.g., "SLY", "APR4")
        output_dir: Directory to save the file (default: current directory)

    Returns:
        Path to the generated NPZ file
    """
    print(f"\n{'='*70}")
    print(f"Processing EOS: {eos_name}")
    print("=" * 70)

    # Load EOS
    print("Loading EOS from bilby...")
    tabular_eos, eos_family = get_bilby_eos(eos_name)
    print("✓ EOS loaded successfully")

    # Extract EOS table data (all in geometric units)
    p_geom = tabular_eos.pressure  # m^-2
    e_geom = tabular_eos.energy_density  # m^-2
    h = tabular_eos.pseudo_enthalpy  # dimensionless

    print(f"  Pressure range: {p_geom.min():.3e} to {p_geom.max():.3e} m^-2")
    print(f"  Energy density range: {e_geom.min():.3e} to {e_geom.max():.3e} m^-2")
    print(f"  Number of EOS points: {len(p_geom)}")

    # Compute density and cs2
    print("Computing density and speed of sound...")
    n_geom, cs2 = compute_density_and_cs2(p_geom, e_geom)
    print(f"  Density range: {n_geom.min():.3e} to {n_geom.max():.3e} m^-3")
    print(f"  cs2 range: {cs2.min():.3f} to {cs2.max():.3f}")

    # Extract M-R-Lambda from TOV solver (all in geometric units)
    mass_geom = eos_family.mass  # m (geometric)
    radius_geom = eos_family.radius  # m (geometric)
    lambda_dimensionless = eos_family.tidal_deformability  # dimensionless

    print("\nTOV solver results:")
    print(f"  Number of M-R points: {len(mass_geom)}")

    # Convert mass and radius to standard units
    masses_Msun = mass_geom * GEOM_TO_MSUN  # Convert to solar masses
    radii_km = radius_geom * GEOM_TO_KM  # Convert to km

    print(f"  Mass range: {masses_Msun.min():.3f} to {masses_Msun.max():.3f} M_sun")
    print(f"  Radius range: {radii_km.min():.3f} to {radii_km.max():.3f} km")
    print(
        f"  Lambda range: {lambda_dimensionless.min():.1f} to {lambda_dimensionless.max():.1f}"
    )
    print(f"  Maximum mass: {eos_family.maximum_mass:.3f} M_sun")

    # Save to NPZ file
    output_dir_path: Path
    if output_dir is None:
        output_dir_path = Path(__file__).parent
    else:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    filename = output_dir_path / f"{eos_name}.npz"

    # Save in JESTER injection format (geometric units for thermodynamic quantities)
    np.savez(
        filename,
        # M-R-Lambda in physical units
        masses_EOS=masses_Msun,  # M_sun
        radii_EOS=radii_km,  # km
        Lambda_EOS=lambda_dimensionless,  # dimensionless
        # Thermodynamic quantities in geometric units (as expected by load_injection_eos)
        n=n_geom,  # m^-3
        p=p_geom,  # m^-2
        e=e_geom,  # m^-2
        cs2=cs2,  # dimensionless
        # Metadata
        eos_name=eos_name,
        source="LALSimulation via bilby",
    )

    print(f"\n✓ EOS data saved to: {filename}")

    # Verify we can load it back
    print("Verifying saved data...")
    loaded = np.load(filename)
    print(f"  Keys: {list(loaded.keys())}")
    print(f"  masses_EOS shape: {loaded['masses_EOS'].shape}")
    print(f"  n shape: {loaded['n'].shape}")

    return filename


def list_available_eos() -> list[str]:
    """List all available LAL EOS names."""
    from bilby.gw.eos.eos import valid_eos_names  # type: ignore[import-untyped]

    return sorted(valid_eos_names)


def main() -> None:
    """Generate NPZ files for a selection of commonly used EOSs."""

    print("=" * 70)
    print("LALSuite EOS to NPZ Converter")
    print("Generates injection-compatible NPZ files for JESTER")
    print("=" * 70)

    # Get output directory
    output_dir = Path(__file__).parent
    print(f"\nOutput directory: {output_dir}")

    # List of EOSs to generate (common ones used in NS physics)
    eos_names = [
        "SLY",
        "SLY230A",
        "MPA1",
        "H4",
        "MS1",
        "SLY4",
        "WFF1",
        "ENG",
        "HQC18",
    ]

    # Show all available EOSs
    print(f"\nAvailable EOSs in bilby: {len(list_available_eos())}")
    print(f"Generating files for {len(eos_names)} selected EOSs...")

    # Generate files
    generated_files = []
    failed_eos = []

    for eos_name in eos_names:
        try:
            filename = generate_eos_file(eos_name, output_dir)
            generated_files.append(filename)
        except Exception as e:
            print(f"\n✗ Failed to generate {eos_name}: {e}")
            failed_eos.append(eos_name)
            continue

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully generated: {len(generated_files)} files")
    print(f"Failed: {len(failed_eos)} EOSs")

    if generated_files:
        print("\nGenerated files:")
        for f in generated_files:
            print(f"  - {f}")

    if failed_eos:
        print("\nFailed EOSs:")
        for name in failed_eos:
            print(f"  - {name}")

    print("\n" + "=" * 70)
    print("To use these files in JESTER, add to your config.yaml:")
    print("=" * 70)
    print(
        """
postprocessing:
  enabled: true
  make_massradius: true
  make_pressuredensity: true
  injection_eos_path: "jesterTOV/tabulated_eos/SLY.npz"
    """
    )


if __name__ == "__main__":
    main()
