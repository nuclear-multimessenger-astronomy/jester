r"""
Using lalsimulation Python bindings, generate EOS files for JESTER.

NOTE: This requires lalsuite dependency, and is not meant to be executed using
the base installation. Install with:

.. code-block:: bash

    uv pip install -e ".[dev]"

This script generates NPZ files containing EOS data in geometric units,
compatible with JESTER's injection_eos_path feature. Output files are written
to ``tabulated_eos/lalsuite/``.

Output format:

- masses_EOS: Solar masses :math:`M_{\odot}` (from LALSuite TOV solver)
- radii_EOS: :math:`\mathrm{km}` (from LALSuite TOV solver)
- Lambda_EOS: dimensionless tidal deformability (from LALSuite TOV solver)
- n: baryon number density in geometric units :math:`m^{-3}` (approximated as
  :math:`\varepsilon / m_\mathrm{avg}`)
- p: pressure in geometric units :math:`m^{-2}`
- e: energy density in geometric units :math:`m^{-2}`
- h: pseudo-enthalpy (dimensionless)
- dloge_dlogp: logarithmic derivative :math:`d\ln\varepsilon / d\ln p` using
  LALSuite's analytic derivative (same function used internally by the LALSuite
  TOV integrator — more accurate than finite differences)
- cs2: speed of sound squared :math:`c_s^2 = p / (\varepsilon \cdot d\ln\varepsilon/d\ln p)`
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

try:
    import lal  # type: ignore[import-untyped]
    import lalsimulation  # type: ignore[import-untyped]
except ImportError as _lal_import_error:
    raise ImportError(
        "lalsuite is required to run this script but is not installed.\n"
        "Install the JESTER dev dependencies:\n"
        '    uv pip install -e ".[dev]"\n'
        "Or install lalsuite directly:\n"
        "    uv pip install lalsuite"
    ) from _lal_import_error

# Physical constants from lal
C_SI = lal.C_SI  # m/s
G_SI = lal.G_SI  # m^3 kg^-1 s^-2
MSUN_SI = lal.MSUN_SI  # kg
MRSUN_SI = lal.MRSUN_SI  # m (solar mass in geometric units = G M_sun / c^2)

# Average nucleon mass for baryon density approximation (MeV)
_m_n = 939.5654205203889  # Neutron mass in MeV
_m_p = 938.2720881604904  # Proton mass in MeV
_m_avg_MeV = (_m_n + _m_p) / 2.0

# m_avg in geometric units (m): MeV → J → kg → m
_MeV_to_J = 1e6 * 1.602176634e-19
_m_avg_geom = (_m_avg_MeV * _MeV_to_J / C_SI**2) * G_SI / C_SI**2  # m

# Pressure sampling grid (SI units: Pa)
# LOG_P_MIN = 26: SLY/SLY4 and similar EOSs extend down to ~10^26 Pa. Using
# a higher cutoff misses the low-pressure crust and biases k2/Lambda.
LOG_P_MIN = 26  # 10^26 Pa — covers full crust for all known LALSuite EOSs
LOG_P_MAX = 37  # 10^37 Pa — well above MTOV central pressure
N_EOS_POINTS = 1000

# Central pressure range for M-R-Lambda family curves (SI units: Pa)
PC_MIN_SI = 1e32  # Pa — well below 0.5 M_sun, captures the full low-mass branch
PC_MAX_SI = 1.3e35  # Pa — above MTOV for most EOSs
N_FAMILY_POINTS = 500


def extract_eos_table(
    eos_name: str,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    r"""
    Extract EOS table from LALSuite for a given EOS name.

    Samples a pressure grid from :math:`10^{26}` to :math:`10^{37}` Pa and
    queries LALSuite for enthalpy, energy density, and the analytic energy
    density derivative at each valid pressure.

    Args:
        eos_name: LALSuite EOS name (e.g., ``"SLY"``, ``"APR4"``).

    Returns:
        Tuple of arrays ``(p, h, e, dloge_dlogp, cs2, n)`` in geometric units:

        - p: pressure :math:`m^{-2}`
        - h: pseudo-enthalpy (dimensionless)
        - e: energy density :math:`m^{-2}`
        - dloge_dlogp: :math:`d\ln\varepsilon / d\ln p` (dimensionless)
        - cs2: speed of sound squared (dimensionless)
        - n: baryon number density :math:`m^{-3}` (approximated as
          :math:`\varepsilon / m_\mathrm{avg}`)

    Raises:
        ValueError: If fewer than 100 valid pressure points are found.
    """
    eos = lalsimulation.SimNeutronStarEOSByName(eos_name)  # type: ignore[attr-defined]

    p_si_arr = 10.0 ** np.linspace(LOG_P_MIN, LOG_P_MAX, N_EOS_POINTS)
    # Convert SI pressure (Pa) to geometric units (m^-2): P_geom = P_SI * G / c^4
    p_geom_arr = p_si_arr * G_SI / C_SI**4

    h_list: list[float] = []
    e_list: list[float] = []
    dedp_list: list[float] = []
    p_valid: list[float] = []

    for p_geom in p_geom_arr:
        try:
            h = lalsimulation.SimNeutronStarEOSPseudoEnthalpyOfPressureGeometerized(  # type: ignore[attr-defined]
                p_geom, eos
            )
            e = lalsimulation.SimNeutronStarEOSEnergyDensityOfPressureGeometerized(  # type: ignore[attr-defined]
                p_geom, eos
            )
            # Analytic dε/dP — same function pointer used by the LALSuite TOV
            # integrator; more accurate than np.gradient finite differences.
            dedp = lalsimulation.SimNeutronStarEOSEnergyDensityDerivOfPressureGeometerized(  # type: ignore[attr-defined]
                p_geom, eos
            )
            h_list.append(h)
            e_list.append(e)
            dedp_list.append(dedp)
            p_valid.append(p_geom)
        except Exception:
            continue  # pressure outside EOS validity range — skip

    if len(p_valid) < 100:
        raise ValueError(
            f"Only {len(p_valid)} valid pressure points for {eos_name} — EOS may be invalid."
        )

    p_arr = np.array(p_valid)
    h_arr = np.array(h_list)
    e_arr = np.array(e_list)
    dedp_arr = np.array(dedp_list)

    # d(ln ε)/d(ln P) = (P/ε) * (dε/dP)
    dloge_dlogp_arr = (p_arr / e_arr) * dedp_arr

    # c_s^2 = dP/dε = P / (ε * d(ln ε)/d(ln P))
    cs2_arr = p_arr / (e_arr * dloge_dlogp_arr)

    # Baryon number density: n ≈ ε / m_avg  (rest-mass approximation)
    n_arr = e_arr / _m_avg_geom  # m^-3

    return p_arr, h_arr, e_arr, dloge_dlogp_arr, cs2_arr, n_arr


def compute_mr_lambda(
    eos_name: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    r"""
    Compute the M-R-:math:`\Lambda` family using the LALSuite TOV solver.

    Integrates the TOV equations over a logarithmic grid of central pressures
    from ``PC_MIN_SI`` to ``PC_MAX_SI`` and collects converged solutions.

    Args:
        eos_name: LALSuite EOS name (e.g., ``"SLY"``, ``"APR4"``).

    Returns:
        Tuple ``(masses_Msun, radii_km, lambdas)`` where:

        - masses_Msun: gravitational mass in :math:`M_\odot`
        - radii_km: circumferential radius in km
        - lambdas: dimensionless tidal deformability
          :math:`\Lambda = \frac{2}{3} k_2 \mathcal{C}^{-5}`
    """
    eos = lalsimulation.SimNeutronStarEOSByName(eos_name)

    pc_arr = np.logspace(np.log10(PC_MIN_SI), np.log10(PC_MAX_SI), N_FAMILY_POINTS)

    masses_Msun: list[float] = []
    radii_km: list[float] = []
    lambdas: list[float] = []

    for pc_si in pc_arr:
        try:
            r_m, m_kg, k2 = lalsimulation.SimNeutronStarTOVODEIntegrate(pc_si, eos)
            m_geom = m_kg * MRSUN_SI / MSUN_SI  # geometric mass (m)
            C = m_geom / r_m  # compactness
            lam = (2.0 / 3.0) * k2 / C**5
            masses_Msun.append(m_kg / MSUN_SI)
            radii_km.append(r_m / 1e3)
            lambdas.append(lam)
        except Exception:
            continue  # solver failed at this central pressure — skip

    masses = np.array(masses_Msun)
    radii = np.array(radii_km)
    lams = np.array(lambdas)

    # Truncate at maximum mass: keep only the stable branch up to M_TOV.
    # Central pressure increases monotonically along pc_arr, so mass increases
    # then turns over. Everything after the peak is the unstable branch.
    i_max = int(np.argmax(masses))
    return masses[: i_max + 1], radii[: i_max + 1], lams[: i_max + 1]


def generate_eos_file(eos_name: str, output_dir: str | Path | None = None) -> Path:
    """
    Generate NPZ file for a single EOS.

    Args:
        eos_name: LALSuite EOS name (e.g., ``"SLY"``, ``"APR4"``).
        output_dir: Directory to save the file. Defaults to
            ``tabulated_eos/lalsuite/`` relative to this script.

    Returns:
        Path to the generated NPZ file.
    """
    print(f"\n{'='*70}")
    print(f"Processing EOS: {eos_name}")
    print("=" * 70)

    # Extract EOS table
    print("Extracting EOS table from LALSuite...")
    p_geom, h, e_geom, dloge_dlogp, cs2, n_geom = extract_eos_table(eos_name)
    print(f"  ✓ {len(p_geom)} valid points")
    print(f"  Pressure range: {p_geom.min():.3e} to {p_geom.max():.3e} m^-2")
    print(f"  Energy density range: {e_geom.min():.3e} to {e_geom.max():.3e} m^-2")
    print(f"  cs2 range: {cs2.min():.3f} to {cs2.max():.3f}")

    # Compute M-R-Lambda family using LALSuite TOV solver
    print("Computing M-R-Lambda family with LALSuite TOV solver...")
    masses_Msun, radii_km, lambdas = compute_mr_lambda(eos_name)

    if len(masses_Msun) == 0:
        raise ValueError(f"No valid TOV solutions found for {eos_name}.")

    print(f"  ✓ {len(masses_Msun)} M-R-Lambda points")
    print(f"  Mass range: {masses_Msun.min():.3f} to {masses_Msun.max():.3f} M_sun")
    print(f"  Radius range: {radii_km.min():.3f} to {radii_km.max():.3f} km")
    print(f"  Lambda range: {lambdas.min():.1f} to {lambdas.max():.1f}")
    print(f"  Maximum mass: {masses_Msun.max():.3f} M_sun")

    # Save to NPZ file
    output_dir_path: Path
    if output_dir is None:
        output_dir_path = Path(__file__).parent / "lalsuite"
    else:
        output_dir_path = Path(output_dir)

    output_dir_path.mkdir(parents=True, exist_ok=True)
    filename = output_dir_path / f"{eos_name}.npz"

    np.savez(
        filename,
        # M-R-Lambda in physical units (from LALSuite TOV solver)
        masses_EOS=masses_Msun,  # M_sun
        radii_EOS=radii_km,  # km
        Lambda_EOS=lambdas,  # dimensionless
        # Thermodynamic quantities in geometric units (as expected by load_injection_eos)
        n=n_geom,  # m^-3
        p=p_geom,  # m^-2
        e=e_geom,  # m^-2
        h=h,  # dimensionless pseudo-enthalpy
        dloge_dlogp=dloge_dlogp,  # dimensionless (analytic, from LALSuite)
        cs2=cs2,  # dimensionless
        # Metadata
        eos_name=eos_name,
        source="LALSimulation",
    )

    print(f"\n✓ Saved to: {filename}")

    # Verify the saved file
    loaded = np.load(filename)
    print(f"  Keys: {list(loaded.keys())}")

    return filename


def list_available_eos() -> list[str]:
    """Return all EOS names loadable by the installed LALSuite version.

    Tries a comprehensive list of known LALSuite EOS names and returns those
    that can be successfully instantiated.
    """
    candidates = [
        "SLY",
        "SLY2",
        "SLY4",
        "SLY9",
        "SLY230A",
        "APR",
        "APR1",
        "APR2",
        "APR3",
        "APR4",
        "APR4_EPP",
        "MS1",
        "MS1B",
        "MS2",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "MPA1",
        "WFF1",
        "WFF2",
        "WFF3",
        "ENG",
        "ALF1",
        "ALF2",
        "ALF3",
        "ALF4",
        "GS1",
        "GS2",
        "BBB2",
        "BGN1H1",
        "BPAL12",
        "FPS",
        "GNH3",
        "KDE0V",
        "KDE0V1",
        "PCL2",
        "RS",
        "SK255",
        "SK272",
        "SKA",
        "SKB",
        "SKI2",
        "SKI3",
        "SKI4",
        "SKI5",
        "SKI6",
        "SKMP",
        "SKT1",
        "SKT2",
        "SKT3",
        "SKX",
        "SQM1",
        "SQM2",
        "SQM3",
        "SV",
        "UU",
        "DD2",
        "AP1",
        "AP2",
        "AP3",
        "AP4",
        "L",
    ]

    available: list[str] = []
    for name in sorted(set(candidates)):
        try:
            lalsimulation.SimNeutronStarEOSByName(name)
            available.append(name)
        except Exception:
            pass

    return sorted(available)


def main() -> None:
    """Generate NPZ files for a selection of commonly used EOSs."""
    print("=" * 70)
    print("LALSuite EOS to NPZ Converter")
    print("Generates injection-compatible NPZ files for JESTER")
    print("=" * 70)

    output_dir = Path(__file__).parent / "lalsuite"
    print(f"\nOutput directory: {output_dir}")

    # Selection of commonly used EOSs available in LALSuite
    eos_names = [
        "SLY",
        "SLY4",
        "SLY230A",
        "APR4_EPP",
        "MPA1",
        "H4",
        "MS1",
        "MS1B",
        "WFF1",
        "WFF2",
        "ENG",
        "HQC18",
    ]

    print(f"\nGenerating files for {len(eos_names)} selected EOSs...")

    generated_files: list[Path] = []
    failed_eos: list[str] = []

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


if __name__ == "__main__":
    main()
