r"""
Outer crust equation of state using Pearson et al. (2018) analytical fits.

This module is a standalone numpy port of the BSk outer crust routines from the
CUTER code (Oertel et al., LUTH-Caen Virgo group). It implements the analytical
fit functions from Pearson et al., Mon. Not. Roy. Astron. Soc. 481 (2018) 3, 2994
for the BSk22 and BSk24 nuclear energy density functionals.

The outer crust covers densities from ~10⁻¹¹ fm⁻³ up to the neutron drip point
(BSk24: 2.56×10⁻⁴ fm⁻³; BSk22: 2.69×10⁻⁴ fm⁻³). Above the drip point, the
inner crust requires a different treatment (Compressible Liquid Drop Model or
similar).

**Reference:** Pearson et al., MNRAS 481, 2994 (2018)
**Original CUTER code:** Oertel et al., MIT License, contact micaela.oertel@obspm.fr
"""

from __future__ import annotations

import math
import numpy as np

# Physical constants (CODATA 2018)
_C_SI = 2.99792458e8  # Speed of light [m/s]
_MEV_SI = 1.602176634e-13  # One MeV [J]
_MN_SI = 1.67492749804e-27  # Neutron mass [kg]
_MN_MEV = _MN_SI * _C_SI**2 / _MEV_SI  # Neutron mass [MeV] ≈ 939.565

# Conversion: energy density from (nb [fm^-3]) * (E/A [MeV]) → [g/cm^3]
# 1 MeV/fm^3 = mev_si / c^2 / fm^3 * (1e15 fm/m)^3 * (1e-3 g/kg) / (1e-2 m/cm)^3
_ECONVERT = _MEV_SI / (_C_SI**2) * 1e42  # ≈ 1.78266e12 g/cm^3 per MeV/fm^3


# ---------------------------------------------------------------------------
# Fit parameter tables (BSk22 and BSk24)
# ---------------------------------------------------------------------------

# Electron chemical potentials [MeV] at each nuclear shell boundary
_MU_ELECTRON: dict[int, list] = {
    22: [
        None,
        0.95,
        2.61,
        4.33,
        4.42,
        5.65,
        8.58,
        11.43,
        13.65,
        14.62,
        15.51,
        18.39,
        20.52,
        21.52,
        22.69,
        24.84,
        25.02,
        26.09,
        26.30,
    ],
    24: [
        None,
        0.95,
        2.61,
        4.33,
        4.42,
        5.65,
        8.58,
        11.43,
        14.36,
        17.57,
        18.18,
        21.13,
        22.75,
        22.93,
        24.14,
        25.69,
        26.14,
    ],
}

# Limiting densities [fm^-3] at each shell boundary
_NLIM: dict[int, list] = {
    22: [
        None,
        4.93e-9,
        1.63e-7,
        8.01e-7,
        8.79e-7,
        1.87e-6,
        6.83e-6,
        1.67e-5,
        2.97e-5,
        3.72e-5,
        4.55e-5,
        7.79e-5,
        1.12e-4,
        1.34e-4,
        1.59e-4,
        2.16e-4,
        2.25e-4,
        2.59e-4,
        2.69e-4,
    ],
    24: [
        None,
        4.93e-9,
        1.63e-7,
        8.01e-7,
        8.79e-7,
        1.87e-6,
        6.83e-6,
        1.67e-5,
        3.47e-5,
        6.63e-5,
        7.54e-5,
        1.22e-4,
        1.58e-4,
        1.64e-4,
        1.95e-4,
        2.39e-4,
        2.56e-4,
    ],
}

# Electron fractions (proton fractions) Z/A at each shell
_YP: dict[int, list] = {
    22: [
        None,
        26 / 56,
        28 / 62,
        28 / 64,
        28 / 66,
        36 / 86,
        34 / 84,
        32 / 82,
        30 / 80,
        28 / 76,
        28 / 78,
        28 / 80,
        42 / 124,
        40 / 122,
        39 / 121,
        38 / 122,
        38 / 124,
        38 / 126,
        38 / 128,
    ],
    24: [
        None,
        26 / 56,
        28 / 62,
        28 / 64,
        28 / 66,
        36 / 86,
        34 / 84,
        32 / 82,
        30 / 80,
        28 / 78,
        28 / 80,
        42 / 124,
        40 / 122,
        39 / 121,
        38 / 120,
        38 / 122,
        38 / 124,
    ],
}

# Energy fit constant [MeV]
_ECONST: dict[int, float] = {22: -9.1536, 24: -9.1536}

# Pressure fit constant [log₁₀(dyn/cm²)]
_PCONST: dict[int, float] = {22: -33.2047, 24: -33.2047}

# Energy fit parameters
_EPARA: dict[int, list] = {
    22: [
        None,
        7.02e8,
        1.133e11,
        6.19e7,
        4.54e6,
        5.46e5,
        15.24,
        0.0683,
        8.86,
        4611,
        48.07,
        2.697,
        81.7,
        7.05,
        1.50,
    ],
    24: [
        None,
        6.59e8,
        9.49e10,
        6.95e7,
        5.63e6,
        6.51e5,
        19.37,
        0.1028,
        4.09,
        6726,
        29.57,
        2.6728,
        19.51,
        4.39,
        1.75,
    ],
}

# Pressure fit parameters
_PPARA: dict[int, list] = {
    22: [
        None,
        6.682,
        5.651,
        0.00459,
        0.14359,
        2.681,
        11.972,
        13.993,
        1.2904,
        2.665,
        -27.787,
        2.0140,
        4.09,
        14.135,
        28.03,
        -1.921,
        1.08,
        14.89,
        0.098,
        11.67,
        4.75,
        -0.037,
        14.10,
        11.9,
    ],
    24: [
        None,
        6.795,
        5.552,
        0.00435,
        0.13963,
        3.636,
        11.943,
        13.848,
        1.3031,
        3.644,
        -30.840,
        2.2322,
        4.65,
        14.290,
        30.08,
        -2.080,
        1.100,
        14.71,
        0.099,
        11.66,
        5.00,
        -0.095,
        14.15,
        9.1,
    ],
}


def get_neutron_drip_density(bsk: int) -> float:
    """Return the neutron drip density (maximum outer crust density) for given BSk model.

    Parameters
    ----------
    bsk : int
        BSk model number (22 or 24)

    Returns
    -------
    float
        Neutron drip density [fm⁻³]
    """
    if bsk not in (22, 24):
        raise ValueError(f"BSk model must be 22 or 24, got {bsk}")
    n_shells = len(_NLIM[bsk]) - 1
    return _NLIM[bsk][n_shells]


def _pearson_fit(nb: float, bsk: int) -> tuple[float, float]:
    """Evaluate Pearson energy (C1) and pressure (C4) fit for outer crust.

    Parameters
    ----------
    nb : float
        Baryon number density [fm⁻³]
    bsk : int
        BSk model number

    Returns
    -------
    tuple[float, float]
        (pressure [MeV/fm³], energy_per_baryon [MeV])
    """
    ep = _EPARA[bsk]
    pp = _PPARA[bsk]

    # Energy per baryon (MeV) — Eq. C1 from Pearson et al. 2018
    w1 = 1.0 / (1.0 + ep[9] * nb)
    w2 = 1.0 / (1.0 + (ep[13] * nb) ** ep[14])
    f1 = (ep[1] * nb) ** (7.0 / 6.0) / (1.0 + math.sqrt(ep[2] * nb))
    f2 = (1.0 + math.sqrt(ep[4] * nb)) / (
        (1.0 + math.sqrt(ep[3] * nb)) * (1.0 + math.sqrt(ep[5] * nb))
    )
    f3 = ep[6] * nb ** ep[7] * (1.0 + ep[8] * nb)
    f4 = (ep[10] * nb) ** ep[11] / (1.0 + ep[12] * nb)
    eps = _ECONST[bsk] + f1 * f2 * w1 + f3 * w2 * (1.0 - w1) + f4 * (1.0 - w2)

    # Pressure — Eq. C4 from Pearson et al. 2018
    # Uses log₁₀(ρ [g/cm³]) as intermediate variable
    rho = nb * (eps + _MN_MEV) * _ECONVERT  # energy density → g/cm^3
    xi = math.log10(rho)

    p1 = (pp[1] + pp[2] * xi + pp[3] * xi**3) / (
        (1.0 + pp[4] * xi) * (1.0 + math.exp(pp[5] * (xi - pp[6])))
    )
    p2 = (pp[7] + pp[8] * xi) / (1.0 + math.exp(pp[9] * (pp[6] - xi)))
    p3 = (pp[10] + pp[11] * xi) / (1.0 + math.exp(pp[12] * (pp[13] - xi)))
    p4 = (pp[14] + pp[15] * xi) / (1.0 + math.exp(pp[16] * (pp[17] - xi)))
    p5 = pp[18] / (1.0 + (pp[20] * (xi - pp[19])) ** 2)
    p6 = pp[21] / (1.0 + (pp[23] * (xi - pp[22])) ** 2)
    log10_p_dyn_cm2 = _PCONST[bsk] + p1 + p2 + p3 + p4 + p5 + p6
    # Convert from dyn/cm² to MeV/fm³:
    # 1 dyn/cm² = 1e-1 Pa; 1 MeV/fm³ = MeV_to_J * fm_inv3_to_SI Pa
    # 1 dyn/cm² = 1e-1 / (1.602176634e-13 * 1e45) MeV/fm³ ≈ 6.242e-33 MeV/fm³
    _dyn_cm2_to_MeV_fm3 = 1e-1 / (_MEV_SI * 1e45)
    pressure = 10.0**log10_p_dyn_cm2 * _dyn_cm2_to_MeV_fm3

    return pressure, eps


def _get_mue_ye(nb: float, bsk: int) -> tuple[float, float]:
    """Return electron chemical potential and fraction at given density.

    Parameters
    ----------
    nb : float
        Baryon density [fm⁻³]
    bsk : int
        BSk model number

    Returns
    -------
    tuple[float, float]
        (mu_e [MeV], y_e)
    """
    n_shells = len(_NLIM[bsk]) - 1
    for ii in range(1, n_shells + 1):
        if nb <= _NLIM[bsk][ii]:
            return _MU_ELECTRON[bsk][ii], _YP[bsk][ii]
    # Clamp to last shell if exceeded
    return _MU_ELECTRON[bsk][n_shells], _YP[bsk][n_shells]


def build_outer_crust_table(
    nbmin: float,
    nbmax: float,
    ndim: int,
    bsk: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Build outer crust EOS table using BSk Pearson analytical fit.

    Computes the equation of state on a logarithmically spaced density grid
    from ``nbmin`` to ``nbmax`` using the Pearson et al. (2018) fit functions.
    If ``nbmax`` exceeds the neutron drip density for the chosen model, it is
    silently clamped to that value.

    Parameters
    ----------
    nbmin : float
        Minimum baryon number density [fm⁻³]
    nbmax : float
        Maximum baryon number density [fm⁻³] (clamped at neutron drip)
    ndim : int
        Number of grid points
    bsk : int
        BSk model number (22 or 24, default 24)

    Returns
    -------
    n_B : np.ndarray, shape (ndim,)
        Baryon number density [fm⁻³]
    p : np.ndarray, shape (ndim,)
        Pressure [MeV fm⁻³]
    e : np.ndarray, shape (ndim,)
        Energy density [MeV fm⁻³]
    cs2 : np.ndarray, shape (ndim,)
        Speed of sound squared :math:`c_s^2 = \partial P/\partial\varepsilon`
    mub : np.ndarray, shape (ndim,)
        Baryon chemical potential [MeV]

    Notes
    -----
    Speed of sound is computed numerically as :math:`c_s^2 = dP/d\varepsilon`
    using second-order finite differences. Causality is enforced by clipping
    to [0, 1].

    The outer crust EOS is largely independent of the nuclear empirical parameters
    (NEPs) that describe the core — it is dominated by experimentally constrained
    nuclear masses and Coulomb lattice energies. This makes it safe to use a fixed
    BSk table in the outer crust even when the core NEPs are sampled.
    """
    if bsk not in (22, 24):
        raise ValueError(f"BSk model must be 22 or 24, got {bsk}")

    nlimit = get_neutron_drip_density(bsk)
    if nbmax > nlimit:
        nbmax = nlimit

    n_B = np.exp(np.linspace(math.log(nbmin), math.log(nbmax), ndim))
    p = np.zeros(ndim)
    e = np.zeros(ndim)
    mub = np.zeros(ndim)

    for ii in range(ndim):
        nb = n_B[ii]
        pressure, eps = _pearson_fit(nb, bsk)
        energy_density = (eps + _MN_MEV) * nb  # MeV/fm³
        p[ii] = pressure
        e[ii] = energy_density
        mub[ii] = (pressure + energy_density) / nb  # MeV

    # Speed of sound: cs² = dP/dε (numerical gradient)
    cs2 = np.gradient(p, e)
    cs2 = np.clip(cs2, 0.0, 1.0)

    return n_B, p, e, cs2, mub
