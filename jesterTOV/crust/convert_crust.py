"""
Small example/utility file to convert existing crust files with other units to the appropriate file format used by jester
"""

import numpy as np
from jesterTOV import utils
from scipy.interpolate import interp1d


def convert_DH_crust(
    interpolate: bool = True, interpolation_ndat: int = 100, max_n_crust: float = 0.09
):
    """
    Convert the units in the DH_crust.dat file to npz format and correct units. The file is saved as DH.npz in the current directory.

    Args:
        interpolate (bool, optional): Whether to interpolate to smoothen. Defaults to True.
        interpolation_ndat (int, optional): Number of interpolation points. Defaults to 100.
        max_n_crust (float, optional): Maximal density of the crust. Defaults to 0.09.
    """
    crust_filename = "./DH_crust.dat"
    crust = np.genfromtxt(crust_filename).T
    _, n, e_g_cm_inv3, p_dyn_cm2 = crust  # e is in g/cm^3, p is in dyn/cm^2

    # Convert:
    e = e_g_cm_inv3 * utils.g_cm_inv3_to_MeV_fm_inv3
    p = p_dyn_cm2 * utils.dyn_cm2_to_MeV_fm_inv3

    # Interpolate the curves if desired
    if interpolate:
        n_interp = np.logspace(
            np.log10(n[0]), 
            np.log10(n[n < max_n_crust][-1]),
            interpolation_ndat
        )
        e = interp1d(n, e, kind="cubic", bounds_error=False)(n_interp)
        p = interp1d(n, p, kind="cubic", bounds_error=False)(n_interp)
        n = n_interp

    # Save:
    np.savez("./DH.npz", n=n, e=e, p=p)


if __name__ == "__main__":
    convert_DH_crust()
