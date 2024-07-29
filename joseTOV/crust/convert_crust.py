"""Small example/utility file to convert existing crust files with other units to the appropriate file format used by jose"""
import numpy as np
from joseTOV import utils
from scipy.interpolate import interp1d

def convert_DH_crust(interpolate: bool = True,
                     interpolation_ndat: int = 10_000):
    crust_filename = "./DH_crust.dat"
    crust = np.genfromtxt(crust_filename).T
    _, n, e_g_cm_inv3, p_dyn_cm2 = crust # e is in g/cm^3, p is in dyn/cm^2

    # Convert:
    e = e_g_cm_inv3 * utils.g_cm_inv3_to_MeV_fm_inv3
    p = p_dyn_cm2 * utils.dyn_cm2_to_MeV_fm_inv3
    
    # Interpolate the curves if desired
    if interpolate:
        n_interp = np.linspace(n[0], n[-1], interpolation_ndat)
        e = interp1d(n, e, kind = "cubic", fill_value = "extrapolate")(n_interp)
        p = interp1d(n, p, kind = "cubic", fill_value = "extrapolate")(n_interp)
        n = n_interp
    
    # Save:
    np.savez("./DH.npz", n=n, e=e, p=p)
    
if __name__ == "__main__":
    convert_DH_crust()