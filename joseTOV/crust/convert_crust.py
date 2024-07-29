"""Small example/utility file to convert existing crust files with other units to the appropriate file format used by jose"""
import numpy as np
from joseTOV import utils

def convert_DH_crust():
    crust_filename = "./DH_crust.dat"
    crust = np.genfromtxt(crust_filename).T
    _, n, e_g_cm_inv3, p_dyn_cm2 = crust # e is in g/cm^3, p is in dyn/cm^2

    # Convert:
    e = e_g_cm_inv3 * utils.g_cm_inv3_to_MeV_fm_inv3
    p = p_dyn_cm2 * utils.dyn_cm2_to_MeV_fm_inv3

    # Save:
    np.savez("./DH.npz", n=n, e=e, p=p)
    
if __name__ == "__main__":
    # convert_DH_crust()
    pass