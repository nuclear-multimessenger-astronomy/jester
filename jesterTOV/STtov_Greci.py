r"""
Post-TOV (modified TOV) equation solver in the scalar tensor theory.

This module modify the standard TOV equations to calculate stellar structure solution in the scalar tensor theory. 
# TODO: Explain methods

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Reference:** Greci et al Phys.Rev.D 111 (2025) 8, 089901 (erratum)
"""

from . import utils
import jax
from jax import lax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, Event, Dopri8

    
def tov_ode_iter(h, y, eos):
    r"""
    """
    # EOS quantities
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]

    # scalar-tensor parameters
    beta_ST = eos["beta_ST"]

    r, m, nu, psi, phi = y

    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)

    #scalar coupling function
    A_phi = jnp.exp(0.5 * beta_ST * jnp.power(phi, 2))
    alpha_phi = beta_ST * phi 

    # Regularization parameter
    EPS = 1e-25
    
    # Modified dpdr to avoid division by zero
    dpdr = -(e + p) * (
        (m + 4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 3) * p) 
        / (r * (r - 2.0 * m + EPS))  # Regularize denominator
        + 0.5 * r * jnp.power(psi, 2) 
        + alpha_phi * psi
    )
    
    # Safe division for drdh (handles dpdr ≈ 0)
    safe_dpdr = jnp.where(
        jnp.abs(dpdr) < EPS, 
        jnp.copysign(EPS, dpdr),  # Preserve sign
        dpdr
    )
    drdh = (e + p) / safe_dpdr  # Numerically stable division
    
    # Remaining equations with regularized denominators
    dmdh = (
        4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 2) * e 
        + 0.5 * r * (r - 2.0 * m) * jnp.power(psi, 2)
    ) * drdh
    
    dnudh = (
        2 * (m + 4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 3) * p) 
        / (r * (r - 2.0 * m + EPS))  # Regularized
        + r * jnp.power(psi, 2)
    ) * drdh
    
    dpsidh = (
        (
            4.0 * jnp.pi * jnp.power(A_phi, 4) * r 
            / (r - 2.0 * m + EPS)  # Regularized
            * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi)
        )
        - (
            2.0 * (r - m) 
            / (r * (r - 2.0 * m + EPS))  # Regularized
            * psi
        )
    ) * drdh

    dphidh = psi * drdh
    return drdh, dmdh, dnudh, dpsidh, dphidh

def tov_ode_iter_tidal(h, y, eos):
    r"""
    """
    # EOS quantities
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    beta_ST = eos["beta_ST"]  # scalar-tensor parameter

    r, m, nu, psi, phi, H0, H0_prime, delta_phi, delta_phi_prime = y
    EPS = 1e-25  # small value to avoid zero division error
    
    # Interpolate EOS
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)
    
    # Scalar field terms
    A_phi = jnp.exp(0.5 * beta_ST * jnp.power(phi, 2))
    # Note that there is a alpha_phi definition difference between Brown (2023) and Greci (2023)
    # Here we use Brown (2023) definition for TOV solver, and use alpha_phi(Greci) = - alpha_phi(Brown) for tidal deformability calcs
    alpha_phi = beta_ST * phi 
    A_phi4 = jnp.power(A_phi, 4)
    four_pi_Aphi4 = 4.0 * jnp.pi * A_phi4
    r2 = r * r
    r3 = r2 * r
    
    # Core equations -----------------------------------------------------------
    # dpdr with regularization
    denom_non_tidal = r - 2.0 * m + EPS
    dpdr = -(e + p) * (
        (m + four_pi_Aphi4 * r3 * p) / (r * denom_non_tidal) 
        + 0.5 * r * jnp.power(psi, 2) 
        + alpha_phi * psi
    )
    
    # Safe division for drdh
    safe_dpdr = jnp.where(jnp.abs(dpdr) < EPS, jnp.copysign(EPS, dpdr), dpdr)
    drdh = (e + p) / safe_dpdr
    
    # Remaining derivatives
    dmdh = (
        four_pi_Aphi4 * r2 * e 
        + 0.5 * r * (r - 2.0 * m) * jnp.power(psi, 2)
    ) * drdh
    
    dnudh = (
        2 * (m + four_pi_Aphi4 * r3 * p) / (r * denom_non_tidal) 
        + r * jnp.power(psi, 2)
    ) * drdh
    
    dpsidh = (
        four_pi_Aphi4 * r / denom_non_tidal * (
            alpha_phi * (e - 3.0 * p) 
            + r * (e - p) * psi
        )
        - 2.0 * (r - m) / (r * denom_non_tidal) * psi
    ) * drdh

    dphidh = psi * drdh

    # TOV should be exactly same with Brown (2023) and Pani (2014) paper
    # Tidal deformabilities (ℓ=2) ----------------------------------------------
    comp = m / r
    denom_pert = r - 2.0 * m + EPS
    
    # Coefficients for H0 equation
    F1 = (4.0 * jnp.pi * jnp.power(r, 3) * A_phi4 * (p - e) + 2.0 * (r - m)) / (r * denom_pert)
    
    F0_num = (
        4.0 * jnp.pi * jnp.power(r, 3) * p * A_phi4 * (r * (dedp + 9.0) - 2.0 * m * (dedp + 13.0))
        + 4.0 * jnp.pi * jnp.power(r, 3) * e * A_phi4 * (dedp + 5.0) * (r - 2.0 * m)
        - 4.0 * jnp.power(r, 2) * (r - 2.0 * m) * jnp.power(psi, 2) * (4.0 * jnp.pi * jnp.power(r, 3) * p * A_phi4 + m)
        - 64.0 * jnp.power(jnp.pi, 2) * jnp.power(r, 6) * jnp.power(p, 2) * jnp.power(A_phi4, 2)
        - 6.0 * r * (r - 2.0 * m)  # ℓ(ℓ+1) = 6 for ℓ=2
        - jnp.power(r, 4) * jnp.power(r - 2.0 * m, 2) * jnp.power(psi, 4)
        - 4.0 * jnp.power(m, 2)
    )
    F0 = F0_num / (jnp.power(r, 2) * jnp.power(r - 2.0 * m, 2))
    
    Fs_num = (
        4.0 * jnp.power(r, 2) * (
            2.0 * jnp.pi * A_phi4 * (
                -alpha_phi * ((dedp - 9.0) * p + (dedp - 1.0) * e) # changed alpha-phi definition to follow Greci et al (2023)
                + 4.0 * r * p * psi
            )
            + (r - 2.0 * m) * jnp.power(psi, 3)
        )
        + 8.0 * m * psi
    )
    Fs = Fs_num / (r * (r - 2.0 * m))
    
    # Coefficients for dphi equation  
    G1 = F1  # Same as F1
    G0 = (
        4.0 * jnp.pi * r * A_phi4 / (r - 2.0 * m) * (
            jnp.power(alpha_phi, 2) * ((dedp + 9.0) * p + (dedp - 7.0) * e) 
            + (e - 3.0 * p) * (-beta_ST)  # α' = - beta for DEF model, Greci et al notation
        )
        - 6.0 / (r * (r - 2.0 * m))  # ℓ(ℓ+1) = 6 for ℓ=2
        - 4.0 * jnp.power(psi, 2)
    )
    Gs = Fs / 4.0  # As defined in paper
    
    # Perturbation derivatives
    dH0dh = H0_prime * drdh
    dH0_primedh = (-F1 * H0_prime - F0 * H0 + Fs * delta_phi) * drdh
    ddelta_phidh = delta_phi_prime * drdh
    ddelta_phi_primedh = (-G1 * delta_phi_prime - G0 * delta_phi + Gs * H0) * drdh

    return drdh, dmdh, dnudh, dpsidh, dphidh, dH0dh, dH0_primedh, ddelta_phidh, ddelta_phi_primedh

# Infinity expansion for boundary conditions:
# From Greci et al Jupyter notebook https://community.wolfram.com/groups/-/m/t/3459453
# Used as exterior basis to match conditions

def H0OnlyQT_jax(M, q, r):
    return 29491200.0 * jnp.power(M, 19.0) / (23.0 * jnp.power(r, 22.0)) - 159004594585600.0 * jnp.power(M, 19.0) * jnp.power(q, 2.0) / (22309287.0 * jnp.power(r, 22.0)) + 6.230429751360435e+16 * jnp.power(M, 19.0) * jnp.power(q, 4.0) / (4216455243.0 * jnp.power(r, 22.0)) - 3.438074420771897e+17 * jnp.power(M, 19.0) * jnp.power(q, 6.0) / (23301463185.0 * jnp.power(r, 22.0)) + 2.5485251228132252e+20 * jnp.power(M, 19.0) * jnp.power(q, 8.0) / (33204585038625.0 * jnp.power(r, 22.0)) - 2.0993833349937666e+20 * jnp.power(M, 19.0) * jnp.power(q, 10.0) / (99613755115875.0 * jnp.power(r, 22.0)) + 7.558555508042696e+17 * jnp.power(M, 19.0) * jnp.power(q, 12.0) / (2554198849125.0 * jnp.power(r, 22.0)) - 2933799166774592.0 * jnp.power(M, 19.0) * jnp.power(q, 14.0) / (150246991125.0 * jnp.power(r, 22.0)) + 14934094118912.0 * jnp.power(M, 19.0) * jnp.power(q, 16.0) / (29515186701.0 * jnp.power(r, 22.0)) - 24576000.0 * jnp.power(M, 19.0) * jnp.power(q, 18.0) / (7436429.0 * jnp.power(r, 22.0)) + 161873920.0 * jnp.power(M, 18.0) / (253.0 * jnp.power(r, 21.0)) - 268178337056768.0 * jnp.power(M, 18.0) * jnp.power(q, 2.0) / (81800719.0 * jnp.power(r, 21.0)) + 2.274123155544045e+16 * jnp.power(M, 18.0) * jnp.power(q, 4.0) / (3681032355.0 * jnp.power(r, 21.0)) - 4822171191567592.0 * jnp.power(M, 18.0) * jnp.power(q, 6.0) / (876436275.0 * jnp.power(r, 21.0)) + 1.4542063357777052e+16 * jnp.power(M, 18.0) * jnp.power(q, 8.0) / (5842908500.0 * jnp.power(r, 21.0)) - 645419377518649.0 * jnp.power(M, 18.0) * jnp.power(q, 10.0) / (1124838000.0 * jnp.power(r, 21.0)) + 3.60184371231355e+16 * jnp.power(M, 18.0) * jnp.power(q, 12.0) / (560919216000.0 * jnp.power(r, 21.0)) - 2316422291259147.0 * jnp.power(M, 18.0) * jnp.power(q, 14.0) / (747892288000.0 * jnp.power(r, 21.0)) + 45672613590959.0 * jnp.power(M, 18.0) * jnp.power(q, 16.0) / (927846678528.0 * jnp.power(r, 21.0)) - 1154725.0 * jnp.power(M, 18.0) * jnp.power(q, 18.0) / (10551296.0 * jnp.power(r, 21.0)) + 24576000.0 * jnp.power(M, 17.0) / (77.0 * jnp.power(r, 20.0)) - 112034631397376.0 * jnp.power(M, 17.0) * jnp.power(q, 2.0) / (74687613.0 * jnp.power(r, 20.0)) + 3.6019803593214464e+16 * jnp.power(M, 17.0) * jnp.power(q, 4.0) / (14115958857.0 * jnp.power(r, 20.0)) - 2.833519571131187e+16 * jnp.power(M, 17.0) * jnp.power(q, 6.0) / (14115958857.0 * jnp.power(r, 20.0)) + 9.891112547831296e+16 * jnp.power(M, 17.0) * jnp.power(q, 8.0) / (127043629713.0 * jnp.power(r, 20.0)) - 567872463104000.0 * jnp.power(M, 17.0) * jnp.power(q, 10.0) / (3849806961.0 * jnp.power(r, 20.0)) + 31632352013824.0 * jnp.power(M, 17.0) * jnp.power(q, 12.0) / (2491051563.0 * jnp.power(r, 20.0)) - 1967882196992.0 * jnp.power(M, 17.0) * jnp.power(q, 14.0) / (4705319619.0 * jnp.power(r, 20.0)) + 12091392.0 * jnp.power(M, 17.0) * jnp.power(q, 16.0) / (3556553.0 * jnp.power(r, 20.0)) + 1114112.0 * jnp.power(M, 16.0) / (7.0 * jnp.power(r, 19.0)) - 6316583848448.0 * jnp.power(M, 16.0) * jnp.power(q, 2.0) / (9258795.0 * jnp.power(r, 19.0)) + 3.3344712951069376e+16 * jnp.power(M, 16.0) * jnp.power(q, 4.0) / (32081724675.0 * jnp.power(r, 19.0)) - 1.0412053914895918e+16 * jnp.power(M, 16.0) * jnp.power(q, 6.0) / (14582602125.0 * jnp.power(r, 19.0)) + 2.6960628142255503e+18 * jnp.power(M, 16.0) * jnp.power(q, 8.0) / (11549420883000.0 * jnp.power(r, 19.0)) - 1.6342736992086124e+18 * jnp.power(M, 16.0) * jnp.power(q, 10.0) / (46197683532000.0 * jnp.power(r, 19.0)) + 1.0581779822765676e+16 * jnp.power(M, 16.0) * jnp.power(q, 12.0) / (4738223952000.0 * jnp.power(r, 19.0)) - 9908370878981.0 * jnp.power(M, 16.0) * jnp.power(q, 14.0) / (219011240448.0 * jnp.power(r, 19.0)) + 546975.0 * jnp.power(M, 16.0) * jnp.power(q, 16.0) / (4358144.0 * jnp.power(r, 19.0)) + 1507328.0 * jnp.power(M, 15.0) / (19.0 * jnp.power(r, 18.0)) - 1494491556352.0 * jnp.power(M, 15.0) * jnp.power(q, 2.0) / (4849845.0 * jnp.power(r, 18.0)) + 23572004454464.0 * jnp.power(M, 15.0) * jnp.power(q, 4.0) / (56581525.0 * jnp.power(r, 18.0)) - 69746327683944.0 * jnp.power(M, 15.0) * jnp.power(q, 6.0) / (282907625.0 * jnp.power(r, 18.0)) + 56461198314712.0 * jnp.power(M, 15.0) * jnp.power(q, 8.0) / (848722875.0 * jnp.power(r, 18.0)) - 2202458527328.0 * jnp.power(M, 15.0) * jnp.power(q, 10.0) / (282907625.0 * jnp.power(r, 18.0)) + 95267171072.0 * jnp.power(M, 15.0) * jnp.power(q, 12.0) / (282907625.0 * jnp.power(r, 18.0)) - 2424832.0 * jnp.power(M, 15.0) * jnp.power(q, 14.0) / (692835.0 * jnp.power(r, 18.0)) + 2252800.0 * jnp.power(M, 14.0) / (57.0 * jnp.power(r, 17.0)) - 1205476363264.0 * jnp.power(M, 14.0) * jnp.power(q, 2.0) / (8729721.0 * jnp.power(r, 17.0)) + 50070815769328.0 * jnp.power(M, 14.0) * jnp.power(q, 4.0) / (305540235.0 * jnp.power(r, 17.0)) - 86903812826108.0 * jnp.power(M, 14.0) * jnp.power(q, 6.0) / (1057639275.0 * jnp.power(r, 17.0)) + 1959948682174993.0 * jnp.power(M, 14.0) * jnp.power(q, 8.0) / (109994484600.0 * jnp.power(r, 17.0)) - 225207514065389.0 * jnp.power(M, 14.0) * jnp.power(q, 10.0) / (146659312800.0 * jnp.power(r, 17.0)) + 427651340075.0 * jnp.power(M, 14.0) * jnp.power(q, 12.0) / (10429106688.0 * jnp.power(r, 17.0)) - 96525.0 * jnp.power(M, 14.0) * jnp.power(q, 14.0) / (661504.0 * jnp.power(r, 17.0)) + 1003520.0 * jnp.power(M, 13.0) / (51.0 * jnp.power(r, 16.0)) - 28169575424.0 * jnp.power(M, 13.0) * jnp.power(q, 2.0) / (459459.0 * jnp.power(r, 16.0)) + 1013852314624.0 * jnp.power(M, 13.0) * jnp.power(q, 4.0) / (16081065.0 * jnp.power(r, 16.0)) - 2712695658496.0 * jnp.power(M, 13.0) * jnp.power(q, 6.0) / (103378275.0 * jnp.power(r, 16.0)) + 3196414281728.0 * jnp.power(M, 13.0) * jnp.power(q, 8.0) / (723647925.0 * jnp.power(r, 16.0)) - 63199271936.0 * jnp.power(M, 13.0) * jnp.power(q, 10.0) / (241215975.0 * jnp.power(r, 16.0)) + 7168.0 * jnp.power(M, 13.0) * jnp.power(q, 12.0) / (1989.0 * jnp.power(r, 16.0)) + 166400.0 * jnp.power(M, 12.0) / (17.0 * jnp.power(r, 15.0)) - 1374576608.0 * jnp.power(M, 12.0) * jnp.power(q, 2.0) / (51051.0 * jnp.power(r, 15.0)) + 8439472838.0 * jnp.power(M, 12.0) * jnp.power(q, 4.0) / (357357.0 * jnp.power(r, 15.0)) - 24597221.0 * jnp.power(M, 12.0) * jnp.power(q, 6.0) / (3094.0 * jnp.power(r, 15.0)) + 11333729111.0 * jnp.power(M, 12.0) * jnp.power(q, 8.0) / (11435424.0 * jnp.power(r, 15.0)) - 417084431.0 * jnp.power(M, 12.0) * jnp.power(q, 10.0) / (11435424.0 * jnp.power(r, 15.0)) + 3003.0 * jnp.power(M, 12.0) * jnp.power(q, 12.0) / (17408.0 * jnp.power(r, 15.0)) + 4864.0 * jnp.power(M, 11.0) / jnp.power(r, 14.0) - 58413792.0 * jnp.power(M, 11.0) * jnp.power(q, 2.0) / (5005.0 * jnp.power(r, 14.0)) + 5785354852.0 * jnp.power(M, 11.0) * jnp.power(q, 4.0) / (675675.0 * jnp.power(r, 14.0)) - 4567772324.0 * jnp.power(M, 11.0) * jnp.power(q, 6.0) / (2027025.0 * jnp.power(r, 14.0)) + 14607952.0 * jnp.power(M, 11.0) * jnp.power(q, 8.0) / (75075.0 * jnp.power(r, 14.0)) - 3712.0 * jnp.power(M, 11.0) * jnp.power(q, 10.0) / (1001.0 * jnp.power(r, 14.0)) + 16896.0 * jnp.power(M, 10.0) / (7.0 * jnp.power(r, 13.0)) - 1569899984.0 * jnp.power(M, 10.0) * jnp.power(q, 2.0) / (315315.0 * jnp.power(r, 13.0)) + 2013679711.0 * jnp.power(M, 10.0) * jnp.power(q, 4.0) / (675675.0 * jnp.power(r, 13.0)) - 11078698957.0 * jnp.power(M, 10.0) * jnp.power(q, 6.0) / (18918900.0 * jnp.power(r, 13.0)) + 3187331519.0 * jnp.power(M, 10.0) * jnp.power(q, 8.0) / (100900800.0 * jnp.power(r, 13.0)) - 693.0 * jnp.power(M, 10.0) * jnp.power(q, 10.0) / (3328.0 * jnp.power(r, 13.0)) + 108800.0 * jnp.power(M, 9.0) / (91.0 * jnp.power(r, 12.0)) - 43769248.0 * jnp.power(M, 9.0) * jnp.power(q, 2.0) / (21021.0 * jnp.power(r, 12.0)) + 703808.0 * jnp.power(M, 9.0) * jnp.power(q, 4.0) / (715.0 * jnp.power(r, 12.0)) - 4736224.0 * jnp.power(M, 9.0) * jnp.power(q, 6.0) / (35035.0 * jnp.power(r, 12.0)) + 80000.0 * jnp.power(M, 9.0) * jnp.power(q, 8.0) / (21021.0 * jnp.power(r, 12.0)) + 7680.0 * jnp.power(M, 8.0) / (13.0 * jnp.power(r, 11.0)) - 7648268.0 * jnp.power(M, 8.0) * jnp.power(q, 2.0) / (9009.0 * jnp.power(r, 11.0)) + 1517584.0 * jnp.power(M, 8.0) * jnp.power(q, 4.0) / (5005.0 * jnp.power(r, 11.0)) - 1578641.0 * jnp.power(M, 8.0) * jnp.power(q, 6.0) / (60060.0 * jnp.power(r, 11.0)) + 4725.0 * jnp.power(M, 8.0) * jnp.power(q, 8.0) / (18304.0 * jnp.power(r, 11.0)) + 3200.0 * jnp.power(M, 7.0) / (11.0 * jnp.power(r, 10.0)) - 231988.0 * jnp.power(M, 7.0) * jnp.power(q, 2.0) / (693.0 * jnp.power(r, 10.0)) + 58724.0 * jnp.power(M, 7.0) * jnp.power(q, 4.0) / (693.0 * jnp.power(r, 10.0)) - 128.0 * jnp.power(M, 7.0) * jnp.power(q, 6.0) / (33.0 * jnp.power(r, 10.0)) + 1568.0 * jnp.power(M, 6.0) / (11.0 * jnp.power(r, 9.0)) - 145616.0 * jnp.power(M, 6.0) * jnp.power(q, 2.0) / (1155.0 * jnp.power(r, 9.0)) + 94613.0 * jnp.power(M, 6.0) * jnp.power(q, 4.0) / (4620.0 * jnp.power(r, 9.0)) - 175.0 * jnp.power(M, 6.0) * jnp.power(q, 6.0) / (528.0 * jnp.power(r, 9.0)) + 208.0 * jnp.power(M, 5.0) / (3.0 * jnp.power(r, 8.0)) - 4664.0 * jnp.power(M, 5.0) * jnp.power(q, 2.0) / (105.0 * jnp.power(r, 8.0)) + 136.0 * jnp.power(M, 5.0) * jnp.power(q, 4.0) / (35.0 * jnp.power(r, 8.0)) + 100.0 * jnp.power(M, 4.0) / (3.0 * jnp.power(r, 7.0)) - 296.0 * jnp.power(M, 4.0) * jnp.power(q, 2.0) / (21.0 * jnp.power(r, 7.0)) + 25.0 * jnp.power(M, 4.0) * jnp.power(q, 4.0) / (56.0 * jnp.power(r, 7.0)) + 110.0 * jnp.power(M, 3.0) / (7.0 * jnp.power(r, 6.0)) - 26.0 * jnp.power(M, 3.0) * jnp.power(q, 2.0) / (7.0 * jnp.power(r, 6.0)) + 50.0 * jnp.power(M, 2.0) / (7.0 * jnp.power(r, 5.0)) - 9.0 * jnp.power(M, 2.0) * jnp.power(q, 2.0) / (14.0 * jnp.power(r, 5.0)) + 3.0 * M / jnp.power(r, 4.0) + jnp.power(r, -3.0)


def PhiPOnlyQT_jax(M, q, r):
    return 93388800.0 * jnp.power(M, 19.0) * q / (253.0 * jnp.power(r, 22.0)) - 1982408089600.0 * jnp.power(M, 19.0) * jnp.power(q, 3.0) / (1312311.0 * jnp.power(r, 22.0)) + 9202652339847040.0 * jnp.power(M, 19.0) * jnp.power(q, 5.0) / (4216455243.0 * jnp.power(r, 22.0)) - 6709158052328800.0 * jnp.power(M, 19.0) * jnp.power(q, 7.0) / (4660292637.0 * jnp.power(r, 22.0)) + 3.095625106257937e+18 * jnp.power(M, 19.0) * jnp.power(q, 9.0) / (6640917007725.0 * jnp.power(r, 22.0)) - 1.4702022486024916e+18 * jnp.power(M, 19.0) * jnp.power(q, 11.0) / (19922751023175.0 * jnp.power(r, 22.0)) + 1.0618124700348515e+17 * jnp.power(M, 19.0) * jnp.power(q, 13.0) / (19922751023175.0 * jnp.power(r, 22.0)) - 327839775523072.0 * jnp.power(M, 19.0) * jnp.power(q, 15.0) / (2213639002575.0 * jnp.power(r, 22.0)) + 7602176.0 * jnp.power(M, 19.0) * jnp.power(q, 17.0) / (7436429.0 * jnp.power(r, 22.0)) + 317521920.0 * jnp.power(M, 18.0) * q / (1771.0 * jnp.power(r, 21.0)) - 54248329565184.0 * jnp.power(M, 18.0) * jnp.power(q, 3.0) / (81800719.0 * jnp.power(r, 21.0)) + 11457631522208.0 * jnp.power(M, 18.0) * jnp.power(q, 5.0) / (13483635.0 * jnp.power(r, 21.0)) - 424943704339756.0 * jnp.power(M, 18.0) * jnp.power(q, 7.0) / (876436275.0 * jnp.power(r, 21.0)) + 919918368991283.0 * jnp.power(M, 18.0) * jnp.power(q, 9.0) / (7011490200.0 * jnp.power(r, 21.0)) - 231217475648777.0 * jnp.power(M, 18.0) * jnp.power(q, 11.0) / (14022980400.0 * jnp.power(r, 21.0)) + 194652894683933.0 * jnp.power(M, 18.0) * jnp.power(q, 13.0) / (224367686400.0 * jnp.power(r, 21.0)) - 926258551895.0 * jnp.power(M, 18.0) * jnp.power(q, 15.0) / (62822952192.0 * jnp.power(r, 21.0)) + 182325.0 * jnp.power(M, 18.0) * jnp.power(q, 17.0) / (5275648.0 * jnp.power(r, 21.0)) + 6684672.0 * jnp.power(M, 17.0) * q / (77.0 * jnp.power(r, 20.0)) - 21515986997248.0 * jnp.power(M, 17.0) * jnp.power(q, 3.0) / (74687613.0 * jnp.power(r, 20.0)) + 4574082043518976.0 * jnp.power(M, 17.0) * jnp.power(q, 5.0) / (14115958857.0 * jnp.power(r, 20.0)) - 2225270466590720.0 * jnp.power(M, 17.0) * jnp.power(q, 7.0) / (14115958857.0 * jnp.power(r, 20.0)) + 4431902475622400.0 * jnp.power(M, 17.0) * jnp.power(q, 9.0) / (127043629713.0 * jnp.power(r, 20.0)) - 47407471925248.0 * jnp.power(M, 17.0) * jnp.power(q, 11.0) / (14115958857.0 * jnp.power(r, 20.0)) + 1697292904448.0 * jnp.power(M, 17.0) * jnp.power(q, 13.0) / (14115958857.0 * jnp.power(r, 20.0)) - 33357824.0 * jnp.power(M, 17.0) * jnp.power(q, 15.0) / (32008977.0 * jnp.power(r, 20.0)) + 5570560.0 * jnp.power(M, 16.0) * q / (133.0 * jnp.power(r, 19.0)) - 2518049131264.0 * jnp.power(M, 16.0) * jnp.power(q, 3.0) / (20369349.0 * jnp.power(r, 19.0)) + 773884913000656.0 * jnp.power(M, 16.0) * jnp.power(q, 5.0) / (6416344935.0 * jnp.power(r, 19.0)) - 63088965880649.0 * jnp.power(M, 16.0) * jnp.power(q, 7.0) / (1283268987.0 * jnp.power(r, 19.0)) + 363573767832881.0 * jnp.power(M, 16.0) * jnp.power(q, 9.0) / (41997894120.0 * jnp.power(r, 19.0)) - 2248987038532789.0 * jnp.power(M, 16.0) * jnp.power(q, 11.0) / (3695814682560.0 * jnp.power(r, 19.0)) + 8772734045155.0 * jnp.power(M, 16.0) * jnp.power(q, 13.0) / (657033721344.0 * jnp.power(r, 19.0)) - 10725.0 * jnp.power(M, 16.0) * jnp.power(q, 15.0) / (272384.0 * jnp.power(r, 19.0)) + 1146880.0 * jnp.power(M, 15.0) * q / (57.0 * jnp.power(r, 18.0)) - 16908159232.0 * jnp.power(M, 15.0) * jnp.power(q, 3.0) / (323323.0 * jnp.power(r, 18.0)) + 493734180064.0 * jnp.power(M, 15.0) * jnp.power(q, 5.0) / (11316305.0 * jnp.power(r, 18.0)) - 495090090644.0 * jnp.power(M, 15.0) * jnp.power(q, 7.0) / (33948915.0 * jnp.power(r, 18.0)) + 455469736.0 * jnp.power(M, 15.0) * jnp.power(q, 9.0) / (230945.0 * jnp.power(r, 18.0)) - 1070016608.0 * jnp.power(M, 15.0) * jnp.power(q, 11.0) / (11316305.0 * jnp.power(r, 18.0)) + 146944.0 * jnp.power(M, 15.0) * jnp.power(q, 13.0) / (138567.0 * jnp.power(r, 18.0)) + 9318400.0 * jnp.power(M, 14.0) * q / (969.0 * jnp.power(r, 17.0)) - 189892217600.0 * jnp.power(M, 14.0) * jnp.power(q, 3.0) / (8729721.0 * jnp.power(r, 17.0)) + 932113326568.0 * jnp.power(M, 14.0) * jnp.power(q, 5.0) / (61108047.0 * jnp.power(r, 17.0)) - 1598428684781.0 * jnp.power(M, 14.0) * jnp.power(q, 7.0) / (392837445.0 * jnp.power(r, 17.0)) + 1363525754411.0 * jnp.power(M, 14.0) * jnp.power(q, 9.0) / (3384445680.0 * jnp.power(r, 17.0)) - 232077111443.0 * jnp.power(M, 14.0) * jnp.power(q, 11.0) / (19554575040.0 * jnp.power(r, 17.0)) + 15015.0 * jnp.power(M, 14.0) * jnp.power(q, 13.0) / (330752.0 * jnp.power(r, 17.0)) + 232960.0 * jnp.power(M, 13.0) * q / (51.0 * jnp.power(r, 16.0)) - 4074131968.0 * jnp.power(M, 13.0) * jnp.power(q, 3.0) / (459459.0 * jnp.power(r, 16.0)) + 82274075648.0 * jnp.power(M, 13.0) * jnp.power(q, 5.0) / (16081065.0 * jnp.power(r, 16.0)) - 152179698688.0 * jnp.power(M, 13.0) * jnp.power(q, 7.0) / (144729585.0 * jnp.power(r, 16.0)) + 10307691008.0 * jnp.power(M, 13.0) * jnp.power(q, 9.0) / (144729585.0 * jnp.power(r, 16.0)) - 494080.0 * jnp.power(M, 13.0) * jnp.power(q, 11.0) / (459459.0 * jnp.power(r, 16.0)) + 36608.0 * jnp.power(M, 12.0) * q / (17.0 * jnp.power(r, 15.0)) - 25723376.0 * jnp.power(M, 12.0) * jnp.power(q, 3.0) / (7293.0 * jnp.power(r, 15.0)) + 194177083.0 * jnp.power(M, 12.0) * jnp.power(q, 5.0) / (119119.0 * jnp.power(r, 15.0)) - 58637561.0 * jnp.power(M, 12.0) * jnp.power(q, 7.0) / (238238.0 * jnp.power(r, 15.0)) + 235028551.0 * jnp.power(M, 12.0) * jnp.power(q, 9.0) / (22870848.0 * jnp.power(r, 15.0)) - 231.0 * jnp.power(M, 12.0) * jnp.power(q, 11.0) / (4352.0 * jnp.power(r, 15.0)) + 7040.0 * jnp.power(M, 11.0) * q / (7.0 * jnp.power(r, 14.0)) - 1362448.0 * jnp.power(M, 11.0) * jnp.power(q, 3.0) / (1001.0 * jnp.power(r, 14.0)) + 13150838.0 * jnp.power(M, 11.0) * jnp.power(q, 5.0) / (27027.0 * jnp.power(r, 14.0)) - 4090336.0 * jnp.power(M, 11.0) * jnp.power(q, 7.0) / (81081.0 * jnp.power(r, 14.0)) + 9760.0 * jnp.power(M, 11.0) * jnp.power(q, 9.0) / (9009.0 * jnp.power(r, 14.0)) + 42240.0 * jnp.power(M, 10.0) * q / (91.0 * jnp.power(r, 13.0)) - 31877096.0 * jnp.power(M, 10.0) * jnp.power(q, 3.0) / (63063.0 * jnp.power(r, 13.0)) + 50453185.0 * jnp.power(M, 10.0) * jnp.power(q, 5.0) / (378378.0 * jnp.power(r, 13.0)) - 1618798.0 * jnp.power(M, 10.0) * jnp.power(q, 7.0) / (189189.0 * jnp.power(r, 13.0)) + 105.0 * jnp.power(M, 10.0) * jnp.power(q, 9.0) / (1664.0 * jnp.power(r, 13.0)) + 19200.0 * jnp.power(M, 9.0) * q / (91.0 * jnp.power(r, 12.0)) - 1251520.0 * jnp.power(M, 9.0) * jnp.power(q, 3.0) / (7007.0 * jnp.power(r, 12.0)) + 227776.0 * jnp.power(M, 9.0) * jnp.power(q, 5.0) / (7007.0 * jnp.power(r, 12.0)) - 7552.0 * jnp.power(M, 9.0) * jnp.power(q, 7.0) / (7007.0 * jnp.power(r, 12.0)) + 13440.0 * jnp.power(M, 8.0) * q / (143.0 * jnp.power(r, 11.0)) - 531766.0 * jnp.power(M, 8.0) * jnp.power(q, 3.0) / (9009.0 * jnp.power(r, 11.0)) + 80291.0 * jnp.power(M, 8.0) * jnp.power(q, 5.0) / (12012.0 * jnp.power(r, 11.0)) - 175.0 * jnp.power(M, 8.0) * jnp.power(q, 7.0) / (2288.0 * jnp.power(r, 11.0)) + 448.0 * jnp.power(M, 7.0) * q / (11.0 * jnp.power(r, 10.0)) - 12302.0 * jnp.power(M, 7.0) * jnp.power(q, 3.0) / (693.0 * jnp.power(r, 10.0)) + 724.0 * jnp.power(M, 7.0) * jnp.power(q, 5.0) / (693.0 * jnp.power(r, 10.0)) + 560.0 * jnp.power(M, 6.0) * q / (33.0 * jnp.power(r, 9.0)) - 1070.0 * jnp.power(M, 6.0) * jnp.power(q, 3.0) / (231.0 * jnp.power(r, 9.0)) + 25.0 * jnp.power(M, 6.0) * jnp.power(q, 5.0) / (264.0 * jnp.power(r, 9.0)) + 20.0 * jnp.power(M, 5.0) * q / (3.0 * jnp.power(r, 8.0)) - 20.0 * jnp.power(M, 5.0) * jnp.power(q, 3.0) / (21.0 * jnp.power(r, 8.0)) + 50.0 * jnp.power(M, 4.0) * q / (21.0 * jnp.power(r, 7.0)) - 5.0 * jnp.power(M, 4.0) * jnp.power(q, 3.0) / (42.0 * jnp.power(r, 7.0)) + 5.0 * jnp.power(M, 3.0) * q / (7.0 * jnp.power(r, 6.0)) + jnp.power(M, 2.0) * q / (7.0 * jnp.power(r, 5.0))


def H0OnlyET_jax(M, q, r):
    return 2.0 * jnp.power(M, 2.0) * jnp.power(q, 2.0) / 3.0 - 1795948544.0 * jnp.power(M, 24.0) * jnp.power(q, 2.0) / (1311.0 * jnp.power(r, 22.0)) + 3026355961364480.0 * jnp.power(M, 24.0) * jnp.power(q, 4.0) / (423876453.0 * jnp.power(r, 22.0)) - 9.934062256455117e+16 * jnp.power(M, 24.0) * jnp.power(q, 6.0) / (7323317001.0 * jnp.power(r, 22.0)) + 1.66948524541456e+22 * jnp.power(M, 24.0) * jnp.power(q, 8.0) / (1387951654614525.0 * jnp.power(r, 22.0)) - 5.494650013306427e+23 * jnp.power(M, 24.0) * jnp.power(q, 10.0) / (1.0409637409608938e+17 * jnp.power(r, 22.0)) + 3.449339897623955e+23 * jnp.power(M, 24.0) * jnp.power(q, 12.0) / (3.122891222882681e+17 * jnp.power(r, 22.0)) - 1.712772250439323e+21 * jnp.power(M, 24.0) * jnp.power(q, 14.0) / (1.9119742180914376e+16 * jnp.power(r, 22.0)) - 1.0276506807305192e+21 * jnp.power(M, 24.0) * jnp.power(q, 16.0) / (2.810602100594413e+18 * jnp.power(r, 22.0)) + 1.6998731009782561e+18 * jnp.power(M, 24.0) * jnp.power(q, 18.0) / (6373247393638125.0 * jnp.power(r, 22.0)) - 70522962808832.0 * jnp.power(M, 24.0) * jnp.power(q, 20.0) / (16328842995465.0 * jnp.power(r, 22.0)) - 301613056.0 * jnp.power(M, 24.0) * jnp.power(q, 22.0) / (13987922949.0 * jnp.power(r, 22.0)) - 172228608.0 * jnp.power(M, 23.0) * jnp.power(q, 2.0) / (253.0 * jnp.power(r, 21.0)) + 1331923272265728.0 * jnp.power(M, 23.0) * jnp.power(q, 4.0) / (409003595.0 * jnp.power(r, 21.0)) - 1.1419773419217792e+16 * jnp.power(M, 23.0) * jnp.power(q, 6.0) / (2045017975.0 * jnp.power(r, 21.0)) + 3431446897968256.0 * jnp.power(M, 23.0) * jnp.power(q, 8.0) / (786545375.0 * jnp.power(r, 21.0)) - 2.373750847096053e+16 * jnp.power(M, 23.0) * jnp.power(q, 10.0) / (14607271250.0 * jnp.power(r, 21.0)) + 259748182786263.0 * jnp.power(M, 23.0) * jnp.power(q, 12.0) / (965770000.0 * jnp.power(r, 21.0)) - 6318613505327741.0 * jnp.power(M, 23.0) * jnp.power(q, 14.0) / (467432680000.0 * jnp.power(r, 21.0)) - 2127504567244611.0 * jnp.power(M, 23.0) * jnp.power(q, 16.0) / (3739461440000.0 * jnp.power(r, 21.0)) + 1.7427823814337472e+16 * jnp.power(M, 23.0) * jnp.power(q, 18.0) / (418819681280000.0 * jnp.power(r, 21.0)) - 721589919151.0 * jnp.power(M, 23.0) * jnp.power(q, 20.0) / (5360891920384.0 * jnp.power(r, 21.0)) - 37791.0 * jnp.power(M, 23.0) * jnp.power(q, 22.0) / (21102592.0 * jnp.power(r, 21.0)) - 11141120.0 * jnp.power(M, 22.0) * jnp.power(q, 2.0) / (33.0 * jnp.power(r, 20.0)) + 1651285448531968.0 * jnp.power(M, 22.0) * jnp.power(q, 4.0) / (1120314195.0 * jnp.power(r, 20.0)) - 4.790558621575209e+17 * jnp.power(M, 22.0) * jnp.power(q, 6.0) / (211739382855.0 * jnp.power(r, 20.0)) + 6.511366545685005e+16 * jnp.power(M, 22.0) * jnp.power(q, 8.0) / (42347876571.0 * jnp.power(r, 20.0)) - 6.968485180692378e+16 * jnp.power(M, 22.0) * jnp.power(q, 10.0) / (146588803515.0 * jnp.power(r, 20.0)) + 2.247140858365107e+16 * jnp.power(M, 22.0) * jnp.power(q, 12.0) / (381130889139.0 * jnp.power(r, 20.0)) - 2818816980099584.0 * jnp.power(M, 22.0) * jnp.power(q, 14.0) / (2450127144465.0 * jnp.power(r, 20.0)) - 328915430605184.0 * jnp.power(M, 22.0) * jnp.power(q, 16.0) / (1905654445695.0 * jnp.power(r, 20.0)) + 187412715008.0 * jnp.power(M, 22.0) * jnp.power(q, 18.0) / (42347876571.0 * jnp.power(r, 20.0)) + 8192.0 * jnp.power(M, 22.0) * jnp.power(q, 20.0) / (440895.0 * jnp.power(r, 20.0)) - 17563648.0 * jnp.power(M, 21.0) * jnp.power(q, 2.0) / (105.0 * jnp.power(r, 19.0)) + 48176937714688.0 * jnp.power(M, 21.0) * jnp.power(q, 4.0) / (72747675.0 * jnp.power(r, 19.0)) - 8.67524307235769e+16 * jnp.power(M, 21.0) * jnp.power(q, 6.0) / (96245174025.0 * jnp.power(r, 19.0)) + 1.2627038878827817e+18 * jnp.power(M, 21.0) * jnp.power(q, 8.0) / (2406129350625.0 * jnp.power(r, 19.0)) - 2.8324664939136896e+18 * jnp.power(M, 21.0) * jnp.power(q, 10.0) / (21655164155625.0 * jnp.power(r, 19.0)) + 1.9253399483808276e+18 * jnp.power(M, 21.0) * jnp.power(q, 12.0) / (173241313245000.0 * jnp.power(r, 19.0)) + 2253864884321707.0 * jnp.power(M, 21.0) * jnp.power(q, 14.0) / (14920304490000.0 * jnp.power(r, 19.0)) - 7.457196187887162e+17 * jnp.power(M, 21.0) * jnp.power(q, 16.0) / (2.217488809536e+16 * jnp.power(r, 19.0)) + 1225448279513.0 * jnp.power(M, 21.0) * jnp.power(q, 18.0) / (5631717611520.0 * jnp.power(r, 19.0)) + 26741.0 * jnp.power(M, 21.0) * jnp.power(q, 20.0) / (13074432.0 * jnp.power(r, 19.0)) - 1572864.0 * jnp.power(M, 20.0) * jnp.power(q, 2.0) / (19.0 * jnp.power(r, 18.0)) + 30971278336.0 * jnp.power(M, 20.0) * jnp.power(q, 4.0) / (104975.0 * jnp.power(r, 18.0)) - 99607926089216.0 * jnp.power(M, 20.0) * jnp.power(q, 6.0) / (282907625.0 * jnp.power(r, 18.0)) + 243697859241888.0 * jnp.power(M, 20.0) * jnp.power(q, 8.0) / (1414538125.0 * jnp.power(r, 18.0)) - 46998257904314.0 * jnp.power(M, 20.0) * jnp.power(q, 10.0) / (1414538125.0 * jnp.power(r, 18.0)) + 2255447520022.0 * jnp.power(M, 20.0) * jnp.power(q, 12.0) / (1414538125.0 * jnp.power(r, 18.0)) + 136801791864.0 * jnp.power(M, 20.0) * jnp.power(q, 14.0) / (1414538125.0 * jnp.power(r, 18.0)) - 6221321152.0 * jnp.power(M, 20.0) * jnp.power(q, 16.0) / (1414538125.0 * jnp.power(r, 18.0)) - 16384.0 * jnp.power(M, 20.0) * jnp.power(q, 18.0) / (1154725.0 * jnp.power(r, 18.0)) - 6995968.0 * jnp.power(M, 19.0) * jnp.power(q, 2.0) / (171.0 * jnp.power(r, 17.0)) + 17041946358784.0 * jnp.power(M, 19.0) * jnp.power(q, 4.0) / (130945815.0 * jnp.power(r, 17.0)) - 205224782811392.0 * jnp.power(M, 19.0) * jnp.power(q, 6.0) / (1527701175.0 * jnp.power(r, 17.0)) + 1.1114967845506816e+16 * jnp.power(M, 19.0) * jnp.power(q, 8.0) / (206239658625.0 * jnp.power(r, 17.0)) - 1.2460283668445272e+16 * jnp.power(M, 19.0) * jnp.power(q, 10.0) / (1649917269000.0 * jnp.power(r, 17.0)) + 5405548485623711.0 * jnp.power(M, 19.0) * jnp.power(q, 12.0) / (59397021684000.0 * jnp.power(r, 17.0)) + 4999186602827.0 * jnp.power(M, 19.0) * jnp.power(q, 14.0) / (198486288000.0 * jnp.power(r, 17.0)) - 113605554383.0 * jnp.power(M, 19.0) * jnp.power(q, 16.0) / (375447840768.0 * jnp.power(r, 17.0)) - 25025.0 * jnp.power(M, 19.0) * jnp.power(q, 18.0) / (10584064.0 * jnp.power(r, 17.0)) - 3088384.0 * jnp.power(M, 18.0) * jnp.power(q, 2.0) / (153.0 * jnp.power(r, 16.0)) + 391057583104.0 * jnp.power(M, 18.0) * jnp.power(q, 4.0) / (6891885.0 * jnp.power(r, 16.0)) - 7139858176.0 * jnp.power(M, 18.0) * jnp.power(q, 6.0) / (143325.0 * jnp.power(r, 16.0)) + 172020744946432.0 * jnp.power(M, 18.0) * jnp.power(q, 8.0) / (10854718875.0 * jnp.power(r, 16.0)) - 15670155663872.0 * jnp.power(M, 18.0) * jnp.power(q, 10.0) / (10854718875.0 * jnp.power(r, 16.0)) - 3725198288384.0 * jnp.power(M, 18.0) * jnp.power(q, 12.0) / (97692469875.0 * jnp.power(r, 16.0)) + 45271423744.0 * jnp.power(M, 18.0) * jnp.power(q, 14.0) / (10854718875.0 * jnp.power(r, 16.0)) + 7936.0 * jnp.power(M, 18.0) * jnp.power(q, 16.0) / (984555.0 * jnp.power(r, 16.0)) - 168960.0 * jnp.power(M, 17.0) * jnp.power(q, 2.0) / (17.0 * jnp.power(r, 15.0)) + 22808576.0 * jnp.power(M, 17.0) * jnp.power(q, 4.0) / (935.0 * jnp.power(r, 15.0)) - 2124241628.0 * jnp.power(M, 17.0) * jnp.power(q, 6.0) / (119119.0 * jnp.power(r, 15.0)) + 1459391631.0 * jnp.power(M, 17.0) * jnp.power(q, 8.0) / (340340.0 * jnp.power(r, 15.0)) - 1859605213.0 * jnp.power(M, 17.0) * jnp.power(q, 10.0) / (9529520.0 * jnp.power(r, 15.0)) - 57914453.0 * jnp.power(M, 17.0) * jnp.power(q, 12.0) / (3465280.0 * jnp.power(r, 15.0)) + 528387.0 * jnp.power(M, 17.0) * jnp.power(q, 14.0) / (1361360.0 * jnp.power(r, 15.0)) + 3861.0 * jnp.power(M, 17.0) * jnp.power(q, 16.0) / (1392640.0 * jnp.power(r, 15.0)) - 73216.0 * jnp.power(M, 16.0) * jnp.power(q, 2.0) / (15.0 * jnp.power(r, 14.0)) + 6965811968.0 * jnp.power(M, 16.0) * jnp.power(q, 4.0) / (675675.0 * jnp.power(r, 14.0)) - 12370991536.0 * jnp.power(M, 16.0) * jnp.power(q, 6.0) / (2027025.0 * jnp.power(r, 14.0)) + 6254576641.0 * jnp.power(M, 16.0) * jnp.power(q, 8.0) / (6081075.0 * jnp.power(r, 14.0)) - 42789671.0 * jnp.power(M, 16.0) * jnp.power(q, 10.0) / (18243225.0 * jnp.power(r, 14.0)) - 7546076.0 * jnp.power(M, 16.0) * jnp.power(q, 12.0) / (2027025.0 * jnp.power(r, 14.0)) + 32.0 * jnp.power(M, 16.0) * jnp.power(q, 14.0) / (45045.0 * jnp.power(r, 14.0)) - 7168.0 * jnp.power(M, 15.0) * jnp.power(q, 2.0) / (3.0 * jnp.power(r, 13.0)) + 20171485856.0 * jnp.power(M, 15.0) * jnp.power(q, 4.0) / (4729725.0 * jnp.power(r, 13.0)) - 139608820654.0 * jnp.power(M, 15.0) * jnp.power(q, 6.0) / (70945875.0 * jnp.power(r, 13.0)) + 171954172043.0 * jnp.power(M, 15.0) * jnp.power(q, 8.0) / (851350500.0 * jnp.power(r, 13.0)) + 1955084741.0 * jnp.power(M, 15.0) * jnp.power(q, 10.0) / (224532000.0 * jnp.power(r, 13.0)) - 8558330503.0 * jnp.power(M, 15.0) * jnp.power(q, 12.0) / (18162144000.0 * jnp.power(r, 13.0)) - 11.0 * jnp.power(M, 15.0) * jnp.power(q, 14.0) / (3328.0 * jnp.power(r, 13.0)) - 105984.0 * jnp.power(M, 14.0) * jnp.power(q, 2.0) / (91.0 * jnp.power(r, 12.0)) + 781056.0 * jnp.power(M, 14.0) * jnp.power(q, 4.0) / (455.0 * jnp.power(r, 12.0)) - 102244488.0 * jnp.power(M, 14.0) * jnp.power(q, 6.0) / (175175.0 * jnp.power(r, 12.0)) + 626256.0 * jnp.power(M, 14.0) * jnp.power(q, 8.0) / (25025.0 * jnp.power(r, 12.0)) + 531288.0 * jnp.power(M, 14.0) * jnp.power(q, 10.0) / (175175.0 * jnp.power(r, 12.0)) - 96.0 * jnp.power(M, 14.0) * jnp.power(q, 12.0) / (7007.0 * jnp.power(r, 12.0)) - 22016.0 * jnp.power(M, 13.0) * jnp.power(q, 2.0) / (39.0 * jnp.power(r, 11.0)) + 989368.0 * jnp.power(M, 13.0) * jnp.power(q, 4.0) / (1485.0 * jnp.power(r, 11.0)) - 20588222.0 * jnp.power(M, 13.0) * jnp.power(q, 6.0) / (135135.0 * jnp.power(r, 11.0)) - 17175089.0 * jnp.power(M, 13.0) * jnp.power(q, 8.0) / (9729720.0 * jnp.power(r, 11.0)) + 1179523.0 * jnp.power(M, 13.0) * jnp.power(q, 10.0) / (2162160.0 * jnp.power(r, 11.0)) + 147.0 * jnp.power(M, 13.0) * jnp.power(q, 12.0) / (36608.0 * jnp.power(r, 11.0)) - 8960.0 * jnp.power(M, 12.0) * jnp.power(q, 2.0) / (33.0 * jnp.power(r, 10.0)) + 2554984.0 * jnp.power(M, 12.0) * jnp.power(q, 4.0) / (10395.0 * jnp.power(r, 10.0)) - 65621.0 * jnp.power(M, 12.0) * jnp.power(q, 6.0) / (2079.0 * jnp.power(r, 10.0)) - 194983.0 * jnp.power(M, 12.0) * jnp.power(q, 8.0) / (93555.0 * jnp.power(r, 10.0)) + 32.0 * jnp.power(M, 12.0) * jnp.power(q, 10.0) / (945.0 * jnp.power(r, 10.0)) - 7104.0 * jnp.power(M, 11.0) * jnp.power(q, 2.0) / (55.0 * jnp.power(r, 9.0)) + 161824.0 * jnp.power(M, 11.0) * jnp.power(q, 4.0) / (1925.0 * jnp.power(r, 9.0)) - 933.0 * jnp.power(M, 11.0) * jnp.power(q, 6.0) / (275.0 * jnp.power(r, 9.0)) - 18399.0 * jnp.power(M, 11.0) * jnp.power(q, 8.0) / (30800.0 * jnp.power(r, 9.0)) - 7.0 * jnp.power(M, 11.0) * jnp.power(q, 10.0) / (1408.0 * jnp.power(r, 9.0)) - 544.0 * jnp.power(M, 10.0) * jnp.power(q, 2.0) / (9.0 * jnp.power(r, 8.0)) + 39916.0 * jnp.power(M, 10.0) * jnp.power(q, 4.0) / (1575.0 * jnp.power(r, 8.0)) + 12514.0 * jnp.power(M, 10.0) * jnp.power(q, 6.0) / (14175.0 * jnp.power(r, 8.0)) - 106.0 * jnp.power(M, 10.0) * jnp.power(q, 8.0) / (1575.0 * jnp.power(r, 8.0)) - 248.0 * jnp.power(M, 9.0) * jnp.power(q, 2.0) / (9.0 * jnp.power(r, 7.0)) + 1856.0 * jnp.power(M, 9.0) * jnp.power(q, 4.0) / (315.0 * jnp.power(r, 7.0)) + 241.0 * jnp.power(M, 9.0) * jnp.power(q, 6.0) / (405.0 * jnp.power(r, 7.0)) + 25.0 * jnp.power(M, 9.0) * jnp.power(q, 8.0) / (4032.0 * jnp.power(r, 7.0)) - 12.0 * jnp.power(M, 8.0) * jnp.power(q, 2.0) / jnp.power(r, 6.0) + 33.0 * jnp.power(M, 8.0) * jnp.power(q, 4.0) / (70.0 * jnp.power(r, 6.0)) + 9.0 * jnp.power(M, 8.0) * jnp.power(q, 6.0) / (70.0 * jnp.power(r, 6.0)) - 100.0 * jnp.power(M, 7.0) * jnp.power(q, 2.0) / (21.0 * jnp.power(r, 5.0)) - 289.0 * jnp.power(M, 7.0) * jnp.power(q, 4.0) / (630.0 * jnp.power(r, 5.0)) - jnp.power(M, 7.0) * jnp.power(q, 6.0) / (140.0 * jnp.power(r, 5.0)) - 22.0 * jnp.power(M, 6.0) * jnp.power(q, 2.0) / (15.0 * jnp.power(r, 4.0)) - 47.0 * jnp.power(M, 6.0) * jnp.power(q, 4.0) / (180.0 * jnp.power(r, 4.0)) + 2.0 * jnp.power(M, 4.0) * jnp.power(q, 2.0) / (3.0 * jnp.power(r, 2.0)) + jnp.power(M, 3.0) * jnp.power(q, 2.0) / (3.0 * r) - 2.0 * M * r + jnp.power(r, 2.0)


def PhiPOnlyET_jax(M, q, r):
    return -0.3333333333333333 * (jnp.power(M, 2.0) * q) - 1048576.0 * jnp.power(M, 24.0) * q / (57.0 * jnp.power(r, 22.0)) - 108047362342912.0 * jnp.power(M, 24.0) * jnp.power(q, 3.0) / (423876453.0 * jnp.power(r, 22.0)) + 1.2223025594248648e+18 * jnp.power(M, 24.0) * jnp.power(q, 5.0) / (1016814398985.0 * jnp.power(r, 22.0)) - 2.415993268667906e+21 * jnp.power(M, 24.0) * jnp.power(q, 7.0) / (1387951654614525.0 * jnp.power(r, 22.0)) + 1.1372707315866402e+23 * jnp.power(M, 24.0) * jnp.power(q, 9.0) / (1.0409637409608938e+17 * jnp.power(r, 22.0)) - 1.56978117441629e+23 * jnp.power(M, 24.0) * jnp.power(q, 11.0) / (4.99662595661229e+17 * jnp.power(r, 22.0)) + 1.2677820661047214e+22 * jnp.power(M, 24.0) * jnp.power(q, 13.0) / (3.52703008702044e+17 * jnp.power(r, 22.0)) + 8.765493299428505e+21 * jnp.power(M, 24.0) * jnp.power(q, 15.0) / (5.756113102017358e+19 * jnp.power(r, 22.0)) - 1.3175059756885878e+22 * jnp.power(M, 24.0) * jnp.power(q, 17.0) / (4.263787482975821e+19 * jnp.power(r, 22.0)) + 4.357109942093411e+19 * jnp.power(M, 24.0) * jnp.power(q, 19.0) / (2.35405796161536e+18 * jnp.power(r, 22.0)) - 3666737926669729.0 * jnp.power(M, 24.0) * jnp.power(q, 21.0) / (1.178630380781568e+16 * jnp.power(r, 22.0)) + 1547.0 * jnp.power(M, 24.0) * jnp.power(q, 23.0) / (2097152.0 * jnp.power(r, 22.0)) - 381026304.0 * jnp.power(M, 23.0) * jnp.power(q, 3.0) / (1771.0 * jnp.power(r, 21.0)) + 336489492363264.0 * jnp.power(M, 23.0) * jnp.power(q, 5.0) / (409003595.0 * jnp.power(r, 21.0)) - 2288720172911296.0 * jnp.power(M, 23.0) * jnp.power(q, 7.0) / (2045017975.0 * jnp.power(r, 21.0)) + 54530206363968.0 * jnp.power(M, 23.0) * jnp.power(q, 9.0) / (76880375.0 * jnp.power(r, 21.0)) - 1344862073331039.0 * jnp.power(M, 23.0) * jnp.power(q, 11.0) / (5842908500.0 * jnp.power(r, 21.0)) + 1844788271586391.0 * jnp.power(M, 23.0) * jnp.power(q, 13.0) / (46743268000.0 * jnp.power(r, 21.0)) - 657087845981487.0 * jnp.power(M, 23.0) * jnp.power(q, 15.0) / (186973072000.0 * jnp.power(r, 21.0)) + 1547821973166531.0 * jnp.power(M, 23.0) * jnp.power(q, 17.0) / (10470492032000.0 * jnp.power(r, 21.0)) - 131287333403.0 * jnp.power(M, 23.0) * jnp.power(q, 19.0) / (58270564352.0 * jnp.power(r, 21.0)) + 109395.0 * jnp.power(M, 23.0) * jnp.power(q, 21.0) / (21102592.0 * jnp.power(r, 21.0)) - 40108032.0 * jnp.power(M, 22.0) * jnp.power(q, 3.0) / (385.0 * jnp.power(r, 20.0)) + 44652955148288.0 * jnp.power(M, 22.0) * jnp.power(q, 5.0) / (124479355.0 * jnp.power(r, 20.0)) - 2032958894531584.0 * jnp.power(M, 22.0) * jnp.power(q, 7.0) / (4705319619.0 * jnp.power(r, 20.0)) + 17319075678208.0 * jnp.power(M, 22.0) * jnp.power(q, 9.0) / (72837765.0 * jnp.power(r, 20.0)) - 2774132700214784.0 * jnp.power(M, 22.0) * jnp.power(q, 11.0) / (42347876571.0 * jnp.power(r, 20.0)) + 115371183150592.0 * jnp.power(M, 22.0) * jnp.power(q, 13.0) / (12455257815.0 * jnp.power(r, 20.0)) - 15246453790208.0 * jnp.power(M, 22.0) * jnp.power(q, 15.0) / (23526598095.0 * jnp.power(r, 20.0)) + 90748965376.0 * jnp.power(M, 22.0) * jnp.power(q, 17.0) / (4705319619.0 * jnp.power(r, 20.0)) - 8339456.0 * jnp.power(M, 22.0) * jnp.power(q, 19.0) / (53348295.0 * jnp.power(r, 20.0)) - 6684672.0 * jnp.power(M, 21.0) * jnp.power(q, 3.0) / (133.0 * jnp.power(r, 19.0)) + 5249385256448.0 * jnp.power(M, 21.0) * jnp.power(q, 5.0) / (33948915.0 * jnp.power(r, 19.0)) - 1746066195088352.0 * jnp.power(M, 21.0) * jnp.power(q, 7.0) / (10693908225.0 * jnp.power(r, 19.0)) + 30531884705802.0 * jnp.power(M, 21.0) * jnp.power(q, 9.0) / (396070675.0 * jnp.power(r, 19.0)) - 131506055976748.0 * jnp.power(M, 21.0) * jnp.power(q, 11.0) / (7403474925.0 * jnp.power(r, 19.0)) + 4110722687299.0 * jnp.power(M, 21.0) * jnp.power(q, 13.0) / (2026214190.0 * jnp.power(r, 19.0)) - 660940017641191.0 * jnp.power(M, 21.0) * jnp.power(q, 15.0) / (6159691137600.0 * jnp.power(r, 19.0)) + 1795939470341.0 * jnp.power(M, 21.0) * jnp.power(q, 17.0) / (876044961792.0 * jnp.power(r, 19.0)) - 6435.0 * jnp.power(M, 21.0) * jnp.power(q, 19.0) / (1089536.0 * jnp.power(r, 19.0)) - 458752.0 * jnp.power(M, 20.0) * jnp.power(q, 3.0) / (19.0 * jnp.power(r, 18.0)) + 106328069632.0 * jnp.power(M, 20.0) * jnp.power(q, 5.0) / (1616615.0 * jnp.power(r, 18.0)) - 13790462592.0 * jnp.power(M, 20.0) * jnp.power(q, 7.0) / (229075.0 * jnp.power(r, 18.0)) + 1360480816336.0 * jnp.power(M, 20.0) * jnp.power(q, 9.0) / (56581525.0 * jnp.power(r, 18.0)) - 51536125009.0 * jnp.power(M, 20.0) * jnp.power(q, 11.0) / (11316305.0 * jnp.power(r, 18.0)) + 23158612446.0 * jnp.power(M, 20.0) * jnp.power(q, 13.0) / (56581525.0 * jnp.power(r, 18.0)) - 874515016.0 * jnp.power(M, 20.0) * jnp.power(q, 15.0) / (56581525.0 * jnp.power(r, 18.0)) + 36736.0 * jnp.power(M, 20.0) * jnp.power(q, 17.0) / (230945.0 * jnp.power(r, 18.0)) - 3727360.0 * jnp.power(M, 19.0) * jnp.power(q, 3.0) / (323.0 * jnp.power(r, 17.0)) + 80154360320.0 * jnp.power(M, 19.0) * jnp.power(q, 5.0) / (2909907.0 * jnp.power(r, 17.0)) - 115607264944.0 * jnp.power(M, 19.0) * jnp.power(q, 7.0) / (5360355.0 * jnp.power(r, 17.0)) + 2987661500984.0 * jnp.power(M, 19.0) * jnp.power(q, 9.0) / (416645775.0 * jnp.power(r, 17.0)) - 13367945464759.0 * jnp.power(M, 19.0) * jnp.power(q, 11.0) / (12221609400.0 * jnp.power(r, 17.0)) + 21903222813317.0 * jnp.power(M, 19.0) * jnp.power(q, 13.0) / (293318625600.0 * jnp.power(r, 17.0)) - 478357587061.0 * jnp.power(M, 19.0) * jnp.power(q, 15.0) / (260727667200.0 * jnp.power(r, 17.0)) + 9009.0 * jnp.power(M, 19.0) * jnp.power(q, 17.0) / (1323008.0 * jnp.power(r, 17.0)) - 93184.0 * jnp.power(M, 18.0) * jnp.power(q, 3.0) / (17.0 * jnp.power(r, 16.0)) + 8672948096.0 * jnp.power(M, 18.0) * jnp.power(q, 5.0) / (765765.0 * jnp.power(r, 16.0)) - 66732268672.0 * jnp.power(M, 18.0) * jnp.power(q, 7.0) / (8933925.0 * jnp.power(r, 16.0)) + 69925152512.0 * jnp.power(M, 18.0) * jnp.power(q, 9.0) / (34459425.0 * jnp.power(r, 16.0)) - 58660306688.0 * jnp.power(M, 18.0) * jnp.power(q, 11.0) / (241215975.0 * jnp.power(r, 16.0)) + 222168704.0 * jnp.power(M, 18.0) * jnp.power(q, 13.0) / (18555075.0 * jnp.power(r, 16.0)) - 24704.0 * jnp.power(M, 18.0) * jnp.power(q, 15.0) / (153153.0 * jnp.power(r, 16.0)) - 219648.0 * jnp.power(M, 17.0) * jnp.power(q, 3.0) / (85.0 * jnp.power(r, 15.0)) + 11074592.0 * jnp.power(M, 17.0) * jnp.power(q, 5.0) / (2431.0 * jnp.power(r, 15.0)) - 1480173854.0 * jnp.power(M, 17.0) * jnp.power(q, 7.0) / (595595.0 * jnp.power(r, 15.0)) + 183740283.0 * jnp.power(M, 17.0) * jnp.power(q, 9.0) / (340340.0 * jnp.power(r, 15.0)) - 938679283.0 * jnp.power(M, 17.0) * jnp.power(q, 11.0) / (19059040.0 * jnp.power(r, 15.0)) + 244740253.0 * jnp.power(M, 17.0) * jnp.power(q, 13.0) / (152472320.0 * jnp.power(r, 15.0)) - 693.0 * jnp.power(M, 17.0) * jnp.power(q, 15.0) / (87040.0 * jnp.power(r, 15.0)) - 8448.0 * jnp.power(M, 16.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 14.0)) + 8929728.0 * jnp.power(M, 16.0) * jnp.power(q, 5.0) / (5005.0 * jnp.power(r, 14.0)) - 7099640.0 * jnp.power(M, 16.0) * jnp.power(q, 7.0) / (9009.0 * jnp.power(r, 14.0)) + 3280691.0 * jnp.power(M, 16.0) * jnp.power(q, 9.0) / (24570.0 * jnp.power(r, 14.0)) - 1198264.0 * jnp.power(M, 16.0) * jnp.power(q, 11.0) / (135135.0 * jnp.power(r, 14.0)) + 488.0 * jnp.power(M, 16.0) * jnp.power(q, 13.0) / (3003.0 * jnp.power(r, 14.0)) - 50688.0 * jnp.power(M, 15.0) * jnp.power(q, 3.0) / (91.0 * jnp.power(r, 13.0)) + 71072272.0 * jnp.power(M, 15.0) * jnp.power(q, 5.0) / (105105.0 * jnp.power(r, 13.0)) - 10623001.0 * jnp.power(M, 15.0) * jnp.power(q, 7.0) / (45045.0 * jnp.power(r, 13.0)) + 5873381.0 * jnp.power(M, 15.0) * jnp.power(q, 9.0) / (194040.0 * jnp.power(r, 13.0)) - 27428833.0 * jnp.power(M, 15.0) * jnp.power(q, 11.0) / (20180160.0 * jnp.power(r, 13.0)) + 63.0 * jnp.power(M, 15.0) * jnp.power(q, 13.0) / (6656.0 * jnp.power(r, 13.0)) - 23040.0 * jnp.power(M, 14.0) * jnp.power(q, 3.0) / (91.0 * jnp.power(r, 12.0)) + 1723584.0 * jnp.power(M, 14.0) * jnp.power(q, 5.0) / (7007.0 * jnp.power(r, 12.0)) - 329328.0 * jnp.power(M, 14.0) * jnp.power(q, 7.0) / (5005.0 * jnp.power(r, 12.0)) + 216144.0 * jnp.power(M, 14.0) * jnp.power(q, 9.0) / (35035.0 * jnp.power(r, 12.0)) - 5664.0 * jnp.power(M, 14.0) * jnp.power(q, 11.0) / (35035.0 * jnp.power(r, 12.0)) - 16128.0 * jnp.power(M, 13.0) * jnp.power(q, 3.0) / (143.0 * jnp.power(r, 11.0)) + 1275212.0 * jnp.power(M, 13.0) * jnp.power(q, 5.0) / (15015.0 * jnp.power(r, 11.0)) - 253378.0 * jnp.power(M, 13.0) * jnp.power(q, 7.0) / (15015.0 * jnp.power(r, 11.0)) + 87641.0 * jnp.power(M, 13.0) * jnp.power(q, 9.0) / (80080.0 * jnp.power(r, 11.0)) - 105.0 * jnp.power(M, 13.0) * jnp.power(q, 11.0) / (9152.0 * jnp.power(r, 11.0)) - 2688.0 * jnp.power(M, 12.0) * jnp.power(q, 3.0) / (55.0 * jnp.power(r, 10.0)) + 6332.0 * jnp.power(M, 12.0) * jnp.power(q, 5.0) / (231.0 * jnp.power(r, 10.0)) - 9047.0 * jnp.power(M, 12.0) * jnp.power(q, 7.0) / (2310.0 * jnp.power(r, 10.0)) + 181.0 * jnp.power(M, 12.0) * jnp.power(q, 9.0) / (1155.0 * jnp.power(r, 10.0)) - 224.0 * jnp.power(M, 11.0) * jnp.power(q, 3.0) / (11.0 * jnp.power(r, 9.0)) + 624.0 * jnp.power(M, 11.0) * jnp.power(q, 5.0) / (77.0 * jnp.power(r, 9.0)) - 249.0 * jnp.power(M, 11.0) * jnp.power(q, 7.0) / (308.0 * jnp.power(r, 9.0)) + 5.0 * jnp.power(M, 11.0) * jnp.power(q, 9.0) / (352.0 * jnp.power(r, 9.0)) - 8.0 * jnp.power(M, 10.0) * jnp.power(q, 3.0) / jnp.power(r, 8.0) + 15.0 * jnp.power(M, 10.0) * jnp.power(q, 5.0) / (7.0 * jnp.power(r, 8.0)) - jnp.power(M, 10.0) * jnp.power(q, 7.0) / (7.0 * jnp.power(r, 8.0)) - 20.0 * jnp.power(M, 9.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 7.0)) + jnp.power(M, 9.0) * jnp.power(q, 5.0) / (2.0 * jnp.power(r, 7.0)) - jnp.power(M, 9.0) * jnp.power(q, 7.0) / (56.0 * jnp.power(r, 7.0)) - 6.0 * jnp.power(M, 8.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 6.0)) + 3.0 * jnp.power(M, 8.0) * jnp.power(q, 5.0) / (28.0 * jnp.power(r, 6.0)) - 6.0 * jnp.power(M, 7.0) * jnp.power(q, 3.0) / (35.0 * jnp.power(r, 5.0)) + 3.0 * jnp.power(M, 7.0) * jnp.power(q, 5.0) / (140.0 * jnp.power(r, 5.0))


def H0OnlyQS_jax(M, q, r):
    return 373555200.0 * jnp.power(M, 19.0) * q / (253.0 * jnp.power(r, 22.0)) - 7929632358400.0 * jnp.power(M, 19.0) * jnp.power(q, 3.0) / (1312311.0 * jnp.power(r, 22.0)) + 3.681060935938816e+16 * jnp.power(M, 19.0) * jnp.power(q, 5.0) / (4216455243.0 * jnp.power(r, 22.0)) - 2.68366322093152e+16 * jnp.power(M, 19.0) * jnp.power(q, 7.0) / (4660292637.0 * jnp.power(r, 22.0)) + 1.2382500425031748e+19 * jnp.power(M, 19.0) * jnp.power(q, 9.0) / (6640917007725.0 * jnp.power(r, 22.0)) - 5.880808994409967e+18 * jnp.power(M, 19.0) * jnp.power(q, 11.0) / (19922751023175.0 * jnp.power(r, 22.0)) + 4.247249880139406e+17 * jnp.power(M, 19.0) * jnp.power(q, 13.0) / (19922751023175.0 * jnp.power(r, 22.0)) - 1311359102092288.0 * jnp.power(M, 19.0) * jnp.power(q, 15.0) / (2213639002575.0 * jnp.power(r, 22.0)) + 30408704.0 * jnp.power(M, 19.0) * jnp.power(q, 17.0) / (7436429.0 * jnp.power(r, 22.0)) + 1270087680.0 * jnp.power(M, 18.0) * q / (1771.0 * jnp.power(r, 21.0)) - 216993318260736.0 * jnp.power(M, 18.0) * jnp.power(q, 3.0) / (81800719.0 * jnp.power(r, 21.0)) + 45830526088832.0 * jnp.power(M, 18.0) * jnp.power(q, 5.0) / (13483635.0 * jnp.power(r, 21.0)) - 1699774817359024.0 * jnp.power(M, 18.0) * jnp.power(q, 7.0) / (876436275.0 * jnp.power(r, 21.0)) + 919918368991283.0 * jnp.power(M, 18.0) * jnp.power(q, 9.0) / (1752872550.0 * jnp.power(r, 21.0)) - 231217475648777.0 * jnp.power(M, 18.0) * jnp.power(q, 11.0) / (3505745100.0 * jnp.power(r, 21.0)) + 194652894683933.0 * jnp.power(M, 18.0) * jnp.power(q, 13.0) / (56091921600.0 * jnp.power(r, 21.0)) - 926258551895.0 * jnp.power(M, 18.0) * jnp.power(q, 15.0) / (15705738048.0 * jnp.power(r, 21.0)) + 182325.0 * jnp.power(M, 18.0) * jnp.power(q, 17.0) / (1318912.0 * jnp.power(r, 21.0)) + 26738688.0 * jnp.power(M, 17.0) * q / (77.0 * jnp.power(r, 20.0)) - 86063947988992.0 * jnp.power(M, 17.0) * jnp.power(q, 3.0) / (74687613.0 * jnp.power(r, 20.0)) + 1.8296328174075904e+16 * jnp.power(M, 17.0) * jnp.power(q, 5.0) / (14115958857.0 * jnp.power(r, 20.0)) - 8901081866362880.0 * jnp.power(M, 17.0) * jnp.power(q, 7.0) / (14115958857.0 * jnp.power(r, 20.0)) + 1.77276099024896e+16 * jnp.power(M, 17.0) * jnp.power(q, 9.0) / (127043629713.0 * jnp.power(r, 20.0)) - 189629887700992.0 * jnp.power(M, 17.0) * jnp.power(q, 11.0) / (14115958857.0 * jnp.power(r, 20.0)) + 6789171617792.0 * jnp.power(M, 17.0) * jnp.power(q, 13.0) / (14115958857.0 * jnp.power(r, 20.0)) - 133431296.0 * jnp.power(M, 17.0) * jnp.power(q, 15.0) / (32008977.0 * jnp.power(r, 20.0)) + 22282240.0 * jnp.power(M, 16.0) * q / (133.0 * jnp.power(r, 19.0)) - 10072196525056.0 * jnp.power(M, 16.0) * jnp.power(q, 3.0) / (20369349.0 * jnp.power(r, 19.0)) + 3095539652002624.0 * jnp.power(M, 16.0) * jnp.power(q, 5.0) / (6416344935.0 * jnp.power(r, 19.0)) - 252355863522596.0 * jnp.power(M, 16.0) * jnp.power(q, 7.0) / (1283268987.0 * jnp.power(r, 19.0)) + 363573767832881.0 * jnp.power(M, 16.0) * jnp.power(q, 9.0) / (10499473530.0 * jnp.power(r, 19.0)) - 2248987038532789.0 * jnp.power(M, 16.0) * jnp.power(q, 11.0) / (923953670640.0 * jnp.power(r, 19.0)) + 8772734045155.0 * jnp.power(M, 16.0) * jnp.power(q, 13.0) / (164258430336.0 * jnp.power(r, 19.0)) - 10725.0 * jnp.power(M, 16.0) * jnp.power(q, 15.0) / (68096.0 * jnp.power(r, 19.0)) + 4587520.0 * jnp.power(M, 15.0) * q / (57.0 * jnp.power(r, 18.0)) - 67632636928.0 * jnp.power(M, 15.0) * jnp.power(q, 3.0) / (323323.0 * jnp.power(r, 18.0)) + 1974936720256.0 * jnp.power(M, 15.0) * jnp.power(q, 5.0) / (11316305.0 * jnp.power(r, 18.0)) - 1980360362576.0 * jnp.power(M, 15.0) * jnp.power(q, 7.0) / (33948915.0 * jnp.power(r, 18.0)) + 1821878944.0 * jnp.power(M, 15.0) * jnp.power(q, 9.0) / (230945.0 * jnp.power(r, 18.0)) - 4280066432.0 * jnp.power(M, 15.0) * jnp.power(q, 11.0) / (11316305.0 * jnp.power(r, 18.0)) + 587776.0 * jnp.power(M, 15.0) * jnp.power(q, 13.0) / (138567.0 * jnp.power(r, 18.0)) + 37273600.0 * jnp.power(M, 14.0) * q / (969.0 * jnp.power(r, 17.0)) - 759568870400.0 * jnp.power(M, 14.0) * jnp.power(q, 3.0) / (8729721.0 * jnp.power(r, 17.0)) + 3728453306272.0 * jnp.power(M, 14.0) * jnp.power(q, 5.0) / (61108047.0 * jnp.power(r, 17.0)) - 6393714739124.0 * jnp.power(M, 14.0) * jnp.power(q, 7.0) / (392837445.0 * jnp.power(r, 17.0)) + 1363525754411.0 * jnp.power(M, 14.0) * jnp.power(q, 9.0) / (846111420.0 * jnp.power(r, 17.0)) - 232077111443.0 * jnp.power(M, 14.0) * jnp.power(q, 11.0) / (4888643760.0 * jnp.power(r, 17.0)) + 15015.0 * jnp.power(M, 14.0) * jnp.power(q, 13.0) / (82688.0 * jnp.power(r, 17.0)) + 931840.0 * jnp.power(M, 13.0) * q / (51.0 * jnp.power(r, 16.0)) - 16296527872.0 * jnp.power(M, 13.0) * jnp.power(q, 3.0) / (459459.0 * jnp.power(r, 16.0)) + 329096302592.0 * jnp.power(M, 13.0) * jnp.power(q, 5.0) / (16081065.0 * jnp.power(r, 16.0)) - 608718794752.0 * jnp.power(M, 13.0) * jnp.power(q, 7.0) / (144729585.0 * jnp.power(r, 16.0)) + 41230764032.0 * jnp.power(M, 13.0) * jnp.power(q, 9.0) / (144729585.0 * jnp.power(r, 16.0)) - 1976320.0 * jnp.power(M, 13.0) * jnp.power(q, 11.0) / (459459.0 * jnp.power(r, 16.0)) + 146432.0 * jnp.power(M, 12.0) * q / (17.0 * jnp.power(r, 15.0)) - 102893504.0 * jnp.power(M, 12.0) * jnp.power(q, 3.0) / (7293.0 * jnp.power(r, 15.0)) + 776708332.0 * jnp.power(M, 12.0) * jnp.power(q, 5.0) / (119119.0 * jnp.power(r, 15.0)) - 117275122.0 * jnp.power(M, 12.0) * jnp.power(q, 7.0) / (119119.0 * jnp.power(r, 15.0)) + 235028551.0 * jnp.power(M, 12.0) * jnp.power(q, 9.0) / (5717712.0 * jnp.power(r, 15.0)) - 231.0 * jnp.power(M, 12.0) * jnp.power(q, 11.0) / (1088.0 * jnp.power(r, 15.0)) + 28160.0 * jnp.power(M, 11.0) * q / (7.0 * jnp.power(r, 14.0)) - 5449792.0 * jnp.power(M, 11.0) * jnp.power(q, 3.0) / (1001.0 * jnp.power(r, 14.0)) + 52603352.0 * jnp.power(M, 11.0) * jnp.power(q, 5.0) / (27027.0 * jnp.power(r, 14.0)) - 16361344.0 * jnp.power(M, 11.0) * jnp.power(q, 7.0) / (81081.0 * jnp.power(r, 14.0)) + 39040.0 * jnp.power(M, 11.0) * jnp.power(q, 9.0) / (9009.0 * jnp.power(r, 14.0)) + 168960.0 * jnp.power(M, 10.0) * q / (91.0 * jnp.power(r, 13.0)) - 127508384.0 * jnp.power(M, 10.0) * jnp.power(q, 3.0) / (63063.0 * jnp.power(r, 13.0)) + 100906370.0 * jnp.power(M, 10.0) * jnp.power(q, 5.0) / (189189.0 * jnp.power(r, 13.0)) - 6475192.0 * jnp.power(M, 10.0) * jnp.power(q, 7.0) / (189189.0 * jnp.power(r, 13.0)) + 105.0 * jnp.power(M, 10.0) * jnp.power(q, 9.0) / (416.0 * jnp.power(r, 13.0)) + 76800.0 * jnp.power(M, 9.0) * q / (91.0 * jnp.power(r, 12.0)) - 5006080.0 * jnp.power(M, 9.0) * jnp.power(q, 3.0) / (7007.0 * jnp.power(r, 12.0)) + 911104.0 * jnp.power(M, 9.0) * jnp.power(q, 5.0) / (7007.0 * jnp.power(r, 12.0)) - 30208.0 * jnp.power(M, 9.0) * jnp.power(q, 7.0) / (7007.0 * jnp.power(r, 12.0)) + 53760.0 * jnp.power(M, 8.0) * q / (143.0 * jnp.power(r, 11.0)) - 2127064.0 * jnp.power(M, 8.0) * jnp.power(q, 3.0) / (9009.0 * jnp.power(r, 11.0)) + 80291.0 * jnp.power(M, 8.0) * jnp.power(q, 5.0) / (3003.0 * jnp.power(r, 11.0)) - 175.0 * jnp.power(M, 8.0) * jnp.power(q, 7.0) / (572.0 * jnp.power(r, 11.0)) + 1792.0 * jnp.power(M, 7.0) * q / (11.0 * jnp.power(r, 10.0)) - 49208.0 * jnp.power(M, 7.0) * jnp.power(q, 3.0) / (693.0 * jnp.power(r, 10.0)) + 2896.0 * jnp.power(M, 7.0) * jnp.power(q, 5.0) / (693.0 * jnp.power(r, 10.0)) + 2240.0 * jnp.power(M, 6.0) * q / (33.0 * jnp.power(r, 9.0)) - 4280.0 * jnp.power(M, 6.0) * jnp.power(q, 3.0) / (231.0 * jnp.power(r, 9.0)) + 25.0 * jnp.power(M, 6.0) * jnp.power(q, 5.0) / (66.0 * jnp.power(r, 9.0)) + 80.0 * jnp.power(M, 5.0) * q / (3.0 * jnp.power(r, 8.0)) - 80.0 * jnp.power(M, 5.0) * jnp.power(q, 3.0) / (21.0 * jnp.power(r, 8.0)) + 200.0 * jnp.power(M, 4.0) * q / (21.0 * jnp.power(r, 7.0)) - 10.0 * jnp.power(M, 4.0) * jnp.power(q, 3.0) / (21.0 * jnp.power(r, 7.0)) + 20.0 * jnp.power(M, 3.0) * q / (7.0 * jnp.power(r, 6.0)) + 4.0 * jnp.power(M, 2.0) * q / (7.0 * jnp.power(r, 5.0))


def PhiPOnlyQS_jax(M, q, r):
    return 137625600.0 * jnp.power(M, 19.0) / (253.0 * jnp.power(r, 22.0)) - 25044285849600.0 * jnp.power(M, 19.0) * jnp.power(q, 2.0) / (7436429.0 * jnp.power(r, 22.0)) + 1.0386679483380224e+16 * jnp.power(M, 19.0) * jnp.power(q, 4.0) / (1405485081.0 * jnp.power(r, 22.0)) - 1.1083481260520846e+18 * jnp.power(M, 19.0) * jnp.power(q, 6.0) / (147575933505.0 * jnp.power(r, 22.0)) + 4.276358632435258e+19 * jnp.power(M, 19.0) * jnp.power(q, 8.0) / (11068195012875.0 * jnp.power(r, 22.0)) - 3.4122519275204542e+19 * jnp.power(M, 19.0) * jnp.power(q, 10.0) / (33204585038625.0 * jnp.power(r, 22.0)) + 4.5715105084355814e+18 * jnp.power(M, 19.0) * jnp.power(q, 12.0) / (33204585038625.0 * jnp.power(r, 22.0)) - 8.537907977396265e+17 * jnp.power(M, 19.0) * jnp.power(q, 14.0) / (99613755115875.0 * jnp.power(r, 22.0)) + 459851552390656.0 * jnp.power(M, 19.0) * jnp.power(q, 16.0) / (2213639002575.0 * jnp.power(r, 22.0)) - 65536.0 * jnp.power(M, 19.0) * jnp.power(q, 18.0) / (52003.0 * jnp.power(r, 22.0)) + 498073600.0 * jnp.power(M, 18.0) / (1771.0 * jnp.power(r, 21.0)) - 130349638000640.0 * jnp.power(M, 18.0) * jnp.power(q, 2.0) / (81800719.0 * jnp.power(r, 21.0)) + 331514716669952.0 * jnp.power(M, 18.0) * jnp.power(q, 4.0) / (105172353.0 * jnp.power(r, 21.0)) - 165519445666736.0 * jnp.power(M, 18.0) * jnp.power(q, 6.0) / (58429085.0 * jnp.power(r, 21.0)) + 5507212513696126.0 * jnp.power(M, 18.0) * jnp.power(q, 8.0) / (4382181375.0 * jnp.power(r, 21.0)) - 5.856179718704707e+16 * jnp.power(M, 18.0) * jnp.power(q, 10.0) / (210344706000.0 * jnp.power(r, 21.0)) + 1378981216484473.0 * jnp.power(M, 18.0) * jnp.power(q, 12.0) / (46743268000.0 * jnp.power(r, 21.0)) - 2.093033358474397e+16 * jnp.power(M, 18.0) * jnp.power(q, 14.0) / (15705738048000.0 * jnp.power(r, 21.0)) + 14826685826357.0 * jnp.power(M, 18.0) * jnp.power(q, 16.0) / (753875426304.0 * jnp.power(r, 21.0)) - 60775.0 * jnp.power(M, 18.0) * jnp.power(q, 18.0) / (1507328.0 * jnp.power(r, 21.0)) + 11206656.0 * jnp.power(M, 17.0) / (77.0 * jnp.power(r, 20.0)) - 18678269390848.0 * jnp.power(M, 17.0) * jnp.power(q, 2.0) / (24895871.0 * jnp.power(r, 20.0)) + 6246198807072256.0 * jnp.power(M, 17.0) * jnp.power(q, 4.0) / (4705319619.0 * jnp.power(r, 20.0)) - 1.473649069109248e+16 * jnp.power(M, 17.0) * jnp.power(q, 6.0) / (14115958857.0 * jnp.power(r, 20.0)) + 1.66641507094784e+16 * jnp.power(M, 17.0) * jnp.power(q, 8.0) / (42347876571.0 * jnp.power(r, 20.0)) - 9022651836532736.0 * jnp.power(M, 17.0) * jnp.power(q, 10.0) / (127043629713.0 * jnp.power(r, 20.0)) + 243121395256832.0 * jnp.power(M, 17.0) * jnp.power(q, 12.0) / (42347876571.0 * jnp.power(r, 20.0)) - 2479639181312.0 * jnp.power(M, 17.0) * jnp.power(q, 14.0) / (14115958857.0 * jnp.power(r, 20.0)) + 42106880.0 * jnp.power(M, 17.0) * jnp.power(q, 16.0) / (32008977.0 * jnp.power(r, 20.0)) + 10027008.0 * jnp.power(M, 16.0) / (133.0 * jnp.power(r, 19.0)) - 3974494584832.0 * jnp.power(M, 16.0) * jnp.power(q, 2.0) / (11316305.0 * jnp.power(r, 19.0)) + 5891336352527072.0 * jnp.power(M, 16.0) * jnp.power(q, 4.0) / (10693908225.0 * jnp.power(r, 19.0)) - 6.006610594366005e+16 * jnp.power(M, 16.0) * jnp.power(q, 6.0) / (160408623375.0 * jnp.power(r, 19.0)) + 4.534986186885946e+17 * jnp.power(M, 16.0) * jnp.power(q, 8.0) / (3849806961000.0 * jnp.power(r, 19.0)) - 3.8909336700647725e+17 * jnp.power(M, 16.0) * jnp.power(q, 10.0) / (23098841766000.0 * jnp.power(r, 19.0)) + 3.6571209266836563e+17 * jnp.power(M, 16.0) * jnp.power(q, 12.0) / (369581468256000.0 * jnp.power(r, 19.0)) - 24255807439991.0 * jnp.power(M, 16.0) * jnp.power(q, 14.0) / (1314067442688.0 * jnp.power(r, 19.0)) + 10725.0 * jnp.power(M, 16.0) * jnp.power(q, 16.0) / (229376.0 * jnp.power(r, 19.0)) + 2228224.0 * jnp.power(M, 15.0) / (57.0 * jnp.power(r, 18.0)) - 264027403264.0 * jnp.power(M, 15.0) * jnp.power(q, 2.0) / (1616615.0 * jnp.power(r, 18.0)) + 1816686703232.0 * jnp.power(M, 15.0) * jnp.power(q, 4.0) / (8083075.0 * jnp.power(r, 18.0)) - 15774907358576.0 * jnp.power(M, 15.0) * jnp.power(q, 6.0) / (121246125.0 * jnp.power(r, 18.0)) + 1350428153472.0 * jnp.power(M, 15.0) * jnp.power(q, 8.0) / (40415375.0 * jnp.power(r, 18.0)) - 11352273008.0 * jnp.power(M, 15.0) * jnp.power(q, 10.0) / (3108875.0 * jnp.power(r, 18.0)) + 17642708288.0 * jnp.power(M, 15.0) * jnp.power(q, 12.0) / (121246125.0 * jnp.power(r, 18.0)) - 318464.0 * jnp.power(M, 15.0) * jnp.power(q, 14.0) / (230945.0 * jnp.power(r, 18.0)) + 6553600.0 * jnp.power(M, 14.0) / (323.0 * jnp.power(r, 17.0)) - 219264332288.0 * jnp.power(M, 14.0) * jnp.power(q, 2.0) / (2909907.0 * jnp.power(r, 17.0)) + 27457227271648.0 * jnp.power(M, 14.0) * jnp.power(q, 4.0) / (305540235.0 * jnp.power(r, 17.0)) - 199469520616378.0 * jnp.power(M, 14.0) * jnp.power(q, 6.0) / (4583103525.0 * jnp.power(r, 17.0)) + 488099722330459.0 * jnp.power(M, 14.0) * jnp.power(q, 8.0) / (54997242300.0 * jnp.power(r, 17.0)) - 77665594008593.0 * jnp.power(M, 14.0) * jnp.power(q, 10.0) / (109994484600.0 * jnp.power(r, 17.0)) + 1343666476931.0 * jnp.power(M, 14.0) * jnp.power(q, 12.0) / (78218300160.0 * jnp.power(r, 17.0)) - 2145.0 * jnp.power(M, 14.0) * jnp.power(q, 14.0) / (38912.0 * jnp.power(r, 17.0)) + 179200.0 * jnp.power(M, 13.0) / (17.0 * jnp.power(r, 16.0)) - 5274612736.0 * jnp.power(M, 13.0) * jnp.power(q, 2.0) / (153153.0 * jnp.power(r, 16.0)) + 564114925568.0 * jnp.power(M, 13.0) * jnp.power(q, 4.0) / (16081065.0 * jnp.power(r, 16.0)) - 479162181632.0 * jnp.power(M, 13.0) * jnp.power(q, 6.0) / (34459425.0 * jnp.power(r, 16.0)) + 1571540384768.0 * jnp.power(M, 13.0) * jnp.power(q, 8.0) / (723647925.0 * jnp.power(r, 16.0)) - 84964553728.0 * jnp.power(M, 13.0) * jnp.power(q, 10.0) / (723647925.0 * jnp.power(r, 16.0)) + 667648.0 * jnp.power(M, 13.0) * jnp.power(q, 12.0) / (459459.0 * jnp.power(r, 16.0)) + 93184.0 * jnp.power(M, 12.0) / (17.0 * jnp.power(r, 15.0)) - 794581696.0 * jnp.power(M, 12.0) * jnp.power(q, 2.0) / (51051.0 * jnp.power(r, 15.0)) + 4753519492.0 * jnp.power(M, 12.0) * jnp.power(q, 4.0) / (357357.0 * jnp.power(r, 15.0)) - 142857509.0 * jnp.power(M, 12.0) * jnp.power(q, 6.0) / (34034.0 * jnp.power(r, 15.0)) + 341843419.0 * jnp.power(M, 12.0) * jnp.power(q, 8.0) / (714714.0 * jnp.power(r, 15.0)) - 723367669.0 * jnp.power(M, 12.0) * jnp.power(q, 10.0) / (45741696.0 * jnp.power(r, 15.0)) + 1155.0 * jnp.power(M, 12.0) * jnp.power(q, 12.0) / (17408.0 * jnp.power(r, 15.0)) + 19968.0 * jnp.power(M, 11.0) / (7.0 * jnp.power(r, 14.0)) - 34722112.0 * jnp.power(M, 11.0) * jnp.power(q, 2.0) / (5005.0 * jnp.power(r, 14.0)) + 1096169384.0 * jnp.power(M, 11.0) * jnp.power(q, 4.0) / (225225.0 * jnp.power(r, 14.0)) - 24147776.0 * jnp.power(M, 11.0) * jnp.power(q, 6.0) / (20475.0 * jnp.power(r, 14.0)) + 185505904.0 * jnp.power(M, 11.0) * jnp.power(q, 8.0) / (2027025.0 * jnp.power(r, 14.0)) - 1984.0 * jnp.power(M, 11.0) * jnp.power(q, 10.0) / (1287.0 * jnp.power(r, 14.0)) + 135168.0 * jnp.power(M, 10.0) / (91.0 * jnp.power(r, 13.0)) - 106489536.0 * jnp.power(M, 10.0) * jnp.power(q, 2.0) / (35035.0 * jnp.power(r, 13.0)) + 383469712.0 * jnp.power(M, 10.0) * jnp.power(q, 4.0) / (225225.0 * jnp.power(r, 13.0)) - 5709620857.0 * jnp.power(M, 10.0) * jnp.power(q, 6.0) / (18918900.0 * jnp.power(r, 13.0)) + 1085909833.0 * jnp.power(M, 10.0) * jnp.power(q, 8.0) / (75675600.0 * jnp.power(r, 13.0)) - 21.0 * jnp.power(M, 10.0) * jnp.power(q, 10.0) / (256.0 * jnp.power(r, 13.0)) + 70400.0 * jnp.power(M, 9.0) / (91.0 * jnp.power(r, 12.0)) - 27389728.0 * jnp.power(M, 9.0) * jnp.power(q, 2.0) / (21021.0 * jnp.power(r, 12.0)) + 2813376.0 * jnp.power(M, 9.0) * jnp.power(q, 4.0) / (5005.0 * jnp.power(r, 12.0)) - 2382944.0 * jnp.power(M, 9.0) * jnp.power(q, 6.0) / (35035.0 * jnp.power(r, 12.0)) + 34688.0 * jnp.power(M, 9.0) * jnp.power(q, 8.0) / (21021.0 * jnp.power(r, 12.0)) + 57600.0 * jnp.power(M, 8.0) / (143.0 * jnp.power(r, 11.0)) - 1630432.0 * jnp.power(M, 8.0) * jnp.power(q, 2.0) / (3003.0 * jnp.power(r, 11.0)) + 15476827.0 * jnp.power(M, 8.0) * jnp.power(q, 4.0) / (90090.0 * jnp.power(r, 11.0)) - 511029.0 * jnp.power(M, 8.0) * jnp.power(q, 6.0) / (40040.0 * jnp.power(r, 11.0)) + 175.0 * jnp.power(M, 8.0) * jnp.power(q, 8.0) / (1664.0 * jnp.power(r, 11.0)) + 2304.0 * jnp.power(M, 7.0) / (11.0 * jnp.power(r, 10.0)) - 50312.0 * jnp.power(M, 7.0) * jnp.power(q, 2.0) / (231.0 * jnp.power(r, 10.0)) + 32672.0 * jnp.power(M, 7.0) * jnp.power(q, 4.0) / (693.0 * jnp.power(r, 10.0)) - 1240.0 * jnp.power(M, 7.0) * jnp.power(q, 6.0) / (693.0 * jnp.power(r, 10.0)) + 3584.0 * jnp.power(M, 6.0) / (33.0 * jnp.power(r, 9.0)) - 95716.0 * jnp.power(M, 6.0) * jnp.power(q, 2.0) / (1155.0 * jnp.power(r, 9.0)) + 25469.0 * jnp.power(M, 6.0) * jnp.power(q, 4.0) / (2310.0 * jnp.power(r, 9.0)) - 25.0 * jnp.power(M, 6.0) * jnp.power(q, 6.0) / (176.0 * jnp.power(r, 9.0)) + 56.0 * jnp.power(M, 5.0) / jnp.power(r, 8.0) - 3064.0 * jnp.power(M, 5.0) * jnp.power(q, 2.0) / (105.0 * jnp.power(r, 8.0)) + 208.0 * jnp.power(M, 5.0) * jnp.power(q, 4.0) / (105.0 * jnp.power(r, 8.0)) + 200.0 * jnp.power(M, 4.0) / (7.0 * jnp.power(r, 7.0)) - 191.0 * jnp.power(M, 4.0) * jnp.power(q, 2.0) / (21.0 * jnp.power(r, 7.0)) + 5.0 * jnp.power(M, 4.0) * jnp.power(q, 4.0) / (24.0 * jnp.power(r, 7.0)) + 100.0 * jnp.power(M, 3.0) / (7.0 * jnp.power(r, 6.0)) - 16.0 * jnp.power(M, 3.0) * jnp.power(q, 2.0) / (7.0 * jnp.power(r, 6.0)) + 48.0 * jnp.power(M, 2.0) / (7.0 * jnp.power(r, 5.0)) - 5.0 * jnp.power(M, 2.0) * jnp.power(q, 2.0) / (14.0 * jnp.power(r, 5.0)) + 3.0 * M / jnp.power(r, 4.0) + jnp.power(r, -3.0)


def H0OnlyES_jax(M, q, r):
    return -4.0 * jnp.power(M, 2.0) * q / 3.0 - 4194304.0 * jnp.power(M, 24.0) * q / (57.0 * jnp.power(r, 22.0)) - 432189449371648.0 * jnp.power(M, 24.0) * jnp.power(q, 3.0) / (423876453.0 * jnp.power(r, 22.0)) + 4.889210237699459e+18 * jnp.power(M, 24.0) * jnp.power(q, 5.0) / (1016814398985.0 * jnp.power(r, 22.0)) - 9.663973074671625e+21 * jnp.power(M, 24.0) * jnp.power(q, 7.0) / (1387951654614525.0 * jnp.power(r, 22.0)) + 4.549082926346561e+23 * jnp.power(M, 24.0) * jnp.power(q, 9.0) / (1.0409637409608938e+17 * jnp.power(r, 22.0)) - 1.56978117441629e+23 * jnp.power(M, 24.0) * jnp.power(q, 11.0) / (1.2491564891530725e+17 * jnp.power(r, 22.0)) + 1.2677820661047214e+22 * jnp.power(M, 24.0) * jnp.power(q, 13.0) / (8.8175752175511e+16 * jnp.power(r, 22.0)) + 8.765493299428505e+21 * jnp.power(M, 24.0) * jnp.power(q, 15.0) / (1.4390282755043396e+19 * jnp.power(r, 22.0)) - 1.3175059756885878e+22 * jnp.power(M, 24.0) * jnp.power(q, 17.0) / (1.0659468707439553e+19 * jnp.power(r, 22.0)) + 4.357109942093411e+19 * jnp.power(M, 24.0) * jnp.power(q, 19.0) / (5.8851449040384e+17 * jnp.power(r, 22.0)) - 3666737926669729.0 * jnp.power(M, 24.0) * jnp.power(q, 21.0) / (2946575951953920.0 * jnp.power(r, 22.0)) + 1547.0 * jnp.power(M, 24.0) * jnp.power(q, 23.0) / (524288.0 * jnp.power(r, 22.0)) - 1524105216.0 * jnp.power(M, 23.0) * jnp.power(q, 3.0) / (1771.0 * jnp.power(r, 21.0)) + 1345957969453056.0 * jnp.power(M, 23.0) * jnp.power(q, 5.0) / (409003595.0 * jnp.power(r, 21.0)) - 9154880691645184.0 * jnp.power(M, 23.0) * jnp.power(q, 7.0) / (2045017975.0 * jnp.power(r, 21.0)) + 218120825455872.0 * jnp.power(M, 23.0) * jnp.power(q, 9.0) / (76880375.0 * jnp.power(r, 21.0)) - 1344862073331039.0 * jnp.power(M, 23.0) * jnp.power(q, 11.0) / (1460727125.0 * jnp.power(r, 21.0)) + 1844788271586391.0 * jnp.power(M, 23.0) * jnp.power(q, 13.0) / (11685817000.0 * jnp.power(r, 21.0)) - 657087845981487.0 * jnp.power(M, 23.0) * jnp.power(q, 15.0) / (46743268000.0 * jnp.power(r, 21.0)) + 1547821973166531.0 * jnp.power(M, 23.0) * jnp.power(q, 17.0) / (2617623008000.0 * jnp.power(r, 21.0)) - 131287333403.0 * jnp.power(M, 23.0) * jnp.power(q, 19.0) / (14567641088.0 * jnp.power(r, 21.0)) + 109395.0 * jnp.power(M, 23.0) * jnp.power(q, 21.0) / (5275648.0 * jnp.power(r, 21.0)) - 160432128.0 * jnp.power(M, 22.0) * jnp.power(q, 3.0) / (385.0 * jnp.power(r, 20.0)) + 178611820593152.0 * jnp.power(M, 22.0) * jnp.power(q, 5.0) / (124479355.0 * jnp.power(r, 20.0)) - 8131835578126336.0 * jnp.power(M, 22.0) * jnp.power(q, 7.0) / (4705319619.0 * jnp.power(r, 20.0)) + 69276302712832.0 * jnp.power(M, 22.0) * jnp.power(q, 9.0) / (72837765.0 * jnp.power(r, 20.0)) - 1.1096530800859136e+16 * jnp.power(M, 22.0) * jnp.power(q, 11.0) / (42347876571.0 * jnp.power(r, 20.0)) + 461484732602368.0 * jnp.power(M, 22.0) * jnp.power(q, 13.0) / (12455257815.0 * jnp.power(r, 20.0)) - 60985815160832.0 * jnp.power(M, 22.0) * jnp.power(q, 15.0) / (23526598095.0 * jnp.power(r, 20.0)) + 362995861504.0 * jnp.power(M, 22.0) * jnp.power(q, 17.0) / (4705319619.0 * jnp.power(r, 20.0)) - 33357824.0 * jnp.power(M, 22.0) * jnp.power(q, 19.0) / (53348295.0 * jnp.power(r, 20.0)) - 26738688.0 * jnp.power(M, 21.0) * jnp.power(q, 3.0) / (133.0 * jnp.power(r, 19.0)) + 20997541025792.0 * jnp.power(M, 21.0) * jnp.power(q, 5.0) / (33948915.0 * jnp.power(r, 19.0)) - 6984264780353408.0 * jnp.power(M, 21.0) * jnp.power(q, 7.0) / (10693908225.0 * jnp.power(r, 19.0)) + 122127538823208.0 * jnp.power(M, 21.0) * jnp.power(q, 9.0) / (396070675.0 * jnp.power(r, 19.0)) - 526024223906992.0 * jnp.power(M, 21.0) * jnp.power(q, 11.0) / (7403474925.0 * jnp.power(r, 19.0)) + 8221445374598.0 * jnp.power(M, 21.0) * jnp.power(q, 13.0) / (1013107095.0 * jnp.power(r, 19.0)) - 660940017641191.0 * jnp.power(M, 21.0) * jnp.power(q, 15.0) / (1539922784400.0 * jnp.power(r, 19.0)) + 1795939470341.0 * jnp.power(M, 21.0) * jnp.power(q, 17.0) / (219011240448.0 * jnp.power(r, 19.0)) - 6435.0 * jnp.power(M, 21.0) * jnp.power(q, 19.0) / (272384.0 * jnp.power(r, 19.0)) - 1835008.0 * jnp.power(M, 20.0) * jnp.power(q, 3.0) / (19.0 * jnp.power(r, 18.0)) + 425312278528.0 * jnp.power(M, 20.0) * jnp.power(q, 5.0) / (1616615.0 * jnp.power(r, 18.0)) - 55161850368.0 * jnp.power(M, 20.0) * jnp.power(q, 7.0) / (229075.0 * jnp.power(r, 18.0)) + 5441923265344.0 * jnp.power(M, 20.0) * jnp.power(q, 9.0) / (56581525.0 * jnp.power(r, 18.0)) - 206144500036.0 * jnp.power(M, 20.0) * jnp.power(q, 11.0) / (11316305.0 * jnp.power(r, 18.0)) + 92634449784.0 * jnp.power(M, 20.0) * jnp.power(q, 13.0) / (56581525.0 * jnp.power(r, 18.0)) - 3498060064.0 * jnp.power(M, 20.0) * jnp.power(q, 15.0) / (56581525.0 * jnp.power(r, 18.0)) + 146944.0 * jnp.power(M, 20.0) * jnp.power(q, 17.0) / (230945.0 * jnp.power(r, 18.0)) - 14909440.0 * jnp.power(M, 19.0) * jnp.power(q, 3.0) / (323.0 * jnp.power(r, 17.0)) + 320617441280.0 * jnp.power(M, 19.0) * jnp.power(q, 5.0) / (2909907.0 * jnp.power(r, 17.0)) - 462429059776.0 * jnp.power(M, 19.0) * jnp.power(q, 7.0) / (5360355.0 * jnp.power(r, 17.0)) + 11950646003936.0 * jnp.power(M, 19.0) * jnp.power(q, 9.0) / (416645775.0 * jnp.power(r, 17.0)) - 13367945464759.0 * jnp.power(M, 19.0) * jnp.power(q, 11.0) / (3055402350.0 * jnp.power(r, 17.0)) + 21903222813317.0 * jnp.power(M, 19.0) * jnp.power(q, 13.0) / (73329656400.0 * jnp.power(r, 17.0)) - 478357587061.0 * jnp.power(M, 19.0) * jnp.power(q, 15.0) / (65181916800.0 * jnp.power(r, 17.0)) + 9009.0 * jnp.power(M, 19.0) * jnp.power(q, 17.0) / (330752.0 * jnp.power(r, 17.0)) - 372736.0 * jnp.power(M, 18.0) * jnp.power(q, 3.0) / (17.0 * jnp.power(r, 16.0)) + 34691792384.0 * jnp.power(M, 18.0) * jnp.power(q, 5.0) / (765765.0 * jnp.power(r, 16.0)) - 266929074688.0 * jnp.power(M, 18.0) * jnp.power(q, 7.0) / (8933925.0 * jnp.power(r, 16.0)) + 279700610048.0 * jnp.power(M, 18.0) * jnp.power(q, 9.0) / (34459425.0 * jnp.power(r, 16.0)) - 234641226752.0 * jnp.power(M, 18.0) * jnp.power(q, 11.0) / (241215975.0 * jnp.power(r, 16.0)) + 888674816.0 * jnp.power(M, 18.0) * jnp.power(q, 13.0) / (18555075.0 * jnp.power(r, 16.0)) - 98816.0 * jnp.power(M, 18.0) * jnp.power(q, 15.0) / (153153.0 * jnp.power(r, 16.0)) - 878592.0 * jnp.power(M, 17.0) * jnp.power(q, 3.0) / (85.0 * jnp.power(r, 15.0)) + 44298368.0 * jnp.power(M, 17.0) * jnp.power(q, 5.0) / (2431.0 * jnp.power(r, 15.0)) - 5920695416.0 * jnp.power(M, 17.0) * jnp.power(q, 7.0) / (595595.0 * jnp.power(r, 15.0)) + 183740283.0 * jnp.power(M, 17.0) * jnp.power(q, 9.0) / (85085.0 * jnp.power(r, 15.0)) - 938679283.0 * jnp.power(M, 17.0) * jnp.power(q, 11.0) / (4764760.0 * jnp.power(r, 15.0)) + 244740253.0 * jnp.power(M, 17.0) * jnp.power(q, 13.0) / (38118080.0 * jnp.power(r, 15.0)) - 693.0 * jnp.power(M, 17.0) * jnp.power(q, 15.0) / (21760.0 * jnp.power(r, 15.0)) - 33792.0 * jnp.power(M, 16.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 14.0)) + 35718912.0 * jnp.power(M, 16.0) * jnp.power(q, 5.0) / (5005.0 * jnp.power(r, 14.0)) - 28398560.0 * jnp.power(M, 16.0) * jnp.power(q, 7.0) / (9009.0 * jnp.power(r, 14.0)) + 6561382.0 * jnp.power(M, 16.0) * jnp.power(q, 9.0) / (12285.0 * jnp.power(r, 14.0)) - 4793056.0 * jnp.power(M, 16.0) * jnp.power(q, 11.0) / (135135.0 * jnp.power(r, 14.0)) + 1952.0 * jnp.power(M, 16.0) * jnp.power(q, 13.0) / (3003.0 * jnp.power(r, 14.0)) - 202752.0 * jnp.power(M, 15.0) * jnp.power(q, 3.0) / (91.0 * jnp.power(r, 13.0)) + 284289088.0 * jnp.power(M, 15.0) * jnp.power(q, 5.0) / (105105.0 * jnp.power(r, 13.0)) - 42492004.0 * jnp.power(M, 15.0) * jnp.power(q, 7.0) / (45045.0 * jnp.power(r, 13.0)) + 5873381.0 * jnp.power(M, 15.0) * jnp.power(q, 9.0) / (48510.0 * jnp.power(r, 13.0)) - 27428833.0 * jnp.power(M, 15.0) * jnp.power(q, 11.0) / (5045040.0 * jnp.power(r, 13.0)) + 63.0 * jnp.power(M, 15.0) * jnp.power(q, 13.0) / (1664.0 * jnp.power(r, 13.0)) - 92160.0 * jnp.power(M, 14.0) * jnp.power(q, 3.0) / (91.0 * jnp.power(r, 12.0)) + 6894336.0 * jnp.power(M, 14.0) * jnp.power(q, 5.0) / (7007.0 * jnp.power(r, 12.0)) - 1317312.0 * jnp.power(M, 14.0) * jnp.power(q, 7.0) / (5005.0 * jnp.power(r, 12.0)) + 864576.0 * jnp.power(M, 14.0) * jnp.power(q, 9.0) / (35035.0 * jnp.power(r, 12.0)) - 22656.0 * jnp.power(M, 14.0) * jnp.power(q, 11.0) / (35035.0 * jnp.power(r, 12.0)) - 64512.0 * jnp.power(M, 13.0) * jnp.power(q, 3.0) / (143.0 * jnp.power(r, 11.0)) + 5100848.0 * jnp.power(M, 13.0) * jnp.power(q, 5.0) / (15015.0 * jnp.power(r, 11.0)) - 1013512.0 * jnp.power(M, 13.0) * jnp.power(q, 7.0) / (15015.0 * jnp.power(r, 11.0)) + 87641.0 * jnp.power(M, 13.0) * jnp.power(q, 9.0) / (20020.0 * jnp.power(r, 11.0)) - 105.0 * jnp.power(M, 13.0) * jnp.power(q, 11.0) / (2288.0 * jnp.power(r, 11.0)) - 10752.0 * jnp.power(M, 12.0) * jnp.power(q, 3.0) / (55.0 * jnp.power(r, 10.0)) + 25328.0 * jnp.power(M, 12.0) * jnp.power(q, 5.0) / (231.0 * jnp.power(r, 10.0)) - 18094.0 * jnp.power(M, 12.0) * jnp.power(q, 7.0) / (1155.0 * jnp.power(r, 10.0)) + 724.0 * jnp.power(M, 12.0) * jnp.power(q, 9.0) / (1155.0 * jnp.power(r, 10.0)) - 896.0 * jnp.power(M, 11.0) * jnp.power(q, 3.0) / (11.0 * jnp.power(r, 9.0)) + 2496.0 * jnp.power(M, 11.0) * jnp.power(q, 5.0) / (77.0 * jnp.power(r, 9.0)) - 249.0 * jnp.power(M, 11.0) * jnp.power(q, 7.0) / (77.0 * jnp.power(r, 9.0)) + 5.0 * jnp.power(M, 11.0) * jnp.power(q, 9.0) / (88.0 * jnp.power(r, 9.0)) - 32.0 * jnp.power(M, 10.0) * jnp.power(q, 3.0) / jnp.power(r, 8.0) + 60.0 * jnp.power(M, 10.0) * jnp.power(q, 5.0) / (7.0 * jnp.power(r, 8.0)) - 4.0 * jnp.power(M, 10.0) * jnp.power(q, 7.0) / (7.0 * jnp.power(r, 8.0)) - 80.0 * jnp.power(M, 9.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 7.0)) + 2.0 * jnp.power(M, 9.0) * jnp.power(q, 5.0) / jnp.power(r, 7.0) - jnp.power(M, 9.0) * jnp.power(q, 7.0) / (14.0 * jnp.power(r, 7.0)) - 24.0 * jnp.power(M, 8.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 6.0)) + 3.0 * jnp.power(M, 8.0) * jnp.power(q, 5.0) / (7.0 * jnp.power(r, 6.0)) - 24.0 * jnp.power(M, 7.0) * jnp.power(q, 3.0) / (35.0 * jnp.power(r, 5.0)) + 3.0 * jnp.power(M, 7.0) * jnp.power(q, 5.0) / (35.0 * jnp.power(r, 5.0))


def PhiPOnlyES_jax(M, q, r):
    return 2.0 * jnp.power(M, 2.0) / 3.0 - 8041005056.0 * jnp.power(M, 24.0) * jnp.power(q, 2.0) / (14421.0 * jnp.power(r, 22.0)) + 1385977507643392.0 * jnp.power(M, 24.0) * jnp.power(q, 4.0) / (423876453.0 * jnp.power(r, 22.0)) - 9.989620396474737e+18 * jnp.power(M, 24.0) * jnp.power(q, 6.0) / (1468731909645.0 * jnp.power(r, 22.0)) + 9.225803659557211e+21 * jnp.power(M, 24.0) * jnp.power(q, 8.0) / (1387951654614525.0 * jnp.power(r, 22.0)) - 1.6043003271337948e+21 * jnp.power(M, 24.0) * jnp.power(q, 10.0) / (462650551538175.0 * jnp.power(r, 22.0)) + 6.755117363294697e+23 * jnp.power(M, 24.0) * jnp.power(q, 12.0) / (6.245782445765362e+17 * jnp.power(r, 22.0)) - 2.710363445407059e+22 * jnp.power(M, 24.0) * jnp.power(q, 14.0) / (1.18967284681245e+17 * jnp.power(r, 22.0)) + 1.1950718113172306e+25 * jnp.power(M, 24.0) * jnp.power(q, 16.0) / (3.597570688760849e+20 * jnp.power(r, 22.0)) - 5.054567548874643e+21 * jnp.power(M, 24.0) * jnp.power(q, 18.0) / (1.75705528144608e+18 * jnp.power(r, 22.0)) + 3.3056744555097832e+22 * jnp.power(M, 24.0) * jnp.power(q, 20.0) / (2.842524988650547e+20 * jnp.power(r, 22.0)) - 3.243763865234007e+16 * jnp.power(M, 24.0) * jnp.power(q, 22.0) / (2.062603166367744e+16 * jnp.power(r, 22.0)) + 1547.0 * jnp.power(M, 24.0) * jnp.power(q, 24.0) / (524288.0 * jnp.power(r, 22.0)) - 443547648.0 * jnp.power(M, 23.0) * jnp.power(q, 2.0) / (1771.0 * jnp.power(r, 21.0)) + 7430031507456.0 * jnp.power(M, 23.0) * jnp.power(q, 4.0) / (6292363.0 * jnp.power(r, 21.0)) - 695487629952512.0 * jnp.power(M, 23.0) * jnp.power(q, 6.0) / (409003595.0 * jnp.power(r, 21.0)) + 15860600113536.0 * jnp.power(M, 23.0) * jnp.power(q, 8.0) / (22472725.0 * jnp.power(r, 21.0)) + 1854140157001254.0 * jnp.power(M, 23.0) * jnp.power(q, 10.0) / (7303635625.0 * jnp.power(r, 21.0)) - 607478734116071.0 * jnp.power(M, 23.0) * jnp.power(q, 12.0) / (2247272500.0 * jnp.power(r, 21.0)) + 1.6931295578153756e+16 * jnp.power(M, 23.0) * jnp.power(q, 14.0) / (233716340000.0 * jnp.power(r, 21.0)) - 426893055188763.0 * jnp.power(M, 23.0) * jnp.power(q, 16.0) / (54083120000.0 * jnp.power(r, 21.0)) + 3.578520927133202e+16 * jnp.power(M, 23.0) * jnp.power(q, 18.0) / (104704920320000.0 * jnp.power(r, 21.0)) - 6233510141877.0 * jnp.power(M, 23.0) * jnp.power(q, 20.0) / (1340222980096.0 * jnp.power(r, 21.0)) + 25857.0 * jnp.power(M, 23.0) * jnp.power(q, 22.0) / (3014656.0 * jnp.power(r, 21.0)) - 149291008.0 * jnp.power(M, 22.0) * jnp.power(q, 2.0) / (1155.0 * jnp.power(r, 20.0)) + 614110969716736.0 * jnp.power(M, 22.0) * jnp.power(q, 4.0) / (1120314195.0 * jnp.power(r, 20.0)) - 2.883604164704051e+16 * jnp.power(M, 22.0) * jnp.power(q, 6.0) / (42347876571.0 * jnp.power(r, 20.0)) + 346354717217408.0 * jnp.power(M, 22.0) * jnp.power(q, 8.0) / (1749912255.0 * jnp.power(r, 20.0)) + 7143052099063808.0 * jnp.power(M, 22.0) * jnp.power(q, 10.0) / (54447269877.0 * jnp.power(r, 20.0)) - 1.7261848214515635e+17 * jnp.power(M, 22.0) * jnp.power(q, 12.0) / (1905654445695.0 * jnp.power(r, 20.0)) + 2.463306532016589e+16 * jnp.power(M, 22.0) * jnp.power(q, 14.0) / (1319299231635.0 * jnp.power(r, 20.0)) - 574469521314688.0 * jnp.power(M, 22.0) * jnp.power(q, 16.0) / (381130889139.0 * jnp.power(r, 20.0)) + 833697187328.0 * jnp.power(M, 22.0) * jnp.power(q, 18.0) / (19249034805.0 * jnp.power(r, 20.0)) - 3137536.0 * jnp.power(M, 22.0) * jnp.power(q, 20.0) / (10669659.0 * jnp.power(r, 20.0)) - 133169152.0 * jnp.power(M, 21.0) * jnp.power(q, 2.0) / (1995.0 * jnp.power(r, 19.0)) + 128568127768576.0 * jnp.power(M, 21.0) * jnp.power(q, 4.0) / (509233725.0 * jnp.power(r, 19.0)) - 1022368992317056.0 * jnp.power(M, 21.0) * jnp.power(q, 6.0) / (3849806961.0 * jnp.power(r, 19.0)) + 1.0601170091752902e+17 * jnp.power(M, 21.0) * jnp.power(q, 8.0) / (2406129350625.0 * jnp.power(r, 19.0)) + 1.2755055261297347e+18 * jnp.power(M, 21.0) * jnp.power(q, 10.0) / (21655164155625.0 * jnp.power(r, 19.0)) - 2.5958300267679514e+17 * jnp.power(M, 21.0) * jnp.power(q, 12.0) / (9117963855000.0 * jnp.power(r, 19.0)) + 1.6227134070813253e+18 * jnp.power(M, 21.0) * jnp.power(q, 14.0) / (366863957460000.0 * jnp.power(r, 19.0)) - 2.797703590745652e+18 * jnp.power(M, 21.0) * jnp.power(q, 16.0) / (1.108744404768e+16 * jnp.power(r, 19.0)) + 341356715454037.0 * jnp.power(M, 21.0) * jnp.power(q, 18.0) / (78844046561280.0 * jnp.power(r, 19.0)) - 6721.0 * jnp.power(M, 21.0) * jnp.power(q, 20.0) / (688128.0 * jnp.power(r, 19.0)) - 655360.0 * jnp.power(M, 20.0) * jnp.power(q, 2.0) / (19.0 * jnp.power(r, 18.0)) + 49009399808.0 * jnp.power(M, 20.0) * jnp.power(q, 4.0) / (425425.0 * jnp.power(r, 18.0)) - 28330659115776.0 * jnp.power(M, 20.0) * jnp.power(q, 6.0) / (282907625.0 * jnp.power(r, 18.0)) + 5361605413888.0 * jnp.power(M, 20.0) * jnp.power(q, 8.0) / (1414538125.0 * jnp.power(r, 18.0)) + 33909814164736.0 * jnp.power(M, 20.0) * jnp.power(q, 10.0) / (1414538125.0 * jnp.power(r, 18.0)) - 11786514354528.0 * jnp.power(M, 20.0) * jnp.power(q, 12.0) / (1414538125.0 * jnp.power(r, 18.0)) + 1338458164964.0 * jnp.power(M, 20.0) * jnp.power(q, 14.0) / (1414538125.0 * jnp.power(r, 18.0)) - 50397087952.0 * jnp.power(M, 20.0) * jnp.power(q, 16.0) / (1414538125.0 * jnp.power(r, 18.0)) + 350976.0 * jnp.power(M, 20.0) * jnp.power(q, 18.0) / (1154725.0 * jnp.power(r, 18.0)) - 51838976.0 * jnp.power(M, 19.0) * jnp.power(q, 2.0) / (2907.0 * jnp.power(r, 17.0)) + 6805873168384.0 * jnp.power(M, 19.0) * jnp.power(q, 4.0) / (130945815.0 * jnp.power(r, 17.0)) - 55166563457312.0 * jnp.power(M, 19.0) * jnp.power(q, 6.0) / (1527701175.0 * jnp.power(r, 17.0)) - 738796077908144.0 * jnp.power(M, 19.0) * jnp.power(q, 8.0) / (206239658625.0 * jnp.power(r, 17.0)) + 1.4811340694832936e+16 * jnp.power(M, 19.0) * jnp.power(q, 10.0) / (1649917269000.0 * jnp.power(r, 17.0)) - 6.6700843335613576e+16 * jnp.power(M, 19.0) * jnp.power(q, 12.0) / (29698510842000.0 * jnp.power(r, 17.0)) + 1344096953129401.0 * jnp.power(M, 19.0) * jnp.power(q, 14.0) / (7542478944000.0 * jnp.power(r, 17.0)) - 18704857702771.0 * jnp.power(M, 19.0) * jnp.power(q, 16.0) / (4693098009600.0 * jnp.power(r, 17.0)) + 7007.0 * jnp.power(M, 19.0) * jnp.power(q, 18.0) / (622592.0 * jnp.power(r, 17.0)) - 1411072.0 * jnp.power(M, 18.0) * jnp.power(q, 2.0) / (153.0 * jnp.power(r, 16.0)) + 159389998336.0 * jnp.power(M, 18.0) * jnp.power(q, 4.0) / (6891885.0 * jnp.power(r, 16.0)) - 9361528576.0 * jnp.power(M, 18.0) * jnp.power(q, 6.0) / (765765.0 * jnp.power(r, 16.0)) - 34191514009088.0 * jnp.power(M, 18.0) * jnp.power(q, 8.0) / (10854718875.0 * jnp.power(r, 16.0)) + 437170363904.0 * jnp.power(M, 18.0) * jnp.power(q, 10.0) / (140970375.0 * jnp.power(r, 16.0)) - 4870862105344.0 * jnp.power(M, 18.0) * jnp.power(q, 12.0) / (8881133625.0 * jnp.power(r, 16.0)) + 308710599424.0 * jnp.power(M, 18.0) * jnp.power(q, 14.0) / (10854718875.0 * jnp.power(r, 16.0)) - 2167808.0 * jnp.power(M, 18.0) * jnp.power(q, 16.0) / (6891885.0 * jnp.power(r, 16.0)) - 405504.0 * jnp.power(M, 17.0) * jnp.power(q, 2.0) / (85.0 * jnp.power(r, 15.0)) + 24589248.0 * jnp.power(M, 17.0) * jnp.power(q, 4.0) / (2431.0 * jnp.power(r, 15.0)) - 2234310352.0 * jnp.power(M, 17.0) * jnp.power(q, 6.0) / (595595.0 * jnp.power(r, 15.0)) - 4198013377.0 * jnp.power(M, 17.0) * jnp.power(q, 8.0) / (2382380.0 * jnp.power(r, 15.0)) + 425842269.0 * jnp.power(M, 17.0) * jnp.power(q, 10.0) / (433160.0 * jnp.power(r, 15.0)) - 184250867.0 * jnp.power(M, 17.0) * jnp.power(q, 12.0) / (1555840.0 * jnp.power(r, 15.0)) + 1102175551.0 * jnp.power(M, 17.0) * jnp.power(q, 14.0) / (304944640.0 * jnp.power(r, 15.0)) - 3663.0 * jnp.power(M, 17.0) * jnp.power(q, 16.0) / (278528.0 * jnp.power(r, 15.0)) - 259072.0 * jnp.power(M, 16.0) * jnp.power(q, 2.0) / (105.0 * jnp.power(r, 14.0)) + 2923899008.0 * jnp.power(M, 16.0) * jnp.power(q, 4.0) / (675675.0 * jnp.power(r, 14.0)) - 1943073856.0 * jnp.power(M, 16.0) * jnp.power(q, 6.0) / (2027025.0 * jnp.power(r, 14.0)) - 4953879404.0 * jnp.power(M, 16.0) * jnp.power(q, 8.0) / (6081075.0 * jnp.power(r, 14.0)) + 5152567744.0 * jnp.power(M, 16.0) * jnp.power(q, 10.0) / (18243225.0 * jnp.power(r, 14.0)) - 44152796.0 * jnp.power(M, 16.0) * jnp.power(q, 12.0) / (2027025.0 * jnp.power(r, 14.0)) + 2096.0 * jnp.power(M, 16.0) * jnp.power(q, 14.0) / (6435.0 * jnp.power(r, 14.0)) - 348160.0 * jnp.power(M, 15.0) * jnp.power(q, 2.0) / (273.0 * jnp.power(r, 13.0)) + 8505963776.0 * jnp.power(M, 15.0) * jnp.power(q, 4.0) / (4729725.0 * jnp.power(r, 13.0)) - 132451952.0 * jnp.power(M, 15.0) * jnp.power(q, 6.0) / (921375.0 * jnp.power(r, 13.0)) - 5406426616.0 * jnp.power(M, 15.0) * jnp.power(q, 8.0) / (16372125.0 * jnp.power(r, 13.0)) + 183798767107.0 * jnp.power(M, 15.0) * jnp.power(q, 10.0) / (2554051500.0 * jnp.power(r, 13.0)) - 520303969.0 * jnp.power(M, 15.0) * jnp.power(q, 12.0) / (162162000.0 * jnp.power(r, 13.0)) + jnp.power(M, 15.0) * jnp.power(q, 14.0) / (64.0 * jnp.power(r, 13.0)) - 4608.0 * jnp.power(M, 14.0) * jnp.power(q, 2.0) / (7.0 * jnp.power(r, 12.0)) + 1935744.0 * jnp.power(M, 14.0) * jnp.power(q, 4.0) / (2695.0 * jnp.power(r, 12.0)) + 6987672.0 * jnp.power(M, 14.0) * jnp.power(q, 6.0) / (175175.0 * jnp.power(r, 12.0)) - 20830608.0 * jnp.power(M, 14.0) * jnp.power(q, 8.0) / (175175.0 * jnp.power(r, 12.0)) + 2749368.0 * jnp.power(M, 14.0) * jnp.power(q, 10.0) / (175175.0 * jnp.power(r, 12.0)) - 11808.0 * jnp.power(M, 14.0) * jnp.power(q, 12.0) / (35035.0 * jnp.power(r, 12.0)) - 145408.0 * jnp.power(M, 13.0) * jnp.power(q, 2.0) / (429.0 * jnp.power(r, 11.0)) + 36596752.0 * jnp.power(M, 13.0) * jnp.power(q, 4.0) / (135135.0 * jnp.power(r, 11.0)) + 6926398.0 * jnp.power(M, 13.0) * jnp.power(q, 6.0) / (135135.0 * jnp.power(r, 11.0)) - 18342487.0 * jnp.power(M, 13.0) * jnp.power(q, 8.0) / (486486.0 * jnp.power(r, 11.0)) + 1703357.0 * jnp.power(M, 13.0) * jnp.power(q, 10.0) / (617760.0 * jnp.power(r, 11.0)) - 63.0 * jnp.power(M, 13.0) * jnp.power(q, 12.0) / (3328.0 * jnp.power(r, 11.0)) - 28672.0 * jnp.power(M, 12.0) * jnp.power(q, 2.0) / (165.0 * jnp.power(r, 10.0)) + 193808.0 * jnp.power(M, 12.0) * jnp.power(q, 4.0) / (2079.0 * jnp.power(r, 10.0)) + 323198.0 * jnp.power(M, 12.0) * jnp.power(q, 6.0) / (10395.0 * jnp.power(r, 10.0)) - 957112.0 * jnp.power(M, 12.0) * jnp.power(q, 8.0) / (93555.0 * jnp.power(r, 10.0)) + 722.0 * jnp.power(M, 12.0) * jnp.power(q, 10.0) / (2079.0 * jnp.power(r, 10.0)) - 4864.0 * jnp.power(M, 11.0) * jnp.power(q, 2.0) / (55.0 * jnp.power(r, 9.0)) + 52224.0 * jnp.power(M, 11.0) * jnp.power(q, 4.0) / (1925.0 * jnp.power(r, 9.0)) + 55563.0 * jnp.power(M, 11.0) * jnp.power(q, 6.0) / (3850.0 * jnp.power(r, 9.0)) - 34537.0 * jnp.power(M, 11.0) * jnp.power(q, 8.0) / (15400.0 * jnp.power(r, 9.0)) + 3.0 * jnp.power(M, 11.0) * jnp.power(q, 10.0) / (128.0 * jnp.power(r, 9.0)) - 400.0 * jnp.power(M, 10.0) * jnp.power(q, 2.0) / (9.0 * jnp.power(r, 8.0)) + 1138.0 * jnp.power(M, 10.0) * jnp.power(q, 4.0) / (225.0 * jnp.power(r, 8.0)) + 77314.0 * jnp.power(M, 10.0) * jnp.power(q, 6.0) / (14175.0 * jnp.power(r, 8.0)) - 556.0 * jnp.power(M, 10.0) * jnp.power(q, 8.0) / (1575.0 * jnp.power(r, 8.0)) - 1376.0 * jnp.power(M, 9.0) * jnp.power(q, 2.0) / (63.0 * jnp.power(r, 7.0)) - 37.0 * jnp.power(M, 9.0) * jnp.power(q, 4.0) / (45.0 * jnp.power(r, 7.0)) + 18493.0 * jnp.power(M, 9.0) * jnp.power(q, 6.0) / (11340.0 * jnp.power(r, 7.0)) - 17.0 * jnp.power(M, 9.0) * jnp.power(q, 8.0) / (576.0 * jnp.power(r, 7.0)) - 72.0 * jnp.power(M, 8.0) * jnp.power(q, 2.0) / (7.0 * jnp.power(r, 6.0)) - 51.0 * jnp.power(M, 8.0) * jnp.power(q, 4.0) / (35.0 * jnp.power(r, 6.0)) + 12.0 * jnp.power(M, 8.0) * jnp.power(q, 6.0) / (35.0 * jnp.power(r, 6.0)) - 464.0 * jnp.power(M, 7.0) * jnp.power(q, 2.0) / (105.0 * jnp.power(r, 5.0)) - 38.0 * jnp.power(M, 7.0) * jnp.power(q, 4.0) / (45.0 * jnp.power(r, 5.0)) + jnp.power(M, 7.0) * jnp.power(q, 6.0) / (28.0 * jnp.power(r, 5.0)) - 22.0 * jnp.power(M, 6.0) * jnp.power(q, 2.0) / (15.0 * jnp.power(r, 4.0)) - 47.0 * jnp.power(M, 6.0) * jnp.power(q, 4.0) / (180.0 * jnp.power(r, 4.0)) + 2.0 * jnp.power(M, 4.0) * jnp.power(q, 2.0) / (3.0 * jnp.power(r, 2.0)) + jnp.power(M, 3.0) * jnp.power(q, 2.0) / (3.0 * r) - 2.0 * M * r + jnp.power(r, 2.0)

# --------------------------------------------
# This part, define matrix multiplication to solve for matching conditions (as well as it's derivative)
# [ H0OnlyQT(M,q,R)   H0OnlyET(M,q,R)   H0OnlyQS(M,q,R)   H0OnlyES(M,q,R) ]   [ cQT ]   [ H0_int(M,q,R) ]
# [ H0OnlyQT'(M,q,R)  H0OnlyET'(M,q,R)  H0OnlyQS'(M,q,R)  H0OnlyES'(M,q,R) ]  [ cET ] = [ H0'_int(M,q,R) ]
# [ φpOnlyQT(M,q,R)   φpOnlyET(M,q,R)   φpOnlyQS(M,q,R)   φpOnlyES(M,q,R) ]  [ cQS ]   [ φp_int(M,q,R) ]
# [ φpOnlyQT'(M,q,R)  φpOnlyET'(M,q,R)  φpOnlyQS'(M,q,R)  φpOnlyES'(M,q,R) ]  [ cES ]   [ φp'_int(M,q,R) ]
# Left matrix from infinity expansion, right matrix from tov solver, and c matrix is what we solve for to determine lambdas.


def build_exterior_basis(M, q, R):
    return [H0OnlyQT_jax(M, q, R), H0OnlyET_jax(M, q, R), H0OnlyQS_jax(M, q, R), H0OnlyES_jax(M, q, R)], [PhiPOnlyQT_jax(M, q, R), PhiPOnlyET_jax(M, q, R), PhiPOnlyQS_jax(M, q, R), PhiPOnlyES_jax(M, q, R)]

def build_exterior_basis_autodiff(M, q, R):
    H0OnlyQT_autodiff_jax = lambda M, q, R: jax.grad(H0OnlyQT_jax, argnums=2)(M, q, R)
    H0OnlyET_autodiff_jax = lambda M, q, R: jax.grad(H0OnlyET_jax, argnums=2)(M, q, R)
    H0OnlyQS_autodiff_jax = lambda M, q, R: jax.grad(H0OnlyQS_jax, argnums=2)(M, q, R)
    H0OnlyES_autodiff_jax = lambda M, q, R: jax.grad(H0OnlyES_jax, argnums=2)(M, q, R)
    PhiPOnlyQT_autodiff_jax = lambda M, q, R: jax.grad(PhiPOnlyQT_jax, argnums=2)(M, q, R)
    PhiPOnlyET_autodiff_jax = lambda M, q, R: jax.grad(PhiPOnlyET_jax, argnums=2)(M, q, R)
    PhiPOnlyQS_autodiff_jax = lambda M, q, R: jax.grad(PhiPOnlyQS_jax, argnums=2)(M, q, R)
    PhiPOnlyES_autodiff_jax = lambda M, q, R: jax.grad(PhiPOnlyES_jax, argnums=2)(M, q, R)
    return [H0OnlyQT_autodiff_jax(M, q, R), H0OnlyET_autodiff_jax(M, q, R), H0OnlyQS_autodiff_jax(M, q, R), H0OnlyES_autodiff_jax(M, q, R)], [PhiPOnlyQT_autodiff_jax(M, q, R), PhiPOnlyET_autodiff_jax(M, q, R), PhiPOnlyQS_autodiff_jax(M, q, R), PhiPOnlyES_autodiff_jax(M, q, R)]

def coeff_solver(interior_sol, exterior_basis, exterior_basis_prime):
    r"""Match interior solution to exterior basis at surface."""
    H0_int, H0_prime_int, delta_phi_int, delta_phi_prime_int = interior_sol
    H0_basis, delta_phi_basis = exterior_basis
    H0_basis_prime, delta_phi_basis_prime = exterior_basis_prime
    
    # Build the matching matrix (4x4 system)
    # We solve particular solutions coeffs: interior = c1*basis1 + c2*basis2 + c3*basis3 + c4*basis4
    # We use this function also for (4x4) = (2x4) system by setting one column to 0 (particular solution) and move one of rhs term to lhs
    # in above case, we leave this function as it is: only input changed.
    A = jnp.array([
        [H0_basis[0], H0_basis[1], H0_basis[2], H0_basis[3]],  # H0 values
        [H0_basis_prime[0], H0_basis_prime[1], H0_basis_prime[2], H0_basis_prime[3]],  # H0' vals
        [delta_phi_basis[0], delta_phi_basis[1], delta_phi_basis[2], delta_phi_basis[3]],  # dphi vals
        [delta_phi_basis_prime[0], delta_phi_basis_prime[1], delta_phi_basis_prime[2], delta_phi_basis_prime[3]]  # dphi' vals
    ])
    
    b = jnp.array([H0_int, H0_prime_int, delta_phi_int, delta_phi_prime_int])

    # Solve for coefficients
    coefficients = jnp.linalg.solve(A, b)
    
    return coefficients


def compute_tidal_deformabilities(coefficients):
    r"""Compute tidal deformabilities from matched coefficients."""
    cQT1, cQT2, cET, cQS1, cQS2, cES = coefficients
    #case 1: scalar = 1, tensor = 0 
    #case 2: scalar = 0, tensor = 1
    double_factorial = 3.0  # (2l-1)!! = 3 for l=2
    
    # Tensor deformability (response to tensor tidal field)
    lambda_T = (1.0 / double_factorial) * (cQT2 / cET) #scalar pert=0
    
    # Scalar deformability (response to scalar tidal field)
    lambda_S = (1.0 / double_factorial) * (cQS1 / cES) #tensor pert=0
    
    # Mixed scalar-tensor deformability
    lambda_ST1 = (1.0 / double_factorial) * (cQT1 / (2.0 * cES)) #tensor pert = 0
    lambda_ST2 = (1.0 / double_factorial) * (2.0 * cQS2 / cET) #scalar pert = 0
    
    return lambda_T, lambda_S, lambda_ST1, lambda_ST2

def tov_solver(eos, pc):
    r"""
    """
    # Extract EOS interpolation arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]

    # Central values and initial conditions
    hc = utils.interp_in_logspace(pc, ps, hs)
    ec = utils.interp_in_logspace(hc, hs, es)
    dedp_c = ec / pc * jnp.interp(hc, hs, dloge_dlogps)
    dhdp_c = 1.0 / (ec + pc)
    dedh_c = dedp_c / dhdp_c

    # Initial values using series expansion near center, GR approximation
    
    dh = -1e-3 * hc
    h0 = hc + dh
    r0 = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
    r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
    m0 = 4.0 * jnp.pi * ec * jnp.power(r0, 3.0) / 3.0
    m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
    psi0 = 0.0

    H0_center = jnp.power(r0, 2)  # ~r^2 for l=2
    H0_prime_center = 2.0 * r0   # derivative
    delta_phi_center = jnp.power(r0, 2)
    delta_phi_prime_center = 2.0 * r0

    ###########################################################
    # initial guess values
    phi_inf_target = eos["phi_inf_tgt"]
    phi0 = eos["phi_c"]
    nu0=0.0
    damping = 0.5
    max_iterations = 1000
    tol = 1e-5


    def run_iteration(phi0_init):
        big = 1e9
        init_state = (
            0,                              # iteration count
            phi0_init,                      # phi0_local
            0.0, 0.0,                       # R_final, M_inf_final
            big,                            # phi_inf_final
            jnp.array([phi0_init], dtype=jnp.float64),  # prev_x
            jnp.array([big], dtype=jnp.float64),        # prev_F
        )

        # Keep forward_solver as a single function (called once per iteration)
        def forward_solver(params):
            phi0_trial = params[0]
            y0 = (r0, m0, nu0, psi0, phi0_trial)

            #------ Stop if mass > 20 Msun
            M_limit = 20.0 * utils.solar_mass_in_meter

            def mass_event(t, y, args, **kwargs):
                return y[1] > M_limit
            #------
            sol_iter = diffeqsolve(
                ODETerm(tov_ode_iter),
                Dopri8(scan_kind="bounded"),
                t0=h0,
                t1=0,
                dt0=dh,
                y0=y0,
                args=eos,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-7, atol=1e-8),
                event=Event(mass_event),
                throw=False,
            )
            
            R = sol_iter.ys[0][-1]
            M_s = sol_iter.ys[1][-1]
            nu_s = sol_iter.ys[2][-1]
            psi_s = sol_iter.ys[3][-1]
            phi_s = sol_iter.ys[4][-1]

            EPS = 1e-99
            nu_s_prime =  2 *M_s / (R * (R - 2.0 * M_s)) + R * jnp.power(psi_s, 2)
            
            front = 2 * psi_s / jnp.sqrt(jnp.power(nu_s_prime,2) + 4 * jnp.power(psi_s,2))
            inside_tanh = jnp.sqrt(jnp.power(nu_s_prime,2) + 4 * jnp.power(psi_s,2)) / (nu_s_prime + 2 / R)
            phi_inf = phi_s + front * jnp.arctanh(inside_tanh)
            # MODIFIED: Return shifted value (phi_inf - target) instead of just phi_inf
            return jnp.array([phi_inf - phi_inf_target]), (R, M_s)

        # Define core step function for scan
        def step_func(state, _):
            i, phi0, R_prev, M_prev, phi_inf_prev, prev_x, prev_F = state
            
            x_curr = jnp.array([phi0])
            F_curr, (R, M) = forward_solver(x_curr)
    
            # Choose step type based on iteration count
            def damped_step():
                step = -damping * F_curr
                return x_curr + step, x_curr, F_curr
    
            def linearized_step():
                dx = x_curr - prev_x
                dF = F_curr - prev_F
                J = dF / (dx + 1e-12)
                step = -0.8 * F_curr / (J + 1e-12)
                return x_curr + jnp.clip(step, -1e6, 1e6), x_curr, F_curr
    
            x_next, new_prev_x, new_prev_F = lax.cond(
                i < 10,
                lambda _: damped_step(),
                lambda _: linearized_step(),
                None
            )
    
            return (
                i+1, x_next[0], R, M, F_curr[0], 
                new_prev_x, new_prev_F
            ), None
    
        # Run phases until convergence
        def phase_loop(state):
            # First run 50 iterations mixing damped/linearized
            state, _ = lax.scan(step_func, state, None, 50)
            
            # Then run 25-step linearized phases until converged
            def cond(state):
                i, _, _, _, phi_inf, _, _ = state
                return (i < max_iterations) & (jnp.abs(phi_inf) >= tol)
                
            state = lax.while_loop(
                cond,
                lambda s: lax.scan(step_func, s, None, 25)[0],
                state
            )
            return state
    
        final_state = phase_loop(init_state)
        i_final, phi0_final, R_final, M_inf_final, phi_inf_final, _, _ = final_state

        # Return NaN if max iteration reached or enclosed mass reached 20 M_sun
        too_big_mass = (M_inf_final / utils.solar_mass_in_meter) > 20.0
        too_many_iters = i_final >= max_iterations
        returnNAN = too_big_mass | too_many_iters

        def nan_branch(_):
            return (jnp.nan,) * 15

        # Calculate tidal deformability using converged phi0 value
        def compute_branch(_):
            # Interior solve
            # There are two interior particular solutions
            # Below start with case 1, H0 = 0 and normalize by setting cET = 0 (2nd case dphi0=0 with cES=0)
            y0_case1 = (r0, m0, nu0, psi0, phi0_final, 0.0, 0.0, delta_phi_center, delta_phi_prime_center) 
            sol_iter_1 = diffeqsolve(
                ODETerm(tov_ode_iter_tidal),
                Dopri8(scan_kind="bounded"),
                t0=h0,
                t1=0,
                dt0=dh,
                y0=y0_case1,
                args=eos,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-7, atol=1e-8),
                throw=False,
            )

            R = sol_iter_1.ys[0][-1]
            M_s = sol_iter_1.ys[1][-1]
            nu_s = sol_iter_1.ys[2][-1]
            psi_s = sol_iter_1.ys[3][-1]
            phi_s = sol_iter_1.ys[4][-1]
            H0_surface_1 = sol_iter_1.ys[5][-1]
            H0_prime_surface_1 = sol_iter_1.ys[6][-1]
            delta_phi_surface_1 = sol_iter_1.ys[7][-1]
            delta_phi_prime_surface_1 = sol_iter_1.ys[8][-1]

            # case 2
            y0_case2 = (r0, m0, nu0, psi0, phi0_final, H0_center, H0_prime_center, 0.0, 0.0) 
            # y0_case2 = (r0, m0, nu0, psi0, phi0_final, H0_center, H0_prime_center,delta_phi_center, delta_phi_prime_center)
            sol_iter_2 = diffeqsolve(
                ODETerm(tov_ode_iter_tidal),
                Dopri8(scan_kind="bounded"),
                t0=h0,
                t1=0,
                dt0=dh,
                y0=y0_case2,
                args=eos,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-7, atol=1e-8),
                throw=False,
            )
            H0_surface_2 = sol_iter_2.ys[5][-1]
            H0_prime_surface_2 = sol_iter_2.ys[6][-1]
            delta_phi_surface_2 = sol_iter_2.ys[7][-1]
            delta_phi_prime_surface_2 = sol_iter_2.ys[8][-1]
            return (R_final, M_inf_final, nu_s, phi_inf_final, psi_s, phi_s, M_s, H0_surface_1,H0_prime_surface_1,delta_phi_surface_1,delta_phi_prime_surface_1,H0_surface_2,H0_prime_surface_2,delta_phi_surface_2,delta_phi_prime_surface_2)

        return lax.cond(returnNAN, nan_branch, compute_branch, operand=None)
    
    R, M_inf, nu_s, phi_inf, psi_s, phi_s, M_s, H0_surface_1,H0_prime_surface_1,delta_phi_surface_1,delta_phi_prime_surface_1,H0_surface_2,H0_prime_surface_2,delta_phi_surface_2,delta_phi_prime_surface_2 = run_iteration(phi0)
    #define scalar charge q (Eq. 4.18 & 4.19)
    nu_s_prime = 2*M_s / (R*(R-2*M_s)) + R*psi_s*psi_s
    q = 2* psi_s/nu_s_prime
    
    exterior_basis_matrix = build_exterior_basis(M_inf, q,R)
    exterior_basis_matrix_prime = build_exterior_basis_autodiff(M_inf, q,R)

    # The idea: we have 6 coefficients with 4 equations. To reduce coefficients, we set two particular cases
    # Case 1: H0 = 0 so cET = 0
    # Case 2: dphi = 0 so cES = 0
    # And then we normalize with one of interior solution coefficient (here case 2)
    # the ratios between coefficients are then used to calculate tidal deformability
    # therefore, normalization has to be consistent for all matrix.
    
    #CASE 1 (Scalar deformability)
    #Set cET = 0, so replace second lhs matrix column with -H01 or -dphi01
    interior_sol = (H0_surface_2, H0_prime_surface_2, delta_phi_surface_2, delta_phi_prime_surface_2)
    exterior_basis_matrix_1 = exterior_basis_matrix
    exterior_basis_matrix_prime_1 = exterior_basis_matrix_prime

    mat1_p0 = jnp.array(exterior_basis_matrix_1[0])
    mat1_p1 = jnp.array(exterior_basis_matrix_1[1])
    mat1_prime_p0 = jnp.array(exterior_basis_matrix_prime_1[0])
    mat1_prime_p1 = jnp.array(exterior_basis_matrix_prime_1[1])
    mat1_p0 = mat1_p0.at[1].set(-H0_surface_1)
    mat1_p1 = mat1_p1.at[1].set(-delta_phi_surface_1)
    mat1_prime_p0 = mat1_prime_p0.at[1].set(-H0_prime_surface_1)
    mat1_prime_p1 = mat1_prime_p1.at[1].set(-delta_phi_prime_surface_1)
    exterior_basis_matrix_1 = (mat1_p0, mat1_p1)
    exterior_basis_matrix_prime_1 = (mat1_prime_p0, mat1_prime_p1)

    coeffs_1 = coeff_solver(interior_sol, exterior_basis_matrix_1, exterior_basis_matrix_prime_1)
    cQT1, c2, cQS1, cES = coeffs_1
    
    # CASE 2 (tensor deformability)
    # Setting cES = 0 
    # so change the coeffs into cQT, cET, cQS, c2, where c2 is particular soluion coeff
    # and replace equation relating to ES to be -H01 or -dphi01
    interior_sol = (H0_surface_2, H0_prime_surface_2, delta_phi_surface_2, delta_phi_prime_surface_2)
    exterior_basis_matrix_2 = exterior_basis_matrix
    exterior_basis_matrix_prime_2 = exterior_basis_matrix_prime

    mat2_part0 = jnp.array(exterior_basis_matrix_2[0])
    mat2_part1 = jnp.array(exterior_basis_matrix_2[1])
    mat2_prime_part0 = jnp.array(exterior_basis_matrix_prime_2[0])
    mat2_prime_part1 = jnp.array(exterior_basis_matrix_prime_2[1])
    mat2_part0 = mat2_part0.at[3].set(-H0_surface_1)
    mat2_part1 = mat2_part1.at[3].set(-delta_phi_surface_1)
    mat2_prime_part0 = mat2_prime_part0.at[3].set(-H0_prime_surface_1)
    mat2_prime_part1 = mat2_prime_part1.at[3].set(-delta_phi_prime_surface_1)
    
    exterior_basis_matrix_2 = (mat2_part0, mat2_part1)
    exterior_basis_matrix_prime_2 = (mat2_prime_part0, mat2_prime_part1)

    coeffs_2 = coeff_solver(interior_sol, exterior_basis_matrix_2, exterior_basis_matrix_prime_2)
    cQT2, cET, cQS2, c2 = coeffs_2

    #Final coefficients
    coeffs = cQT1, cQT2, cET, cQS1, cQS2, cES

    lambda_T, lambda_S, lambda_ST1, lambda_ST2 = compute_tidal_deformabilities(coeffs)
    #lambda_ST1 should have same value with lambda_ST2
    
    #Jordan frame conversion
    beta_ST = eos["beta_ST"]
    A_phi_inf = jnp.exp(0.5 * beta_ST * jnp.power(phi_inf_target, 2))
    A_phi_s = jnp.exp(0.5 * beta_ST * jnp.power(phi_s, 2))
    R_jordan = A_phi_s*R
    M_inf_jordan = (1/A_phi_inf)*(M_inf + (beta_ST*phi_inf_target*(-q*M_inf)))
    #Tidal deforms dimensionless, multiplied by ADM mass ^-5
    Lambda_T_J = lambda_T * jnp.power(M_inf, -5.0)
    Lambda_S_J = ( jnp.exp(2*beta_ST*jnp.power(phi_inf_target, 2))/(4*beta_ST*beta_ST*phi_inf_target*phi_inf_target) ) * lambda_S * jnp.power(M_inf, -5.0)
    Lambda_ST1_J = (-jnp.exp(beta_ST*jnp.power(phi_inf_target, 2))/(2*beta_ST*phi_inf_target))*lambda_ST1 * jnp.power(M_inf, -5.0) #or lambda_ST2, must be same
    Lambda_ST2_J = (-jnp.exp(beta_ST*jnp.power(phi_inf_target, 2))/(2*beta_ST*phi_inf_target))*lambda_ST2 * jnp.power(M_inf, -5.0) #or lambda_ST1, must be same
    return M_inf_jordan, R_jordan, Lambda_T_J, Lambda_S_J, Lambda_ST1_J, Lambda_ST2_J
