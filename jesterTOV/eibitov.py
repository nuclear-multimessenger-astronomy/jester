r"""
Modified TOV equation solver within Eddington-Inspired Born-Infeld theory [Phys. Rev. Lett. 105, 011101].
Mass, radius, and tidal deformability calculated with dependence to a cosmological constant.

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Reference:** I. Prasetyo et al Phys. Rev. D 104, 084029
"""

from . import utils
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, RESULTS
import equinox as eqx

def tov_ode(h, y, eos):
    r"""
    Solve TOV equations in Eddington-inspired Born-Infeld (EiBI) gravity.
    
    This function implements the modified Tolman-Oppenheimer-Volkoff (TOV) equations
    within the framework of EiBI gravity, which introduces a new parameter $\kappa$
    that modifies the gravitational field equations. The equations reduce to standard
    GR when $\kappa \to 0$.
    
    The pressure gradient equation in EiBI gravity becomes:
    
    .. math::
        \frac{dp}{dr} = -\frac{1}{4\pi\kappa} \frac{
            \frac{r}{2\kappa}\left(\frac{1}{ab} + \frac{a}{b^3} - 2\right) + \frac{2m}{r^2} + \frac{\Lambda_{\mathrm{cosmo}} r}{3\lambda}
        }{
            \left[\frac{4}{A - B} + \frac{3}{B} + \frac{1}{A}\frac{de}{dp}\right]
            \left[1 - \frac{2m}{r} - \frac{\Lambda_{\mathrm{cosmo}} r^2}{3\lambda}\right]
        }
    
    where:
    - $a = \sqrt{1 + 8\pi\kappa\varepsilon}$, $b = \sqrt{1 - 8\pi\kappa p}$
    - $A = 1 + 8\pi\kappa\varepsilon$, $B = 1 - 8\pi\kappa p$
    - $\lambda = \kappa\Lambda_{\mathrm{cosmo}} + 1$
    
    The mass equation becomes:
    
    .. math::
        \frac{dm}{dr} = \frac{r^2}{4\kappa}\left(2 - \frac{3}{ab} + \frac{a}{b^3}\right)
    
    The system automatically switches to standard GR equations when $|\kappa| < 10^4$
    for numerical stability.
    
    Args:
        h (float): Enthalpy value (integration variable).
        y (tuple): Current state vector containing:
            - r: Radial coordinate [Geom]
            - m: Gravitational mass [Geom]
            - yr: Tidal variable for deformability calculation
        eos (dict): Equation of state dictionary containing:
            - "p": Pressure array
            - "h": Enthalpy array  
            - "e": Energy density array
            - "dloge_dlogp": Derivative d(log ε)/d(log p)
            - "kappa": EiBI gravity parameter $\kappa$ [Geom]
            - "Lambda_cosmo": Cosmological constant parameter
    
    Returns:
        tuple: Derivatives (dr/dh, dm/dh, dyr/dh) for the ODE solver.
    
    Note:
        Small EPS values are added throughout to prevent division by zero and
        ensure numerical stability near the stellar surface and in the GR limit.
    """
    # fetch the eos arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    r, m, yr = y
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)

    K=eos["kappa"]
    Lambda_cosmo = eos["Lambda_cosmo"]
    lambda_small = K*Lambda_cosmo + 1 #Eq. 8 from https://arxiv.org/pdf/2109.05718
    EPS = 1e-19 #small value to avoid zero division error
    #close to surface, A and B -> 1. Small EPS added to make sure 1/(A-B) etc is safe.
    B = jnp.sqrt(1-8*jnp.pi*K*p) + EPS
    A = jnp.sqrt(1+8*jnp.pi*K*e) + 1.5*EPS
    b=B
    a=A
    AA = 1+8*jnp.pi*K*e + EPS
    BB = 1-8*jnp.pi*K*p + 1.2*EPS
    
    A_per_BBB = jnp.sqrt((1+8*jnp.pi*K*e)/(1-8*jnp.pi*K*p)) / BB
    AB = jnp.sqrt((1-8*jnp.pi*K*p)*(1+8*jnp.pi*K*e))
    
    Q1 = 1/dedp #cs2(p0, G1, G2, G3,p)
    Q2 = -Q1*B/A

    term1 = (r / (2*K)) * (1/(a*b + EPS) + a/(b*b*b + EPS) - 2) + (2*m) / (r*r + EPS) + Lambda_cosmo*r/(3*lambda_small)
    denominator1 = (4/(AA - BB + EPS)) + (3/(BB + EPS)) + (1/(AA + EPS)) * dedp
    denominator2 = 1 - 2*m/r - Lambda_cosmo*r*r/(3*lambda_small)
    
    dpdr = -1/(4*jnp.pi*K) * term1 / (denominator1 * denominator2)
    
    dpdr_GR = -e * m / (r * r) * (1 + p / (e)) * (1 + 4 * jnp.pi * r*r*r * p / m) / (1 - 2 * m / r)
    dpdr = jnp.where(jnp.abs(K) < 1e4, dpdr_GR, dpdr)  # turn to GR when K<0.01 km^2 for numerical stability
    dmdr = (r*r / (4 * K)) * (2 - 3/(a*b + EPS) + a/(b*b*b + EPS))
    dmdr_GR = 4.0 * jnp.pi * r*r * e
    dmdr = jnp.where(jnp.abs(K) < 1e4, dmdr_GR, dmdr)

    #For tidal deformability
    dbetadr = -2 * dpdr * (2*jnp.pi*K*(3/BB +1/AA*dedp)+1/(e+p))
    expalpha = 1/(1-2*m/r-Lambda_cosmo*r*r/(3*lambda_small))

    f_func = ((r * expalpha / K) * (1 / (A * B) - 1)+ expalpha / r+ 1 / r)
    
    g_func = ((2 * expalpha * A / (K * B*B*B))* (2- (4 / (AA - BB +EPS))* 1/((4 / (AA - BB+EPS))+ (1 / AA) * dedp+ (3 / BB)))- ((2 * (2 + 1) * expalpha / (r*r))+ (2 * expalpha / K)+ dbetadr*dbetadr))

    drdh = (e + p) / dpdr
    dmdh = dmdr * drdh
    dyrdr = -f_func*yr - r*g_func - yr*yr/r + yr/r
    dyrdh = dyrdr * drdh
    dydt = drdh, dmdh, dyrdh #output
    return dydt


def calc_k2_eibi(R, M, y_R, eos):
    r"""
    Calculate the even-parity tidal Love number k2 following Ref. [Phys. Rev. D 104, 084029]. Uses geometric units (G = 1).
    
    Signature intentionally matches the example:
        calc_k2_ST(R, M, y_R)
    where:
      R   : radius (geometric units)
      M   : mass   (geometric units)
      y_R : y(R) = r * (dH/dr)/H evaluated at the surface
    """

    # constants / choices
    YL = 2.0
    LAM = eos["Lambda_cosmo"]   # chosen default (see notes)
    K = eos["kappa"]   # choose K = YL as a best-effort mapping from Fortran
    # L = LAM * K + 1.0 #original
    L = LAM * K + 1.0

    # compactness
    C = M / R

    # eps as in Fortran but with GSS=MSS=1 removed (geometrized units)
    eps = (LAM / L) * (C ** 2)


    # coefficients a_?_2 and b_?_2 (copied from Fortran)
    a12 = 0.0
    a22 = C ** 2 / 3.0
    a32 = 113.0 * C ** 2 / 84.0
    a42 = 13.0 * C ** 2 / 9.0
    
    b12 = 15.0 / (8.0 * C ** 3)
    b22 = 0.0
    b32 = 1787.0 / (392.0 * C ** 3)
    b42 = (-(3305.0 / 448.0 + 5.0 * (jnp.pi ** 2) / 7.0 + 15.0 * (jnp.log(2.0) ** 2) / 7.0)
           / C ** 3)
    # derivatives
    DQ2 = jax.grad(Q2_fn)
    DP2 = jax.grad(P2_fn)
    DT2 = jax.grad(T2_fn)
    DS2 = jax.grad(S2_fn)

    # evaluate helper combinations (matching Fortran names)
    QB2 = y_R * Q2_fn(C) + C * DQ2(C)
    PB2 = y_R * P2_fn(C) + C * DP2(C)
    TB2 = y_R * T2_fn(C) + C * DT2(C)
    SB2 = y_R * S2_fn(C) + C * DS2(C)

    # KN2 and KD2 as in Fortran
    KN2 = (a12 + eps * a32) * QB2 + (a22 + eps * a42) * PB2 + eps * (a12 * TB2 + a22 * SB2)
    KD2 = (b12 + eps * b32) * QB2 + (b22 + eps * b42) * PB2 + eps * (b12 * TB2 + b22 * SB2)

    # double factorial factor: DOUBFAC(2*YL - 1)
    n_df = int(2 * int(YL) - 1)
    dfac = double_factorial(n_df)

    K2 = - (dfac / 2.0) * (KN2 / KD2)
    return K2
    
def double_factorial(n):
    def body(i, val):
        return val * i
    n = jnp.asarray(n, int)
    res = jax.lax.cond(
        n <= 0,
        lambda _: 1,
        lambda _: jax.lax.fori_loop(1, n + 1, 
            lambda i, v: jax.lax.cond(i % 2 == n % 2, lambda _: v * i, lambda _: v, None), 
            1),
        None
    )
    return res
    
def Q2_fn(C):
    """
    Translation of the Q2(C).
    Adapted from https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
    XX = -1 + 1/C
    Q2 = 6*(XX-1)*(XX+1)*( ... ) with natural logs
    """
    C = jnp.asarray(C)
    XX = -1.0 + 1.0 / C

    term1 = - ( -1.0 + XX )**2 / (48.0 * (1.0 + XX)**2)
    term2 = ( -1.0 + XX ) / (6.0 * (1.0 + XX))
    term3 = - (1.0 + XX) / (6.0 * ( -1.0 + XX ))
    term4 = (1.0 + XX)**2 / (48.0 * ( -1.0 + XX )**2)
    term5 = ( - jnp.log( -1.0 + XX ) + jnp.log( 1.0 + XX ) ) / 4.0

    inner = term1 + term2 + term3 + term4 + term5
    Q2_val = 6.0 * ( -1.0 + XX ) * ( 1.0 + XX ) * inner
    return Q2_val

def P2_fn(C):
    """
    Translation of the P2(C).
    Adapted from https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
    P2 = (3*(1-XX)^2*(1+XX))/(XX-1)  which simplifies sign-wise like Fortran
    """
    C = jnp.asarray(C)
    XX = -1.0 + 1.0 / C
    # keep algebraic structure faithful to Fortran
    P2_val = (3.0 * (1.0 - XX)**2 * (1.0 + XX)) / (-1.0 + XX)
    return P2_val
    
def T2_fn(C):
    """
    Translation of the T2(C).
    Adapted from https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
    Requires a PLG(C) function available in the same scope.
    """
    C = jnp.asarray(C)
    XX = -1.0 + 1.0 / C

    # polynomial pieces
    F32 = (235.0/56.0) + (1357.0*XX)/56.0 - (1469.0*XX**2)/84.0 - (1987.0*XX**3)/84.0 \
          + (2057.0*XX**4)/168.0 + (389.0*XX**5)/56.0 - (8.0*XX**6)/7.0

    F42 = -(25.0/14.0) + (153.0*XX)/14.0 + (171.0*XX**2)/14.0 - (17.0*XX**3)/14.0 \
          - (25.0*XX**4)/7.0 - (4.0*XX**5)/7.0

    F52 = -(25.0/14.0) + (34.0*XX)/7.0 - (93.0*XX**2)/14.0 + (13.0*XX**3)/7.0 + (4.0*XX**4)/7.0

    # the Fortran piece: ( -24/7 )*( DLOG(XX-1)*DLOG((XX+1)/4) + 2*PLG(C) )
    F62 = - (24.0/7.0) * ( jnp.log(XX - 1.0) * jnp.log((XX + 1.0) / 4.0) + 2.0 * PLG_fn(C) )

    T2_val = F32 / ((XX + 1.0) * (XX - 1.0)**2) \
             + F42 / (XX + 1.0) * jnp.log(XX - 1.0) \
             + F52 * (XX + 1.0) / (XX - 1.0) * jnp.log(XX + 1.0) \
             + (XX**2 - 1.0) * F62

    return T2_val

def PLG_fn(C):
    """
    Approximates PolyLog(2, (1 - x)/2) with x = 1/C - 1.
    Adapted from https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
    Region-based polynomial fit:
      - For 0.01 < x <= 0.5 : 12th-degree polynomial fit
      - For 0 < x <= 0.01   : 5894.17 - 5875.92 / x**0.001
      - Else                : 0
    """
    x = jnp.asarray(C)

    # region 1 polynomial
    poly1 = (
        -4.900128181312306
        + 0.00016932349271285354 / x**2
        - 0.06607300001104759 / x
        + 67.81366827244638 * x
        - 661.3708294454829 * x**2
        + 4714.740202946316 * x**3
        - 23715.678000133867 * x**4
        + 84334.3662190509 * x**5
        - 213246.7648131581 * x**6
        + 383601.9350809346 * x**7
        - 486336.6093429251 * x**8
        + 424030.9280758699 * x**9
        - 241718.87649564294 * x**10
        + 81048.35624350267 * x**11
        - 12113.292172158068 * x**12
    )

    # region 2 (small x)
    poly2 = 5894.173801180173 - 5875.9257455803145 / (x**0.001)

    # combine using jnp.where
    plg_val = jnp.where(
        (x > 0.01) & (x <= 0.5),
        poly1,
        jnp.where((x > 0.0) & (x <= 0.01), poly2, 0.0),
    )

    return plg_val

def S2_fn(C):
    """
    Translation of the S2(C).
    Adapted from https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
    """
    C = jnp.asarray(C)
    XX = -1.0 + 1.0 / C

    F12 = 1.0 + 21.0*XX/4.0 - 46.0*XX**2/7.0 - 59.0*XX**3/4.0 \
          - XX**4/7.0 + 6.0*XX**5 + 8.0*XX**6/7.0

    F22 = (3.0/56.0) * (113.0 * jnp.log(XX - 1.0) + 15.0 * jnp.log(XX + 1.0))

    S2_val = F12 / (XX**2 - 1.0) + (XX**2 - 1.0) * F22
    return S2_val
    
# @eqx.filter_jit
def tov_solver(eos, pc):
    r"""
    Solve TOV equations for a single neutron star in EiBI gravity.
    
    This function computes the structure of a single neutron star by solving the
    modified Tolman-Oppenheimer-Volkoff equations in Eddington-inspired Born-Infeld
    gravity for a given central pressure. The solution includes mass, radius, and
    tidal Love number calculations.
    
    The integration starts from the stellar center using a series expansion for
    initial conditions and proceeds outward until the surface (where pressure
    drops to zero) is reached. The method uses a Dopri5 ODE solver with adaptive
    stepsize control for numerical accuracy.
    
    The tidal Love number $k_2$ is computed using the surface value of the tidal
    variable $y_R$ through:
    
    .. math::
        k_2 = \text{calc\_k2\_eibi}(R, M, y_R, \text{eos})
    
    Args:
        eos (dict): Equation of state dictionary containing:
            - "p": Pressure array [geometric units]
            - "h": Enthalpy array
            - "e": Energy density array [geometric units]
            - "dloge_dlogp": Derivative d(log ε)/d(log p)
            - "kappa": EiBI gravity parameter $\kappa$ [km²]
            - "Lambda_cosmo": Cosmological constant parameter
        pc (float): Central pressure value [geometric units] at which to solve the TOV equations.
    
    Returns:
        tuple: A tuple containing:
            - M: Gravitational mass [geometric units]
            - R: Circumferential radius [geometric units]
            - k2: Tidal Love number
    
    Note:
        The function filters out unsuccessful integrations (non-positive mass or
        solver failures) by returning NaN values. Initial conditions are derived
        from a series expansion around the stellar center in GR (assuming EiBI 
        correction is small) to ensure numerical stability.
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

    # Initial values using series expansion near center
    dh  = -1e-3 * hc
    h0  = hc + dh
    r0  = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
    r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
    m0  = 4.0 * jnp.pi * ec * jnp.power(r0, 3.0) / 3.0
    m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
    H0  = r0 * r0
    b0  = 2.0 * r0
    yr0 = 2.0

    y0  = (r0, m0, yr0)

    sol = diffeqsolve(
        ODETerm(tov_ode),
        Dopri5(scan_kind="bounded"),
        t0=h0,
        t1=0,
        dt0=None,
        y0=y0,
        args=eos,
        saveat=SaveAt(t1=True),
        stepsize_controller=PIDController(rtol=1e-6, atol=1e-7),
        throw=False,
    )
    
    R = sol.ys[0][-1]
    M = sol.ys[1][-1]
    y_R = sol.ys[2][-1]

    # Filter out unsuccessful results
    R = jnp.where((sol.result == RESULTS.successful) & (M>0), R, jnp.nan)
    M = jnp.where((sol.result == RESULTS.successful) & (M>0), M, jnp.nan)
    y_R = jnp.where((sol.result == RESULTS.successful) & (M>0), y_R, jnp.nan)

    k2 = calc_k2_eibi(R, M, y_R, eos)
    return M, R, k2
