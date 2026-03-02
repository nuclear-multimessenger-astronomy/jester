r"""
Modified TOV equation solver within Eddington-Inspired Born-Infeld (EiBI) gravity.

This module implements the modified Tolman-Oppenheimer-Volkoff equations in
Eddington-inspired Born-Infeld gravity, which introduces a parameter $\kappa$
that modifies gravitational interactions. The equations reduce to standard
General Relativity in the limit $\kappa \to 0$.

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Reference:**
- M. Bañados and P. G. Ferreira, Phys. Rev. Lett. 105, 011101 (2010)
- I. Prasetyo et al., Phys. Rev. D 104, 084029 (2021)
- Implementation of tidal deformability adapted from Fortran code by Ilham Prasetyo,
  https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
"""

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, RESULTS

from jesterTOV import utils
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, TOVSolution


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

    # Guard h against non-positive values. Diffrax's adaptive step controller
    # evaluates the ODE at tentative points that can overshoot t1=0 into h <= 0.
    # interp_in_logspace computes jnp.log(h), which is NaN for h < 0, producing
    # a NaN that propagates into the adjoint backward pass and corrupts gradients.
    h_safe = jnp.maximum(h, hs[0])

    e = utils.interp_in_logspace(h_safe, hs, es)
    p = utils.interp_in_logspace(h_safe, hs, ps)
    dedp = e / p * jnp.interp(h_safe, hs, dloge_dlogps)

    K = eos["kappa"]
    Lambda_cosmo = eos["Lambda_cosmo"]
    lambda_small = K * Lambda_cosmo + 1  # Eq. 8 from https://arxiv.org/pdf/2109.05718
    EPS = 1e-19  # small value to avoid zero division error
    # close to surface, A and B -> 1. Small EPS added to make sure 1/(A-B) etc is safe.
    B = jnp.sqrt(1 - 8 * jnp.pi * K * p) + EPS
    A = jnp.sqrt(1 + 8 * jnp.pi * K * e) + 1.5 * EPS
    b = B
    a = A
    AA = 1 + 8 * jnp.pi * K * e + EPS
    BB = 1 - 8 * jnp.pi * K * p + 1.2 * EPS

    A_per_BBB = jnp.sqrt((1 + 8 * jnp.pi * K * e) / (1 - 8 * jnp.pi * K * p)) / BB
    AB = jnp.sqrt((1 - 8 * jnp.pi * K * p) * (1 + 8 * jnp.pi * K * e))

    Q1 = 1 / dedp  # cs2(p0, G1, G2, G3,p)
    Q2 = -Q1 * B / A

    term1 = (
        (r / (2 * K)) * (1 / (a * b + EPS) + a / (b * b * b + EPS) - 2)
        + (2 * m) / (r * r + EPS)
        + Lambda_cosmo * r / (3 * lambda_small)
    )
    denominator1 = (4 / (AA - BB + EPS)) + (3 / (BB + EPS)) + (1 / (AA + EPS)) * dedp
    denominator2 = 1 - 2 * m / r - Lambda_cosmo * r * r / (3 * lambda_small)

    dpdr = -1 / (4 * jnp.pi * K) * term1 / (denominator1 * denominator2)

    dpdr_GR = (
        -e
        * m
        / (r * r)
        * (1 + p / (e))
        * (1 + 4 * jnp.pi * r * r * r * p / m)
        / (1 - 2 * m / r)
    )
    dpdr = jnp.where(
        jnp.abs(K) < 1e4, dpdr_GR, dpdr
    )  # turn to GR when K<0.01 km^2 for numerical stability
    dmdr = (r * r / (4 * K)) * (2 - 3 / (a * b + EPS) + a / (b * b * b + EPS))
    dmdr_GR = 4.0 * jnp.pi * r * r * e
    dmdr = jnp.where(jnp.abs(K) < 1e4, dmdr_GR, dmdr)

    # For tidal deformability
    dbetadr = -2 * dpdr * (2 * jnp.pi * K * (3 / BB + 1 / AA * dedp) + 1 / (e + p))
    expalpha = 1 / (1 - 2 * m / r - Lambda_cosmo * r * r / (3 * lambda_small))

    f_func = (r * expalpha / K) * (1 / (A * B) - 1) + expalpha / r + 1 / r

    g_func = (2 * expalpha * A / (K * B * B * B)) * (
        2
        - (4 / (AA - BB + EPS))
        * 1
        / ((4 / (AA - BB + EPS)) + (1 / AA) * dedp + (3 / BB))
    ) - ((2 * (2 + 1) * expalpha / (r * r)) + (2 * expalpha / K) + dbetadr * dbetadr)

    drdh = (e + p) / dpdr
    dmdh = dmdr * drdh
    dyrdr = -f_func * yr - r * g_func - yr * yr / r + yr / r
    dyrdh = dyrdr * drdh
    dydt = drdh, dmdh, dyrdh  # output
    return dydt


def calc_k2_eibi(R, M, y_R, eos):
    r"""
    Calculate the even-parity tidal Love number $k_2$ for EiBI gravity.

    Computes the tidal Love number $k_2$ (dimensionless quadrupolar tidal deformability)
    following the method from I. Prasetyo et al., Phys. Rev. D 104, 084029 (2021).
    Uses geometric units (:math:`G = c = 1`).

    The calculation uses compactness :math:`C = M/R` and involves several helper
    functions (:math:`Q_2`, :math:`P_2`, :math:`T_2`, :math:`S_2`) and their derivatives.

    The final expression is:
    .. math::
        k_2 = -\frac{(2\ell-1)!!}{2} \frac{K_{N2}}{K_{D2}}

    where for :math:`\ell=2`, :math:`(2\ell-1)!! = 3!! = 3`.

    Args:
        R (float or Array): Stellar radius [geometric units].
        M (float or Array): Gravitational mass [geometric units].
        y_R (float or Array): Tidal variable :math:`y(R) = r(dH/dr)/H` evaluated at the surface.
        eos (dict): EOS dictionary containing:
            - "kappa": EiBI parameter $\kappa$ [Geom]
            - "Lambda_cosmo": Cosmological constant $\Lambda_\mathrm{cosmo}$

    Returns:
        Array: Tidal Love number $k_2$ [dimensionless].

    Note:
        Signature intentionally matches `calc_k2_ST(R, M, y_R)` for consistency
        with other TOV solvers in the codebase.
    """

    # constants / choices
    YL = 2.0
    LAM = eos["Lambda_cosmo"]  # chosen default (see notes)
    K = eos["kappa"]  # choose K = YL as a best-effort mapping from Fortran
    # L = LAM * K + 1.0 #original
    L = LAM * K + 1.0

    # compactness
    C = M / R

    # eps as in Fortran but with GSS=MSS=1 removed (geometrized units)
    eps = (LAM / L) * (C**2)

    # coefficients a_?_2 and b_?_2 (copied from Fortran)
    a12 = 0.0
    a22 = C**2 / 3.0
    a32 = 113.0 * C**2 / 84.0
    a42 = 13.0 * C**2 / 9.0

    b12 = 15.0 / (8.0 * C**3)
    b22 = 0.0
    b32 = 1787.0 / (392.0 * C**3)
    b42 = (
        -(3305.0 / 448.0 + 5.0 * (jnp.pi**2) / 7.0 + 15.0 * (jnp.log(2.0) ** 2) / 7.0)
        / C**3
    )
    # Use pre-computed gradients
    DQ2 = DQ2_fn
    DP2 = DP2_fn
    DT2 = DT2_fn
    DS2 = DS2_fn

    # evaluate helper combinations (matching Fortran names)
    QB2 = y_R * Q2_fn(C) + C * DQ2(C)
    PB2 = y_R * P2_fn(C) + C * DP2(C)
    TB2 = y_R * T2_fn(C) + C * DT2(C)
    SB2 = y_R * S2_fn(C) + C * DS2(C)

    # KN2 and KD2 as in Fortran
    KN2 = (
        (a12 + eps * a32) * QB2
        + (a22 + eps * a42) * PB2
        + eps * (a12 * TB2 + a22 * SB2)
    )
    KD2 = (
        (b12 + eps * b32) * QB2
        + (b22 + eps * b42) * PB2
        + eps * (b12 * TB2 + b22 * SB2)
    )

    # double factorial factor: DOUBFAC(2*YL - 1)
    n_df = int(2 * int(YL) - 1)
    dfac = double_factorial(n_df)

    K2 = -(dfac / 2.0) * (KN2 / KD2)
    return K2


def double_factorial(n):
    r"""
    Compute the double factorial :math:`n!!` for integer :math:`n`.

    The double factorial is defined as:
    .. math::
        n!! = n \times (n-2) \times (n-4) \times \cdots \times
        \begin{cases}
            2 & \text{if } n \text{ even} \\
            1 & \text{if } n \text{ odd}
        \end{cases}

    For :math:`n \le 0`, returns 1 by convention.

    Args:
        n (int or Array): Integer input.

    Returns:
        Array: Double factorial of n.

    Note:
        Uses JAX control flow (lax.cond, fori_loop) for compatibility with JAX
        transformations (grad, vmap, jit).
    """

    def body(i, val):
        return val * i

    n = jnp.asarray(n, int)
    res = jax.lax.cond(
        n <= 0,
        lambda _: 1,
        lambda _: jax.lax.fori_loop(
            1,
            n + 1,
            lambda i, v: jax.lax.cond(
                i % 2 == n % 2, lambda _: v * i, lambda _: v, None
            ),
            1,
        ),
        None,
    )
    return res


def Q2_fn(C):
    r"""
    Helper function :math:`Q_2(C)` for tidal Love number calculation in EiBI gravity.

    Defined in terms of compactness :math:`C = M/R` as:
    .. math::
        Q_2(C) = 6(XX-1)(XX+1) \left[-\frac{(XX-1)^2}{48(XX+1)^2} + \frac{XX-1}{6(XX+1)}
                  - \frac{XX+1}{6(XX-1)} + \frac{(XX+1)^2}{48(XX-1)^2}
                  + \frac{1}{4}\ln\frac{XX+1}{XX-1}\right]

    where :math:`XX = -1 + 1/C`.

    Args:
        C (float or Array): Compactness :math:`C = M/R` [dimensionless].

    Returns:
        Array: Value of :math:`Q_2(C)`.

    References:
        Implementation adapted from Fortran code by Ilham Prasetyo,
        https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
    """
    C = jnp.asarray(C)
    XX = -1.0 + 1.0 / C

    term1 = -((-1.0 + XX) ** 2) / (48.0 * (1.0 + XX) ** 2)
    term2 = (-1.0 + XX) / (6.0 * (1.0 + XX))
    term3 = -(1.0 + XX) / (6.0 * (-1.0 + XX))
    term4 = (1.0 + XX) ** 2 / (48.0 * (-1.0 + XX) ** 2)
    term5 = (-jnp.log(-1.0 + XX) + jnp.log(1.0 + XX)) / 4.0

    inner = term1 + term2 + term3 + term4 + term5
    Q2_val = 6.0 * (-1.0 + XX) * (1.0 + XX) * inner
    return Q2_val


def P2_fn(C):
    r"""
    Helper function :math:`P_2(C)` for tidal Love number calculation in EiBI gravity.

    Defined in terms of compactness :math:`C = M/R` as:
    .. math::
        P_2(C) = \frac{3(1-XX)^2(1+XX)}{XX-1}

    where :math:`XX = -1 + 1/C`.

    Args:
        C (float or Array): Compactness :math:`C = M/R` [dimensionless].

    Returns:
        Array: Value of :math:`P_2(C)`.

    References:
        Implementation adapted from Fortran code by Ilham Prasetyo,
        https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
    """
    C = jnp.asarray(C)
    XX = -1.0 + 1.0 / C
    # keep algebraic structure faithful to Fortran
    P2_val = (3.0 * (1.0 - XX) ** 2 * (1.0 + XX)) / (-1.0 + XX)
    return P2_val


def T2_fn(C):
    r"""
    Helper function :math:`T_2(C)` for tidal Love number calculation in EiBI gravity.

    Defined in terms of compactness :math:`C = M/R` as:
    .. math::
        T_2(C) = \frac{F_{32}}{(XX+1)(XX-1)^2} + \frac{F_{42}}{XX+1}\ln(XX-1)
                 + F_{52}\frac{XX+1}{XX-1}\ln(XX+1) + (XX^2-1)F_{62}

    where :math:`XX = -1 + 1/C`, and :math:`F_{32}`, :math:`F_{42}`, :math:`F_{52}`,
    :math:`F_{62}` are polynomial expressions in :math:`XX`.

    This function requires `PLG_fn(C)` (dilogarithm approximation) to be defined.

    Args:
        C (float or Array): Compactness :math:`C = M/R` [dimensionless].

    Returns:
        Array: Value of :math:`T_2(C)`.

    References:
        Implementation adapted from Fortran code by Ilham Prasetyo,
        https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
    """
    C = jnp.asarray(C)
    XX = -1.0 + 1.0 / C

    # polynomial pieces
    F32 = (
        (235.0 / 56.0)
        + (1357.0 * XX) / 56.0
        - (1469.0 * XX**2) / 84.0
        - (1987.0 * XX**3) / 84.0
        + (2057.0 * XX**4) / 168.0
        + (389.0 * XX**5) / 56.0
        - (8.0 * XX**6) / 7.0
    )

    F42 = (
        -(25.0 / 14.0)
        + (153.0 * XX) / 14.0
        + (171.0 * XX**2) / 14.0
        - (17.0 * XX**3) / 14.0
        - (25.0 * XX**4) / 7.0
        - (4.0 * XX**5) / 7.0
    )

    F52 = (
        -(25.0 / 14.0)
        + (34.0 * XX) / 7.0
        - (93.0 * XX**2) / 14.0
        + (13.0 * XX**3) / 7.0
        + (4.0 * XX**4) / 7.0
    )

    # the Fortran piece: ( -24/7 )*( DLOG(XX-1)*DLOG((XX+1)/4) + 2*PLG(C) )
    F62 = -(24.0 / 7.0) * (
        jnp.log(XX - 1.0) * jnp.log((XX + 1.0) / 4.0) + 2.0 * PLG_fn(C)
    )

    T2_val = (
        F32 / ((XX + 1.0) * (XX - 1.0) ** 2)
        + F42 / (XX + 1.0) * jnp.log(XX - 1.0)
        + F52 * (XX + 1.0) / (XX - 1.0) * jnp.log(XX + 1.0)
        + (XX**2 - 1.0) * F62
    )

    return T2_val


def PLG_fn(C):
    r"""
    Approximation of the dilogarithm :math:`\mathrm{Li}_2((1-x)/2)` for EiBI tidal calculation.

    Computes an approximation to the dilogarithm (polylogarithm of order 2)
    needed for the :math:`T_2(C)` function. Uses piecewise polynomial fits:

    .. math::
        \mathrm{PLG}(C) \approx \mathrm{Li}_2\left(\frac{1-x}{2}\right)

    where :math:`x = 1/C - 1`.

    For :math:`0.01 < x \le 0.5`: 12th-degree polynomial fit.
    For :math:`0 < x \le 0.01`: approximation :math:`5894.17 - 5875.92 / x^{0.001}`.
    Otherwise returns 0.

    Args:
        C (float or Array): Compactness :math:`C = M/R` [dimensionless].

    Returns:
        Array: Approximation of :math:`\mathrm{Li}_2((1-x)/2)`.

    References:
        Implementation adapted from Fortran code by Ilham Prasetyo,
        https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
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
    r"""
    Helper function :math:`S_2(C)` for tidal Love number calculation in EiBI gravity.

    Defined in terms of compactness :math:`C = M/R` as:
    .. math::
        S_2(C) = \frac{F_{12}}{XX^2 - 1} + (XX^2 - 1)F_{22}

    where :math:`XX = -1 + 1/C`, :math:`F_{12}` is a 6th-degree polynomial in
    :math:`XX`, and :math:`F_{22} = \frac{3}{56}[113\ln(XX-1) + 15\ln(XX+1)]`.

    Args:
        C (float or Array): Compactness :math:`C = M/R` [dimensionless].

    Returns:
        Array: Value of :math:`S_2(C)`.

    References:
        Implementation adapted from Fortran code by Ilham Prasetyo,
        https://github.com/ilhamdotP/fortran2mathematica-TOV-EiBI-momin-tidal
    """
    C = jnp.asarray(C)
    XX = -1.0 + 1.0 / C

    F12 = (
        1.0
        + 21.0 * XX / 4.0
        - 46.0 * XX**2 / 7.0
        - 59.0 * XX**3 / 4.0
        - XX**4 / 7.0
        + 6.0 * XX**5
        + 8.0 * XX**6 / 7.0
    )

    F22 = (3.0 / 56.0) * (113.0 * jnp.log(XX - 1.0) + 15.0 * jnp.log(XX + 1.0))

    S2_val = F12 / (XX**2 - 1.0) + (XX**2 - 1.0) * F22
    return S2_val


# Pre-compute gradients outside function for performance
DQ2_fn = jax.grad(Q2_fn)
DP2_fn = jax.grad(P2_fn)
DT2_fn = jax.grad(T2_fn)
DS2_fn = jax.grad(S2_fn)


class EiBITOVSolver(TOVSolverBase):
    r"""
    Solver for modified TOV equations in Eddington-inspired Born-Infeld (EiBI) gravity.

    This class implements the Tolman-Oppenheimer-Volkoff equations within the
    EiBI gravity framework, which introduces a parameter $\kappa$ that modifies
    the gravitational field equations. The equations reduce to standard GR when
    $\kappa \to 0$. The solver also includes a cosmological constant term
    $\Lambda_\mathrm{cosmo}$.

    The modified TOV equations are derived from the EiBI action and include
    additional terms proportional to $\kappa$:

    .. math::
        \frac{dp}{dr} = -\frac{1}{4\pi\kappa} \frac{
            \frac{r}{2\kappa}\left(\frac{1}{ab} + \frac{a}{b^3} - 2\right) + \frac{2m}{r^2} + \frac{\Lambda_\mathrm{cosmo} r}{3\lambda}
        }{
            \left[\frac{4}{A - B} + \frac{3}{B} + \frac{1}{A}\frac{de}{dp}\right]
            \left[1 - \frac{2m}{r} - \frac{\Lambda_\mathrm{cosmo} r^2}{3\lambda}\right]
        }

    .. math::
        \frac{dm}{dr} = \frac{r^2}{4\kappa}\left(2 - \frac{3}{ab} + \frac{a}{b^3}\right)

    where:
    - $a = \sqrt{1 + 8\pi\kappa\varepsilon}$, $b = \sqrt{1 - 8\pi\kappa p}$
    - $A = 1 + 8\pi\kappa\varepsilon$, $B = 1 - 8\pi\kappa p$
    - $\lambda = \kappa\Lambda_\mathrm{cosmo} + 1$

    For numerical stability, the system automatically switches to standard GR
    equations when $|\kappa| < 10^4$ (effectively the GR limit).

    The tidal Love number $k_2$ is computed using the method from
    I. Prasetyo et al., Phys. Rev. D 104, 084029 (2021).

    References:
        - Banados & Ferreira, Phys. Rev. Lett. 105, 011101 (2010)
        - I. Prasetyo et al., Phys. Rev. D 104, 084029 (2021)
        - Implementation adapted from Fortran code by Ilham Prasetyo

    # TODO: check the default values here
    Example:
        >>> from jesterTOV.tov.eibi import EiBITOVSolver
        >>> from jesterTOV.eos import MetaModelEOS
        >>> solver = EiBITOVSolver()
        >>> eos = MetaModelEOS()
        >>> eos_data = eos.construct_eos(...)
        >>> solution = solver.solve(eos_data, pc=1e-5, tov_params={"kappa": 1e4, "Lambda_cosmo": 1.4657e-52})
        >>> print(f"Mass: {solution.M:.3f}, Radius: {solution.R:.3f}, k2: {solution.k2:.3f}")
    """

    def solve(
        self, eos_data: EOSData, pc: float, tov_params: dict[str, float]
    ) -> TOVSolution:
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
            eos_data: EOS quantities (type-safe dataclass) containing:
                - ps: Pressure array [geometric units]
                - hs: Enthalpy array
                - es: Energy density array [geometric units]
                - dloge_dlogps: Derivative d(log ε)/d(log p)
            pc (float): Central pressure value [geometric units] at which to solve the TOV equations.
            tov_params: EiBI gravity parameters as returned by :meth:`~jesterTOV.tov.base.TOVSolverBase.fetch_params`.
                Must contain:
                - ``kappa``: EiBI gravity parameter :math:`\kappa` [Geom]
                - ``Lambda_cosmo``: Cosmological constant [1/m²]

        Returns:
            TOVSolution: Mass, radius, and Love number in geometric units.

        Note:
            The function filters out unsuccessful integrations (non-positive mass or
            solver failures) by returning NaN values. Initial conditions are derived
            from a series expansion around the stellar center in GR (assuming EiBI
            correction is small) to ensure numerical stability.
        """
        # Extract EOS interpolation arrays

        kappa = tov_params["kappa"]
        Lambda_cosmo = tov_params["Lambda_cosmo"]
        # Convert EOSData to dictionary for ODE solver
        eos_dict = {
            "p": eos_data.ps,
            "h": eos_data.hs,
            "e": eos_data.es,
            "dloge_dlogp": eos_data.dloge_dlogps,
            # Add modification parameters
            "kappa": kappa,
            "Lambda_cosmo": Lambda_cosmo,
        }

        ps = eos_data.ps
        hs = eos_data.hs
        es = eos_data.es
        dloge_dlogps = eos_data.dloge_dlogps
        # Central values and initial conditions
        hc = utils.interp_in_logspace(pc, ps, hs)
        ec = utils.interp_in_logspace(hc, hs, es)
        dloge_dlogps = eos_data.dloge_dlogps
        dedp_c = ec / pc * jnp.interp(hc, hs, dloge_dlogps)
        dhdp_c = 1.0 / (ec + pc)
        dedh_c = dedp_c / dhdp_c

        # Initial values using series expansion near center
        dh = -1e-3 * hc
        h0 = hc + dh
        r0 = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
        r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
        m0 = 4.0 * jnp.pi * ec * jnp.power(r0, 3.0) / 3.0
        m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
        yr0 = 2.0

        y0 = (r0, m0, yr0)

        sol = diffeqsolve(
            ODETerm(tov_ode),
            Dopri5(scan_kind="bounded"),
            t0=h0,
            t1=0,
            dt0=dh,
            y0=y0,
            args=eos_dict,
            saveat=SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=1e-6, atol=1e-7),
            throw=False,
        )

        R = sol.ys[0][-1]  # type: ignore[index]
        M = sol.ys[1][-1]  # type: ignore[index]
        y_R = sol.ys[2][-1]  # type: ignore[index]

        # Filter out unsuccessful results
        # TODO: check if the NaNs here can break the inference
        success = (sol.result == RESULTS.successful) & (M > 0)  # type: ignore[operator]
        R = jnp.where(success, R, jnp.nan)
        M = jnp.where(success, M, jnp.nan)
        y_R = jnp.where(success, y_R, jnp.nan)

        k2 = calc_k2_eibi(R, M, y_R, eos_dict)
        return TOVSolution(M=M, R=R, k2=k2)  # type: ignore[arg-type]

    def get_required_parameters(self) -> list[str]:
        r"""
        Return the additional theory parameters required by EiBI gravity.

        EiBI gravity requires two parameters beyond the standard EOS:
        1. $\kappa$ - EiBI gravity parameter [geometric units], controls deviation from GR
        2. $\Lambda_\mathrm{cosmo}$ - Cosmological constant [1/m²]

        Returns:
            list[str]: ["kappa", "Lambda_cosmo"]
        """
        return ["kappa", "Lambda_cosmo"]
