# this whole script is written in geometric unit
from . import utils
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController


def sigma_func(p, e, m, r, lambda_BL, lambda_DY, lambda_HB, gamma, alpha, beta):
    A = 1.0 / (1.0 - 2.0 * m / r)
    dpdr = -(e + p) * (m + 4.0 * jnp.pi * r * r * r * p) / r / r * A
    sigma = 0.0
    # models reviewed in https://doi.org/10.1140/epjc/s10052-020-8361-4
    # in Eq. 12, the power of epsilon should be 2
    sigma += -lambda_BL * r * r / 3.0 * (e + 3.0 * p) * (e + p) * A
    sigma += lambda_DY * 2.0 * m / r * p
    sigma += -(1.0 / lambda_HB - 1.0) * r / 2.0 * dpdr
    # post-Newtonian modification
    sigma += gamma * 2.0 * m / r * p * jnp.tanh(alpha * (m / r - beta))
    return sigma


def tov_ode(h, y, eos):
    # fetch the eos arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    # actual equations
    r, m, H, b = y
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)

    # evalute the sigma and dsigmadp
    sigma = sigma_func(
        p, e, m, r,
        eos["lambda_BL"],
        eos["lambda_DY"],
        eos["lambda_HB"],
        eos["gamma"],
        eos["alpha"],
        eos["beta"],
    )
    dsigmadp_fn = jax.grad(sigma_func, argnums=0)  # Gradient w.r.t. p
    dsigmade_fn = jax.grad(sigma_func, argnums=1)  # Gradient w.r.t. p
    dsigmadp = dsigmadp_fn(
        p, e, m, r,
        eos["lambda_BL"],
        eos["lambda_DY"],
        eos["lambda_HB"],
        eos["gamma"],
        eos["alpha"],
        eos["beta"],
    )
    dsigmadp += dedp * dsigmade_fn(
        p, e, m, r,
        eos["lambda_BL"],
        eos["lambda_DY"],
        eos["lambda_HB"],
        eos["gamma"],
        eos["alpha"],
        eos["beta"],
    )

    A = 1.0 / (1.0 - 2.0 * m / r)
    # terms for radius and mass
    dpdr = -(e + p) * (m + 4.0 * jnp.pi * r * r * r * p) / r / r * A
    dpdr += -2.0 * sigma / r  # adding non-GR contribution
    # terms for tidal deformability
    C1 = 2.0 / r + A * (2.0 * m / (r * r) + 4.0 * jnp.pi * r * (p - e))
    C0 = A * (
        -6 / (r * r)
        + 4.0 * jnp.pi * (e + p) * dedp / (1.0 - dsigmadp)
        + 4.0 * jnp.pi * (5.0 * e + 9.0 * p)
        + 16.0 * jnp.pi * sigma
    ) - jnp.power(
            2.0 * (m + 4.0 * jnp.pi * r * r * r * p) / (r * (r - 2.0 * m)),
            2.0
        )

    drdh = (e + p) / dpdr
    dmdh = 4.0 * jnp.pi * r * r * e * drdh
    dHdh = b * drdh
    dbdh = -(C0 * H + C1 * b) * drdh

    dydt = drdh, dmdh, dHdh, dbdh

    return dydt


def calc_k2(R, M, H, b):

    y = R * b / H
    C = M / R

    num = (
        (8.0 / 5.0)
        * jnp.power(1 - 2 * C, 2.0)
        * jnp.power(C, 5.0)
        * (2 * C * (y - 1) - y + 2)
    )
    den = (
        2
        * C
        * (
            4 * (y + 1) * jnp.power(C, 4)
            + (6 * y - 4) * jnp.power(C, 3)
            + (26 - 22 * y) * C * C
            + 3 * (5 * y - 8) * C
            - 3 * y
            + 6
        )
    )
    den -= (
        3
        * jnp.power(1 - 2 * C, 2)
        * (2 * C * (y - 1) - y + 2)
        * jnp.log(1.0 / (1 - 2 * C))
    )

    return num / den


def tov_solver(eos, pc):
    # fetch the eos arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    # central values
    hc = utils.interp_in_logspace(pc, ps, hs)
    ec = utils.interp_in_logspace(hc, hs, es)
    dedp_c = ec / pc * jnp.interp(hc, hs, dloge_dlogps)
    dhdp_c = 1.0 / (ec + pc)
    dedh_c = dedp_c / dhdp_c

    # initial values
    dh = -1e-3 * hc
    h0 = hc + dh
    r0 = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
    r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
    m0 = 4.0 * jnp.pi * ec * jnp.power(r0, 3.0) / 3.0
    m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
    H0 = r0 * r0
    b0 = 2.0 * r0

    y0 = (r0, m0, H0, b0)

    sol = diffeqsolve(
        ODETerm(tov_ode),
        Dopri5(scan_kind="bounded"),
        t0=h0,
        t1=0,
        dt0=dh,
        y0=y0,
        args=eos,
        saveat=SaveAt(t1=True),
        stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
    )

    R = sol.ys[0][-1]
    M = sol.ys[1][-1]
    H = sol.ys[2][-1]
    b = sol.ys[3][-1]

    k2 = calc_k2(R, M, H, b)

    return M, R, k2
