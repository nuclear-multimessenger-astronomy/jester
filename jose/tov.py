# this whole script is written in geometric unit
import jax.numpy as jnp
from diffrax import (
    diffeqsolve,
    ODETerm,
    Dopri5,
    SaveAt,
    PIDController
)


def tov_ode(h, y, eos):
    r, m, H, b = y
    e = eos.energy_density_from_pseudo_enthalpy(h)
    p = eos.pressure_from_pseudo_enthalpy(h)
    dedp = e / p * eos.dloge_dlogp_from_pseudo_enthalpy(h)

    A = 1. / (1. - 2. * m / r)
    C1 = 2. / r + A * (2. * m / (r * r) + 4. * jnp.pi * r * (p - e))
    C0 = A * (- 6 / (r * r) + 4. * jnp.pi * (e + p) * dedp
              + 4. * jnp.pi * (5. * e + 9. * p)) - \
        jnp.power(2. * (m + 4. * jnp.pi * r * r * r * p) / (r * (r - 2. * m)), 2.)

    drdh = -r * (r - 2. * m) / (m + 4. * jnp.pi * r * r * r * p)
    dmdh = 4. * jnp.pi * r * r * e * drdh
    dHdh = b * drdh
    dbdh = -(C0 * H + C1 * b) * drdh

    dydt = drdh, dmdh, dHdh, dbdh

    return dydt


def calc_k2(R, M, H, b):

    y = R * b / H
    C = M / R

    num = (8. / 5.) * jnp.power(1 - 2 * C, 2.) * jnp.power(C, 5.) * \
        (2 * C * (y - 1) - y + 2)
    den = 2 * C * (4 * (y + 1) * jnp.power(C, 4) + (6 * y - 4) * jnp.power(C, 3) +
                   (26 - 22 * y) * C * C + 3 * (5 * y - 8) * C - 3 * y + 6)
    den -= 3 * jnp.power(1 - 2 * C, 2) * (2 * C * (y - 1) - y + 2) * \
        jnp.log(1.0 / (1 - 2 * C))

    return num / den


def tov_solver(eos, pc):

    # central values
    hc = eos.pseudo_enthalpy_from_pressure(pc)
    ec = eos.energy_density_from_pseudo_enthalpy(hc)
    dedp_c = ec / pc * eos.dloge_dlogp_from_pseudo_enthalpy(hc)
    dhdp_c = 1. / (ec + pc)
    dedh_c = dedp_c / dhdp_c

    # initial values
    dh = -1e-3 * hc
    h0 = hc + dh
    r0 = jnp.sqrt(3. * (-dh) / 2. / jnp.pi / (ec + 3. * pc))
    r0 *= 1. - 0.25 * (ec - 3. * pc - 0.6 * dedh_c) * (-dh) / (ec + 3. * pc)
    m0 = 4. * jnp.pi * ec * jnp.power(r0, 3.) / 3.
    m0 *= 1. - 0.6 * dedh_c * (-dh) / ec
    H0 = r0 * r0
    b0 = 2. * r0

    y0 = (r0, m0, H0, b0)

    sol = diffeqsolve(
        ODETerm(tov_ode),
        Dopri5(),
        t0=h0,
        t1=0,
        dt0=dh,
        y0=y0,
        args=eos,
        saveat=SaveAt(ts=[h0, 0]),
        stepsize_controller=PIDController(rtol=1e-5, atol=1e-6)
    )

    R = sol.ys[0][-1]
    M = sol.ys[1][-1]
    H = sol.ys[2][-1]
    b = sol.ys[3][-1]

    # take one final Euler step to get to the surface
    #    y1 = [R, M, H, b]
    #    dydt1 = tov_ode(h1, y1, eos)
    #    # take one extra step towards the surface
    #    y1 = jnp.array(y1) + jnp.array(dydt1) * (0. - h1)
    #
    #    R, M, H, b = y1
    k2 = calc_k2(R, M, H, b)

    return M, R, k2
