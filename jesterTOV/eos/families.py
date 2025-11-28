r"""Neutron star family construction utilities."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .. import utils, tov, ptov, STtov


def locate_lowest_non_causal_point(cs2):
    r"""
    Find the first point where the equation of state becomes non-causal.

    The speed of sound squared :math:`c_s^2 = dp/d\varepsilon` must satisfy
    :math:`c_s^2 \leq 1` (in units where :math:`c = 1`) for causality.
    This function locates the first density where this condition is violated.

    Args:
        cs2 (Array): Speed of sound squared values.

    Returns:
        int: Index of first non-causal point, or -1 if EOS is everywhere causal.
    """
    mask = cs2 >= 1.0
    any_ones = jnp.any(mask)
    indices = jnp.arange(len(cs2))
    masked_indices = jnp.where(mask, indices, len(cs2))
    first_index = jnp.min(masked_indices)
    return jnp.where(any_ones, first_index, -1)


def construct_family(eos: tuple, ndat: Int = 50, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    Solve the TOV equations and generate mass-radius-tidal deformability relations.

    This function constructs a neutron star family by solving the Tolman-Oppenheimer-Volkoff (TOV)
    equations for a range of central pressures. The TOV equations describe the hydrostatic
    equilibrium of a spherically symmetric, static star:

    .. math::
        \frac{dm}{dr} &= 4\pi r^2 \varepsilon(r) \\
        \frac{dp}{dr} &= -\frac{[\varepsilon(r) + p(r)][m(r) + 4\pi r^3 p(r)]}{r[r - 2m(r)]}

    Args:
        eos (tuple): Tuple of the EOS data (ns, ps, hs, es, dloge_dlogps).
        ndat (int, optional): Number of datapoints used when constructing the central pressure grid. Defaults to 50.
        min_nsat (int, optional): Starting density for central pressure in numbers of :math:`n_0`
                                 (assumed to be 0.16 :math:`\mathrm{fm}^{-3}`). Defaults to 2.

    Returns:
        tuple: A tuple containing:

            - :math:`\log(p_c)`: Logarithm of central pressures [geometric units]
            - :math:`M`: Gravitational masses [:math:`M_{\odot}`]
            - :math:`R`: Circumferential radii [:math:`\mathrm{km}`]
            - :math:`\Lambda`: Dimensionless tidal deformabilities
    """
    # Construct the dictionary
    ns, ps, hs, es, dloge_dlogps = eos
    eos_dict = dict(p=ps, h=hs, e=es, dloge_dlogp=dloge_dlogps)

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    # end at pc at pmax at which it is causal
    cs2 = ps / es / dloge_dlogps
    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2)]

    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    def solve_single_pc(pc):
        """Solve for single pc value"""
        return tov.tov_solver(eos_dict, pc)

    ms, rs, ks = jax.vmap(solve_single_pc)(pcs)

    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass and the radius to km
    ms /= utils.solar_mass_in_meter
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    # Limit masses to be below MTOV
    pcs, ms, rs, lambdas = utils.limit_by_MTOV(pcs, ms, rs, lambdas)

    # Get a mass grid and interpolate, since we might have dropped provided some duplicate points
    mass_grid = jnp.linspace(jnp.min(ms), jnp.max(ms), ndat)
    rs = jnp.interp(mass_grid, ms, rs)
    lambdas = jnp.interp(mass_grid, ms, lambdas)
    pcs = jnp.interp(mass_grid, ms, pcs)

    ms = mass_grid

    return jnp.log(pcs), ms, rs, lambdas


def construct_family_nonGR(eos: tuple, ndat: Int = 50, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    Solve modified TOV equations with beyond-GR corrections.

    This function extends the standard TOV equations to include phenomenological
    modifications that parameterize deviations from General Relativity. The modified
    pressure gradient equation becomes:

    .. math::
        \frac{dp}{dr} = -\frac{[\varepsilon(r) + p(r)][m(r) + 4\pi r^3 p(r)]}{r[r - 2m(r)]} - \frac{2\sigma(r)}{r}

    where :math:`\sigma(r)` contains the non-GR corrections parameterized by
    :math:`\lambda_{\mathrm{BL}}`, :math:`\lambda_{\mathrm{DY}}`, :math:`\lambda_{\mathrm{HB}}`,
    and post-Newtonian parameters :math:`\alpha`, :math:`\beta`, :math:`\gamma`.

    Args:
        eos (tuple): Extended EOS data including GR modification parameters.
        ndat (int, optional): Number of datapoints for central pressure grid. Defaults to 50.
        min_nsat (int, optional): Starting density in units of :math:`n_0`. Defaults to 2.

    Returns:
        tuple: A tuple containing:

            - :math:`\log(p_c)`: Logarithm of central pressures [geometric units]
            - :math:`M`: Gravitational masses [:math:`M_{\odot}`]
            - :math:`R`: Circumferential radii [:math:`\mathrm{km}`]
            - :math:`\Lambda`: Dimensionless tidal deformabilities
    """
    # Construct the dictionary
    (
        ns,
        ps,
        hs,
        es,
        dloge_dlogps,
        alpha,
        beta,
        gamma,
        lambda_BL,
        lambda_DY,
        lambda_HB,
    ) = eos
    eos_dict = dict(
        p=ps,
        h=hs,
        e=es,
        dloge_dlogp=dloge_dlogps,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lambda_BL=lambda_BL,
        lambda_DY=lambda_DY,
        lambda_HB=lambda_HB,
    )

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    # end at pc at pmax at which it is causal
    cs2 = ps / es / dloge_dlogps
    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2)]

    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    def solve_single_pc(pc):
        """Solve for single pc value"""
        return ptov.tov_solver(eos_dict, pc)

    ms, rs, ks = jax.vmap(solve_single_pc)(pcs)

    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass and the radius to km
    ms /= utils.solar_mass_in_meter
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    # Limit masses to be below MTOV
    pcs, ms, rs, lambdas = utils.limit_by_MTOV(pcs, ms, rs, lambdas)

    # Get a mass grid and interpolate, since we might have dropped provided some duplicate points
    mass_grid = jnp.linspace(jnp.min(ms), jnp.max(ms), ndat)
    rs = jnp.interp(mass_grid, ms, rs)
    lambdas = jnp.interp(mass_grid, ms, lambdas)
    pcs = jnp.interp(mass_grid, ms, pcs)

    ms = mass_grid

    return jnp.log(pcs), ms, rs, lambdas


def construct_family_ST(eos: tuple, ndat: Int = 50, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    # TODO:
    (updated later)
    """

    # Construct the dictionary
    ns, ps, hs, es, dloge_dlogps, beta_STs, phi_cs, nu_cs = eos
    # Here's EoS dict names defined
    eos_dict = dict(
        p=ps,
        h=hs,
        e=es,
        dloge_dlogp=dloge_dlogps,
        beta_ST=beta_STs,
        phi_c=phi_cs,
        nu_c=nu_cs,
    )

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    pc_max = eos_dict["p"][-1]
    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    def solve_single_pc(pc):
        """Solve for single pc value"""
        return STtov.tov_solver(eos_dict, pc)

    ms, rs, ks = jax.vmap(solve_single_pc)(pcs)

    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass and the radius to km
    ms /= utils.solar_mass_in_meter
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    # Limit masses to be below MTOV
    pcs, ms, rs, lambdas = utils.limit_by_MTOV(pcs, ms, rs, lambdas)

    # Get a mass grid and interpolate, since we might have dropped provided some duplicate points
    mass_grid = jnp.linspace(jnp.min(ms), jnp.max(ms), ndat)
    rs = jnp.interp(mass_grid, ms, rs)
    lambdas = jnp.interp(mass_grid, ms, lambdas)
    pcs = jnp.interp(mass_grid, ms, pcs)

    ms = mass_grid

    return jnp.log(pcs), ms, rs, lambdas


def construct_family_ST_sol(eos: tuple, ndat: Int = 1, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    # TODO: complete the description
    Also output stellar structure solution via sol_iter (interior) and solext (exterior)
    """

    # Construct the dictionary
    ns, ps, hs, es, dloge_dlogps, beta_STs, phi_cs, nu_cs = eos
    # Here's EoS dict names defined
    eos_dict = dict(
        p=ps,
        h=hs,
        e=es,
        dloge_dlogp=dloge_dlogps,
        beta_ST=beta_STs,
        phi_c=phi_cs,
        nu_c=nu_cs,
    )

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    pc_max = eos_dict["p"][-1]
    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    def solve_single_pc(pc):
        """Solve for single pc value"""
        return STtov.tov_solver_printsol(eos_dict, pc)

    ms, rs, ks, sol_iter, solext = jax.vmap(solve_single_pc)(pcs)

    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass and the radius to km
    ms /= utils.solar_mass_in_meter
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    # Limit masses to be below MTOV
    pcs, ms, rs, lambdas = utils.limit_by_MTOV(pcs, ms, rs, lambdas)

    # Get a mass grid and interpolate, since we might have dropped provided some duplicate points
    mass_grid = jnp.linspace(jnp.min(ms), jnp.max(ms), ndat)
    rs = jnp.interp(mass_grid, ms, rs)
    lambdas = jnp.interp(mass_grid, ms, lambdas)
    pcs = jnp.interp(mass_grid, ms, pcs)

    ms = mass_grid

    return jnp.log(pcs), ms, rs, lambdas, sol_iter, solext
