# FIXME: move to a different place

r"""Neutron star family construction utilities."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from diffrax import Solution

from jesterTOV import utils, tov, ptov, STtov, STtov_Greci, eibitov
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


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
        eos (tuple): Tuple of (ns, ps, hs, es, dloge_dlogps, cs2) EOS data.
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
    # Unpack EOS
    ns, ps, hs, es, dloge_dlogps, cs2s = eos
    eos_dict = dict(p=ps, h=hs, e=es, dloge_dlogp=dloge_dlogps)

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    # End at pc at pmax at which it is causal
    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2s)]

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
        eos (tuple): Extended EOS data (ns, ps, hs, es, dloge_dlogps, alpha, beta, gamma,
            lambda_BL, lambda_DY, lambda_HB, cs2) including GR modification parameters.
        ndat (int, optional): Number of datapoints for central pressure grid. Defaults to 50.
        min_nsat (int, optional): Starting density in units of :math:`n_0`. Defaults to 2.

    Returns:
        tuple: A tuple containing:

            - :math:`\log(p_c)`: Logarithm of central pressures [geometric units]
            - :math:`M`: Gravitational masses [:math:`M_{\odot}`]
            - :math:`R`: Circumferential radii [:math:`\mathrm{km}`]
            - :math:`\Lambda`: Dimensionless tidal deformabilities
    """
    # Unpack EOS
    (
        ns,
        ps,
        hs,
        es,
        dloge_dlogps,
        cs2s,
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
        cs2=cs2s,
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

    # End at pc at pmax at which it is causal
    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2s)]

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
    ns, ps, hs, es, dloge_dlogps, cs2s, beta_STs, phi_cs, nu_cs = eos
    # Here's EoS dict names defined
    eos_dict = dict(
        p=ps,
        h=hs,
        e=es,
        dloge_dlogp=dloge_dlogps,
        cs2=cs2s,
        beta_ST=beta_STs,
        phi_c=phi_cs,
        nu_c=nu_cs,
    )

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2s)]
    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    def solve_single_pc(pc: Array) -> tuple[float, float, int]:
        """Solve for single pc value (returns scalars, vmap will vectorize)"""
        return STtov.tov_solver(eos_dict, pc)  # type: ignore[return-value]

    ms, rs, ks = jax.vmap(solve_single_pc)(pcs)  # type: ignore[misc]
    ms = jnp.asarray(ms)
    rs = jnp.asarray(rs)
    ks = jnp.asarray(ks)

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
    Solution,
    Solution,
]:
    r"""
    # TODO: complete the description
    Also output stellar structure solution via sol_iter (interior) and solext (exterior)
    """

    # Construct the dictionary
    ns, ps, hs, es, dloge_dlogps, cs2s, beta_STs, phi_cs, nu_cs = eos
    # Here's EoS dict names defined
    eos_dict = dict(
        p=ps,
        h=hs,
        e=es,
        dloge_dlogp=dloge_dlogps,
        cs2=cs2s,
        beta_ST=beta_STs,
        phi_c=phi_cs,
        nu_c=nu_cs,
    )

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2s)]
    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    def solve_single_pc(pc: Array) -> tuple[float, float, int, Solution, Solution]:
        """Solve for single pc value (returns scalars, vmap will vectorize)"""
        return STtov.tov_solver_printsol(eos_dict, pc)  # type: ignore[return-value]

    ms, rs, ks, sol_iter, solext = jax.vmap(solve_single_pc)(pcs)  # type: ignore[misc]
    ms = jnp.asarray(ms)
    rs = jnp.asarray(rs)
    ks = jnp.asarray(ks)

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

def construct_family_eibi(eos: tuple, ndat: Int = 50, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    Construct neutron star family in Eddington-inspired Born-Infeld (EiBI) gravity.
    
    This function generates a sequence of neutron star solutions by solving the modified
    TOV equations in EiBI gravity across a range of central pressures. The EiBI framework
    introduces a parameter $\kappa$ that modifies gravitational interactions, reducing to
    standard General Relativity when $\kappa \to 0$.
    
    The solutions are computed for central pressures ranging from a minimum determined
    by nuclear saturation density ($n_{\mathrm{sat}} = 0.16$ fm⁻³) to a maximum where
    the equation of state becomes acausal (sound speed exceeds speed of light).
    
    The tidal deformability $\Lambda$ is calculated from the Love number $k_2$ and
    compactness $C = M/R$ using:
    

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
    
    The resulting mass-radius relation is truncated at the maximum TOV mass and
    interpolated to ensure uniform sampling.
    
    Args:
        eos (tuple): Extended EOS tuple containing:
            - ns: Baryon density array [fm⁻³]
            - ps: Pressure array [geometric units]
            - hs: Enthalpy array
            - es: Energy density array [geometric units]
            - dloge_dlogps: Derivative d(log ε)/d(log p)
            - kappa: EiBI gravity parameter $\kappa$ [m²]
            - Lambda_cosmo: Cosmological constant parameter
        ndat (int, optional): Number of points in central pressure grid. Defaults to 50.
        min_nsat (float, optional): Starting density in units of nuclear saturation density. 
                                    Defaults to 2.
    
    Returns:
        tuple: A tuple containing:
            - $\log(p_c)$: Logarithm of central pressures [geometric units]
            - $M$: Gravitational masses [$M_{\odot}$]
            - $R$: Circumferential radii [km]
            - $\Lambda$: Dimensionless tidal deformabilities
    
    Note:
        The function automatically handles unit conversions from geometric units to
        physical units (solar masses and kilometers) and ensures the resulting
        mass-radius relation represents only stable configurations.
    """

    # Construct the dictionary
    (
        ns,
        ps,
        hs,
        es,
        dloge_dlogps,
        cs2s,
        kappa,
        Lambda_cosmo,
    ) = eos
    eos_dict = dict(
        p=ps,
        h=hs,
        e=es,
        dloge_dlogp=dloge_dlogps,
        cs2=cs2s,
        kappa=kappa,
        Lambda_cosmo=Lambda_cosmo,
    )

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    # end at pc at pmax at which it is causal
    # cs2 = ps / es / dloge_dlogps
    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2s)]

    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)

    def solve_single_pc(pc):
        """Solve for single pc value"""
        return eibitov.tov_solver(eos_dict, pc)

    ms, rs, ks = jax.vmap(solve_single_pc)(pcs)

    cs = ms / rs

    # convert the mass to solar mass and the radius to km
    ms /= utils.solar_mass_in_meter
    rs /= 1e3

    # calculate the tidal deformability
    lambdas = 2.0 / 3.0 * ks * jnp.power(cs, -5.0)

    pcs, ms, rs, lambdas = utils.limit_by_MTOV_and_interpolate(pcs, ms, rs, lambdas, ndat)

    return jnp.log(pcs), ms, rs, lambdas
    
def construct_family_ST(eos: tuple, ndat: Int = 50, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    (updated later)
    """

    # Construct the dictionary
    ns, ps, hs, es, dloge_dlogps, cs2s, beta_STs, phi_cs, phi_inf_tgts = eos
    #Here's EoS dict names defined 
    eos_dict = dict(p=ps, h=hs, e=es, dloge_dlogp=dloge_dlogps, cs2=cs2s, beta_ST = beta_STs, phi_c=phi_cs, phi_inf_tgt=phi_inf_tgts)

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2s)]
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
    pcs, ms, rs, lambdas = utils.limit_by_MTOV_and_interpolate(pcs, ms, rs, lambdas, ndat)
    
    return jnp.log(pcs), ms, rs, lambdas

def construct_family_ST_Greci(eos: tuple, ndat: Int = 50, min_nsat: Float = 2) -> tuple[
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
    Float[Array, "ndat"],
]:
    r"""
    (updated later)
    """

    # Construct the dictionary
    ns, ps, hs, es, dloge_dlogps,cs2s, beta_STs, phi_cs, phi_inf_tgts = eos
    #Here's EoS dict names defined 
    eos_dict = dict(p=ps, h=hs, e=es, dloge_dlogp=dloge_dlogps, cs2=cs2s, beta_ST = beta_STs, phi_c=phi_cs, phi_inf_tgt=phi_inf_tgts)

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2s)]
    pcs = jnp.logspace(jnp.log10(pc_min), jnp.log10(pc_max), num=ndat)
    def solve_single_pc(pc):
        """Solve for single pc value"""
        return STtov_Greci.tov_solver(eos_dict, pc)
    #this one returns dimensionless Lambdas
    ms, rs, LsT, LsS, LsST1, LsST2 = jax.vmap(solve_single_pc)(pcs)
    
    # calculate the compactness
    cs = ms / rs

    # convert the mass to solar mass and the radius to km

    ms /= utils.solar_mass_in_meter
    rs /= 1e3

    # calculate the tidal deformability
    # @TODO: Need more efficient interpolation + filtering
    pcs, ms, rs, lambdasT = utils.limit_by_MTOV_and_interpolate(pcs, ms, rs, LsT, ndat)
    _, _, _, lambdasS = utils.limit_by_MTOV_and_interpolate(pcs, ms, rs, LsS, ndat)
    _, _, _, lambdasST1 = utils.limit_by_MTOV_and_interpolate(pcs, ms, rs, LsST1, ndat)
    _, _, _, lambdasST2 = utils.limit_by_MTOV_and_interpolate(pcs, ms, rs, LsST2, ndat)
    return jnp.log(pcs), ms, rs, lambdasT, lambdasS, lambdasST1, lambdasST2
# For diagnostic, used in example file
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
    ns, ps, hs, es, dloge_dlogps, cs2s, beta_STs, phi_cs, nu_cs = eos
    # Here's EoS dict names defined
    eos_dict = dict(
        p=ps,
        h=hs,
        e=es,
        dloge_dlogp=dloge_dlogps,
        cs2=cs2s,
        beta_ST=beta_STs,
        phi_c=phi_cs,
        nu_c=nu_cs,
    )

    # calculate the pc_min
    pc_min = utils.interp_in_logspace(
        min_nsat * 0.16 * utils.fm_inv3_to_geometric, ns, ps
    )

    pc_max = eos_dict["p"][locate_lowest_non_causal_point(cs2s)]
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