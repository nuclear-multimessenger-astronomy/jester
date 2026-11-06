r"""Constraint from the collapse of the remnant of GW170817 to a black hole.

If the remnant collapsed, its baryonic mass exceeded the threshold :math:`M_\mathrm{coll,b}` of the EOS. Following
Koehn et al. (2025), the baryonic mass of the remnant is

.. math::
    M_\mathrm{rem,b} = M_{1,b} + M_{2,b} - M_\mathrm{ej,dyn} - M_\mathrm{disk},

with masses and ejecta from the joint inference of GW170817, AT2017gfo and the afterglow of GRB170817A, and the
threshold is either the baryonic TOV mass or the baryonic Kepler mass limit.

The original analysis samples over the discrete EOS jointly with the binary. Here the binary is marginalised with
posterior samples that are independent of the EOS being evaluated. The Kepler limit uses the fit of Eq. (27) of the
paper, with its Gaussian error of :math:`0.06\,M_\odot`, since the RNS code cannot run inside a sampler.
"""

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from jax.scipy.stats import norm
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.inference.base import LikelihoodBase

DATA_FILE = Path(__file__).parent.parent / "data" / "koehn2024" / "postmerger_gw170817.npz"
MIN_LOG_LIKELIHOOD = -1000.0
# Proton mass in solar masses times (fm/km)^3, the convention of the original analysis.
BARYON_MASS = 8.40893833e-4
KEPLER_SLOPE, KEPLER_OFFSET, KEPLER_STD = 1.737, -0.698, 0.06
TOV_MASS_TOLERANCE = 1e-4


def _baryonic_ode(h, y, eos):
    r, m, _ = y
    h_safe = jnp.maximum(h, eos["h"][0])
    p = utils.interp_in_logspace(h_safe, eos["h"], eos["p"])
    n_fm = utils.interp_in_logspace(h_safe, eos["h"], eos["n"]) / utils.fm_inv3_to_geometric
    drdh = -r * (r - 2.0 * m) / (m + 4.0 * jnp.pi * r**3 * p)
    e = utils.interp_in_logspace(h_safe, eos["h"], eos["e"])
    dmdh = 4.0 * jnp.pi * r**2 * e * drdh
    r_km = r * 1e-3
    dmbdh = BARYON_MASS * 4.0 * jnp.pi * n_fm * r_km**2 / jnp.sqrt(1.0 - 2.0 * m / r) * drdh * 1e-3
    return drdh, dmdh, dmbdh


def masses_of_central_pressure(pc: Float, eos: dict[str, Array]) -> tuple[Float, Float]:
    """Solve the TOV equations with the baryonic mass as extra variable, same scheme as ``GRTOVSolver``.

    Parameters
    ----------
    pc : Float
        Central pressure in geometric units.
    eos : dict[str, Array]
        EOS arrays ``n``, ``p``, ``h``, ``e``, ``dloge_dlogp`` in geometric units.

    Returns
    -------
    tuple[Float, Float]
        Gravitational and baryonic mass in solar masses.
    """
    hc = utils.interp_in_logspace(pc, eos["p"], eos["h"])
    ec = utils.interp_in_logspace(hc, eos["h"], eos["e"])
    nc = utils.interp_in_logspace(hc, eos["h"], eos["n"]) / utils.fm_inv3_to_geometric
    dedp_c = ec / pc * jnp.interp(hc, eos["h"], eos["dloge_dlogp"])
    dedh_c = dedp_c * (ec + pc)
    dh = -1e-3 * hc
    r0 = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
    r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
    m0 = 4.0 * jnp.pi * ec * r0**3 / 3.0 * (1.0 - 0.6 * dedh_c * (-dh) / ec)
    mb0 = BARYON_MASS * 4.0 * jnp.pi * nc * (r0 * 1e-3) ** 3 / 3.0
    sol = diffeqsolve(
        ODETerm(_baryonic_ode),
        Dopri5(scan_kind="bounded"),
        t0=hc + dh,
        t1=0,
        dt0=dh,
        y0=(r0, m0, mb0),
        args=eos,
        saveat=SaveAt(t1=True),
        stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
        throw=False,
    )
    gravitational = sol.ys[1][-1] / utils.solar_mass_in_meter  # type: ignore[index]
    return gravitational, sol.ys[2][-1]  # type: ignore[index]


class PostmergerLikelihood(LikelihoodBase):
    r"""Requirement that the remnant of GW170817 exceeded the collapse threshold.

    Parameters
    ----------
    threshold : {"tov", "kepler"}, optional
        Baryonic TOV mass (default of the paper) or baryonic Kepler limit.
    n_samples : int | None, optional
        Number of binary samples to use, all by default.
    data_file : str | Path | None, optional
        NPZ file with arrays ``mass_1``, ``mass_2``, ``log10_mdisk`` and ``log10_mej_dyn``.
    """

    def __init__(
        self,
        threshold: Literal["tov", "kepler"] = "tov",
        n_samples: int | None = None,
        data_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        data = np.load(DATA_FILE if data_file is None else data_file)
        self.threshold = threshold
        stop = n_samples if n_samples is not None else len(data["mass_1"])
        self.mass_1 = jnp.array(data["mass_1"][:stop], dtype=jnp.float64)
        self.mass_2 = jnp.array(data["mass_2"][:stop], dtype=jnp.float64)
        self.ejected = 10.0 ** jnp.array(data["log10_mdisk"][:stop], dtype=jnp.float64) + 10.0 ** jnp.array(
            data["log10_mej_dyn"][:stop], dtype=jnp.float64
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """Evaluate the log-likelihood.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain ``logpc_EOS`` and the EOS arrays ``n``, ``p``, ``h``, ``e`` and ``dloge_dlogp``.
            Gravitational and baryonic masses come from the same integration, so they are consistent.

        Returns
        -------
        Float
            Log of the fraction of binary samples whose remnant exceeds the threshold, floor -1000.
        """
        eos = {name: params[name] for name in ["n", "p", "h", "e", "dloge_dlogp"]}
        pcs = jnp.power(10.0, params["logpc_EOS"])
        masses, baryonic = jax.vmap(lambda pc: masses_of_central_pressure(pc, eos))(pcs)
        index = jnp.argmax(masses)
        mtov = masses[index]

        remnant = (
            jnp.interp(self.mass_1, masses, baryonic) + jnp.interp(self.mass_2, masses, baryonic) - self.ejected
        )
        supported = (self.mass_1 <= mtov) & (self.mass_2 <= mtov)
        if self.threshold == "tov":
            exceeds = jnp.clip((remnant - baryonic[index] + TOV_MASS_TOLERANCE) / (2 * TOV_MASS_TOLERANCE), 0.0, 1.0)
        else:
            exceeds = norm.cdf((remnant - (KEPLER_SLOPE * mtov + KEPLER_OFFSET)) / KEPLER_STD)
        likelihood = jnp.mean(jnp.where(supported, exceeds, 0.0))
        safe = jnp.where(likelihood > 0.0, likelihood, 1.0)
        return jnp.where(likelihood > 0.0, jnp.maximum(jnp.log(safe), MIN_LOG_LIKELIHOOD), MIN_LOG_LIKELIHOOD)
