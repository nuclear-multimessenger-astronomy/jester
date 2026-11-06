r"""Ejecta mass of GRB211211A, under the assumption that it originates from a binary neutron star merger.

Following Koehn et al. (2025) and the ``gwem-resampling`` routine of NMMA, the posterior on the total ejecta
mass from the light curve of Kunert et al. (2024) acts as a likelihood. The binary is unknown, so the
likelihood is marginalised over a flat prior on chirp mass and mass ratio and over the parameters
:math:`\zeta` (fraction of the disk ejected as wind) and :math:`\alpha` (error of the fit for the dynamical
ejecta). This is done with a fixed set of draws, which keeps the likelihood deterministic.

The coefficients are those in the NMMA code that produced the published results, which differ slightly from the
rounded values in the text of the paper.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from jesterTOV.inference.base import LikelihoodBase

DATA_FILE = Path(__file__).parent.parent / "data" / "koehn2024" / "grb211211a.npz"
GEOM_MSUN_KM = 1.476625038050125
MIN_LOG_LIKELIHOOD = -1000.0
LOG_ZERO = -1e5

CHIRP_MASS_RANGE = (0.609385394307287, 3.2210370841956593)
MASS_RATIO_RANGE = (0.125, 1.0)
COMPONENT_MASS_RANGE = (0.7, 3.7)
ALPHA_STD = 4e-4


def dynamical_ejecta_mass(m1: Float, m2: Float, c1: Float, c2: Float) -> Float:
    """Dynamical ejecta mass in solar masses, Kruger and Foucart, Phys. Rev. D 101, 103002 (2020)."""
    a, b, c, n = -9.3335, 114.17, -337.56, 1.5465
    mdyn = m1 * (a / c1 + b * (m2 / m1) ** n + c * c1) + m2 * (a / c2 + b * (m1 / m2) ** n + c * c2)
    return jnp.maximum(0.0, 1e-3 * mdyn)


def log10_disk_mass(total_mass: Float, q: Float, mtov: Float, r16: Float) -> Float:
    """Log10 of the disk mass in solar masses, with q = m2 / m1 <= 1 and r16 in solar masses."""
    threshold = (-3.606 * mtov / r16 + 2.38) * mtov
    xi = 0.5 * jnp.tanh(5.879 * (q - 0.886))
    a = -1.725 - 2.337 * xi
    b = -0.564 - 0.437 * xi
    return jnp.maximum(-3.0, a * (1.0 + b * jnp.tanh((0.958 - total_mass / threshold) / 0.057)))


class GRBEjectaLikelihood(LikelihoodBase):
    r"""Likelihood of the total ejecta mass of GRB211211A.

    Parameters
    ----------
    n_samples : int, optional
        Number of draws of (chirp mass, mass ratio, :math:`\alpha`, :math:`\zeta`).
    seed : int, optional
        Seed of the fixed draws.
    data_file : str | Path | None, optional
        NPZ file with ``mej`` and ``log_density``, the KDE of the total ejecta mass in solar masses.
    """

    def __init__(self, n_samples: int = 4000, seed: int = 42, data_file: str | Path | None = None) -> None:
        super().__init__()
        data = np.load(DATA_FILE if data_file is None else data_file)
        self.mej_grid = jnp.array(data["mej"], dtype=jnp.float64)
        self.log_density = jnp.array(data["log_density"], dtype=jnp.float64)

        key_mc, key_q, key_alpha, key_zeta = jax.random.split(jax.random.key(seed), 4)
        # Oversample and keep the draws inside the constraint on the component masses.
        n_draws = 10 * n_samples
        mc = jax.random.uniform(key_mc, (n_draws,), minval=CHIRP_MASS_RANGE[0], maxval=CHIRP_MASS_RANGE[1])
        q = jax.random.uniform(key_q, (n_draws,), minval=MASS_RATIO_RANGE[0], maxval=MASS_RATIO_RANGE[1])
        total = mc * (1 + q) ** 1.2 / q**0.6
        m1, m2 = total / (1 + q), total * q / (1 + q)
        keep = np.flatnonzero(
            np.asarray((m1 >= COMPONENT_MASS_RANGE[0]) & (m1 <= COMPONENT_MASS_RANGE[1])
                       & (m2 >= COMPONENT_MASS_RANGE[0]) & (m2 <= COMPONENT_MASS_RANGE[1]))
        )[:n_samples]
        self.m1, self.m2 = m1[keep], m2[keep]
        self.alpha = ALPHA_STD * jax.random.normal(key_alpha, (keep.size,))
        self.zeta = jax.random.uniform(key_zeta, (keep.size,))

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """Evaluate the log-likelihood.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain ``masses_EOS`` in solar masses and ``radii_EOS`` in km.

        Returns
        -------
        Float
            Log-likelihood with a floor of -1000. Binaries that the EOS cannot support contribute zero.
        """
        masses = params["masses_EOS"]
        radii = params["radii_EOS"]
        mtov = jnp.max(masses)
        r1 = jnp.interp(self.m1, masses, radii, right=0.0)
        r2 = jnp.interp(self.m2, masses, radii, right=0.0)
        r16 = jnp.interp(1.6, masses, radii, right=0.0) / GEOM_MSUN_KM
        supported = (r1 > 0.0) & (r2 > 0.0) & (r16 > 0.0)

        c1 = self.m1 / jnp.where(supported, r1, 1.0) * GEOM_MSUN_KM
        c2 = self.m2 / jnp.where(supported, r2, 1.0) * GEOM_MSUN_KM
        mdyn = dynamical_ejecta_mass(self.m1, self.m2, c1, c2) + self.alpha
        log10_mdisk = log10_disk_mass(self.m1 + self.m2, self.m2 / self.m1, mtov, jnp.where(r16 > 0.0, r16, 1.0))
        ejecta = mdyn + self.zeta * 10.0**log10_mdisk

        valid = supported & (mdyn >= 0.0) & (ejecta <= self.mej_grid[-1])
        logp = jnp.interp(ejecta, self.mej_grid, self.log_density)
        log_likelihood = logsumexp(jnp.where(valid, logp, LOG_ZERO)) - jnp.log(self.m1.size)
        return jnp.maximum(jnp.nan_to_num(log_likelihood, nan=MIN_LOG_LIKELIHOOD), MIN_LOG_LIKELIHOOD)
