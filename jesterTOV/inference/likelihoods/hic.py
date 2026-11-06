r"""Heavy-ion collision constraint on the pressure of neutron-rich matter.

The ASY-EOS measurement of the elliptic flow ratio is translated into a density-dependent skew-normal
distribution on the pressure in beta equilibrium [hic1]_. The likelihood of an EOS follows Eq. (19) of
[hic2]_,

.. math::
    \mathcal{L}(\mathrm{EOS}|\mathrm{HIC}) = \int \mathrm{d}n\, P(p(n, \mathrm{EOS}), n | \mathrm{HIC})\, C(n),

where :math:`C(n)` is the sensitivity of the experiment.

References
----------
.. [hic1] Huth et al., "Constraining neutron-star matter with microscopic and macroscopic collisions,"
   Nature 606, 276 (2022).
.. [hic2] Koehn et al., "From existing and new nuclear and astrophysical constraints to stringent limits
   on the equation of state of neutron-rich dense matter," Phys. Rev. X 15, 021014 (2025).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.inference.base import LikelihoodBase

DATA_FILE = Path(__file__).parent.parent / "data" / "koehn2024" / "hic.npz"


class HICLikelihood(LikelihoodBase):
    r"""Likelihood of the ASY-EOS heavy-ion collision data.

    Parameters
    ----------
    data_file : str | Path | None, optional
        NPZ file with arrays ``n`` [fm⁻³], ``xi``, ``omega``, ``alpha`` (skew-normal location, scale and
        shape of the pressure in MeV fm⁻³) and ``log_sensitivity``, all on the same density grid.
        Defaults to the tables of Koehn et al. (2025).
    """

    def __init__(self, data_file: str | Path | None = None) -> None:
        super().__init__()
        data = np.load(DATA_FILE if data_file is None else data_file)
        self.n_grid = jnp.array(data["n"])
        self.xi = jnp.array(data["xi"])
        self.omega = jnp.array(data["omega"])
        self.alpha = jnp.array(data["alpha"])
        self.log_sensitivity = jnp.array(data["log_sensitivity"])

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """Evaluate the log-likelihood.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain the EOS arrays ``n`` and ``p`` in geometric units.

        Returns
        -------
        Float
            Log-likelihood, up to a constant.
        """
        n = params["n"] / utils.fm_inv3_to_geometric
        p = params["p"] / utils.MeV_fm_inv3_to_geometric
        z = (jnp.interp(self.n_grid, n, p) - self.xi) / self.omega
        log_skewnorm = jnp.log(2.0) + norm.logpdf(z) + norm.logcdf(self.alpha * z) - jnp.log(self.omega)
        return logsumexp(log_skewnorm + self.log_sensitivity)
