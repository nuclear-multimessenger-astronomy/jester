r"""Perturbative QCD constraint matched at the TOV density.

A point :math:`(n_L, \epsilon_L, p_L)` of a candidate EOS can only be connected to the pQCD EOS at
:math:`\mu_H` through a stable and causal interpolation if the pressure difference obeys
:math:`\Delta p_\mathrm{min} \leq p_H - p_L \leq \Delta p_\mathrm{max}` [pqcd1]_. Marginalising this
condition over the missing higher orders and the renormalisation scale gives a weighted average over
tabulated :math:`(n_H, p_H)` draws at :math:`\mu_H = 2.6` GeV [pqcd2]_. Following [pqcd3]_ the EOS is
matched at its TOV density.

References
----------
.. [pqcd1] Komoltsev and Kurkela, Phys. Rev. Lett. 128, 202701 (2022).
.. [pqcd2] Gorda, Komoltsev, Kurkela and Mazeliauskas, JHEP 06, 002 (2023).
.. [pqcd3] Koehn et al., Phys. Rev. X 15, 021014 (2025).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.inference.base import LikelihoodBase

DATA_FILE = Path(__file__).parent.parent / "data" / "koehn2024" / "pqcd_miho.npz"
MIN_LOG_LIKELIHOOD = -1000.0
MeV_TO_GeV = 1e-3


class PQCDLikelihood(LikelihoodBase):
    r"""Likelihood of matching an EOS to pQCD at the TOV density.

    Parameters
    ----------
    data_file : str | Path | None, optional
        NPZ file with the scalar ``mu_H`` [GeV] and the arrays ``n_H`` [fm⁻³], ``p_H`` [GeV fm⁻³] and
        ``weight``. Defaults to the MiHO draws used by Koehn et al. (2025).
    """

    def __init__(self, data_file: str | Path | None = None) -> None:
        super().__init__()
        data = np.load(DATA_FILE if data_file is None else data_file)
        self.mu_H = float(data["mu_H"])
        self.n_H = jnp.array(data["n_H"])
        self.p_H = jnp.array(data["p_H"])
        self.weight = jnp.array(data["weight"])

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """Evaluate the log-likelihood.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain ``masses_EOS``, ``logpc_EOS`` and the EOS arrays ``n``, ``p``, ``e``, all with
            geometric units for the latter four.

        Returns
        -------
        Float
            Log-likelihood. EOS that cannot be matched get a floor value of -1000.
        """
        n = params["n"] / utils.fm_inv3_to_geometric
        p = params["p"] / utils.MeV_fm_inv3_to_geometric * MeV_TO_GeV
        e = params["e"] / utils.MeV_fm_inv3_to_geometric * MeV_TO_GeV
        log10_pc_tov = params["logpc_EOS"][jnp.argmax(params["masses_EOS"])]
        p_tov = jnp.power(10.0, log10_pc_tov) / utils.MeV_fm_inv3_to_geometric * MeV_TO_GeV

        n_tov = jnp.interp(p_tov, p, n)
        e_tov = jnp.interp(n_tov, n, e)
        mu_tov = (e_tov + p_tov) / n_tov

        delta_p = self.p_H - p_tov
        delta_p_min = 0.5 * (self.mu_H**2 / mu_tov - mu_tov) * n_tov
        delta_p_max = 0.5 * (self.mu_H - mu_tov**2 / self.mu_H) * self.n_H
        allowed = (delta_p_min < delta_p) & (delta_p < delta_p_max) & (n_tov < self.n_H * mu_tov / self.mu_H)

        likelihood = jnp.mean(allowed * self.weight)
        safe = jnp.where(likelihood > 0.0, likelihood, 1.0)
        return jnp.where(likelihood > 0.0, jnp.log(safe), MIN_LOG_LIKELIHOOD)
