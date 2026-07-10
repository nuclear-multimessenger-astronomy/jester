r"""Gaussian likelihood on the pressure at a fixed density, defined purely in EOS space.

Unlike the mass-radius-based likelihoods, this only needs the EOS's own
``n``/``p`` grids (already present in every :class:`~jesterTOV.inference.transforms.transform.JesterTransform`
output) -- no TOV family construction is required to *evaluate* it, making it a cheap,
smooth, and analytically interpretable target for validating samplers (SMC, NVI, ...)
against each other.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.inference.base.likelihood import LikelihoodBase


class EOSPressureAtDensityLikelihood(LikelihoodBase):
    r"""Gaussian log-likelihood on the EOS pressure at a fixed density.

    Models a synthetic "measurement" of :math:`p(n_0)` -- the pressure at a
    given density :math:`n_0` (in units of nuclear saturation density
    :math:`n_{\rm sat} = 0.16\,\mathrm{fm}^{-3}`) -- as a Gaussian centred on
    *mean_pressure* with standard deviation *std_pressure* (both in
    :math:`\mathrm{MeV\,fm}^{-3}`). The predicted pressure is obtained by
    linearly interpolating the EOS's own :math:`(n, p)` grid at :math:`n_0`.

    Parameters
    ----------
    density_nsat : float
        Density at which to evaluate the pressure, in units of :math:`n_{\rm sat}`.
    mean_pressure : float
        Mean of the Gaussian target, in :math:`\mathrm{MeV\,fm}^{-3}`.
    std_pressure : float
        Standard deviation of the Gaussian target, in :math:`\mathrm{MeV\,fm}^{-3}`.

    Examples
    --------
    >>> likelihoods:
    >>>   - type: "eos_pressure_gaussian"
    >>>     enabled: true
    >>>     density_nsat: 3.0
    >>>     mean_pressure: 115.7
    >>>     std_pressure: 15.0
    """

    density_nsat: float
    mean_pressure: float
    std_pressure: float

    def __init__(
        self,
        density_nsat: float,
        mean_pressure: float,
        std_pressure: float,
    ) -> None:
        super().__init__()
        self.density_nsat = float(density_nsat)
        self.mean_pressure = float(mean_pressure)
        self.std_pressure = float(std_pressure)
        # Precompute the target density in jester's internal geometric units so the
        # transform's raw "n"/"p" grids can be interpolated directly, with no need to
        # convert the (potentially large) EOS arrays themselves.
        self._density_geometric = (
            self.density_nsat * 0.16 * utils.fm_inv3_to_geometric
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        r"""Evaluate the Gaussian log-likelihood at the EOS-predicted pressure.

        Parameters
        ----------
        params : dict
            Must contain:

            - ``'n'``: 1-D array of densities (jester geometric units), ascending.
            - ``'p'``: 1-D array of corresponding pressures (jester geometric units).

        Returns
        -------
        Float
            Gaussian log-likelihood :math:`\ln \mathcal{N}(p(n_0);\,\mu,\sigma^2)`.
        """
        n: Float[Array, " n_points"] = params["n"]
        p: Float[Array, " n_points"] = params["p"]

        p_geometric = jnp.interp(self._density_geometric, n, p)
        p_pred = p_geometric / utils.MeV_fm_inv3_to_geometric

        residual = (p_pred - self.mean_pressure) / self.std_pressure
        return -0.5 * residual**2 - jnp.log(self.std_pressure * jnp.sqrt(2 * jnp.pi))
