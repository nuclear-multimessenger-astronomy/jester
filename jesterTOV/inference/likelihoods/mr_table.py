r"""Mass-radius likelihoods evaluated on tabulated two-dimensional posteriors.

This covers measurements whose posterior is available as a grid or whose kernel density estimate has been
tabulated: the X-ray bursters 4U 1702-429 and SAX J1808.4-3658, the qLMXBs in :math:`\omega` Cen and 47 Tuc,
HESS J1731-347 and the NICER posteriors. Tabulation keeps the files small enough for git and makes the
evaluation independent of the number of posterior samples, in contrast to a KDE.

The likelihood of a measurement is an integral along the M-R curve of the EOS,

.. math::
    \mathcal{L}(\mathrm{EOS}) = \int_0^{M_\mathrm{TOV}} \mathrm{d}M\,
    P(M, R(M, \mathrm{EOS}) | d)

which may be divided by :math:`M_\mathrm{TOV}` to include a flat mass prior up to the TOV mass.

The files of Koehn et al. (2025) do not all follow the formula in the paper, and
:func:`load_koehn2024_mr_likelihood` reproduces the files, which define the published numbers:

- NICER, HESS and SAX include the factor :math:`1/M_\mathrm{TOV}`, Eq. (11) of the paper does not.
- 4U 1702-429 and the qLMXBs do not include it.
- The qLMXB file multiplies the two posteriors inside one mass integral, assuming one star, while the
  paper quotes the product of two integrals. Pass ``combine="product"`` for the latter.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from jesterTOV.inference.base import LikelihoodBase

DATA_DIR = Path(__file__).parent.parent / "data" / "koehn2024"
MIN_LOG_LIKELIHOOD = -1000.0
LOG_ZERO = -1e5
MAX_MASS = 5.0


@dataclass(frozen=True)
class MRTable:
    """Posterior on a rectilinear (R, M) grid.

    Attributes
    ----------
    radius : Float[Array, " n_r"]
        Radius nodes in km, strictly increasing.
    mass : Float[Array, " n_m"]
        Mass nodes in solar masses, strictly increasing.
    values : Float[Array, "n_r n_m"]
        Posterior density if ``log_values`` is False, otherwise its logarithm.
    log_values : bool
        Whether interpolation is done in the logarithm, which suits a smooth KDE. Measured grids are
        interpolated linearly in the density, with zero outside the grid.
    """

    radius: Float[Array, " n_r"]
    mass: Float[Array, " n_m"]
    values: Float[Array, "n_r n_m"]
    log_values: bool

    @classmethod
    def from_file(cls, path: str | Path) -> "MRTable":
        """Load a table from an NPZ file with keys ``radius``, ``mass``, ``values``, ``log_values``."""
        data = np.load(path)
        return cls(
            jnp.array(data["radius"], dtype=jnp.float64),
            jnp.array(data["mass"], dtype=jnp.float64),
            jnp.array(data["values"], dtype=jnp.float64),
            bool(data["log_values"]),
        )

    def log_prob(self, radius: Float[Array, " n"], mass: Float[Array, " n"]) -> Float[Array, " n"]:
        """Return the log density by bilinear interpolation."""
        i = jnp.clip(jnp.searchsorted(self.radius, radius, side="right") - 1, 0, self.radius.size - 2)
        j = jnp.clip(jnp.searchsorted(self.mass, mass, side="right") - 1, 0, self.mass.size - 2)
        t = (radius - self.radius[i]) / (self.radius[i + 1] - self.radius[i])
        u = (mass - self.mass[j]) / (self.mass[j + 1] - self.mass[j])
        value = (
            (1 - t) * (1 - u) * self.values[i, j]
            + t * (1 - u) * self.values[i + 1, j]
            + (1 - t) * u * self.values[i, j + 1]
            + t * u * self.values[i + 1, j + 1]
        )
        inside = (
            (radius >= self.radius[0])
            & (radius <= self.radius[-1])
            & (mass >= self.mass[0])
            & (mass <= self.mass[-1])
        )
        if self.log_values:
            return jnp.where(inside, value, LOG_ZERO)
        positive = inside & (value > 0.0)
        return jnp.where(positive, jnp.log(jnp.where(positive, value, 1.0)), LOG_ZERO)


class MRTableLikelihood(LikelihoodBase):
    r"""Likelihood of one or more tabulated M-R posteriors along the EOS M-R curve.

    Parameters
    ----------
    name : str
        Name of the measurement, for bookkeeping.
    tables : list[MRTable]
        The posteriors.
    combine : {"mixture", "product", "joint_integrand"}, optional
        ``mixture`` is the weighted arithmetic mean of the separate likelihoods, used for different
        analyses of the same star. ``product`` multiplies the separate likelihoods, for different stars.
        ``joint_integrand`` multiplies the posteriors inside a single mass integral.
    weights : list[float] | None, optional
        Mixture weights, equal by default.
    divide_by_mtov : bool, optional
        Whether to include the flat mass prior :math:`1/M_\mathrm{TOV}`.
    mass_step : float, optional
        Spacing of the fixed mass grid in solar masses. Only the nodes below :math:`M_\mathrm{TOV}` count,
        as in the original analysis. For narrow posteriors near the TOV mass this choice matters.
    upper_pad : float, optional
        Masses up to :math:`M_\mathrm{TOV}` plus this value enter the integral, at the radius of the TOV star.
    """

    def __init__(
        self,
        name: str,
        tables: list[MRTable],
        combine: Literal["mixture", "product", "joint_integrand"] = "mixture",
        weights: list[float] | None = None,
        divide_by_mtov: bool = True,
        mass_step: float = 0.02,
        upper_pad: float = 0.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.tables = tables
        self.combine = combine
        self.divide_by_mtov = divide_by_mtov
        self.upper_pad = upper_pad
        self.mass_step = mass_step
        self.mass_grid = mass_step * jnp.arange(int(np.ceil(MAX_MASS / mass_step)))
        n_tables = len(tables)
        self.log_weights = jnp.log(jnp.ones(n_tables) / n_tables if weights is None else jnp.array(weights))

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """Evaluate the log-likelihood.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain ``masses_EOS`` in solar masses and ``radii_EOS`` in km.

        Returns
        -------
        Float
            Log-likelihood, with a floor of -1000.
        """
        masses_eos = params["masses_EOS"]
        mtov = jnp.max(masses_eos)
        radii = jnp.interp(self.mass_grid, masses_eos, params["radii_EOS"])
        counted = self.mass_grid < mtov + self.upper_pad
        logps = jnp.stack([table.log_prob(radii, self.mass_grid) for table in self.tables])
        logps = jnp.where(counted, logps, LOG_ZERO)

        # With 1/M_TOV the integral is the mean over the counted nodes, otherwise their sum times the spacing.
        if self.divide_by_mtov:
            log_norm = -jnp.log(jnp.maximum(jnp.sum(counted), 1))
        else:
            log_norm = jnp.log(self.mass_step)
        if self.combine == "joint_integrand":
            log_likelihood = logsumexp(jnp.sum(logps, axis=0)) + log_norm
        else:
            per_table = logsumexp(logps, axis=1) + log_norm
            if self.combine == "product":
                log_likelihood = jnp.sum(per_table)
            else:
                log_likelihood = logsumexp(self.log_weights + per_table)
        log_likelihood = jnp.nan_to_num(log_likelihood, nan=MIN_LOG_LIKELIHOOD, neginf=MIN_LOG_LIKELIHOOD)
        return jnp.maximum(log_likelihood, MIN_LOG_LIKELIHOOD)


def load_koehn2024_mr_likelihood(name: str) -> MRTableLikelihood:
    """Create an M-R likelihood reproducing the stored per-EOS files of Koehn et al. (2025).

    Parameters
    ----------
    name : str
        One of ``4U_1702``, ``SAX``, ``J1731``, ``qLMXBs``, ``J0030``, ``J0740``.

    Returns
    -------
    MRTableLikelihood
        The likelihood.
    """

    def table(file_stem: str) -> MRTable:
        return MRTable.from_file(DATA_DIR / f"mr_{file_stem}.npz")

    def mixture_weights(stems: list[str]) -> list[float]:
        # Each analysis was normalised over the EOS set before averaging, which enters as a relative weight.
        return [float(np.load(DATA_DIR / f"mr_{stem}.npz")["mixture_weight"]) for stem in stems]

    match name:
        case "4U_1702":
            return MRTableLikelihood(name, [table(name)], divide_by_mtov=False, mass_step=0.01)
        case "SAX" | "J1731":
            return MRTableLikelihood(name, [table(name)])
        case "qLMXBs":
            return MRTableLikelihood(
                name, [table("wCen"), table("X5")], combine="joint_integrand", divide_by_mtov=False, upper_pad=0.01
            )
        case "J0030" | "J0740":
            stems = [f"{name}_amsterdam", f"{name}_maryland"]
            return MRTableLikelihood(name, [table(stem) for stem in stems], weights=mixture_weights(stems))
        case _:
            raise ValueError(f"Unknown M-R measurement '{name}'.")
