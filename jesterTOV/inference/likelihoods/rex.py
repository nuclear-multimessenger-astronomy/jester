r"""Symmetry energy constraints from atomic nuclei: PREX-II, CREX and the :math:`^{208}` Pb dipole.

Neutron-skin thicknesses from parity-violating electron scattering (PREX-II on :math:`^{208}` Pb, CREX
on :math:`^{48}` Ca) and the electric dipole polarizability of :math:`^{208}` Pb are mapped through a
collection of energy density functionals to a posterior on the symmetry energy :math:`E_\mathrm{sym}` and
its slope :math:`L_\mathrm{sym}`. As in Koehn et al. (2025), this posterior directly acts as the
likelihood of an EOS with these nuclear empirical parameters,

.. math::
    \mathcal{L}(\mathrm{EOS}|\mathrm{PREX}) = P(E_\mathrm{sym}, L_\mathrm{sym}|\mathrm{PREX}).

The posterior is stored as the logarithm of a kernel density estimate tabulated on a grid.

References
----------
.. [rex1] Adhikari et al., Phys. Rev. Lett. 126, 172502 (2021).
.. [rex2] Adhikari et al., Phys. Rev. Lett. 129, 042501 (2022).
.. [rex3] Koehn et al., Phys. Rev. X 15, 021014 (2025).
"""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jesterTOV.inference.base import LikelihoodBase

DATA_DIR = Path(__file__).parent.parent / "data" / "koehn2024"
REX_EXPERIMENTS = ("PREX", "CREX", "dipole")


class TabulatedPosterior2D:
    """Bilinear interpolation of a log density tabulated on a rectilinear grid, with clamping at the edges.

    Parameters
    ----------
    x : Float[Array, " n_x"]
        Nodes of the first parameter.
    y : Float[Array, " n_y"]
        Nodes of the second parameter.
    log_density : Float[Array, "n_x n_y"]
        Log density on the nodes.
    """

    def __init__(
        self,
        x: Float[Array, " n_x"],
        y: Float[Array, " n_y"],
        log_density: Float[Array, "n_x n_y"],
    ) -> None:
        self.x = x
        self.y = y
        self.log_density = log_density

    def logpdf(self, x: Float, y: Float) -> Float:
        """Return the interpolated log density at a point."""
        i = jnp.clip(jnp.searchsorted(self.x, x, side="right") - 1, 0, self.x.size - 2)
        j = jnp.clip(jnp.searchsorted(self.y, y, side="right") - 1, 0, self.y.size - 2)
        t = jnp.clip((x - self.x[i]) / (self.x[i + 1] - self.x[i]), 0.0, 1.0)
        u = jnp.clip((y - self.y[j]) / (self.y[j + 1] - self.y[j]), 0.0, 1.0)
        return (
            (1 - t) * (1 - u) * self.log_density[i, j]
            + t * (1 - u) * self.log_density[i + 1, j]
            + (1 - t) * u * self.log_density[i, j + 1]
            + t * u * self.log_density[i + 1, j + 1]
        )


def load_rex_posterior(experiment_name: str) -> TabulatedPosterior2D:
    """Load the tabulated posterior of an experiment.

    Parameters
    ----------
    experiment_name : str
        One of ``PREX``, ``CREX`` or ``dipole``.

    Returns
    -------
    TabulatedPosterior2D
        Posterior on (E_sym, L_sym).
    """
    if experiment_name not in REX_EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment_name}', must be one of {REX_EXPERIMENTS}.")
    data = np.load(DATA_DIR / f"rex_{experiment_name}.npz")
    return TabulatedPosterior2D(
        jnp.array(data["E_sym"], dtype=jnp.float64),
        jnp.array(data["L_sym"], dtype=jnp.float64),
        jnp.array(data["log_density"], dtype=jnp.float64),
    )


class REXLikelihood(LikelihoodBase):
    """Likelihood of a measurement that constrains (E_sym, L_sym).

    Parameters
    ----------
    experiment_name : str
        Name of the experiment: ``PREX``, ``CREX`` or ``dipole``.
    posterior : Any
        Object with a method ``logpdf(E_sym, L_sym)``, such as :class:`TabulatedPosterior2D`.

    Raises
    ------
    ValueError
        If ``experiment_name`` is not known.
    """

    experiment_name: str
    posterior: Any

    def __init__(self, experiment_name: str, posterior: Any) -> None:
        super().__init__()
        if experiment_name not in REX_EXPERIMENTS:
            raise ValueError(f"Unknown experiment '{experiment_name}', must be one of {REX_EXPERIMENTS}.")
        self.experiment_name = experiment_name
        self.posterior = posterior

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """Evaluate the log-likelihood.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain ``E_sym`` and ``L_sym`` in MeV.

        Returns
        -------
        Float
            Log density of the posterior at (E_sym, L_sym).
        """
        return self.posterior.logpdf(params["E_sym"], params["L_sym"])
