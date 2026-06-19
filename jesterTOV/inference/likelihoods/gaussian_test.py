r"""Analytical Gaussian test likelihood — no TOV solver required.

This likelihood is intended for integration tests that need to validate
the jester inference pipeline without the computational cost of the TOV
solver. It evaluates a multivariate isotropic Gaussian on the raw
sampled parameters.
"""

import jax.numpy as jnp
from jaxtyping import Float

from jesterTOV.inference.base import LikelihoodBase


class GaussianTestLikelihood(LikelihoodBase):
    r"""Isotropic Gaussian log-likelihood on raw sampled parameters.

    Evaluates :math:`\log \mathcal{L}(\theta) = -\frac{1}{2\sigma^2}
    \sum_i (\theta_i - \mu_i)^2` directly on the parameter vector,
    without requiring any EOS construction or TOV solve.

    Parameters
    ----------
    mean : dict[str, float]
        Mean of the Gaussian for each parameter. Only parameters present
        in this dict contribute to the likelihood; any extra parameters in
        the input are ignored.
    std : float
        Isotropic standard deviation (same for all dimensions).
    """

    mean: dict[str, float]
    std: float

    def __init__(self, mean: dict[str, float], std: float) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def evaluate(self, params: dict[str, Float]) -> Float:  # type: ignore[override]
        """Evaluate the log-likelihood.

        Parameters
        ----------
        params : dict[str, Float]
            Parameter dictionary (JAX-traceable values).

        Returns
        -------
        Float
            Log-likelihood value.
        """
        log_l: Float = jnp.array(0.0)
        for name, mu in self.mean.items():
            log_l = log_l + -0.5 * ((params[name] - mu) / self.std) ** 2
        return log_l
