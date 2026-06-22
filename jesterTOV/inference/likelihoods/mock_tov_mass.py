r"""Mock TOV maximum-mass likelihood for testing and synthetic inference studies.

This module implements a Gaussian likelihood on the maximum TOV mass :math:`M_\mathrm{TOV}`,
modelling a hypothetical future measurement that constrains both a lower and an upper bound
on the maximum neutron star mass supported by the equation of state.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.inference.base.likelihood import LikelihoodBase


class MockTOVMassLikelihood(LikelihoodBase):
    r"""Gaussian likelihood on the maximum TOV mass.

    This likelihood places a Gaussian constraint directly on :math:`M_\mathrm{TOV}`,
    the maximum gravitational mass an EOS can support.  It is useful for mock-data
    studies where one wants to assess how well a hypothetical future measurement
    (providing both a lower and an upper bound on the maximum mass) would constrain
    the EOS.

    The log-likelihood is

    .. math::

        \ln \mathcal{L}(M_\mathrm{TOV}) =
            -\frac{1}{2} \left(\frac{M_\mathrm{TOV} - \mu}{\sigma}\right)^2

    where :math:`\mu` is the measured central value and :math:`\sigma` is the
    :math:`1\sigma` uncertainty, which simultaneously encodes the lower and upper
    bounds on the maximum mass.

    Parameters
    ----------
    mean : float
        Central value of the Gaussian constraint on :math:`M_\mathrm{TOV}`
        (:math:`M_\odot`).
    std : float
        :math:`1\sigma` width of the Gaussian constraint (:math:`M_\odot`).
    penalty_value : float, optional
        Log-likelihood value returned when the TOV solver fails (i.e. when
        :math:`M_\mathrm{TOV}` is not a valid mass).  Default is ``0.0``.

    Examples
    --------
    >>> from jesterTOV.inference.likelihoods.mock_tov_mass import MockTOVMassLikelihood
    >>> import jax.numpy as jnp
    >>> likelihood = MockTOVMassLikelihood(mean=2.0, std=0.1)
    >>> params = {"masses_EOS": jnp.linspace(0.5, 2.1, 50)}
    >>> log_like = likelihood.evaluate(params)
    """

    mean: float
    std: float
    penalty_value: float

    def __init__(
        self,
        mean: float,
        std: float,
        penalty_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.penalty_value = penalty_value

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        r"""Evaluate the Gaussian log-likelihood on :math:`M_\mathrm{TOV}`.

        Parameters
        ----------
        params : dict
            Must contain ``'masses_EOS'``: 1-D array of neutron star masses
            (:math:`M_\odot`) from the EOS M-R relation.  The maximum value is
            taken as :math:`M_\mathrm{TOV}`.

        Returns
        -------
        Float
            Gaussian log-likelihood :math:`\ln \mathcal{L}(M_\mathrm{TOV})`.
            Returns ``penalty_value`` when the TOV solution is invalid.
        """
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        z = (mtov - self.mean) / self.std
        log_likelihood = -0.5 * z**2

        log_likelihood = jnp.where(
            jnp.isfinite(mtov),
            log_likelihood,
            self.penalty_value,
        )

        return log_likelihood
