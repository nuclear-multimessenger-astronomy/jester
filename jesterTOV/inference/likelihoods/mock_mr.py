r"""Mock mass-radius likelihood for testing and synthetic inference studies.

This module implements a Gaussian mock mass-radius likelihood, modelling a
synthetic observation of a neutron star as a bivariate normal distribution in
the (mass, radius) plane.  Its main purpose is to allow end-to-end tests of the
inference pipeline without requiring real observational data, and to support
mock-data studies where one wants to assess how well a hypothetical future
measurement would constrain the EOS.

The evaluation strategy mirrors :class:`~jesterTOV.inference.likelihoods.nicer.NICERLikelihood`:
masses are pre-sampled once from the joint distribution at initialisation (using
:class:`~jesterTOV.inference.base.prior.MultivariateGaussianPrior`), and at every
likelihood call the EOS M-R relation is used to predict the radius at each
pre-sampled mass.  The joint log-probability is then computed using the same
``MultivariateGaussianPrior.log_prob`` and averaged with the log-sum-exp trick.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.base.prior import MultivariateGaussianPrior
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class MockMassRadiusLikelihood(LikelihoodBase):
    r"""Likelihood for a synthetic bivariate Gaussian mass-radius measurement.

    The "observation" is a 2-D Gaussian centred on *(mean_mass, mean_radius)*
    with standard deviations *std_mass* and *std_radius* and Pearson correlation
    *correlation*.  The log-likelihood for a model M-R curve is estimated by
    Monte-Carlo integration: masses are pre-sampled from the joint distribution
    at initialisation via
    :class:`~jesterTOV.inference.base.prior.MultivariateGaussianPrior`, the EOS
    is used to interpolate the predicted radius at each sampled mass, and the
    bivariate normal log-pdf is evaluated and averaged.

    The covariance matrix is built from the given standard deviations and
    correlation as

    .. math::

        \Sigma = \begin{pmatrix}
            \sigma_M^2 & \rho\,\sigma_M\sigma_R \\
            \rho\,\sigma_M\sigma_R & \sigma_R^2
        \end{pmatrix}

    Parameters
    ----------
    psr_name : str
        Identifier for this measurement (e.g., ``"PSR0"``).  Used only for
        logging and bookkeeping.
    mean_mass : float
        Mean of the observed mass distribution (:math:`M_\odot`).
    mean_radius : float
        Mean of the observed radius distribution (km).
    std_mass : float
        Standard deviation of the mass measurement (:math:`M_\odot`).
    std_radius : float
        Standard deviation of the radius measurement (km).
    correlation : float
        Pearson correlation coefficient between mass and radius,
        :math:`\rho \in (-1, 1)`.
    penalty_value : float, optional
        Log-likelihood penalty applied when the pre-sampled mass exceeds
        :math:`M_\mathrm{TOV}` (default: ``-99999.0``).
    N_masses_evaluation : int, optional
        Number of (mass, radius) samples drawn at initialisation for
        Monte-Carlo integration (default: ``100``).
    N_masses_batch_size : int, optional
        Batch size passed to :func:`jax.lax.map` for memory-efficient
        processing of the mass samples (default: ``20``).
    seed : int, optional
        Random seed used for the deterministic mass pre-sampling
        (default: ``42``).
    """

    psr_name: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    seed: int
    distribution: MultivariateGaussianPrior
    fixed_mass_samples: Float[Array, " n_samples"]

    def __init__(
        self,
        psr_name: str,
        mean_mass: float,
        mean_radius: float,
        std_mass: float,
        std_radius: float,
        correlation: float,
        penalty_value: float = 0.0,
        N_masses_evaluation: int = 100,
        N_masses_batch_size: int = 20,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.psr_name = psr_name
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size
        self.seed = seed

        if not -1.0 < correlation < 1.0:
            raise ValueError(
                f"Correlation for {psr_name} must be strictly between -1 and 1, "
                f"got {correlation}."
            )

        # Build covariance matrix from standard deviations and correlation
        cov = jnp.array(
            [
                [std_mass**2, correlation * std_mass * std_radius],
                [correlation * std_mass * std_radius, std_radius**2],
            ]
        )
        mean = jnp.array([mean_mass, mean_radius])

        self.distribution = MultivariateGaussianPrior(
            parameter_names=["mass", "radius"],
            mean=mean,
            cov=cov,
        )

        # Pre-sample (mass, radius) pairs from the joint distribution once at
        # initialisation so that each likelihood evaluation is deterministic.
        logger.info(
            f"Pre-sampling {N_masses_evaluation} masses with seed={seed} "
            f"for mock observation '{psr_name}'"
        )
        key = jax.random.key(seed)
        samples = self.distribution.sample(key, N_masses_evaluation)
        # Keep only the mass component; radius comes from the EOS during evaluation
        self.fixed_mass_samples = samples["mass"]
        logger.info(
            f"Mock '{psr_name}' mass sample range: "
            f"[{float(jnp.min(self.fixed_mass_samples)):.3f}, "
            f"{float(jnp.max(self.fixed_mass_samples)):.3f}] Msun"
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        r"""Evaluate the log likelihood for given EOS parameters.

        For each pre-sampled mass the corresponding radius is obtained by
        linear interpolation of the EOS M-R curve.  The bivariate normal
        log-pdf is computed via
        :meth:`~jesterTOV.inference.base.prior.MultivariateGaussianPrior.log_prob`
        and the Monte-Carlo average is taken via the log-sum-exp trick.

        Parameters
        ----------
        params : dict
            Must contain:

            - ``'masses_EOS'``: 1-D array of neutron star masses (:math:`M_\odot`)
              from the EOS M-R relation, sorted in ascending order.
            - ``'radii_EOS'``: 1-D array of corresponding radii (km).

        Returns
        -------
        Float
            Log likelihood :math:`\ln \mathcal{L}`.
        """
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        radii_EOS: Float[Array, " n_points"] = params["radii_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        def process_sample(mass: Float) -> Float:
            """Evaluate log-pdf at (mass, EOS-interpolated radius)."""
            radius = jnp.interp(mass, masses_EOS, radii_EOS)
            logpdf = self.distribution.log_prob({"mass": mass, "radius": radius})
            penalty = jnp.where(mass > mtov, self.penalty_value, 0.0)
            return logpdf + penalty

        logprobs = jax.lax.map(
            process_sample,
            self.fixed_mass_samples,
            batch_size=self.N_masses_batch_size,
        )

        n_samples = logprobs.shape[0]
        return logsumexp(logprobs) - jnp.log(n_samples)
