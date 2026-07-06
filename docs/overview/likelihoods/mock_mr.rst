.. _likelihood-mock-mr:

Mock mass-radius measurements
==============================

The mock mass-radius likelihood models a synthetic neutron star observation as a bivariate Gaussian in the mass-radius plane. It has no counterpart in real data: its purpose is to let users test the inference pipeline end to end without downloading real observational data, and to run mock-data studies that forecast how well a hypothetical future mass-radius measurement (e.g. from a next-generation X-ray timing mission) would constrain the EOS.

Each mock observation is specified by a mean mass :math:`\mu_M`, a mean radius :math:`\mu_R`, their standard deviations :math:`\sigma_M` and :math:`\sigma_R`, and a Pearson correlation coefficient :math:`\rho \in (-1, 1)`. These define a covariance matrix

.. math::
    :label: mock_mr_covariance

    \Sigma = \begin{pmatrix}
        \sigma_M^2 & \rho\,\sigma_M\sigma_R \\
        \rho\,\sigma_M\sigma_R & \sigma_R^2
    \end{pmatrix} \, ,

so that the "observation" is represented by :math:`\mathcal{N}\!\left((\mu_M, \mu_R), \Sigma\right)`.
This mimics the joint mass-radius observation from actual X-ray pulse profile modelling sources (e.g., those observed by NICER), while being analytical and therefore cheap to evaluate for testing end-to-end EOS inference pipelines.

Numerical implementation
--------------------------

Because the EOS predicts radius as a function of mass, the log-likelihood is estimated by Monte Carlo integration over the mass direction. At initialisation, a fixed set of masses :math:`\{m_i\}_{i=1}^{N}` is drawn once from the marginal mass distribution implied by :math:`\mathcal{N}\!\left((\mu_M, \mu_R), \Sigma\right)`, using a fixed random seed so that the likelihood is deterministic across evaluations.

At every likelihood call, the EOS mass-radius curve is used to interpolate the predicted radius :math:`R(m_i)` at each pre-sampled mass, and the bivariate Gaussian log-pdf is evaluated at each pair :math:`(m_i, R(m_i))`. The Monte Carlo estimate of the log-likelihood is then the log-mean of these values (mimicing, e.g., the NICER mass-radius likelihood), which is computed via log-sum-exp for numerical stability:

.. math::
    :label: mock_mr_likelihood

    \ln P(\theta_{\rm{EOS}} \mid d_{\rm{mock}}) = \ln\!\left(\frac{1}{N}\sum_{i=1}^{N} \exp\!\bigl[\ln\mathcal{N}\bigl((m_i, R(m_i)) \mid (\mu_M, \mu_R), \Sigma\bigr)\bigr]\right) \, .

Pre-sampled masses that exceed the maximum mass :math:`M_{\rm{TOV}}` predicted by the EOS cannot be interpolated meaningfully; these instead receive a configurable log-likelihood penalty (``penalty_value``, default ``0.0``, i.e. no penalty).

Multiple mock observations
------------------------------

The ``mock_mr`` likelihood type reads a JSON file listing one or more mock observations (see :ref:`yaml-reference` for the full field list). One :class:`~jesterTOV.inference.likelihoods.mock_mr.MockMassRadiusLikelihood` instance is created per entry in the file, and all instances are combined as independent measurements.

Usage
-----

.. code-block:: yaml

   likelihoods:
     - type: "mock_mr"
       enabled: true
       json_file: "./mock_observations.json"
       N_masses_evaluation: 100
       seed: 42

.. code-block:: json

   [
     {
       "name": "PSR0",
       "mean_mass": 1.4,
       "mean_radius": 12.0,
       "std_mass": 0.1,
       "std_radius": 0.1,
       "correlation": 0.1
     }
   ]

See ``examples/inference/smc_random_walk/mock_mr/`` for a complete working configuration.

Further resources
-------------------

* API reference: :class:`jesterTOV.inference.likelihoods.mock_mr.MockMassRadiusLikelihood`
* Config class for usage in Bayesian inference workflows: :class:`jesterTOV.inference.config.schemas.likelihoods.MockMassRadiusLikelihoodConfig`
