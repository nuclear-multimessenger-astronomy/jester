.. _sampler-smc:

Sequential Monte Carlo (SMC)
=============================

SMC is the recommended default sampler for EOS inference. It anneals a particle population from the prior (:math:`\lambda = 0`) to the posterior (:math:`\lambda = 1`) through a sequence of tempered distributions :math:`\pi_\lambda(\theta) \propto p(\theta)\, \mathcal{L}(\theta)^\lambda`, with :math:`\lambda = 1/T` the so-called inverse temperature.

**How it works.** At each annealing step, the next temperature :math:`\lambda` is chosen adaptively so that the effective sample size (ESS) after importance reweighting stays at ``target_ess`` x N. Particles are resampled with systematic resampling, then refreshed by running ``n_mcmc_steps`` MCMC transitions with one of two kernels: Gaussian Random Walk (SMC-RW) or NUTS (SMC-NUTS). The loop terminates when :math:`\lambda = 1`.

Evidence is accumulated as :math:`\log Z = \sum_t \Delta \log Z_t`, where each increment is computed from the importance weights at step *t*.

The sampler is implemented in ``blackjax``. For more information about the inner workings, check out the ``blackjax`` source code here: https://github.com/blackjax-devs/blackjax. The ``blackjax`` SMC api is available in the documentation here: https://blackjax-devs.github.io/blackjax/autoapi/blackjax/smc/index.html.


Gaussian Random Walk kernel (``smc-rw``)
-----------------------------------------

The proposal covariance is estimated from the current particle cloud and scaled by ``random_walk_sigma``\ :sup:`2`:

.. math::

   \Sigma_\text{prop} = \sigma^2 \cdot \hat{\Sigma}_\text{particles}

The covariance shape adapts every tempering step; only the overall scale is fixed. This makes the kernel well-suited to posteriors whose shape changes significantly during annealing.

**This is the default sampler** and runs comfortably on a laptop without a GPU.

.. code-block:: yaml

   sampler:
     type: smc-rw
     n_particles: 5000       # number of SMC particles
     n_mcmc_steps: 10        # number of MCMC steps per annealing level
     target_ess: 0.9         # target ESS to compute next temperature
     random_walk_sigma: 0.5  # proposal = sigma^2 * empirical covariance

NUTS kernel (``smc-nuts``) — experimental
-------------------------------------------

.. warning::
   SMC-NUTS has not been thoroughly validated. Cross-check results with SMC-RW.

Uses the No-U-Turn Sampler as the refreshment kernel, which exploits gradient information via JAX automatic differentiation. The inverse mass matrix is adapted each annealing step using an eigen-decomposition of the Hessian evaluated at the highest-log-posterior particle, with SoftAbs regularisation. The step size is adapted with a simple dual-averaging update targeting ``target_acceptance``.

.. code-block:: yaml

   sampler:
     type: smc-nuts
     n_particles: 10000
     n_mcmc_steps: 5
     target_ess: 0.9
     init_step_size: 0.01       # initial leapfrog step size
     mass_matrix_base: 0.2      # diagonal mass matrix baseline
     mass_matrix_param_scales:  # optional per-parameter overrides
       K_sat: 0.5
     target_acceptance: 0.7
     adaptation_rate: 0.3

Diagnostics
-----------

After sampling, ``plot_diagnostics()`` produces a three-panel figure showing the temperature schedule, ESS evolution, and acceptance rate over annealing steps. These are saved automatically to the output directory as ``smc_diagnostics.png``.

API reference
-------------

* :class:`jesterTOV.inference.samplers.blackjax.smc.random_walk.BlackJAXSMCRandomWalkSampler`
* :class:`jesterTOV.inference.samplers.blackjax.smc.nuts.BlackJAXSMCNUTSSampler`
* :class:`jesterTOV.inference.samplers.blackjax.smc.base.BlackjaxSMCSampler` (base class)
