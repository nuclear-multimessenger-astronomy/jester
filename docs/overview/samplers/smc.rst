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

.. _sampler-smc-partial-posteriors:

Path of partial posteriors (``smc-partial-posteriors-rw``)
============================================================

A second, orthogonal SMC mode tempers by *number of GW events included* rather than by :math:`\lambda`. Instead of annealing the whole combined likelihood from prior to posterior, each configured GW event's likelihood term is turned on one at a time, so the particle population tracks the sequence of partial posteriors :math:`\pi_t(\theta) \propto p(\theta)\, \prod_{i=1}^{t} \mathcal{L}_i(\theta)` as events :math:`1, \ldots, t` are assimilated. This is Chopin's Iterated Batch Importance Sampling (IBIS), presented as the "path of partial posteriors" in Dai, Heng, Jacob & Whiteley, "An invitation to sequential Monte Carlo samplers" (`arXiv:2007.11936 <https://arxiv.org/abs/2007.11936>`_). It is useful for visualizing how a population-level EOS posterior evolves as more BNS events are assimilated, and supports sequential N → N+1 updating via ``warm_start_from`` (below).

**Turning an event on in a single SMC step is measurably biased** for informative events: the underlying ``blackjax.smc.base.step`` reweights particles *after* the MCMC rejuvenation move, which is only a good approximation for small target-to-target jumps — true of the small, ESS-adaptive :math:`\lambda` increments used by ``smc-rw``/``smc-nuts`` above, but not of a whole-event jump. Each event is therefore ramped in over ``n_substeps_per_event`` small fractional mask increments, matching the source paper's own suggestion of "a geometric path between successive partial posteriors".

Requires at least one ``gw`` likelihood event in the configuration; reuses the same Gaussian Random Walk kernel and covariance adaptation as ``smc-rw``.

.. code-block:: yaml

   sampler:
     type: smc-partial-posteriors-rw
     n_particles: 5000
     n_mcmc_steps: 2              # RW rejuvenation steps per fractional sub-step
     random_walk_sigma: 0.5
     event_order: null            # null = use the order events appear in the gw likelihood block
     n_substeps_per_event: 8      # fractional mask steps (0 -> 1) ramping in each event
     warm_start_from: null        # path to a previous run's HDF5 result; null = start from the prior

``plot_diagnostics()`` for this sampler shows ESS, acceptance rate, and cumulative log evidence per event instead of per annealing step.

**Sequential N → N+1 updating (** ``warm_start_from`` **).** When a new GW event becomes available, a follow-up run can resume from a previous run's converged posterior instead of the prior: set ``warm_start_from`` to the previous run's saved ``InferenceResult`` HDF5 path, and list the previously-covered events followed by the new one(s) in the ``gw`` likelihood block, in the same order. The sampler resamples its initial particles from that file's posterior and validates that its recorded ``event_order`` is a strict prefix of the new run's — the already-covered events are *not* replayed (their mask entries start, and stay, at 1); only the newly appended event(s) go through the fractional ramp-in described above. This is the same incremental-Bayes usage IBIS is designed for: MCMC rejuvenation after the new weight update corrects for reusing an old converged population as the starting point, so no additional correction is needed.

API reference
-------------

* :class:`jesterTOV.inference.samplers.blackjax.smc.random_walk.BlackJAXSMCRandomWalkSampler`
* :class:`jesterTOV.inference.samplers.blackjax.smc.nuts.BlackJAXSMCNUTSSampler`
* :class:`jesterTOV.inference.samplers.blackjax.smc.partial_posteriors.BlackJAXPartialPosteriorsRandomWalkSampler`
* :class:`jesterTOV.inference.samplers.blackjax.smc.base.BlackjaxSMCSampler` (base class)
