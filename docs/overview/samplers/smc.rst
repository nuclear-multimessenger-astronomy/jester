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

Adaptive step size
^^^^^^^^^^^^^^^^^^^

By default, ``random_walk_sigma`` is a single fixed scale used for the whole run. For high-SNR signals (e.g. ET), the posterior narrows quickly during annealing, and a fixed sigma can cause the acceptance rate to collapse in later tempering steps. Setting ``adaptive_step_size: true`` instead lets each particle's proposal scale adapt continuously toward a specified target acceptance rate.

In this mode, ``random_walk_sigma`` is treated only as a *starting value*. However, before sampling begins, ``n_pretune_steps`` pilot Metropolis steps run on the initial prior particles to calibrate a reasonable starting scale, correcting for a poorly chosen ``random_walk_sigma``. During annealing, the per-particle scale is then adapted every MCMC step to drive the empirical acceptance rate toward ``target_acceptance_rate`` (default 0.234, the standard optimal value for random-walk Metropolis in high dimensions). The updating is done through ``blackjax``'s ``update_scale_from_acceptance_rate``.

.. code-block:: yaml

   sampler:
     type: smc-rw
     n_particles: 5000
     n_mcmc_steps: 10
     target_ess: 0.9
     random_walk_sigma: 0.5          # starting scale, pretuned before annealing
     adaptive_step_size: true        # adapt scale per particle instead of using a fixed sigma
     target_acceptance_rate: 0.234   # Roberts-Rosenthal optimal value
     n_pretune_steps: 20             # pilot steps on prior particles; 0 disables pretuning

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

**The MCMC move within each sub-step targets the *old*, pre-substep mask, not the new one.** A naive implementation would move the particles with an MCMC kernel that targets the new mask and then reweight by the new/old ratio — this is only a good approximation of the incremental log-evidence for small target-to-target jumps, and is measurably biased for a whole-event jump. Moving under the old mask instead is the "resample-move" construction of Gilks & Berzuini (2001) — also used, via a different code path, by ``blackjax.smc.tempered.build_kernel`` and by Chopin's reference ``particles`` package — which makes the per-substep log-evidence increment exact at *any* mask-jump size: a move that leaves the old target invariant cannot change the particles' marginal distribution, so reweighting the post-move particles by the plain incremental likelihood ratio is unbiased regardless of step size. Each event is nonetheless still ramped in over an adaptive sequence of small fractional mask increments, matching the source paper's own suggestion of "a geometric path between successive partial posteriors": the sampler reuses the exact same ESS-targeting bisection search (``blackjax.smc.ess.ess_solver`` + ``blackjax.smc.solver.dichotomy``) that drives ``smc-rw``'s adaptive :math:`\lambda` schedule above, applied to the mask fraction instead of :math:`\lambda`. This sub-stepping is now purely an MCMC-mixing / proposal-adaptation concern rather than a correctness requirement for the evidence; it is possible because the mask-weighted logposterior is linear in the fraction of the single event being ramped in (every other likelihood term cancels in the successive-target log-weight difference) — exactly the structure ``ess_solver`` assumes. The number of sub-steps per event is uncapped: the search keeps taking ESS-targeting increments until the mask reaches 1.0.

Requires at least one ``gw`` likelihood event in the configuration; reuses the same Gaussian Random Walk kernel and covariance adaptation as ``smc-rw``. The config is split into two levels: the top level only orchestrates which events are assimilated, in what order, and warm-start bookkeeping; the nested ``inner`` block fully specifies the adaptive SMC-RW loop used to ramp in each event.

.. code-block:: yaml

   sampler:
     type: smc-partial-posteriors-rw
     event_order: null            # null = use the order events appear in the gw likelihood block
     warm_start_from: null        # path to a previous run's HDF5 result; null = start from the prior
     inner:
       n_particles: 5000
       n_mcmc_steps: 2              # RW rejuvenation steps per fractional sub-step
       random_walk_sigma: 0.5
       target_ess: 0.9              # target ESS for the sub-step bisection search

``plot_diagnostics()`` for this sampler produces two kinds of plot: the overall across-events plot showing ESS, acceptance rate, and cumulative log evidence per event instead of per annealing step, plus one per-event sub-step diagnostics plot (mask fraction / ESS / acceptance vs. sub-step, mirroring the ``smc-rw``/``smc-nuts`` three-panel style above) saved under ``outdir/substep_diagnostics/``.

**Sequential N → N+1 updating (** ``warm_start_from`` **).** When a new GW event becomes available, a follow-up run can resume from a previous run's converged posterior instead of the prior: set ``warm_start_from`` to the previous run's saved ``InferenceResult`` HDF5 path, and list the previously-covered events followed by the new one(s) in the ``gw`` likelihood block, in the same order. The source run does **not** have to be a partial-posteriors run itself — any sampler (``smc-rw``, ``smc-nuts``, ``flowmc``, ...) works, as long as its posterior is converged on those events. "Already covered" is derived directly from the previous run's *saved config* (its ``likelihoods`` section), not from sampler-specific metadata, so a plain lambda-tempered ``smc-rw`` posterior over N events is just as valid a starting point as a previous partial-posteriors run.

Two checks keep this safe:

* The previous run's configured GW events must be a strict prefix of this run's event order (mask-index alignment) — the already-covered events are *not* replayed (their mask entries start, and stay, at 1); only the newly appended event(s) go through the adaptive ramp-in described above.
* The previous run's **always-on** (non-GW) likelihoods — ChiEFT, NICER, radio, EOS/TOV constraints, ... — must exactly match this run's (same types, same kwargs, e.g. the same ChiEFT data path). The cumulative logZ bookkeeping only makes sense if that always-on part of the posterior is unchanged; a mismatch raises a ``ValueError`` rather than silently producing a wrong evidence.

This is the same incremental-Bayes usage IBIS is designed for: MCMC rejuvenation after the new weight update corrects for reusing an old converged population as the starting point, so no additional correction is needed.

API reference
-------------

* :class:`jesterTOV.inference.samplers.blackjax.smc.random_walk.BlackJAXSMCRandomWalkSampler`
* :class:`jesterTOV.inference.samplers.blackjax.smc.nuts.BlackJAXSMCNUTSSampler`
* :class:`jesterTOV.inference.samplers.blackjax.smc.partial_posteriors.BlackJAXPartialPosteriorsRandomWalkSampler`
* :class:`jesterTOV.inference.samplers.blackjax.smc.base.BlackjaxSMCSampler` (base class)
