.. _sampler-nested:

Nested Sampling (NS-AW)
=======================

Nested sampling targets the Bayesian evidence :math:`Z = \int \mathcal{L}(\theta)\,\pi(\theta)\,\mathrm{d}\theta` directly and yields the posterior as a by-product. JESTER uses the BlackJAX implementation from the `handley-lab fork <https://github.com/handley-lab/blackjax>`_ with an adaptive Differential Evolution acceptance-walk kernel.

**How it works.** A set of ``n_live`` live points is maintained, all satisfying :math:`\mathcal{L} > \mathcal{L}^*` for a current likelihood threshold :math:`\mathcal{L}^*`. Each iteration removes the ``n_delete_frac * n_live`` lowest-likelihood points (dead points), adds the same number of new points drawn by running the acceptance-walk kernel from a randomly selected live point, and advances the evidence integral. The loop terminates when the remaining evidence in the live points satisfies

.. math::

   \Delta \log Z < \texttt{termination\_dlogz}.

**Batch deletion:** Compared to the usual CPU implementation in, e.g., ``dynesty`` (as in ``bilby``), the GPU implementation allows for batch deletion of multiple live points per iteration, which can speed up convergence at the cost of iteration granularity. An ``n_delete_frac`` of 0.5 is recommended.

**Unit-cube sampling:** All parameters are mapped to :math:`[0,1]` before sampling. Uniform parameters are rescaled linearly; ``MultivariateGaussianPrior`` parameters use the probability integral transform. Proposals wrap at the :math:`[0,1]` boundary via modulo.

**Acceptance-walk kernel:** The kernel is a bilby-style adaptive Differential Evolution walk. Starting from a live point, it proposes ``n_target`` accepted steps, taking at most ``max_mcmc`` total steps and at most ``max_proposals`` attempts per step. More details can be found in :cite:`Prathaban:2025qgg`.

**Post-processing:** After the loop, dead-point weights are computed by `anesthetic <https://anesthetic.readthedocs.io/>`_ using the :math:`(\log\mathcal{L}, \log\mathcal{L}_\text{birth})` pairs. The weighted samples are then resampled to approximately ESS unweighted posterior samples for downstream use.

Configuration
-------------

.. code-block:: yaml

   sampler:
     type: blackjax-ns-aw
     n_live: 1000          # live-point population size
     n_delete_frac: 0.5    # fraction removed per iteration
     n_target: 60          # target accepted steps in acceptance walk
     max_mcmc: 5000        # max total MCMC steps per new live point
     max_proposals: 1000   # max proposal attempts per MCMC step
     termination_dlogz: 0.1

Larger ``n_live`` gives more accurate evidence estimates and denser posterior coverage at the cost of runtime. ``n_delete_frac`` trades iteration granularity against speed; the default of 0.5 matches the bilby nested-sampling setup.

API reference
-------------

* :class:`jesterTOV.inference.samplers.blackjax.nested_sampling.acceptance_walk.BlackJAXNSAWSampler`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
