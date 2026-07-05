.. _sampler-flowmc:

FlowMC
======

FlowMC combines a local MCMC kernel with a global normalizing-flow proposal trained on samples collected during a warm-up phase. It is well-suited to posteriors with complex geometry (strong correlations, multi-modality) where the random-walk kernel struggles to mix efficiently. The method is discussed at length in Ref. :cite:`Gabrie:2021tlu`, and the software paper can be found in Ref. :cite:`Wong:2022xvh`.

**How it works.** Sampling proceeds in two sequential phases.

*Training phase* (``n_loop_training`` loops): Each loop runs ``n_local_steps`` local MCMC transitions followed by ``n_global_steps`` normalizing-flow proposals, then trains the flow for ``n_epochs`` epochs on all accepted samples so far. The flow is a masked coupling rational-quadratic spline (RQS) with ``num_layers`` coupling layers, each conditioned by a network of width ``hidden_size``.

*Production phase* (``n_loop_production`` loops): The flow is frozen; sampling alternates between local and global moves at the finer thinning ``output_thinning``. Only production samples are returned.

The local kernel is a Gaussian Random Walk by default; MALA is also supported and uses gradient information for potentially better mixing.

.. note::
   We are using an older version of FlowMC, namely ``v0.4.5``. The main development of flowMC has migrated to a new clone of the repository: https://github.com/GW-JAX-Team/flowMC

Configuration
-------------

.. code-block:: yaml

   sampler:
     type: flowmc
     n_chains: 1000             # number of parallel MCMC chains
     n_loop_training: 10        # number of training loops, combining MCMC sampling and flow training
     n_loop_production: 10      # production loops after training, with frozen flow
     n_local_steps: 100         # local MCMC steps per loop
     n_global_steps: 100        # normalizing-flow proposals per loop
     n_epochs: 30               # training epochs per training loop
     learning_rate: 0.001       # Adam learning rate for flow training
     train_thinning: 5          # downsample to store every N-th training sample
     output_thinning: 5         # downsample to store every N-th production sample

Total production samples = ``n_chains x (n_local_steps / output_thinning + n_global_steps / output_thinning) x n_loop_production``.

``train_thinning`` and ``output_thinning`` must not exceed ``n_local_steps`` or ``n_global_steps`` respectively; an error is raised at construction if this is violated.

API reference
-------------

* :class:`jesterTOV.inference.samplers.flowmc.FlowMCSampler`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
