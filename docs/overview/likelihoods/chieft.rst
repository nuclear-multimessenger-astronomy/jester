.. _likelihood-chieft:

Chiral effective field theory (ChiEFT)
======================================

Chiral effective field theory (ChiEFT) provides theoretical constraints on the equation of state of nuclear matter at low densities, through a systematic expansion of the nuclear many-body Hamiltonian in terms of low-energy degrees of freedom (nucleons and pions). 
The expansion is truncated at a certain order, and the Hamiltonian can be solved numerically. 
The theoretical uncertainties in the expansion can be estimated systematically, and the output is an uncertainty band :math:`[p_-, p_+]` for the pressure as a function of energy density, which can be used to constrain the EOS parameters in Bayesian inference, in the low-density region (up to :math:`2n_{\rm{sat}}`).
For the details of the precise calculations of the chiEFT constraint used in ``jester``, we refer to the discussion in :cite:`Koehn:2024set`. 
Here, we limit ourselves to how the likelihood function is implemented in ``jester`` and how it can be used in practice.

The likelihood is implemented by the class :class:`jesterTOV.inference.likelihoods.chieft.ChiEFTLikelihood`, which takes as input two files ``low_filename`` and ``high_filename`` containing the lower and upper bounds of the chiEFT constraint, respectively, as described above.
By default, ``jester`` loads the pressure bounds from :cite:`Koehn:2024set`, but users can also provide their own files with different chiEFT constraints if they wish.
With these bounds loaded, the likelihood function of a particular EOS, which predicts a pressure :math:`p(\theta_{\rm{EOS}} ; n)` at a given density :math:`n`, is defined by

.. math::
    :label: chiEFT_likelihood

    P(\theta_{\rm{EOS}} | \chi {\rm{EFT}}) \propto \exp\left( \int_{0.75 n_{\rm{sat}}}^{n_{\rm{break}}} \frac{\log f(p(\theta_{\rm{EOS}} ; n), n)}{n_{\rm{break}} - 0.75 n_{\rm{sat}}} {\rm{d}} n \right) \, ,

where the integration is terminated at the density :math:`n_{\rm{break}}` where the chiEFT prediction breaks down. This means that the likelihood can only be evaluated for EOSs that either freely sample this parameter, or fix it otherwise. 
Here, the function :math:`f(p, n)` is a score function used to smoothly taper off the likelihood function around the chiEFT bounds, and is defined as (see ``arXiv:2402.04172v3``, Sec. III A for details)

.. math::
    :label: score_function

    f(p, n) = \begin{cases}
    \exp \left( -\beta \frac{p(n) - p_{+}}{p_{+} - p_{-}} \right) & \text{if } p(n) \geq p_{+} \\
    \exp \left( -\beta \frac{p_{-} - p(n)}{p_{+} - p_{-}} \right) & \text{if } p(n) \leq p_{-} \\
    1 & \text{else.}
    \end{cases}

where :math:`p_-(n)` and :math:`p_+(n)` are the lower and upper bounds of the chiEFT constraint at density :math:`n`, respectively, and :math:`\sigma(n)` is a smoothing parameter that controls how quickly the likelihood tapers off around the bounds.
Following :cite:`Koehn:2024set`, we set :math:`\beta = 6` in ``jester``, so that 75% of the weight is contained within the interval :math:`[p_-, p_+]`, and the likelihood tapers off smoothly outside of this interval.
Note that the lower bound of the integration in Eq. :eq:`chiEFT_likelihood` is set to :math:`0.75 n_{\rm{sat}}`, which is chosen somewhat arbitrarily, but this choice has been found to not impact the EOS inference. 
The pressure bounds, as well as a visualization of the score function, is shown below (which mimics Fig. 2 in :cite:`Koehn:2024set`):

.. plot:: overview/likelihoods/chieft_score_function.py

   Score function :math:`f(p, n)` as a function of pressure :math:`p` and density :math:`n`.
   Dashed lines show the chiEFT band boundaries :math:`p_-` (lower) and :math:`p_+` (upper)
   from :cite:`Koehn:2024set`.  Inside the band :math:`f = 1`; outside it decays exponentially
   with slope :math:`\beta = 6`.

Further resources
-------------------

.. TODO: fix the broken config class link, it is not clickable?

* API reference: :class:`jesterTOV.inference.likelihoods.chieft.ChiEFTLikelihood`
* Config class for usage in Bayesian inference workflows: :class:`jesterTOV.inference.config.schemas.likelihoods.ChiEFTLikelihoodConfig` 

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
