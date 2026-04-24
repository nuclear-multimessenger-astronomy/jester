.. _likelihood-radio:

Radio timing constraints
=========================

Mass measurements of the heaviest pulsars known so far provide a lower bound on the maximum mass of a (non-rotating) neutron star supported by a given EOS, and thus provide important constraints on the EOS parameters.
More information on the techniques involved here can be found in :cite:`Demorest:2010bx`.

The input of the class is a name (to uniquely identify the likelihood and inform other users which neutron star observation is being used), and the mass measurement through the mean and standard deviation of the Gaussian distribution that describes the measurement.

Given this, the likelihood function is given by

.. math::
    :label: radio_likelihood

    P(\theta_{\rm{EOS}} | d_{\rm{radio}}) \propto \frac{1}{M_{\rm{TOV}} - m_{\rm{min}}} \int_{m_{\rm{min}}}^{M_{\rm{TOV}}} P(M|d_{\rm{radio}}) {\rm{d}} M \, ,

where :math:`M_{\rm{TOV}}` is the maximum mass of a non-rotating neutron star predicted by the EOS parameters :math:`\theta_{\rm{EOS}}`, and :math:`m_{\rm{min}}` is a lower bound on the physical neutron star mass (set to :math:`0.1\,M_\odot` by default, well below any observed neutron star).
The :math:`1/(M_{\rm{TOV}} - m_{\rm{min}})` prefactor follows from assuming a uniform prior on the true mass over :math:`[m_{\rm{min}},\, M_{\rm{TOV}}]`.

We now consider the case when the mass measurement is a Gaussian, i.e.,

.. math::
    :label: gaussian_mass_measurement

    P(M|d_{\rm{radio}}) = \mathcal{N}(M | \mu, \sigma) \, ,

where :math:`\mu` and :math:`\sigma` are the mean and standard deviation of the Gaussian distribution.
Denoting the cumulative density function of the Gaussian distribution as :math:`\Phi(x)`, the likelihood function can be evaluated analytically as

.. math::
    :label: gaussian_radio_likelihood

    P(\theta_{\rm{EOS}} | d_{\rm{radio}}) \propto \frac{\Phi\!\left( \frac{M_{\rm{TOV}} - \mu}{\sigma} \right) - \Phi\!\left( \frac{m_{\rm{min}} - \mu}{\sigma} \right)}{M_{\rm{TOV}} - m_{\rm{min}}} \, .

Numerical implementation
--------------------------

The log-likelihood is computed in log-space throughout to avoid numerical underflow when combining with other log-likelihoods.
Introducing the standardised z-scores

.. math::

    z_{\rm{upper}} = \frac{M_{\rm{TOV}} - \mu}{\sigma}\,, \qquad
    z_{\rm{lower}} = \frac{m_{\rm{min}} - \mu}{\sigma}\,,

the log of the CDF difference is computed using the identity

.. math::

    \ln\!\bigl[\Phi(z_{\rm{upper}}) - \Phi(z_{\rm{lower}})\bigr]
    = \ln \Phi(z_{\rm{upper}}) + \ln\!\left[1 - \exp\!\bigl(\ln\Phi(z_{\rm{lower}}) - \ln\Phi(z_{\rm{upper}})\bigr)\right]\,,

which is evaluated with ``jnp.log1p`` to avoid catastrophic cancellation when the two CDF values are close.
The full log-likelihood then reads

.. math::

    \ln P(\theta_{\rm{EOS}} | d_{\rm{radio}})
    = \ln\!\bigl[\Phi(z_{\rm{upper}}) - \Phi(z_{\rm{lower}})\bigr]
      - \ln(M_{\rm{TOV}} - m_{\rm{min}})\,.

EOS configurations for which :math:`M_{\rm{TOV}} \le m_{\rm{min}}` (indicating a TOV integration failure or an unphysical EOS) receive a large negative penalty value instead of the above expression.
Any remaining ``NaN`` or infinite values are also replaced by this penalty to avoid numerical issues in the sampler.

.. note::

   While the class is named ``RadioTimingLikelihood``, the mass measurement need not come from pulsar timing specifically.
   Any Gaussian mass measurement — whether from pulse-profile modelling, gravitational waves, or another technique — can be incorporated.
   Moreover, Eq. :eq:`radio_likelihood` can in principle accommodate non-Gaussian mass measurements, but this is not yet implemented in ``jester`` and is left for a future release.

Further resources
-------------------

* API reference: :class:`jesterTOV.inference.likelihoods.radio.RadioTimingLikelihood`
* Config class for usage in Bayesian inference workflows: :class:`jesterTOV.inference.config.schemas.likelihoods.RadioLikelihoodConfig`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
