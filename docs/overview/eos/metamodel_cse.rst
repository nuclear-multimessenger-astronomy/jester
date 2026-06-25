.. _eos-metamodel-cse:

Metamodel + CSE
===============

This page describes the Metamodel + CSE EOS parametrization and its implementation in ``jester``.
Before reading this page, it may help to first read the :ref:`eos-metamodel` page, since the CSE
extension builds directly on top of the standard metamodel.

The specific implementation follows Ref. :cite:`Koehn:2024set`.
Moreover, you can find the datasets for this paper in `this webpage <https://multi-messenger.physik.uni-potsdam.de/eos_constraints/>`_ . 

Physical motivation
-------------------

The metamodel Taylor expansion around saturation density provides a well-motivated description of
nuclear matter up to roughly :math:`2{-}3\,n_\mathrm{sat}`, where the coefficients of the expansion
(the NEPs) are constrained by nuclear theory and experiment.
At the higher densities reached in the cores of the most massive neutron stars, however, the metamodel expansion might break down, as new degrees of freedom could appear (such as deconfined quark matter), such that the nuclear matter description is no longer valid. Therefore, one needs to switch to an agnostic and flexible framework at the higher densities.

The speed-of-sound extension (CSE) approach handles this by abandoning the metamodel at a
user-specified break density :math:`n_\mathrm{break}` and instead specifying the squared speed of
sound :math:`c_s^2(n)` directly on a grid of density nodes above :math:`n_\mathrm{break}`.
The full thermodynamic EOS in this high-density region is then recovered by integration.
This keeps the well-understood NEP parametrization at low-to-intermediate densities while
remaining agnostic about the high-density physics.

Thermodynamic integration above the break density
--------------------------------------------------

Given :math:`c_s^2(n)` as a function of baryon number density, the baryon chemical potential
:math:`\mu`, the pressure :math:`p`, and the energy density :math:`\varepsilon` can all be
obtained from the fundamental thermodynamic relations.
The definition of the speed of sound, together with the Gibbs-Duhem relation
:math:`dp = n\,d\mu`, gives

.. math::
   :label: dlnmu_dlnn_cse

   \frac{d\ln\mu}{d\ln n} = c_s^2 \, ,

which integrates to

.. math::
   :label: mu_cse

   \mu(n) = \mu_\mathrm{break}\,\exp\!\left[\int_{n_\mathrm{break}}^{n} \frac{c_s^2(n')}{n'}\,dn'\right].

The pressure follows from :math:`dp = n\,d\mu = c_s^2\,\mu\,dn`:

.. math::
   :label: p_cse

   p(n) = p_\mathrm{break} + \int_{n_\mathrm{break}}^{n} c_s^2(n')\,\mu(n')\,dn' \, ,

and the first law of thermodynamics, :math:`d\varepsilon = \mu\,dn`, gives

.. math::
   :label: e_cse

   \varepsilon(n) = \varepsilon_\mathrm{break} + \int_{n_\mathrm{break}}^{n} \mu(n')\,dn' \, .

The boundary values :math:`\mu_\mathrm{break}`, :math:`p_\mathrm{break}`, and
:math:`\varepsilon_\mathrm{break}` are read off from the metamodel EOS at :math:`n = n_\mathrm{break}`,
ensuring continuity of the full EOS at the transition.

The example below shows the pressure and speed of sound for a representative EOS with a
four-node CSE extension starting at :math:`n_\mathrm{break} = 2\,n_\mathrm{sat}`.

.. plot:: overview/eos/metamodel_cse_plot.py

   Pressure (top) and squared speed of sound (bottom) versus baryon number density for a
   representative Metamodel + CSE EOS.
   The grey, orange, and green shaded bands indicate the crust, the metamodel core, and the
   CSE extension region, respectively.
   The dashed vertical line marks the break density :math:`n_\mathrm{break} = 2\,n_\mathrm{sat}` for this example, although note that ``jester`` can vary this density as a parameter on-the-fly.

Parameters
----------

The CSE parametrization adds three groups of parameters on top of the nine NEPs of the
standard metamodel:

- **Break density** :math:`n_\mathrm{break}` — the density at which the metamodel is replaced
  by the CSE. In inference, this is sampled from a prior that typically spans
  :math:`1{-}2\,n_\mathrm{sat}`.

- **Node positions** :math:`n_{\mathrm{CSE},i}` — the density grid points above
  :math:`n_\mathrm{break}` at which the speed of sound is specified.
  In ``jester`` these are parametrized as normalized fractions
  :math:`u_i \in [0, 1]` via
  :math:`n_{\mathrm{CSE},i} = n_\mathrm{break} + u_i\,(n_\mathrm{max} - n_\mathrm{break})`,
  so the parameter names are ``n_CSE_0_u``, ``n_CSE_1_u``, etc.
  The fractions are sorted internally to enforce a monotonic density grid.

- **Speed-of-sound values** :math:`c_{s,i}^2` — the squared speed of sound (in units of :math:`c^2`)
  at each node, plus a final value at :math:`n_\mathrm{max}`.
  The parameter names are ``cs2_CSE_0``, ``cs2_CSE_1``, ..., ``cs2_CSE_nb_CSE``.
  Between nodes, the speed of sound is linearly interpolated.

For a model with ``nb_CSE`` nodes, the total free parameter count is
:math:`9\text{ (NEPs)} + 1\text{ (}n_\mathrm{break}\text{)} + 2\,N_\mathrm{CSE} + 1`,
where :math:`N_\mathrm{CSE}` is the number of internal nodes.
The default in the provided example configurations is ``nb_CSE = 8``.

Usage example
-------------

**Configuration file:**

.. code-block:: yaml

   eos:
     type: metamodel_cse
     ndat_metamodel: 100
     nmax_nsat: 25.0
     nb_CSE: 8
     crust_name: DH

**Prior file** (partial; ``nb_CSE = 8`` example):

.. code-block:: python

   E_sat = UniformPrior(-16.1, -15.9, parameter_names=["E_sat"])
   K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
   # ... other NEPs ...
   nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])
   n_CSE_0_u = UniformPrior(0.0, 1.0, parameter_names=["n_CSE_0_u"])
   cs2_CSE_0 = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_0"])
   # ... remaining n_CSE_i_u and cs2_CSE_i pairs ...

A complete working example is in ``examples/inference/blackjax-ns-aw/NICER_J0030/``.
For a step-by-step guide to running inference, see the :doc:`/inference/quickstart`.

Further resources
-----------------

* API reference: :class:`jesterTOV.eos.metamodel.MetaModel_with_CSE_EOS_model`
* Metamodel background: :ref:`eos-metamodel`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
