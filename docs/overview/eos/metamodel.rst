.. _eos-metamodel:

Metamodel
=========

This page describes the metamodel EOS parametrization and its implementation in ``jester``. 

General background
----------------------

The metamodel, or a 'model-of-a-model', is a model for the homogeneoous nucleonic EOS, and was introduced in Refs. The metamodel parametrization is described in Refs. :cite:`Margueron:2017eqc` and :cite:`Margueron:2017lup`. 
It describes nuclear matter with different isoscalar (is) density

.. math::
   :label: n_0

   n_0 = n_n + n_p

(which we will also denote simply by :math:`n` below) and isovector (iv) density

.. math::
   :label: n_1

   n_1 = n_n - n_p

where :math:`n_n` and :math:`n_p` are the neutron and proton number densities, respectively.
By introducing the asymmetry parameter

.. math::
   :label: delta

   \delta = n_1/n_0
   
the two possible extreme values describing particular types of nuclear matter are:

* Symmetric nuclear matter (SNM) with :math:`\delta = 0` (i.e., :math:`n_n = n_p`) and
* Pure neutron matter (PNM) with :math:`\delta = 1` (i.e., :math:`n_p = 0`).

The energy per nucleon of this nuclear matter can be expanded as follows:

.. math::
   :label: energy_per_nucleon

   \tfrac{E}{A}(n_0, n_1) = e_{\rm{is}}(n_0) + \delta^2 e_{\rm{iv}}(n_0) + \mathcal{O}(\delta^4)

where :math:`e_{\rm{is}}(n_0)` and :math:`e_{\rm{iv}}(n_0)` are the isoscalar and isovector contributions to the energy per nucleon, respectively.
Note that :math:`n_1` enters implicitly in the asymmetry parameter :math:`\delta`.
Often, the isovector energy is also called the symmetry energy.
We define the saturation density :math:`n_\mathrm{sat}` of symmetric nuclear matter as the density at which the symmetric matter pressure reaches zero. 
The two terms in the expansion are then each Taylor expanded in the parameter :math:`x` defined as:

.. math::
   :label: x

   x = \frac{n_0 - n_\mathrm{sat}}{3 n_\mathrm{sat}}

This Taylor expansion introduces the nuclear empirical parameters (NEPs) as the coefficients of the expansion, which are related to physical properties of nuclear matter at saturation density.
Their expansions are given by:

.. math::
   :label: metamodel_expansions
   
   e_{\rm{is}} &= E_\mathrm{sat} + \tfrac{1}{2} K_\mathrm{sat} x^2 + \tfrac{1}{3!} Q_\mathrm{sat} x^3 + \tfrac{1}{4!} Z_\mathrm{sat} x^4 + \ldots \\
   e_{\rm{iv}} &= E_\mathrm{sym} + L_\mathrm{sym} x + \tfrac{1}{2} K_\mathrm{sym} x^2 + \tfrac{1}{3!} Q_\mathrm{sym} x^3 + \tfrac{1}{4!} Z_\mathrm{sym} x^4 + \ldots

The figure below shows an example of the energy per nucleon :math:`E/A` for symmetric nuclear matter and pure neutron matter computed from the metamodel with fiducial NEP values.

.. plot:: overview/eos/metamodel_nep_plot.py

   Energy per nucleon :math:`E/A` for symmetric nuclear matter (:math:`\delta = 0`, red)
   and pure neutron matter (:math:`\delta = 1`, blue) computed from the meta-model with
   fiducial NEP values (:math:`E_\mathrm{sat} = -16\,\mathrm{MeV}`,
   :math:`K_\mathrm{sat} = 230\,\mathrm{MeV}`,
   :math:`E_\mathrm{sym} = 32\,\mathrm{MeV}`,
   :math:`L_\mathrm{sym} = 60\,\mathrm{MeV}`).

Before continuining the discussion of the metamodel we provide a few relations that are helpful for the discussion above when reading the ``jester`` metamodel source.
First, one can observe that

.. math::
   :label: n_over_nsat
   
   \frac{n}{n_{\rm{sat}}} = 1 + 3x

Second, by introducing the proton fraction 

.. math::
   :label: def_proton_fraction
   
   Y_p = n_p/n

one can also easily show that

.. math::
   :label: delta_proton_fraction
   
   \delta = 1 - 2 Y_p


Kinetic energy
---------------

The expansion given above does not account for the kinetic energy contribution to the energy per nucleon. 
In the metamodel approach, this is modeled using a nonrelativistic free Fermi gas (FG) and by taking into account the effective mass due to the momentum-dependent nuclear interaction (see Section IIIA in Ref. :cite:`Margueron:2017eqc` for details).
The kinetic energy per nucleon is then given by:

.. math::
   :label: kinetic_energy

   t^{\rm{FG}}(n_0, n_1) = \frac{t_{\rm{sat}^{\rm{FG}}}}{2} \left( \frac{n_0}{n_{\rm{sat}}} \right)^{2/3} \left[ \left( 1 + \kappa_{\rm{sat}} \frac{n_0}{n_{\rm{sat}}} \right) f_1(\delta) + + \kappa_{\rm{sym}} \frac{n_0}{n_{\rm{sat}}} f_2(\delta) \right]

where

.. math::
   :label: t_sat_FG

   t_{\rm{sat}}^{\rm{FG}} = \frac{3 \hbar^2}{10m} \left( \tfrac{3}{2} \pi^2 n_{\rm{sat}} \right)^{2/3}

:math:`m` is the nucleonic mass, and the functions :math:`f_1`, :math:`f_2` are defined as:

.. math::
   :label: f_1_and_f_2

   f_1(\delta) &= (1 + \delta)^{5/3} + (1 - \delta)^{5/3} \\
   f_2(\delta) &= \delta \left[ (1 + \delta)^{5/3} + (1 - \delta)^{5/3} \right]


Potential energy functional
----------------------------

The expansion shown in Eq. :eq:`energy_per_nucleon` is a rather simplistic model which has some defaults. 
Indeed, as explained in detail in Sec. III in Ref. :cite:`Margueron:2017eqc`, for some choices of the NEPs, the potential energy is not zero at zero density, which is unphysical.
This is to be expected, as the expansion in Eq. :eq:`energy_per_nucleon` is only valid around saturation density, and therefore does not necessarily incorporate the correct low-density behavior of the EOS.

To solve this issue, the potential energy functional is slightly adjusted. 
In ``jester``, we take the functional form called "ELFc" in Ref. :cite:`Margueron:2017eqc`, with modifications proposed by Ref. :cite:`Somasundaram:2020chb`.
This introduces an additional density-dependent term at lower densities to satisfy the zero-density limit, and drops to zero at higher densities, to conserve the behavior of the expansion introduced above at densities around and above saturation density.

We start by first rewriting the original expansion of the potential energy as follows:

.. math::
   :label: metamodel_potential_energy_expansion

   e_{\rm{pot}}(n_0, n_1) = \sum_{\alpha \geq 0}^N \frac{1}{\alpha!} \left( c_\alpha^{\rm{is}} + c_\alpha^{\rm{iv}} \delta^2 \right) x^\alpha

where, in this form, the coefficients :math:`c_\alpha^{\rm{is}}` and :math:`c_\alpha^{\rm{iv}}` are the NEPs as introduced above.
Note that the expansion is truncated at order :math:`N`, which is set to 4 by default in ``jester``.
Instead, the ELFc functional form of the potential energy is given by

.. math::
   :label: ELFc_expansion

   v^N_{ELFc}(n_0, n_1) = \sum_{\alpha \geq 0}^N \frac{1}{\alpha!}( v_{\alpha}^{is}+ v_{\alpha}^{iv} \delta^2) 
   x^\alpha - (a_N^{is} + a_N^{iv}\delta^2) x^{N+1} \exp \left(-b \frac{n_0}{n_{sat}} \right) 

The coefficients :math:`a_N^{is}` and :math:`a_N^{iv}` are determined by the condition that the potential energy goes to zero at zero density, which therefore results in

.. math::
   :label: a_N_coefficients

   a_N^{is} &= -\sum_{\alpha \geq 0}^{N} \frac{1}{\alpha!} v_{\alpha}^{is} (-3)^{N + 1 - \alpha} \\
   a_N^{iv} &= -\sum_{\alpha \geq 0}^{N} \frac{1}{\alpha!} v_{\alpha}^{iv} (-3)^{N + 1 - \alpha}

A more compact form of the ELFc expansion given by Eq. :eq:`ELFc_expansion` can be obtained as follows

.. math::
   :label: ELFc_expansion_compact
   
   v^N_{ELFc}(n_0, n_1)=\sum_{\alpha \geq 0}^N \frac{1}{\alpha!}(v_{\alpha}^{is} + v_{\alpha}^{iv} \delta^2) x^\alpha u^N_{ELFc, \alpha}(x)

which introduces a new function :math:`u^N_{ELFc,\alpha}` defined as

.. math::
   :label: u_ELFc
   
   u^N_{ELFc, \alpha}(x) = 1 - (-3x)^{N + 1 - \alpha} \exp(-b n_0/n_{sat})

In Eq. :eq:`ELFc_expansion`, a single parameter :math:`b` controls how fast the additional term drops to zero at higher densities, and Ref. :cite:`Margueron:2017eqc` suggests a value of :math:`b = 10 \ln(2) \approx 6.93`. 
However, in ``jester``, we follow the modifications proposed by Ref. :cite:`Somasundaram:2020chb` density, and instead, define two parameters :math:`b_{\rm{sat}}` and :math:`b_{\rm{PNM}}` controlling this low-density behavior in symmetric nuclear matter and pure neutron matter, respectively.
The values for these parameters are given in Table III of Ref. :cite:`Somasundaram:2020chb`.
Therefore, building on top of the expansion in Eq. :eq:`ELFc_expansion_compact`, we define the potential energy functionals for pure neutron matter and symmetric nuclear matter in ``jester`` as

.. math::
   :label: ELFc_expansion_compact
   
   e_{\rm{PNM/SNM}}(n) &= \sum_{\alpha \geq 0}^N \frac{1}{\alpha!} v_{\rm{PNM/SNM}, \alpha} x^\alpha u^N_{\rm{sym/PNM}, \alpha}(x) \\
   u^N_{PNM/SNM}(x) &= 1 - (-3x)^{N + 1 - \alpha} \exp(-b_{\rm{sym/PNM}} n_0/n_{sat})

As in the original metamodel, the expansion coefficients :math:`v_{\rm{PNM/SNM}, \alpha}` are related to the NEPs.
The complete expressions are given in Appendix B of Ref. :cite:`Somasundaram:2020chb`.

Finally, the potential energy and kinetic energy are combined to give the total energy per nucleon as follows:

.. math::
   :label: total_energy_per_nucleon

   \tfrac{E}{A}(n_0, n_1) = t^{\rm{FG}}(n_0, n_1) + e_{\rm{pot}}(n_0, n_1)

where 

.. math::
   :label: potential_energy_form_combined

   e_{\rm{pot}}(n_0, n_1) = e_{\rm{is}}(n_0) + \delta^2 e_{\rm{iv}}(n_0)

Using the metamodel in ``jester``
----------------------------------


Further resources
-------------------

* API reference: :class:`jesterTOV.eos.metamodel.MetaModel_EOS_model`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
