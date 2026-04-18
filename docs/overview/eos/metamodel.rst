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

The energy density is the sum of a kinetic and a potential term

.. math::
   :label: energy_density

   e_{\rm{MM}}(n, \delta) = t^*(n, \delta) + e^{\rm{pot}}(n, \delta)

In the metamodel, the potential energy of this matter can be expanded as follows:

.. math::
   :label: energy_per_nucleon

   e^{\rm{pot}}(n, \delta) = e_{\rm{is}}(n) + \delta^2 e_{\rm{iv}}(n) + \mathcal{O}(\delta^4)

where :math:`e_{\rm{is}}(n)` and :math:`e_{\rm{iv}}(n)` are the isoscalar and isovector contributions to the energy per nucleon, respectively.
Note that :math:`n_1` enters implicitly in the asymmetry parameter :math:`\delta`.
Often, the isovector energy is also called the symmetry energy.
We define the saturation density :math:`n_\mathrm{sat}` of symmetric nuclear matter as the density at which the symmetric matter pressure reaches zero. 

Below, we explain how both the kinetic and potential energy terms are modeled in ``jester`` in more detail.

Kinetic energy
---------------

In the metamodel approach, the kinetic energy is modeled using a nonrelativistic free Fermi gas (FG) and by taking into account the effective mass due to the momentum-dependent nuclear interaction (see Section IIIA in Ref. :cite:`Margueron:2017eqc` for details).
The kinetic energy per nucleon is then given by:

.. math::
   :label: kinetic_energy

   t^{\rm{FG}}(n, \delta) = \frac{t_{\rm{sat}}^{\rm{FG}}}{2} \left( \frac{n}{n_{\rm{sat}}} \right)^{2/3} \left[ \left( 1 + \kappa_{\rm{sat}} \frac{n}{n_{\rm{sat}}} \right) f_1(\delta) + + \kappa_{\rm{sym}} \frac{n}{n_{\rm{sat}}} f_2(\delta) \right]

where the constant :math:`t_{\rm{sat}}^{\rm{FG}}` is defined as

.. math::
   :label: t_sat_FG

   t_{\rm{sat}}^{\rm{FG}} = \frac{3 \hbar^2}{10m} \left( \tfrac{3}{2} \pi^2 n_{\rm{sat}} \right)^{2/3}

:math:`m` is the nucleonic mass, and the functions :math:`f_1`, :math:`f_2` are defined as:

.. math::
   :label: f_1_and_f_2

   f_1(\delta) &= (1 + \delta)^{5/3} + (1 - \delta)^{5/3} \\
   f_2(\delta) &= \delta \left[ (1 + \delta)^{5/3} + (1 - \delta)^{5/3} \right]

Potential energy
-----------------

As explained above, the potential energy is modeled as a Taylor expansion around saturation density, and the expansion is truncated at some order. 
In practice, the expansion parameter is :math:`x`, defined as:

.. math::
   :label: x   

   x = \frac{n - n_\mathrm{sat}}{3 n_\mathrm{sat}}

This Taylor expansion introduces the nuclear empirical parameters (NEPs) as the coefficients of the expansion, which are related to physical properties of nuclear matter at saturation density.
Their expansions in the original metamodel paper are given by:

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

The expansion shown in Eq. :eq:`energy_per_nucleon` is a rather simplistic model which has some defaults. 
Indeed, as explained in detail in Sec. III in Ref. :cite:`Margueron:2017eqc`, for some choices of the NEPs, the potential energy is not zero at zero density, which is unphysical.
This is to be expected, as the expansion in Eq. :eq:`energy_per_nucleon` is only valid around saturation density, and therefore does not necessarily incorporate the correct low-density behavior of the EOS.
To solve this issue, the potential energy functional is slightly adjusted.
For this, let us start by rewriting the original expansion of the potential energy shown above as follows:

.. math::
   :label: ELFc_expansion_without_u_factor

   e^{\rm{pot}}(n, \delta) = \sum_{j = 0}^N \frac{1}{j!} \left( v_{\rm{sat}} + v_{\rm{sym}} \delta^2 \right) x^j 

Where the coefficients are related to the nuclear empirical parameters (as we discuss below in more detail).
In our case, we truncate the expansion at order :math:`N = 4`. 
The modification proposed by :cite:`Margueron:2017eqc` and called "ELFc" is to include an additional term in the potential energy functional to describe the low-density behavior. 
This can equivalently also be incorporated as a multiplicative factor in the expansion above, as follows:

.. math::
   :label: ELFc_expansion

   e^{\rm{pot}}(n, \delta) = \sum_{j = 0}^N \frac{1}{j!} \left( v_{\rm{sat}} + v_{\rm{sym}} \delta^2 \right) x^j u_j(n, \delta)

where the functions :math:`u_j(n, \delta)` are defined as:

.. math::
   :label: u_j_function_Margueron

   u_j(n, \delta) = 1 - (-3x)^{N + 1 - j} \exp\left( -b \frac{n}{n_{\rm{sat}}} \right)

where the parameter :math:`b` controls the strength of the low-density modification.
This introduces an additional density-dependent term at lower densities to satisfy the zero-density limit, and drops to zero at higher densities, to conserve the behavior of the expansion introduced above at densities around and above saturation density.

In practice, we follow a slight modification to this ELFc expansion, following by Ref. :cite:`Somasundaram:2020chb` and Ref. :cite:`Grams:2021lzx`.
In Eq. :eq:`ELFc_expansion` and as defined in the original metamodel paper, a single parameter :math:`b` controls how fast the additional term drops to zero at higher densities, and Ref. :cite:`Margueron:2017eqc` suggests a value of :math:`b = 10 \ln(2) \approx 6.93`. 
However, in ``jester``, we follow the modifications proposed by Ref. :cite:`Somasundaram:2020chb` density, and instead, define two parameters :math:`b_{\rm{sat}}` and :math:`b_{\rm{PNM}}` controlling this low-density behavior in symmetric nuclear matter and pure neutron matter, respectively.
Their values for these parameters are given in Table III of Ref. :cite:`Somasundaram:2020chb`.
Following :cite:`Grams:2021lzx` , these parameters are then used to define an asymmetry-dependent parameter :math:`b(\delta)` that is used as :math:`b` in the expansions above.

.. math::
   :label: beta_function_of_delta

   b(\delta) = b_{\rm{sat}} + (b_{\rm{sym}}) \delta^2


Finally, Eq. :eq:`ELFc_expansion` where :math:`b = b(\delta)` is the form of the potential energy functional used in ``jester``.
Within this description, the mapping from the nuclear empirical parameters, which are the coefficients of the Taylor expansion, to the coefficients :math:`v_{\rm{sat}}` and :math:`v_{\rm{sym}}` is given by the expresssions in Appendix B of Ref. :cite:`Somasundaram:2020chb`.

NOTE: This describes the potential energy up to quadratic order in the asymmetry parameter :math:`\delta`. ``jester`` also has support for non-quadratic contributions, as described in Ref. :cite:`Somasundaram:2020chb`. 


Useful relations
------------------

The relations below can help guide you in the metamodel source code in ``jester``, as they can be used to rewrite some equations to other forms.
First, note that :math:`\frac{n}{n_{\rm{sat}}}`, which appears in the exponentials of the potential energy functionals, can be rewritten to

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

.. Using the metamodel in ``jester``
.. ----------------------------------


Further resources
-------------------

* API reference: :class:`jesterTOV.eos.metamodel.MetaModel_EOS_model`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
