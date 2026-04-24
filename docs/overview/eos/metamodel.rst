.. _eos-metamodel:

.. TODO: change the f_3^* notation to f_1^{***} et cetera for f_1^*, f_1^**, f_1^***

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

The potential energy, in the metamodel, is modeled as a Taylor expansion around saturation density, and the expansion is truncated at some order. 
In practice, the expansion parameter is :math:`x`, defined as:

.. math::
   :label: x   

   x = \frac{n - n_\mathrm{sat}}{3 n_\mathrm{sat}}

This Taylor expansion introduces the nuclear empirical parameters (NEPs) as the coefficients of the expansion, which are related to physical properties of nuclear matter at saturation density.
Using these definitions, most equations and metamodel source code is written as function of :math:`x` and :math:`\delta` instead of :math:`n_0` and :math:`n_1`, as this makes the equations more compact and easier to read.

The Taylor expansions in the original metamodel paper are given by:

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

In ``jester``, this is then extended to higher orders as

.. TODO: need to figure out where this comes from

.. math::
   :label: kinetic_energy_extended

   t(n, \delta) = \frac{t_{\rm{sat}}^{\rm{FG}}}{2} (1 + 3x)^{2/3} \left[ f_1 + (1 + 3x) f^{*} + (1 + 3x)^2 f^{*}_{2} + (1 + 3x)^3 f^{*}_{3} \right]

where each of the functions :math:`f_{*}^{(j)}` are defined as

.. math::
   :label: f_star_functions

   f^*(\delta) &= (\kappa_{\text{sat}} + \kappa_{\text{sym}}\delta)(1 + \delta)^{5/3} + (\kappa_{\text{sat}} - \kappa_{\text{sym}}\delta)(1 - \delta)^{5/3} \\
   f^*_2(\delta) &= (\kappa_{\text{sat}2} + \kappa_{\text{sym}2}\delta)(1 + \delta)^{5/3} + (\kappa_{\text{sat}2} - \kappa_{\text{sym}2}\delta)(1 - \delta)^{5/3} \\
   f^*_3(\delta) &= (\kappa_{\text{sat}3} + \kappa_{\text{sym}3}\delta)(1 + \delta)^{5/3} + (\kappa_{\text{sat}3} - \kappa_{\text{sym}3}\delta)(1 - \delta)^{5/3}



Potential energy
-----------------

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

   b(\delta) = b_{\rm{sat}} + b_{\rm{sym}} \delta^2


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

Computing the pressure
------------------------

After having defined how the energy is computed, let us have a look at computing the pressure, which is defined in Eq. (9) in :cite:`Margueron:2017eqc` as

.. math::
   :label: pressure_margueron_eq

      P(n, \delta) = n^2 \frac{\partial e(n, \delta)}{\partial n}

where now :math:`e(n, \delta) = e_{\rm{kin}}(n, \delta) + e_{\rm{pot}}(n, \delta)`.
By using the above expressions and the chain rule, this can be rewritten as

.. math::
   :label: pressure_rewritten

   P(n, \delta) = \frac{n_{\rm{sat}}}{3} (1 + 3x)^2 \frac{\partial e}{\partial x}

which makes the expression easier to compute. 
By using the final form of Eq. :eq:`kinetic_energy_extended` as a function of :math:`x`, one can find that

.. math::
   :label: pressure_kinetic_contribution

   P_{\rm{kin}}(n, \delta) = \frac{1}{3} n_{\text{sat}} t_{\rm{sat}}^{\rm{FG}} (1 + 3x)^{5/3} \left( f_1 + \frac{5}{2}(1 + 3x)f^* + 4(1 + 3x)^2 f^*_2 + \frac{11}{2}(1 + 3x)^3 f^*_3 \right)

which is easily shown by carrying out the derivative. 

For the contribution from the potential energy, we can first compute an intermediate result, namely the derivate of :math:`u_j` with respect to :math:`x`:

.. math::
   :label: duj_dx

   \frac{\partial u_j(n, \delta)}{\partial x} = \left[ N + 1 - j - 3 bx \right] \left[ - (-3)^{N - j} x^{N + 1 - j} e^{-b (1 + 3x)} \right]

From this, it is then easy to derive that 

.. math::
   :label: x_times_duj_dx

   x \frac{\partial u_j(n, \delta)}{\partial x} = \left[ N + 1 - j - 3 bx \right] \left[ u_j - 1 \right]

which will appear in the derivative and is therefore a useful quantity to precompute. 
Having this at hand, the contribution to the pressure from the potential energy can be computed as

.. math::
   :label: pressure_potential_contribution

   P_{\rm{pot}}(n, \delta) = v_0 \frac{\partial u_0}{\partial x} + \sum_{j = 1}^N \frac{1}{j!} v_j \left( \frac{\partial u_j}{\partial x} x + u_j j \right) x^{j - 1} \, ,

where we have defined the shorthand

.. math::
   :label: v_j_shorthand

   v_j = v_{\rm{sat},j} + v_{\rm{sym},j} \delta^2

With these equations at hand, the ``jester`` code for computing the pressure in the metamodel can be easily traced back.

.. TODO: need to add the j label to the v's above!!!

Adding electrons
----------------

So far, we have ignored the presence of leptons. We will, for the moment, assume the neutron start to only contain electrons, so that charge neutrality requires that the electron number density :math:`n_e` is equal to the proton number density :math:`n_p`. The Fermi momentum is given by 

.. math::
   :label: fermi_momentum_electron

   k_{F,e} = (3 \pi^2 n_e)^{1/3} \hbar c = (\tfrac{3}{2} \pi^2 n)^{1/3} (1 - \delta)^{1/3} \hbar c \, ,

where we have used that :math:`n_e = n_p = n (1 - \delta)/2`. We will define the parameter :math:`x_e` defined as

.. math::
   :label: x_electron

   x_e = \frac{k_{F,e}}{m_e} \, .

.. TODO: need to ask Guilherme how to derive this?

The total energy density (including the mass term) is given by 

.. math::
   :label: energy_density_electron

   \varepsilon_e = C_e f(x_e) \, ,

where :math:`C_e` is a defined as

.. math::
   :label: C_electron

   C_e = \frac{m_e^4}{8 \pi^2 \hbar^3 c^3} \, ,

and the function :math:`f(x_e)` is defined as

.. math::
   :label: f_x_electron

   f(x_e) = x_e (1 + 2 x_e^2) \sqrt{1 + x_e^2} - \sinh^{-1}(x_e) \, .

The pressure is defined similarly as for the nucleons above, so that we find

.. math::
   :label: pressure_electron

   P_e = -\varepsilon_e + \frac{8}{3} C_e x_e^3 \sqrt{1 + x_e^2} \, .

and the contribution to the isoscalar compressibility is given by

.. math::
   :label: isoscalar_compressibility_electron

   K_{\rm{is,e}} = \frac{8 C_e}{n} \frac{x_e^3 (3 + 4 x_e^2)}{\sqrt{1 + x_e^2}} - \frac{9}{n} (\varepsilon_e + P_e) \, .

Speed-of-sound
------------------

For the metamodel, the sound speed in units of the speed of light :math:`c_s` is defined as

.. math::
   :label: speed_of_sound

   c_s^2 = \frac{\partial P}{\partial \varepsilon}
   
where the total energy density, which includes the rest mass contribution, is given by

.. math::
   :label: energy_density_total

   \varepsilon = n(mc^2 + e)

This can be rewritten as

.. math::
   :label: speed_of_sound_rewritten

   c_s^2 = \frac{\partial P}{\partial n} \left( \frac{\partial \varepsilon}{\partial n} \right)^{-1} \, .

The second factor is readily computed to yield

.. math::
   :label: derivative_energy_density

   \frac{\partial \varepsilon}{\partial n} = m c^2 + e + n \frac{\partial e}{\partial n} = m c^2 + e + \frac{P}{n} = \frac{\varepsilon + P}{n} = h \, .

Therefore, the speed of sound can also be rewritten as

.. math::
   :label: speed_of_sound_metamodel

   c_s^2(x, \delta) = \frac{K_{\rm{is}}(x, \delta)}{9\left[ mc^2 + e + \frac{P(x, \delta)}{n} \right]} \, ,

where :math:`P` is the pressure and :math:`\epsilon` is the energy density. This is Eq. (10) in the Margueron et al paper. 
Here, we have defined the isoscalar compressibility as

.. math::
   :label: isoscalar_compressibility

   K_{\rm{is}}(x, \delta) = 9 \frac{\partial P}{\partial n} = 9 n^2 \frac{\partial^2 e(x, \delta)}{\partial n^2} + 18 \frac{P(x, \delta)}{n} \, .

Above, we have already computed the isoscalar compressibility contribution for electrons. In order to compute the isoscalar compressibility contribution from the nucleons, we can use similar calculations as done for the pressure calculation, and is most easily done by considering the kinetic and potential contributions separately, similar to how we computed the pressure above.
The result for the kinetic contribution, not including the second term (which just reuses the results from above) is given by:

.. math::
   :label: isoscalar_compressibility_kinetic_contribution

   K_{\rm{is,kin}}^{(1)}(x, \delta) = t_{\text{sat}} (1 + 3x)^{2/3} \left[ -f_1 + 5(1 + 3x)f^* + 20(1 + 3x)^2 f^*_2 + 44(1 + 3x)^3 f^*_3 \right]

The contribution from the potential energy to the first term is given by

.. math::
   :label: isoscalar_compressibility_potential_contribution

   K_{\rm{is,pot}}^{(1)}(x, \delta) = (1 + 3x)^2 \sum_{j=0}^{N} \frac{v_{j}(\delta)}{j!} x^{j-2}  \left[ j(j - 1)u_{j}\delta_{j \ge 2} + 2j x u'_{j} \delta_{j \ge 1} + x^2 u''_{j} \right]\, ,

where :math:`\delta_{j\geq a}` restricts the sum to terms with :math:`j \geq a`, and primes are shorthand for derivatives with respect to :math:`x`. 
One can show the intermediate result that

.. math::
   :label: x_squared_times_second_derivative_uj

   x^2 u''_{j} = \left[ -(N + 1 - j)(N - j) + 6xb(N + 1 - j) - 9x^2b^2 \right] (1 - u_{j}) \, , 

which helps in organizing the computations here. 


:math:`\beta`-equilibrium
--------------------------

Matter inside the neutron star is expected to be in :math:`\beta`-equilibrium, which implies that the proton fraction is a function of density, and is therefore no longer an additional variable, so that pressure becomes a function of density alone. Here, we will describe how the proton fraction can be computed, assuming :math:`n,p,e` matter, meaning that the equilibrium is between the following reactions

.. math::
   :label: beta_equilibrium_reactions

   n &\leftrightarrow p + e^- + \bar{\nu}_e \, , \\
   p + e^- &\leftrightarrow n + \nu_e \, .

This leads to the following relations on the chemical potentials of the different species:

.. math::
   :label: chemical_potential_relations

   m_n c^2 + \mu_n + \mu_{\nu_e} &= m_p c^2 + \mu_p + \mu_e \, , \\
   \mu_{\nu_e} &= \mu_{\bar{\nu}_e} \, .

We assume that neutrinos escape freely, so that :math:`\mu_{\nu_e} = \mu_{\bar{\nu}_e} = 0`. 
Moreover, we use the approximation that :math:`m_n \approx m_p`, so that the mass terms cancel out.
With these assumptions, the chemical potential relations can be rewritten as

.. math::
   :label: chemical_potential_relations_rewritten

   \mu_n = \mu_p + \mu_e \, .

The chemical potential for electrons is simply equal to the Fermi energy, so (as defined above):

.. math::
   :label: chemical_potential_electron

   \mu_e = k_{F,e} = \hbar c (3 \pi^2 n)^{1/3} (1 - \delta)^{1/3} \, .

For the difference between the neutron and proton chemical potentials, we can use the following relation:

.. TODO: how to show this?

.. math::
   :label: chemical_potential_difference

   \mu_n - \mu_p = 2 \frac{\partial e(n, \delta)}{\partial \delta} = 4 \delta e_{\rm{sym}} \, .


TODO: finish the calculation

For the proton fraction :math:`Y_p`, we can use the relation between :math:`\delta` and :math:`Y_p` shown above, and ... so we find the cubic equation 

.. math::
   :label: cubic_equation_proton_fraction

   y^3 + \frac{(3 \pi^2 n) \hbar c}{8 e_{\rm{sym}}} y - \frac{1}{2} = 0 \, .

This is of the form 

.. math::
   :label: cubic_equation_general

   y^3 + a y^2 + b y  + c = 0 \, , \quad a = 0 \, , \quad b = \frac{(3 \pi^2 n) \hbar c}{8 e_{\rm{sym}}} \, , \quad c = -\frac{1}{2} \, .

This can be solved analytically (see for instance Section 6 of :cite:`Press:2007ipz`). 


Implementation in ``jester``
------------------------------

In ``jester``, at the time of writing, there are a few specifics about the implementation of the metamodel EOS parametrization that are worth noting:

All of the physics described above has been implemented in ``jester``, but the following features are not readily exposed to the user or the Bayesian inference workflow:

* The modifications due to the effective mass are ignored, such that :math:`m=m^*` in the kinetic energy expression. 
* The non-quadratic contributions to the potential energy. 
* In ``jester``, we take take an average nucleonic mass :math:`m = (m_n + m_p)/2` for the nucleonic mass term, instead of separately accounting for the neutron and proton mass.

In particular, the metamodel code (in :class:`jesterTOV.eos.metamodel.MetaModel_EOS_model`) has the equations implemented for them, but the ``construct_eos`` method does not currently allow the user to easily use these modifications, outside of specifying these parameters in the ``__init__`` method of the class. These modifications are controlled by parameters called ``kappa_*`` for the kinetic energy and ``v_nq`` for the potential energy, but these are set to zero by default when initializing the metamodel class, and are not exposed to downstream functions. In particular, the Bayesian inference workflow for the moment does not recognize these parameters. Future releases of ``jester`` will likely include support for these features during sampling.



Further resources
-------------------

* API reference: :class:`jesterTOV.eos.metamodel.MetaModel_EOS_model`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
