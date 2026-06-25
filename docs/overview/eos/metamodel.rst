.. _eos-metamodel:

Metamodel
=========

This page describes the metamodel EOS parametrization and its implementation in ``jester``.

General background
----------------------

The metamodel, or a 'model-of-a-model', is a model for the homogeneous nucleonic EOS, introduced and described in Refs. :cite:`Margueron:2017eqc` and :cite:`Margueron:2017lup`. 
It describes nuclear matter with different isoscalar (is) density

.. math::
   :label: n_0

   n_0 = n_n + n_p

(which we will also denote simply by :math:`n` below) and isovector (iv) density

.. math::
   :label: n_1

   n_1 = n_n - n_p

where :math:`n_n` and :math:`n_p` are the neutron and proton number densities, respectively.
We also define the asymmetry parameter

.. math::
   :label: delta

   \delta = n_1/n_0

which quantifies the isospin asymmetry of nuclear matter. The two possible extreme values of :math:`\delta` describe particular types of nuclear matter:

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
Note that this implies that 

.. math::
   :label: n_over_nsat
   
   \frac{n}{n_{\rm{sat}}} = 1 + 3x

Which is a useful relation to have when deriving the equations below. Moreover, we can also easily see that

.. math::
   :label: delta_proton_fraction
   
   \delta = 1 - 2 Y_p

where :math:`Y_p` is the proton fraction, defined as :math:`Y_p = n_p/n`.

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

   t_0(x, \delta) = \frac{t_{\rm{sat}}}{2} (1+3x)^{2/3} \big[ f^{\rm{FG}}(\delta) &+ (1+3x) f^*(\delta) \\
   &+ (1+3x)^2 f^{**}(\delta) + (1+3x)^3 f^{***}(\delta) \big]

where the constant :math:`t_{\rm{sat}}` is defined as

.. math::
   :label: t_sat

   t_{\rm{sat}} = \frac{3 \hbar^2}{10m} \left( \tfrac{3}{2} \pi^2 n_{\rm{sat}} \right)^{2/3}

:math:`m` is the nucleonic mass, and the auxiliary functions :math:`f^{\rm{FG}}`, :math:`f^*`, :math:`f^{**}`, and :math:`f^{***}` are defined as

.. math::
   :label: f_FG_and_f_star_functions

   f^{\rm{FG}}(\delta) &= (1 + \delta)^{5/3} + (1 - \delta)^{5/3} \\
   f^*(\delta) &= (\kappa_{{\rm sat}} + \kappa_{{\rm sym}}\delta)(1 + \delta)^{5/3} + (\kappa_{{\rm sat}} - \kappa_{{\rm sym}}\delta)(1 - \delta)^{5/3} \\
   f^{**}(\delta) &= (\kappa_{{\rm sat},2} + \kappa_{{\rm sym},2}\delta)(1 + \delta)^{5/3} + (\kappa_{{\rm sat},2} - \kappa_{{\rm sym},2}\delta)(1 - \delta)^{5/3} \\
   f^{***}(\delta) &= (\kappa_{{\rm sat},3} + \kappa_{{\rm sym},3}\delta)(1 + \delta)^{5/3} + (\kappa_{{\rm sat},3} - \kappa_{{\rm sym},3}\delta)(1 - \delta)^{5/3}

Therefore, the kinetic energy depends on 6 additional free parameters :math:`\kappa_i`, which control the effective mass and its density dependence.
In ``jester``, these parameters are set to zero by default, which means that the kinetic energy is modeled as a free Fermi gas without effective mass modifications, but the code does allow for these modifications to be included if desired by the user.

Potential energy
-----------------

The expansion shown in Eq. :eq:`energy_per_nucleon` is a rather simplistic model which has some shortcomings. 
Indeed, as explained in detail in Sec. III in Ref. :cite:`Margueron:2017eqc`, for some choices of the NEPs, the potential energy is not zero at zero density, which is unphysical.
This is to be expected, as the expansion in Eq. :eq:`energy_per_nucleon` is only valid around saturation density, and therefore does not necessarily incorporate the correct low-density behavior of the EOS.
To solve this issue, the potential energy functional is slightly adjusted.
The modification proposed by :cite:`Margueron:2017eqc` and called "ELFc" is to include an additional term in the potential energy functional to describe the low-density behavior.

In ``jester``, we therefore start from a different ansatz for the potential energy functional (which however is very similar to the ELFc potential ansatz), namely:

.. math::
   :label: e_pot_ansatz_guilherme

   e^{\rm{pot}}(x, \delta) = \sum_{\alpha = 0}^N \frac{v_{\alpha}(\delta)}{\alpha!} x^\alpha u_{\alpha}(x, \delta) \, .

In our case, we truncate the expansion at order :math:`N = 4`.
The coefficients of the expansion are defined as 

.. math::
   :label: v_alpha_coefficients

   v_{\alpha}(\delta) = v_{\alpha,{\rm sat}} + v_{\alpha,{\rm sym2}} \delta^2 + v_{\alpha, nq}(\delta)

which defines the coefficients :math:`v_{\alpha,{\rm sat}}` and :math:`v_{\alpha,{\rm sym2}}` as the isoscalar and isovector contributions to the potential energy expansion coefficients, respectively, and :math:`v_{\alpha, nq}(\delta)` as the non-quadratic contributions to the potential energy expansion coefficients.
The functions :math:`u_\alpha(x, \delta)` are defined as

.. math::
   :label: u_alpha_function

   u_\alpha(x, \delta) = 1 - (-3x)^{N + 1 - \alpha} e^{-b(\delta)(1 + 3x)}

where the parameter :math:`b(\delta)` controls the strength of the low-density modification.
This introduces an additional density-dependent term at lower densities to satisfy the zero-density limit, and drops to zero at higher densities, to conserve the behavior of the expansion introduced above at densities around and above saturation density.
In practice, we follow a slight modification to this ELFc expansion, following by Ref. :cite:`Somasundaram:2020chb` and Ref. :cite:`Grams:2021lzx`.
In Eq. :eq:`e_pot_ansatz_guilherme` and as defined in the original metamodel paper, a single parameter :math:`b` controls how fast the additional term drops to zero at higher densities.
However, in ``jester``, we follow the modifications proposed by Ref. :cite:`Somasundaram:2020chb` and instead define two parameters :math:`b_{\rm{sat}}` and :math:`b_{\rm{sym}}` controlling this low-density behavior in symmetric nuclear matter and pure neutron matter, respectively.
Their values for these parameters are given in Table III of Ref. :cite:`Somasundaram:2020chb`.
Following :cite:`Grams:2021lzx`, these parameters are then used to define an asymmetry-dependent parameter :math:`b(\delta)` as:

.. math::
   :label: b_function_of_delta

   b(\delta) = b_{\rm{sat}} + b_{\rm{sym}} \delta^2

Within this description, the mapping from the nuclear empirical parameters, which are the coefficients of the Taylor expansion originally provided in :eq:`metamodel_expansions`, to the coefficients :math:`v_{\alpha,{\rm sat}}` and :math:`v_{\alpha,{\rm sym2}}` is given by the expressions in Appendix B of Ref. :cite:`Somasundaram:2020chb`.
Explicitly, the isoscalar coefficients are:

.. math::
   :label: v_alpha_sat_coefficients

   v_{0,{\rm sat}} &= E_{\rm sat} - t_{\rm sat}(1 + \kappa_{\rm sat} + \kappa_{{\rm sat},2} + \kappa_{{\rm sat},3}) \\
   v_{1,{\rm sat}} &= -t_{\rm sat}(2 + 5\kappa_{\rm sat} + 8\kappa_{{\rm sat},2} + 11\kappa_{{\rm sat},3}) \\
   v_{2,{\rm sat}} &= K_{\rm sat} - 2t_{\rm sat}(-1 + 5\kappa_{\rm sat} + 20\kappa_{{\rm sat},2} + 44\kappa_{{\rm sat},3}) \\
   v_{3,{\rm sat}} &= Q_{\rm sat} - 2t_{\rm sat}(4 - 5\kappa_{\rm sat} + 40\kappa_{{\rm sat},2} + 220\kappa_{{\rm sat},3}) \\
   v_{4,{\rm sat}} &= Z_{\rm sat} - 8t_{\rm sat}(-7 + 5\kappa_{\rm sat} - 10\kappa_{{\rm sat},2} + 110\kappa_{{\rm sat},3})

and the isovector coefficients (combined with the non-quadratic contributions evaluated at :math:`\delta = 1`, i.e., in pure neutron matter) are:

.. math::
   :label: v_alpha_sym2_coefficients

   v_{0,{\rm sym2}} &= E_{\rm sym} - t_{\rm sat}\left[2^{2/3}(1 + \kappa_{\rm NM} + \kappa_{{\rm NM},2} + \kappa_{{\rm NM},3})\right. \\
                   &\qquad\qquad \left. - (1 + \kappa_{\rm sat} + \kappa_{{\rm sat},2} + \kappa_{{\rm sat},3})\right] - v_{0,nq}(\delta\!=\!1) \\
   v_{1,{\rm sym2}} &= L_{\rm sym} - t_{\rm sat}\left[2^{2/3}(2 + 5\kappa_{\rm NM} + 8\kappa_{{\rm NM},2} + 11\kappa_{{\rm NM},3})\right. \\
                   &\qquad\qquad \left. - (2 + 5\kappa_{\rm sat} + 8\kappa_{{\rm sat},2} + 11\kappa_{{\rm sat},3})\right] - v_{1,nq}(\delta\!=\!1) \\
   v_{2,{\rm sym2}} &= K_{\rm sym} - 2t_{\rm sat}\left[2^{2/3}(-1 + 5\kappa_{\rm NM} + 20\kappa_{{\rm NM},2} + 44\kappa_{{\rm NM},3})\right. \\
                   &\qquad\qquad \left. - (-1 + 5\kappa_{\rm sat} + 20\kappa_{{\rm sat},2} + 44\kappa_{{\rm sat},3})\right] - v_{2,nq}(\delta\!=\!1) \\
   v_{3,{\rm sym2}} &= Q_{\rm sym} - 2t_{\rm sat}\left[2^{2/3}(4 - 5\kappa_{\rm NM} + 40\kappa_{{\rm NM},2} + 220\kappa_{{\rm NM},3})\right. \\
                   &\qquad\qquad \left. - (4 - 5\kappa_{\rm sat} + 40\kappa_{{\rm sat},2} + 220\kappa_{{\rm sat},3})\right] - v_{3,nq}(\delta\!=\!1) \\
   v_{4,{\rm sym2}} &= Z_{\rm sym} - 8t_{\rm sat}\left[2^{2/3}(-7 + 5\kappa_{\rm NM} - 10\kappa_{{\rm NM},2} + 110\kappa_{{\rm NM},3})\right. \\
                   &\qquad\qquad \left. - (-7 + 5\kappa_{\rm sat} - 10\kappa_{{\rm sat},2} + 110\kappa_{{\rm sat},3})\right] - v_{4,nq}(\delta\!=\!1)

where :math:`\kappa_{\rm NM} = \kappa_{\rm sat} + \kappa_{\rm sym}`, :math:`\kappa_{{\rm NM},2} = \kappa_{{\rm sat},2} + \kappa_{{\rm sym},2}`, and :math:`\kappa_{{\rm NM},3} = \kappa_{{\rm sat},3} + \kappa_{{\rm sym},3}` are the effective mass parameters for pure neutron matter.
Note that this introduces the neutron matter (NM) constants

.. math::
   :label: kappa_NM_constants

   \kappa_{\rm NM} = \kappa_{\rm sat} + \kappa_{\rm sym} \, , \\
   \kappa_{{\rm NM},2} = \kappa_{{\rm sat},2} + \kappa_{{\rm sym},2} \, , \\
   \kappa_{{\rm NM},3} = \kappa_{{\rm sat},3} + \kappa_{{\rm sym},3}

Therefore, on top of the NEPs, this full metamodel requires a description of the effective mass and its density dependence, which is controlled by the 6 parameters :math:`\kappa_i` introduced already above.
On top of this, we require input for the :math:`b_{\rm{sat}}` and :math:`b_{\rm{sym}}` parameters controlling the low-density behavior of the potential energy functional. 

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
Note that here we have used that

.. math::
   :label: derivative_x_with_respect_to_n

   \frac{\partial x}{\partial n} = \frac{1}{3 n_{\rm{sat}}} \, ,

which is a useful relation to use in the calculations to follow.
By using the final form of Eq. :eq:`kinetic_energy` as a function of :math:`x`, one can find that

.. math::
   :label: pressure_kinetic_contribution

   P_{\rm{kin}}(x, \delta) = \frac{1}{3} n_{\rm{sat}} t_{\rm{sat}} (1 + 3x)^{5/3} \big( f^{\rm{FG}} &+ \frac{5}{2}(1 + 3x)f^* \\ 
    &+ 4(1 + 3x)^2 f^{**} + \frac{11}{2}(1 + 3x)^3 f^{***} \big)

which is easily shown by carrying out the derivative. 

For the contribution from the potential energy, we can first compute an intermediate result, namely the derivative of :math:`u_\alpha` with respect to :math:`x`, which we denote :math:`u'_\alpha`:

.. math::
   :label: du_alpha_dx

   u'_\alpha \equiv \frac{\partial u_\alpha(x, \delta)}{\partial x} = \left[ N + 1 - \alpha - 3 b(\delta) x \right] \left[ - (-3)^{N - \alpha} x^{N + 1 - \alpha} e^{-b(\delta)(1 + 3x)} \right]

From this, it is then easy to derive the useful shorthand

.. math::
   :label: x_times_du_alpha_dx

   x u'_\alpha \equiv x \frac{\partial u_\alpha(x, \delta)}{\partial x} = \left[ N + 1 - \alpha - 3 b(\delta) x \right] \left[ u_\alpha - 1 \right]

which will appear in the derivative and is therefore a useful quantity to precompute.
Having this at hand, the contribution to the pressure from the potential energy can be computed as

.. math::
   :label: pressure_potential_contribution

   P_{\rm{pot}}(x, \delta) = \frac{n_{\rm{sat}}}{3}(1+3x)^2 \sum_{\alpha = 0}^N \frac{v_\alpha(\delta)}{\alpha!} x^{\alpha - 1} \left[ \alpha \, u_\alpha \, \delta_{\alpha \ge 1} + x u'_\alpha \right] \, ,

With these equations at hand, the ``jester`` code for computing the pressure in the metamodel can be easily traced back.

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

For the metamodel, the sound speed in units of the speed of light :math:`c` is defined as

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

   K_{\rm{is,kin}}^{(1)}(x, \delta) = t_{\rm{sat}} (1 + 3x)^{2/3} \left[ -f^{\rm{FG}} + 5(1 + 3x)f^* + 20(1 + 3x)^2 f^{**} + 44(1 + 3x)^3 f^{***} \right]

The contribution from the potential energy to the first term is given by

.. math::
   :label: isoscalar_compressibility_potential_contribution

   K_{\rm{is,pot}}^{(1)}(x, \delta) = (1 + 3x)^2 \sum_{\alpha=0}^{N} \frac{v_{\alpha}(\delta)}{\alpha!} x^{\alpha-2}  \left[ \alpha(\alpha - 1)u_{\alpha}\delta_{\alpha \ge 2} + 2\alpha x u'_{\alpha} \delta_{\alpha \ge 1} + x^2 u''_{\alpha} \right]\, ,

where :math:`\delta_{\alpha\geq a}` restricts the sum to terms with :math:`\alpha \geq a`, and primes are shorthand for derivatives with respect to :math:`x`.
One can show the intermediate result that

.. math::
   :label: x_squared_times_second_derivative_u_alpha

   x^2 u''_{\alpha} = \left[ -(N + 1 - \alpha)(N - \alpha) + 6xb(\delta)(N + 1 - \alpha) - 9x^2b(\delta)^2 \right] (1 - u_{\alpha}) \, ,

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

For the proton fraction :math:`Y_p`, we can use the relation between :math:`\delta` and :math:`Y_p` shown above. 
Filling this in and setting this equal to the electron chemical potential defined above, we can find the following cubic equation for the proton fraction:

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

All of the physics described above has been implemented in ``jester``, but the following features are **not** incorporated in the metamodel or readily exposed to the user or the Bayesian inference workflow:

* The modifications due to the effective mass are ignored, such that :math:`m=m^*` in the kinetic energy expression. 
* Contributions from muons are neglected.
* The beta-equilibrium equation is not solved, so we use an approximate relation for the proton fraction calculation.
* The non-quadratic contributions to the potential energy. 
* In ``jester``, we take an average nucleonic mass :math:`m = (m_n + m_p)/2` for the nucleonic mass term, instead of separately accounting for the neutron and proton mass.

In particular, the metamodel code (in :class:`jesterTOV.eos.metamodel.MetaModel_EOS_model`) has the equations implemented for them, but the ``construct_eos`` method does not currently allow the user to easily use these modifications, outside of specifying these parameters in the ``__init__`` method of the class. These modifications are controlled by parameters called ``kappa_*`` for the kinetic energy and ``v_nq`` for the potential energy, but these are set to zero by default when initializing the metamodel class, and are not exposed to downstream functions. In particular, the Bayesian inference workflow for the moment does not recognize these parameters. Future releases of ``jester`` will likely include support for these features during sampling.



Further resources
-------------------

* API reference: :class:`jesterTOV.eos.metamodel.MetaModel_EOS_model`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
