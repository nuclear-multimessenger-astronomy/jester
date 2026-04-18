.. _tov-gr:

General Relativity TOV Solver
==============================

This page describes the standard Tolman-Oppenheimer-Volkoff (TOV) equations in general relativity and their numerical implementation in ``jester``.

Background
----------

The structure of a spherically symmetric, non-rotating neutron star in hydrostatic equilibrium is governed by the TOV equations, first derived by Tolman :cite:`Tolman:1939jz` and Oppenheimer & Volkoff :cite:`Oppenheimer:1939ne`.
In geometric units (:math:`G = c = 1`), the equations read

.. math::
   :label: tov_pressure

   \frac{dp}{dr} = -\frac{(\varepsilon + p)(m + 4\pi r^3 p)}{r(r - 2m)}

.. math::
   :label: tov_mass

   \frac{dm}{dr} = 4\pi r^2 \varepsilon

where :math:`r` is the circumferential radius, :math:`p(r)` is the pressure, :math:`\varepsilon(r)` is the energy density, and :math:`m(r)` is the gravitational mass enclosed within radius :math:`r`.
All calculations in ``jester`` are carried out in geometric units, and the results are converted to physical units (solar masses and kilometres) only at the final output stage.

Reformulation in terms of the specific enthalpy
-------------------------------------------------

In practice it is numerically advantageous to integrate the TOV equations using the specific enthalpy

.. math::
   :label: enthalpy_definition

   h = \int_0^p \frac{dp'}{\varepsilon(p') + p'}

as the independent variable rather than the radius :math:`r`.
The enthalpy decreases monotonically from its central value :math:`h_c` to zero at the stellar surface, which gives a natural stopping condition for the integration.
Inverting the relation :math:`dh/dr = -(1/(\varepsilon + p))(dp/dr)` and substituting Eq. :eq:`tov_pressure` yields

.. math::
   :label: dr_dh

   \frac{dr}{dh} = -\frac{r(r - 2m)}{m + 4\pi r^3 p}

and consequently

.. math::
   :label: dm_dh

   \frac{dm}{dh} = 4\pi r^2 \varepsilon \frac{dr}{dh}

These are the two equations integrated by the ``GRTOVSolver`` in ``jester``.
The pressure :math:`p` and energy density :math:`\varepsilon` at a given enthalpy value are recovered by interpolating the pre-computed EOS table (stored in ``EOSData``).

Tidal deformability
-------------------

In addition to the stellar structure, ``jester`` computes the tidal deformability :math:`\Lambda`, which is directly measurable from gravitational wave observations and is defined as

.. math::
   :label: tidal_deformability

   \Lambda = \frac{2}{3} k_2 C^{-5}

where :math:`C = M/R` is the compactness and :math:`k_2` is the second tidal Love number :cite:`Flanagan:2007ix,Hinderer:2007mb`.
To compute :math:`k_2`, a linear perturbation :math:`H(r)` to the metric is evolved alongside the stellar structure.
Expressed in terms of the enthalpy, the two auxiliary ODEs are

.. math::
   :label: dH_dh

   \frac{dH}{dh} = \beta \frac{dr}{dh}

.. math::
   :label: dbeta_dh

   \frac{d\beta}{dh} = -\left(C_0 H + C_1 \beta\right) \frac{dr}{dh}

where :math:`\beta \equiv dH/dr` and the coefficients :math:`C_0`, :math:`C_1` depend on the local stellar structure via

.. math::
   :label: C1_coeff

   C_1 = \frac{2}{r} + A\!\left(\frac{2m}{r^2} + 4\pi r (p - \varepsilon)\right)

.. math::
   :label: C0_coeff

   C_0 = A\!\left[-\frac{6}{r^2} + 4\pi\!\left(\varepsilon + p\right)\frac{d\varepsilon}{dp} + 4\pi\!\left(5\varepsilon + 9p\right)\right] - \left(\frac{2(m + 4\pi r^3 p)}{r(r - 2m)}\right)^2

with :math:`A = (1 - 2m/r)^{-1}` being the metric function.
At the stellar surface, the Love number :math:`k_2` is extracted from the ratio :math:`y = R\,\beta(R)/H(R)` through the closed-form expression (see :cite:`Hinderer:2007mb`)

.. math::
   :label: k2_formula

   k_2 = \frac{\tfrac{8}{5}C^5(1-2C)^2\bigl[2C(y-1) - y + 2\bigr]}
             {D(C, y)}

where the denominator :math:`D(C, y)` is

.. math::
   :label: k2_denominator

   \begin{aligned}
   D(C, y) &= 2C\!\left[4(y+1)C^4 + (6y-4)C^3 + (26-22y)C^2 + 3(5y-8)C - 3y + 6\right] \\
                     &\quad - 3(1-2C)^2\bigl[2C(y-1) - y + 2\bigr]\ln\!\frac{1}{1-2C}
   \end{aligned}

Numerical implementation
-------------------------

The four coupled ODEs — Eqs. :eq:`dr_dh`, :eq:`dm_dh`, :eq:`dH_dh`, :eq:`dbeta_dh` — are integrated from :math:`h_c` down to zero using the Dormand-Prince 5th-order Runge-Kutta method (Dopri5) provided by the `Diffrax <https://docs.kidger.site/diffrax/>`_ library, with an adaptive PID step-size controller (relative tolerance :math:`10^{-5}`, absolute tolerance :math:`10^{-6}`).

**Initial conditions.** The state vector :math:`(r, m, H, \beta)` is initialised just off-centre at :math:`h_0 = h_c - 10^{-3} h_c` using a series expansion valid near :math:`r = 0`:

.. math::
   :label: initial_r

   r_0 \approx \sqrt{\frac{3(-\Delta h)}{2\pi(\varepsilon_c + 3p_c)}}
              \left[1 - \frac{(\varepsilon_c - 3p_c - 0.6\,d\varepsilon/dh|_c)\,(-\Delta h)}
                            {4(\varepsilon_c + 3p_c)}\right]

.. math::
   :label: initial_m

   m_0 \approx \frac{4\pi \varepsilon_c r_0^3}{3}
              \left[1 - \frac{0.6\,(d\varepsilon/dh|_c)\,(-\Delta h)}{\varepsilon_c}\right]

.. math::
   :label: initial_Hb

   H_0 = r_0^2, \qquad \beta_0 = 2r_0

where :math:`\Delta h = h_0 - h_c = -10^{-3} h_c`.

**Enthalpy clamping.** Diffrax's adaptive step controller can evaluate the ODE at trial points where :math:`h \leq 0`, causing undefined logarithmic operations in the EOS interpolation routines. ``jester`` guards against this by clamping the enthalpy to the minimum tabulated value before any EOS lookup, which freezes the ODE derivatives at their surface value without affecting the converged solution.

**Family construction.** To build a full mass-radius-tidal family, the TOV equations are solved for a logarithmically spaced grid of central pressures :math:`p_c` running from a minimum density (set by ``min_nsat``, in units of saturation density :math:`n_\mathrm{sat} = 0.16\,\mathrm{fm}^{-3}`) up to the maximum pressure at which the EOS remains causal (:math:`c_s^2 < 1`). JAX's ``vmap`` is used to evaluate all central pressures in parallel, making the family construction JIT-compilable and fully differentiable. The stable branch of the :math:`M(R)` curve (up to the maximum mass :math:`M_\mathrm{TOV}`) is retained, and the family is re-sampled onto a uniform mass grid for downstream inference.

Further resources
-----------------

* API reference: :class:`jesterTOV.tov.gr.GRTOVSolver`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
