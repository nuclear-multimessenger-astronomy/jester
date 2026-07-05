.. _eos-spectral:

Spectral decomposition
======================

This page describes the spectral EOS parametrization and its implementation in ``jester``.
The implementation follows :cite:`Lindblom:2010bb` and matches the LALSuite function
``XLALSimNeutronStarEOS4ParameterSpectralDecomposition`` as close as possible.

Physical motivation
-------------------

Any barotropic EOS is fully determined (up to a constant) by the adiabatic index

.. math::
   :label: gamma_def_spectral

   \Gamma(p) = \frac{\varepsilon + p}{p}\frac{dp}{d\varepsilon} \, ,

which must be positive for thermodynamic stability but is otherwise a fairly slowly varying
function for realistic equations of state.
Parametrizing :math:`\Gamma(p)` — rather than :math:`\varepsilon(p)` directly — has a key
advantage: a simple positivity constraint on :math:`\Gamma` is far easier to enforce than
the non-negativity and monotonicity conditions on :math:`\varepsilon(p)` itself.
The spectral approach of :cite:`Lindblom:2010bb` represents :math:`\Gamma(p)` using a
polynomial expansion in the dimensionless log-pressure, guaranteeing a physically valid EOS
for any choice of coefficients.

Parametrization
---------------

Introducing the dimensionless log-pressure

.. math::
   :label: x_spectral

   x = \log(p/p_0) \, ,

where :math:`p_0` is a reference pressure at the lower boundary of the spectral region
(matched to the crust, which is fixed to SLy by default in LALSuite), the adiabatic index is expanded as

.. math::
   :label: gamma_expansion

   \log \Gamma(x) = \gamma_0 + \gamma_1 x + \gamma_2 x^2 + \gamma_3 x^3 \, .

The exponential of this polynomial is always positive, ensuring thermodynamic stability by
construction.  The coefficient :math:`\gamma_0` sets the adiabatic index at the reference
pressure, :math:`\gamma_0 = \log \Gamma(p_0)`, while the higher-order coefficients control
how :math:`\Gamma` evolves across the spectral domain :math:`x \in [0, x_\mathrm{max}]`.

Given :math:`\Gamma(x)`, the energy density is obtained by integrating the first-order ODE
:math:`d\varepsilon/dp = (\varepsilon + p)/(p\Gamma)`.
Writing :math:`\mu(x) = \exp[-\int_0^x 1/\Gamma(x')\,dx']`, the solution is

.. math::
   :label: epsilon_spectral

   \varepsilon(x) = \frac{\varepsilon_0}{\mu(x)} + \frac{p_0}{\mu(x)} \int_0^x
   \frac{\mu(x')\,e^{x'}}{\Gamma(x')}\,dx' \, ,

where :math:`\varepsilon_0 = \varepsilon(p_0)` is fixed by matching to the crust at :math:`x = 0`.
Both integrals are evaluated numerically using 10-point Gauss-Legendre quadrature,
exactly as in LALSuite.

Physical validity constraints
------------------------------

Not every combination of :math:`(\gamma_0, \gamma_1, \gamma_2, \gamma_3)` produces a physically
useful EOS.  LALSuite requires :math:`\Gamma(x) \in [0.6, 4.5]` throughout the spectral domain;
values outside this range indicate an acausal or thermodynamically unstable EOS.
``jester`` tracks the number of grid points where this bound is violated in the
``extra_constraints`` field of the returned :class:`~jesterTOV.tov.data_classes.EOSData`,
and the :class:`~jesterTOV.inference.likelihoods.constraints.ConstraintGammaLikelihood`
applies a penalty to suppress such samples during inference.

The figure below shows :math:`\Gamma(x)` and the pressure-density relation for three
representative parameter sets; the grey band marks the valid adiabatic index range.
The label 'radio posterior mean' refers to the mean spectral coefficients from a radio-timing-only inference run, which are used as the center of the Gaussian reparametrization described in the next section below.

.. TODO: limit the bottom panel to xlim = left = min density at which we define the EOS generation

.. plot:: overview/eos/spectral_plot.py

   Top: adiabatic index :math:`\Gamma(x)` across the spectral domain for three representative
   parameter sets.  The grey band marks the valid range :math:`[0.6, 4.5]`.
   Bottom: the corresponding pressure-density relation above the crust.

Usage
-----

**Configuration file:**

.. code-block:: yaml

   eos:
     type: spectral
     crust_name: SLy
     n_points_high: 100

**Prior file:**

.. code-block:: python

   gamma_0 = UniformPrior(0.2, 2.0, parameter_names=["gamma_0"])
   gamma_1 = UniformPrior(-1.6, 1.7, parameter_names=["gamma_1"])
   gamma_2 = UniformPrior(-0.6, 0.6, parameter_names=["gamma_2"])
   gamma_3 = UniformPrior(-0.02, 0.02, parameter_names=["gamma_3"])

Complete examples are in ``examples/inference/spectral/``.

Reparametrization
-----------------

With a flat prior on :math:`(\gamma_0, \gamma_1, \gamma_2, \gamma_3)`, a large fraction of samples
produce numerically broken EOS: when :math:`\Gamma` drops below 1, the integrating factor
:math:`\mu(x)` collapses exponentially to zero, causing the energy density to diverge and the
subsequent TOV integration to return NaN.
This is not a bug — it reflects the fact that much of the flat prior volume corresponds to
unphysical high-density matter — but it makes inference inefficient.

A more efficient alternative is to draw the spectral coefficients from a prior that is already
concentrated on physically reasonable EOS.  In ``jester`` this is achieved with a
**Gaussian reparametrization** inspired by the posterior from a radio-timing-only inference run.
The idea is that the radio constraints (maximum observed pulsar mass :math:`\approx 2\,M_\odot`)
already rule out the softest EOS and loosely constrain the spectral coefficients to a
four-dimensional ellipsoid.
A multivariate Gaussian is fitted to this radio posterior by computing the weighted empirical mean
:math:`\boldsymbol{\mu}_\mathrm{radio}` and covariance :math:`\Sigma_\mathrm{radio}`, and then the
Cholesky factor :math:`L` of :math:`\Sigma_\mathrm{radio}` is widened by a scale factor
:math:`\sigma_\mathrm{scale} = 1.5` to remain broader than the radio posterior.
The reparametrized prior then places a standard normal over the whitened coordinates

.. math::
   :label: reparam_forward

   \boldsymbol{\gamma} = \boldsymbol{\mu}_\mathrm{radio} + L_\mathrm{wide}\,\tilde{\boldsymbol{\gamma}},
   \qquad \tilde{\boldsymbol{\gamma}} \sim \mathcal{N}(\mathbf{0}, I_4) \, ,

where :math:`L_\mathrm{wide} = \sigma_\mathrm{scale}\,L`.
The sampler works entirely in the :math:`\tilde{\boldsymbol{\gamma}}` space and the transform
maps back to physical spectral coefficients inside :meth:`~jesterTOV.eos.spectral.SpectralDecomposition_EOS_model.construct_eos`.

The numerical values of :math:`\boldsymbol{\mu}_\mathrm{radio}` and :math:`L` are
hardcoded in :class:`~jesterTOV.eos.spectral.SpectralDecomposition_EOS_model` and were derived
from a BlackJAX SMC run with radio constraints only using fiducial pulsar masses.
For reference, the hardcoded constants are (rounded to 4 decimal places):

.. math::

   \boldsymbol{\mu}_\mathrm{radio} =
   \begin{pmatrix}
     0.9223 \\ 0.3418 \\ -0.0831 \\ 0.0041
   \end{pmatrix}, \qquad
   L =
   \begin{pmatrix}
     0.3412 & 0.0000 & 0.0000 & 0.0000 \\
    -0.2120 & 0.1153 & 0.0000 & 0.0000 \\
     0.0319 & -0.0374 & 0.0083 & 0.0000 \\
    -0.0014 & 0.0024 & -0.0008 & 0.0002
   \end{pmatrix}.

To use the reparametrization, set ``reparametrized: true`` in the config and replace the
flat :math:`\gamma` priors with a standard ``MultivariateGaussianPrior``:

**Configuration file:**

.. code-block:: yaml

   eos:
     type: spectral
     crust_name: SLy
     reparametrized: true

**Prior file:**

.. code-block:: python

   spectral_reparam = MultivariateGaussianPrior(
       parameter_names=["gamma_0_tilde", "gamma_1_tilde", "gamma_2_tilde", "gamma_3_tilde"],
   )

A complete working example is in ``examples/inference/spectral_reparam/``.

Further resources
-----------------

* API reference: :class:`jesterTOV.eos.spectral.SpectralDecomposition_EOS_model`
* Prior class: :class:`jesterTOV.inference.base.prior.MultivariateGaussianPrior`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
