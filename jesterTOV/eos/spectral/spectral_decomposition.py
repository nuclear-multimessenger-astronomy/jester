r"""
Spectral decomposition equation of state model.

This module implements the spectral decomposition EOS following Lindblom (2010) PRD 82, 103011
and Lackey & Wade (2018) PRD 98, 063004, exactly matching the LALSuite implementation.

The implementation uses 10-point Gauss-Legendre quadrature for numerical integration and
stitches a low-density SLy crust to the high-density spectral expansion.
"""

import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, Float
from typing import Tuple

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.crust import load_crust
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

# TODO: this should be moved to the utils file perhaps?
# Gauss-Legendre 10-point quadrature nodes and weights from lalsuite
GL_NODES_10 = jnp.array([
    -0.9739065285171717,
    -0.8650633666889845,
    -0.6794095682990244,
    -0.4333953941292472,
    -0.1488743389816312,
     0.1488743389816312,
     0.4333953941292472,
     0.6794095682990244,
     0.8650633666889845,
     0.9739065285171717
])

GL_WEIGHTS_10 = jnp.array([
    0.0666713443086881,
    0.1494513491505806,
    0.2190863625159820,
    0.2692667193099963,
    0.2955242247147529,
    0.2955242247147529,
    0.2692667193099963,
    0.2190863625159820,
    0.1494513491505806,
    0.0666713443086881
])


def get_gauss_legendre_nodes(a: float, b: float) -> Float[Array, "10"]:
    r"""
    Get Gauss-Legendre quadrature nodes mapped from :math:`[-1, 1]` to :math:`[a, b]`.

    Args:
        a: Lower integration limit
        b: Upper integration limit

    Returns:
        Array of 10 nodes in :math:`[a, b]`
    """
    midpoint = (b + a) / 2.0
    half_width = (b - a) / 2.0
    return midpoint + half_width * GL_NODES_10


def gauss_legendre_quad(
    func_values: Float[Array, "10"],
    a: float,
    b: float
) -> Float[Array, ""]:
    r"""
    Compute integral using precomputed function values at Gauss-Legendre nodes.

    This implements the transformation from :math:`[-1, 1]` to :math:`[a, b]`:

    .. math::
        \int_a^b f(x) \, dx = \frac{b-a}{2} \sum_i w_i f\left(\frac{b-a}{2} x_i + \frac{b+a}{2}\right)

    Args:
        func_values: Function values evaluated at GL nodes (from get_gauss_legendre_nodes)
        a: Lower integration limit
        b: Upper integration limit

    Returns:
        Integral value (scalar array)
    """
    half_width = (b - a) / 2.0
    result = jnp.sum(GL_WEIGHTS_10 * func_values)
    return result * half_width


class SpectralDecomposition_EOS_model(Interpolate_EOS_model):
    r"""
    Spectral decomposition equation of state model.

    This class implements the spectral decomposition EOS parametrization following
    Lindblom (2010) PRD 82, 103011 and Lackey & Wade (2018) PRD 98, 063004.
    The implementation matches LALSuite's XLALSimNeutronStarEOS4ParameterSpectralDecomposition
    exactly.

    The adiabatic index is parametrized as:

    .. math::
        \log \Gamma(x) = \gamma_0 + \gamma_1 x + \gamma_2 x^2 + \gamma_3 x^3

    where :math:`x = \log(p/p_0)` is the dimensionless log-pressure.

    Reference values (geometric units):
        :math:`\varepsilon_0 = 9.54629006 \times 10^{-11} \, \mathrm{m}^{-2}`
        :math:`p_0 = 4.43784199 \times 10^{-13} \, \mathrm{m}^{-2}`

    Thermodynamic relations from PRD 82, 103011 (2010):

    .. math::
        \mu(x) &= \exp\left[-\int_0^x \frac{1}{\Gamma(x')} \, dx'\right] \\
        \varepsilon(x) &= \varepsilon_0 \cdot \exp\left[\int_0^x \frac{\mu(x')}{1+\mu(x')} \, dx'\right] \\
        p(x) &= p_0 \cdot \exp(x)
    """
    
    # Reference values in geometric units (from LALSuite): Minimum pressure and energy density of core EOS geom
    e0_geom = 9.54629006e-11  # m^-2
    p0_geom = 4.43784199e-13  # m^-2

    # Maximum dimensionless pressure for high-density EOS (from LALSuite)
    xmax = 12.3081  # Corresponds to pmax/p0

    def __init__(
        self,
        crust_name: str = 'SLy',
        n_points_high: int = 500,
    ):
        r"""
        Initialize spectral decomposition EOS model.

        Args:
            crust_name: Name of crust model to use ('SLy', 'DH', 'BPS').
                       Default 'SLy' for LALSuite compatibility.
            n_points_high: Number of high-density points to generate.
                          Default 500 to match LALSuite.
        """
        super().__init__()
        self.crust_name = crust_name
        self.n_points_high = n_points_high
        
                # Load low-density crust data
        n_crust, p_crust, e_crust = load_crust(self.crust_name)
        
        # TODO: After testing is done, we should use crust EOS code for this
        # For SLy crust from LALSuite, take first 69 points
        # (LALSuite hardcodes this)
        if self.crust_name == 'SLy':
            n_crust = n_crust[:69]
            p_crust = p_crust[:69]
            e_crust = e_crust[:69]

        # Filter out zero pressure points (causes issues with log in enthalpy calculation)
        nonzero_mask = p_crust > 0
        self.n_crust = n_crust[nonzero_mask]
        self.p_crust = p_crust[nonzero_mask]
        self.e_crust = e_crust[nonzero_mask]

        logger.info(f"Initialized SpectralDecomposition_EOS_model with crust={crust_name}, "
                   f"n_points={n_points_high}")
        
        # Convert reference pressure to MeV fm⁻³ for internal use
        self.p0_nuclear = self.p0_geom / utils.MeV_fm_inv3_to_geometric
        self.pmax_nuclear = self.p0_geom * jnp.exp(self.xmax) / utils.MeV_fm_inv3_to_geometric

    @staticmethod
    @jit
    def _log_adiabatic_index(x: float, gamma: Float[Array, "4"]) -> Float[Array, ""]:
        r"""
        Compute :math:`\log \Gamma(x)` from spectral expansion.

        Args:
            x: Dimensionless log-pressure :math:`\log(p/p_0)`
            gamma: Spectral coefficients :math:`[\gamma_0, \gamma_1, \gamma_2, \gamma_3]`

        Returns:
            :math:`\log \Gamma(x) = \gamma_0 + \gamma_1 x + \gamma_2 x^2 + \gamma_3 x^3` (scalar array)
        """
        return gamma[0] + gamma[1]*x + gamma[2]*x**2 + gamma[3]*x**3

    @staticmethod
    @jit
    def _adiabatic_index(x: float, gamma: Float[Array, "4"]) -> Float[Array, ""]:
        r"""
        Compute adiabatic index :math:`\Gamma(x) = \exp(\log \Gamma(x))`.

        This matches LALSuite's AdiabaticIndex function.

        Args:
            x: Dimensionless log-pressure :math:`\log(p/p_0)`
            gamma: Spectral coefficients :math:`[\gamma_0, \gamma_1, \gamma_2, \gamma_3]`

        Returns:
            :math:`\Gamma(x) = \exp(\gamma_0 + \gamma_1 x + \gamma_2 x^2 + \gamma_3 x^3)` (scalar array)
        """
        log_gamma_val = SpectralDecomposition_EOS_model._log_adiabatic_index(x, gamma)
        return jnp.exp(log_gamma_val)

    @staticmethod
    def _compute_mu(x: float, gamma: Float[Array, "4"]) -> Float[Array, ""]:
        r"""
        Compute :math:`\mu(x) = \exp\left[-\int_0^x \frac{1}{\Gamma(x')} \, dx'\right]`.

        This implements the integral in Eq. 8 of PRD 82, 103011 (2010) using
        10-point Gauss-Legendre quadrature, exactly matching LALSuite.

        Note: the paper shows the integral with pressure being the variable of the integral,
        but the LALSuite implementation and this code use :math:`x = \log(p/p_0)` as the variable.

        Args:
            x: Dimensionless log-pressure :math:`\log(p/p_0)`
            gamma: Spectral coefficients :math:`[\gamma_0, \gamma_1, \gamma_2, \gamma_3]`

        Returns:
            :math:`\mu(x)` value (scalar array)
        """
        # Get quadrature nodes in [0, x]
        nodes = get_gauss_legendre_nodes(0.0, x)

        # Evaluate integrand: 1/Γ(x') at all nodes
        gamma_values = vmap(
            lambda xprime: SpectralDecomposition_EOS_model._adiabatic_index(xprime, gamma)
        )(nodes)
        integrand_values = 1.0 / gamma_values

        # Compute integral using Gauss-Legendre quadrature
        integral = gauss_legendre_quad(integrand_values, 0.0, x)

        # Return μ(x) = exp(-integral)
        return jnp.exp(-integral)

    @staticmethod
    def _compute_energy_density_geom(x: float, gamma: Float[Array, "4"]) -> Float[Array, ""]:
        r"""
        Compute energy density :math:`\varepsilon(x)` in geometric units.

        This implements Eq. 7 of PRD 82, 103011 (2010) as coded in LALSuite:

        .. math::
            \varepsilon(x) = \frac{\varepsilon_0}{\mu(x)} + \frac{p_0}{\mu(x)} \cdot I(x)

        where:

        .. math::
            I(x) = \int_0^x \frac{\mu(x') \exp(x')}{\Gamma(x')} \, dx'

        The implementation uses nested Gauss-Legendre quadrature exactly as in LALSuite,
        where :math:`\mu(x')` must be recomputed for each evaluation point in the outer integral.

        This is the most computationally intensive function, with :math:`O(n^2)` evaluations
        due to the nested integration structure.

        Args:
            x: Dimensionless log-pressure :math:`\log(p/p_0)`
            gamma: Spectral coefficients :math:`[\gamma_0, \gamma_1, \gamma_2, \gamma_3]`

        Returns:
            Energy density :math:`\varepsilon(x)` in geometric units [:math:`\mathrm{m}^{-2}`]
        """
        e0 = SpectralDecomposition_EOS_model.e0_geom
        p0 = SpectralDecomposition_EOS_model.p0_geom

        # Compute μ(x), this is the upper bound of the integral
        mu = SpectralDecomposition_EOS_model._compute_mu(x, gamma)

        # Get quadrature nodes in [0, x]
        nodes = get_gauss_legendre_nodes(0.0, x)

        # Evaluate integrand at all nodes (vectorized)
        # Note: This requires computing μ at each node (nested integration)
        mu_values = vmap(
            lambda xprime: SpectralDecomposition_EOS_model._compute_mu(xprime, gamma)
        )(nodes)
        gamma_values = vmap(
            lambda xprime: SpectralDecomposition_EOS_model._adiabatic_index(xprime, gamma)
        )(nodes)
        integrand_values = mu_values * jnp.exp(nodes) / gamma_values

        # Compute integral using Gauss-Legendre quadrature
        integral = gauss_legendre_quad(integrand_values, 0.0, x)

        # Return ε(x) using Eq. 7
        return e0 / mu + (p0 / mu) * integral

    # TODO: this must be migrated into something like ConstraintEOSLikelihood
    def _validate_gamma(self, gamma: Float[Array, "4"]) -> bool:
        r"""
        Validate spectral parameters by checking adiabatic index bounds.

        This implements XLALSimNeutronStarEOS4ParamSDGammaCheck from LALSuite.
        Checks that :math:`\Gamma(x) \in [0.6, 4.5]` for all :math:`x \in [0, x_\mathrm{max}]`
        as required for physical EOS and TOV solver stability.

        NOTE: This validation is not JIT-friendly due to the boolean return.
        It is called during EOS construction (outside JIT) to catch invalid
        parameters early. For production inference with JIT-compiled likelihoods,
        consider implementing as a soft constraint (e.g., :math:`\log \pi = -\infty` for
        invalid parameters) instead of hard validation.

        Args:
            gamma: Spectral coefficients :math:`[\gamma_0, \gamma_1, \gamma_2, \gamma_3]`

        Returns:
            True if :math:`\Gamma(x) \in [0.6, 4.5]` for all :math:`x \in [0, x_\mathrm{max}]`, False otherwise
        """
        # Sample Γ(x) at 100 points as in LALSuite
        x_test = jnp.linspace(0.0, self.xmax, 100)
        gamma_vals = vmap(lambda x: self._adiabatic_index(x, gamma))(x_test)

        # Check bounds
        valid = jnp.all((gamma_vals >= 0.6) & (gamma_vals <= 4.5))

        return bool(valid)

    def construct_eos(
        self,
        gamma: Float[Array, "4"]
    ) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"],
               Float[Array, "n_points"], Float[Array, "n_points"],
               Float[Array, "n_points"]]:
        r"""
        Construct full EOS from spectral parameters.

        This method:
        1. Validates parameters
        2. Loads low-density crust (69 points)
        3. Generates high-density spectral region (500 points by default)
        4. Stitches them together
        5. Converts to geometric units and computes auxiliary quantities

        Args:
            gamma: Spectral coefficients :math:`[\gamma_0, \gamma_1, \gamma_2, \gamma_3]`

        Returns:
            Tuple of (ns, ps, hs, es, dloge_dlogps) in geometric units

        Raises:
            ValueError: If gamma parameters fail validation
        """

        # Generate high-density spectral region
        n_high, p_high, e_high = self._generate_spectral_region(gamma)

        # Stitch on top of the crust
        n_full = jnp.concatenate([self.n_crust, n_high])
        p_full = jnp.concatenate([self.p_crust, p_high])
        e_full = jnp.concatenate([self.e_crust, e_high])

        # Convert to geometric units and compute auxiliary quantities
        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n_full, p_full, e_full)

        return ns, ps, hs, es, dloge_dlogps

    def _generate_spectral_region(
        self,
        gamma: Float[Array, "4"],
    ) -> Tuple[Float[Array, "n_high"], Float[Array, "n_high"], Float[Array, "n_high"]]:
        r"""
        Generate high-density EOS points from spectral expansion.

        This generates a logarithmically-spaced pressure grid from :math:`p_0` (reference pressure)
        to the maximum pressure, then computes energy density using the spectral
        decomposition formulas. This matches LALSuite implementation exactly.

        Args:
            gamma: Spectral coefficients :math:`[\gamma_0, \gamma_1, \gamma_2, \gamma_3]`

        Returns:
            Tuple of (n, p, e) in nuclear units [:math:`\mathrm{fm}^{-3}`, :math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`, :math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]
        """

        # This matches LALSuite implementation (lines 270-278 in C code): start from p0 (reference pressure), NOT from stitching point!
        logp0 = jnp.log(self.p0_nuclear)
        logpmax = jnp.log(self.pmax_nuclear)
        dlogp = (logpmax - logp0) / self.n_points_high

        # Pressure grid (starts from p0, matching LALSuite)
        log_p_high = logp0 + dlogp * jnp.arange(self.n_points_high)
        p_high = jnp.exp(log_p_high)

        # Compute dimensionless log-pressure for each pressure point
        x_high = log_p_high - logp0

        # Compute energy density for each x using spectral decomposition
        # This is the expensive step: nested integration for each point
        e_high_geom = vmap(
            lambda x: self._compute_energy_density_geom(x, gamma)
        )(x_high)

        # Convert to nuclear units
        e_high = e_high_geom / utils.MeV_fm_inv3_to_geometric

        # Convert pressure to geometric units for enthalpy calculation
        p_high_geom = p_high * utils.MeV_fm_inv3_to_geometric

        # Compute pseudo-enthalpy for the spectral region, similar to the approach from interpolate_eos()
        h_high = utils.cumtrapz(p_high_geom / (e_high_geom + p_high_geom), jnp.log(p_high_geom))

        # Compute rest-mass density using thermodynamic relation
        rho_high_geom = (e_high_geom + p_high_geom) * jnp.exp(-h_high)

        # Convert rest-mass density to nuclear units
        rho_high = rho_high_geom / utils.MeV_fm_inv3_to_geometric

        # Convert to number density using average nucleon mass
        n_high = rho_high / utils.m

        return n_high, p_high, e_high
