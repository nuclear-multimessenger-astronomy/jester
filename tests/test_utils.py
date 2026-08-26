"""Unit tests for utils module."""

import pytest
import jax.numpy as jnp
from hypothesis import given, strategies as st
from jesterTOV import utils


class TestConstants:
    """Test physical constants and unit conversions."""

    def test_physical_constants_values(self):
        """Test that physical constants have expected values."""
        assert abs(utils.c - 299792458.0) < 1e-6
        assert abs(utils.G - 6.6743e-11) < 1e-15
        assert abs(utils.Msun - 1.988409870698051e30) < 1e20
        assert abs(utils.hbarc - 197.3269804593025) < 1e-10
        assert abs(utils.m_p - 938.2720881604904) < 1e-10
        assert abs(utils.m_n - 939.5654205203889) < 1e-10

    def test_unit_conversions_consistency(self):
        """Test that unit conversions are self-consistent."""
        # Test fm to meter conversion
        assert abs(utils.fm_to_m * utils.m_to_fm - 1.0) < 1e-15

        # Test MeV to Joule conversion
        assert abs(utils.MeV_to_J * utils.J_to_MeV - 1.0) < 1e-15

        # Test pressure unit conversions
        assert abs(utils.MeV_fm_inv3_to_SI * utils.SI_to_MeV_fm_inv3 - 1.0) < 1e-10


class TestInterpolationFunctions:
    """Test interpolation utility functions."""

    def test_interp_in_logspace_basic(self):
        """Test basic functionality of log-space interpolation."""
        x = jnp.array([1.0, 10.0, 100.0])
        y = jnp.array([1.0, 100.0, 10000.0])  # y = x^2

        # Test interpolation at known points
        result = utils.interp_in_logspace(10.0, x, y)
        assert abs(result - 100.0) < 1e-10

        # Test interpolation at intermediate point
        result = utils.interp_in_logspace(jnp.sqrt(10.0), x, y)
        expected = 10.0  # sqrt(10)^2 = 10
        assert abs(result - expected) < 1e-6

    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_interp_in_logspace_monotonic(self, x_val):
        """Test that log-space interpolation preserves monotonicity."""
        x = jnp.array([0.1, 1.0, 10.0])
        y = jnp.array([0.01, 1.0, 100.0])  # Monotonically increasing

        result = utils.interp_in_logspace(x_val, x, y)
        assert result > 0  # Should be positive

    def test_cubic_spline(self):
        """Test cubic spline interpolation."""
        xp = jnp.linspace(0, 2 * jnp.pi, 10)
        fp = jnp.sin(xp)
        xq = jnp.linspace(0, 2 * jnp.pi, 25)

        result = utils.cubic_spline(xq, xp, fp)
        expected = jnp.sin(xq)

        # Should be close to true sine function
        assert jnp.mean(jnp.abs(result - expected)) < 0.1


class TestCumtrapz:
    """Test cumulative trapezoidal integration."""

    def test_cumtrapz_linear_function(self):
        """Test cumulative integration of linear function."""
        x = jnp.linspace(0, 1, 11)
        y = 2 * x  # Linear function: y = 2x

        result = utils.cumtrapz(y, x)

        # Analytical result for integral of 2x from 0 to x is x^2
        expected = x**2
        expected = expected.at[0].set(1e-30)  # First element set to small value

        # Check that results match (except first element which is set to 1e-30)
        assert jnp.allclose(result[1:], expected[1:], rtol=1e-10)

    def test_cumtrapz_constant_function(self):
        """Test cumulative integration of constant function."""
        x = jnp.linspace(0, 2, 21)
        y = jnp.ones_like(x) * 3  # Constant function: y = 3

        result = utils.cumtrapz(y, x)

        # Analytical result for integral of 3 from 0 to x is 3x
        expected = 3 * x
        expected = expected.at[0].set(1e-30)

        assert jnp.allclose(result[1:], expected[1:], rtol=1e-10)

    def test_cumtrapz_shape_validation(self):
        """Test that cumtrapz validates input shapes."""
        x = jnp.array([1, 2, 3])
        y = jnp.array([1, 2])  # Different length

        with pytest.raises(AssertionError):
            utils.cumtrapz(y, x)


class TestCubicRootForProtonFraction:
    """Tests for the NR A,B-only cubic root solver used in proton fraction.

    Covers the three physical regimes identified in the investigation:
      1. Esym > 0  → one real root in (0, 1), returned correctly.
      2. Esym < 0 (moderate, D < 0) → guard returns 0.
      3. Esym << 0 (large negative, D > 0) → guard returns 0, not a
         spurious root near y=1 (pre-existing bug in the Cardano formula).
    """

    def _make_coeffs(self, esym_mev: float, n_fm3: float):
        """Build [a, b, c, d] coefficients for the beta-equilibrium cubic."""
        a = jnp.array([8.0 * esym_mev])
        b = jnp.array([0.0])
        c = jnp.array([utils.hbarc * (3.0 * jnp.pi**2 * n_fm3) ** (1.0 / 3.0)])
        d = jnp.array([-4.0 * esym_mev - (utils.m_n - utils.m_p)])
        return jnp.stack([a, b, c, d], axis=1)  # shape [1, 4]

    def test_physical_root_back_substitution(self):
        """Root satisfies the polynomial to near machine precision when Esym > 0."""
        esym = 32.0  # MeV, typical physical value
        n = 0.32  # fm^-3, twice saturation density

        coeffs = self._make_coeffs(esym, n)
        y = utils.cubic_root_for_proton_fraction(coeffs)[0]

        a, _, c, d = coeffs[0]
        residual = float(jnp.abs(a * y**3 + c * y + d))
        assert residual < 1e-8, f"back-substitution residual {residual:.2e} too large"

    def test_physical_root_in_range(self):
        """Root y = xp^(1/3) is in (0, 1) when Esym > 0."""
        for esym in [28.0, 32.0, 45.0]:
            for n in [0.16, 0.32, 0.64, 1.0]:
                coeffs = self._make_coeffs(esym, n)
                y = float(utils.cubic_root_for_proton_fraction(coeffs)[0])
                assert 0.0 < y < 1.0, f"y={y} out of (0,1) for Esym={esym}, n={n}"

    def test_moderate_negative_esym_returns_zero(self):
        """Guard returns 0 for moderately negative Esym (unphysical regime)."""
        # Esym = -5 MeV: a < 0, guard fires immediately
        coeffs = self._make_coeffs(-5.0, 0.32)
        y = float(utils.cubic_root_for_proton_fraction(coeffs)[0])
        assert y == 0.0

    def test_large_negative_esym_returns_zero(self):
        """Guard returns 0 for large negative Esym (spurious-root regime).

        When |Esym| > c/4 the discriminant D > 0 and the old Cardano formula
        returned a spurious real root near y = 1 (xp ≈ 1).  The NR A,B-only
        formula with the a > 0 guard must return 0 instead.

        The reference case is from script 10: n ≈ 0.79 fm⁻³, Esym ≈ -140 MeV,
        where the old code gave xp ≈ 0.9995.
        """
        coeffs = self._make_coeffs(-140.0, 0.79)
        y = float(utils.cubic_root_for_proton_fraction(coeffs)[0])
        assert y == 0.0, (
            f"Expected y=0 for large-negative Esym but got y={y:.4f}; "
            "old Cardano formula returned a spurious root near y=1 here"
        )

    def test_zero_esym_returns_zero(self):
        """Guard returns 0 at the boundary Esym = 0."""
        coeffs = self._make_coeffs(0.0, 0.32)
        y = float(utils.cubic_root_for_proton_fraction(coeffs)[0])
        assert y == 0.0

    def test_vectorized_over_densities(self):
        """Vectorised call (multiple density points) has no NaN/Inf."""
        esym = 32.0
        n_arr = jnp.linspace(0.08, 0.8, 50)
        a = 8.0 * esym * jnp.ones_like(n_arr)
        b = jnp.zeros_like(n_arr)
        c = utils.hbarc * jnp.power(3.0 * jnp.pi**2 * n_arr, 1.0 / 3.0)
        d = (-4.0 * esym - (utils.m_n - utils.m_p)) * jnp.ones_like(n_arr)
        coeffs = jnp.stack([a, b, c, d], axis=1)
        ys = utils.cubic_root_for_proton_fraction(coeffs)
        assert jnp.all(jnp.isfinite(ys)), "NaN/Inf in vectorised output"  # type: ignore[arg-type]
        assert jnp.all(ys >= 0.0)  # type: ignore[operator]
        assert jnp.all(ys <= 1.0)  # type: ignore[operator]


class TestLimitByMTOV:
    """Test the limit_by_MTOV function."""

    def test_limit_by_mtov_basic(self):
        """Test basic functionality of MTOV limiting."""
        # Create sample data with maximum at index 5
        pc = jnp.linspace(1, 10, 10)
        m = jnp.array(
            [1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.1, 1.9, 1.7, 1.5]
        )  # Peak at index 5
        r = jnp.linspace(10, 8, 10)
        l = jnp.linspace(100, 50, 10)

        pc_new, m_new, r_new, l_new = utils.limit_by_MTOV(pc, m, r, l)

        # Check that arrays have same shape
        assert pc_new.shape == pc.shape
        assert m_new.shape == m.shape
        assert r_new.shape == r.shape
        assert l_new.shape == l.shape

        # Check that mass is non-decreasing up to maximum
        max_idx = jnp.argmax(m_new)
        assert jnp.all(jnp.diff(m_new[: max_idx + 1]) >= 0)

    def test_limit_by_mtov_sorting(self):
        """Test that output is sorted by mass."""
        pc = jnp.array([1, 2, 3, 4, 5])
        m = jnp.array([1.0, 2.0, 1.5, 1.8, 1.2])  # Not sorted
        r = jnp.array([10, 9, 8, 7, 6])
        l = jnp.array([100, 90, 80, 70, 60])

        pc_new, m_new, r_new, l_new = utils.limit_by_MTOV(pc, m, r, l)

        # Output should be sorted by mass
        assert jnp.all(jnp.diff(m_new) >= 0)


class TestCalculateRestMassDensity:
    """Test rest mass density calculation."""

    def test_calculate_rest_mass_density_monotonic(self):
        """Test that rest mass density calculation produces reasonable results."""
        e = jnp.linspace(100, 1000, 20)  # Energy density
        p = jnp.linspace(10, 300, 20)  # Pressure

        # Test the actual function to see what error occurs
        rho = utils.calculate_rest_mass_density(e, p)

        # Basic checks
        assert jnp.all(rho > 0)
        assert jnp.all(jnp.isfinite(rho))
        assert len(rho) == len(e)


class TestSigmoid:
    """Test sigmoid function."""

    def test_sigmoid_properties(self):
        """Test basic properties of sigmoid function."""
        x = jnp.linspace(-10, 10, 21)
        y = utils.sigmoid(x)

        # Should be between 0 and 1
        assert jnp.all(y >= 0)
        assert jnp.all(y <= 1)

        # Should be monotonically increasing
        assert jnp.all(jnp.diff(y) >= 0)

        # Should approach limits (be more lenient with numerical precision)
        assert utils.sigmoid(-10.0) < 1e-4
        assert abs(utils.sigmoid(100.0) - 1.0) < 1e-10  # Very close to 1
        assert abs(utils.sigmoid(0.0) - 0.5) < 1e-15


@pytest.mark.parametrize(
    "test_input,expected,tolerance",
    [
        (0.0, 0.5, 1e-15),
        (10.0, 1.0, 1e-4),  # Very close to 1
        (-10.0, 0.0, 1e-4),  # Very close to 0
    ],
)
def test_sigmoid_specific_values(test_input, expected, tolerance):
    """Test sigmoid function at specific values."""
    result = utils.sigmoid(test_input)
    assert abs(result - expected) < tolerance


def _reference_lambda_tilde(lambda_1, lambda_2, eta):
    """Plain-Python re-implementation of arXiv:1402.5156 eq. 5, independent of
    utils.lambda_tilde_from_lambda1_lambda2 (not imported from et-bgr-jester, which
    is not a jester dependency), used to cross-check that function."""
    import math

    eta2 = eta * eta
    seta = math.sqrt(1.0 - 4.0 * eta)
    return (8.0 / 13.0) * (
        (1.0 + 7.0 * eta - 31.0 * eta2) * (lambda_1 + lambda_2)
        + seta * (1.0 + 9.0 * eta - 11.0 * eta2) * (lambda_1 - lambda_2)
    )


class TestGravitationalWaveBinaryConversions:
    """Test chirp-mass/mass-ratio/Lambda_tilde helpers used by GWFisherLikelihood."""

    @pytest.mark.parametrize(
        "lambda_1,lambda_2,eta",
        [
            (300.0, 500.0, 0.2475),  # m1=1.5, m2=1.3
            (100.0, 100.0, 0.25),  # equal mass: eta -> 0.25, sqrt(1-4eta) -> 0
            (800.0, 50.0, 0.16),  # strongly asymmetric
        ],
    )
    def test_lambda_tilde_matches_reference_formula(self, lambda_1, lambda_2, eta):
        """utils.lambda_tilde_from_lambda1_lambda2 must match a hand-written
        re-implementation of arXiv:1402.5156 eq. 5."""
        result = utils.lambda_tilde_from_lambda1_lambda2(
            jnp.asarray(lambda_1), jnp.asarray(lambda_2), jnp.asarray(eta)
        )
        expected = _reference_lambda_tilde(lambda_1, lambda_2, eta)
        assert float(result) == pytest.approx(expected, rel=1e-10)

    def test_lambda_tilde_equal_mass_limit(self):
        """At eta=0.25 (equal mass), sqrt(1-4*eta) vanishes and Lambda_tilde reduces
        to a pure function of (Lambda_1 + Lambda_2)."""
        result = utils.lambda_tilde_from_lambda1_lambda2(
            jnp.asarray(300.0), jnp.asarray(300.0), jnp.asarray(0.25)
        )
        eta, eta2 = 0.25, 0.0625
        expected = (8.0 / 13.0) * (1.0 + 7.0 * eta - 31.0 * eta2) * (300.0 + 300.0)
        assert float(result) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("m1,m2", [(1.5, 1.3), (2.0, 1.1), (1.4, 1.4)])
    def test_component_masses_from_chirp_mass_and_mass_ratio_roundtrips(self, m1, m2):
        """Recovering (m1, m2) from their own (chirp_mass, mass_ratio) must return
        the original values, without depending on bilby (an optional dependency)."""
        chirp_mass = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        mass_ratio = m2 / m1

        recovered_m1, recovered_m2 = (
            utils.component_masses_from_chirp_mass_and_mass_ratio(
                jnp.asarray(chirp_mass), jnp.asarray(mass_ratio)
            )
        )

        assert float(recovered_m1) == pytest.approx(m1, rel=1e-10)
        assert float(recovered_m2) == pytest.approx(m2, rel=1e-10)

    def test_symmetric_mass_ratio_from_mass_ratio(self):
        """eta = q/(1+q)^2; eta(1) = 0.25 (equal mass, the maximum of eta)."""
        assert float(
            utils.symmetric_mass_ratio_from_mass_ratio(jnp.asarray(1.0))
        ) == pytest.approx(0.25)
        q = jnp.linspace(0.05, 1.0, 20)
        eta = utils.symmetric_mass_ratio_from_mass_ratio(q)
        # Monotonically increasing on (0, 1].
        assert jnp.all(jnp.diff(eta) > 0)
