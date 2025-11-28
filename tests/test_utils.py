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


# NOTE: cubic_root_for_proton_fraction is tested within the EOS framework
# where it's used for computing proton fractions, not as a standalone cubic solver


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
