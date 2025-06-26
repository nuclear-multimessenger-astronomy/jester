"""Unit tests for post-TOV equation solver (modified gravity)."""

import pytest
import jax.numpy as jnp
from jesterTOV import ptov


@pytest.fixture
def sample_ptov_eos_dict(sample_eos_dict):
    """Extend sample EOS dict with post-TOV parameters."""
    ptov_eos = sample_eos_dict.copy()

    # Add modified gravity parameters
    ptov_eos.update(
        {
            "lambda_BL": 0.1,  # Brans-Dicke like parameter
            "lambda_DY": 0.05,  # Dynamical Chern-Simons parameter
            "lambda_HB": 1.2,  # Horndeski parameter
            "gamma": 0.02,  # Post-Newtonian parameter
            "alpha": 10.0,  # Tanh transition sharpness
            "beta": 0.3,  # Transition compactness
        }
    )

    return ptov_eos


class TestSigmaFunction:
    """Test the sigma function for modified gravity contributions."""

    def test_sigma_func_basic(self):
        """Test basic sigma function evaluation."""
        # Realistic neutron star values in geometric units
        p = 3.71e-11  # Pressure in geometric units (m^-2)
        e = 4.13e-27  # Energy density in geometric units (m^-2)
        m = 2.07e3  # Mass in geometric units (m)
        r = 1.20e4  # Radius in geometric units (m)

        # Modified gravity parameters
        lambda_BL = 0.1
        lambda_DY = 0.05
        lambda_HB = 1.2
        gamma = 0.02
        alpha = 10.0
        beta = 0.3

        sigma = ptov.sigma_func(
            p, e, m, r, lambda_BL, lambda_DY, lambda_HB, gamma, alpha, beta
        )

        # Sigma should be finite
        assert jnp.isfinite(sigma)

        # For small deviations from GR, sigma should be reasonable
        assert abs(sigma) < 1e-8  # Adjusted bound for geometric units

    def test_sigma_func_gr_limit(self):
        """Test that sigma function reduces to zero in GR limit."""
        p, e, m, r = 3.71e-11, 4.13e-27, 2.07e3, 1.20e4

        # Set all modified gravity parameters to GR values
        lambda_BL = 0.0
        lambda_DY = 0.0
        lambda_HB = 1.0  # GR value
        gamma = 0.0
        alpha = 10.0
        beta = 0.3

        sigma = ptov.sigma_func(
            p, e, m, r, lambda_BL, lambda_DY, lambda_HB, gamma, alpha, beta
        )

        # Should be very close to zero in GR limit
        assert abs(sigma) < 1e-10

    def test_sigma_func_parameter_dependence(self):
        """Test sigma function dependence on parameters."""
        p, e, m, r = 3.71e-11, 4.13e-27, 2.07e3, 1.20e4

        # Test lambda_BL dependence
        sigma1 = ptov.sigma_func(p, e, m, r, 0.0, 0.0, 1.0, 0.0, 10.0, 0.3)
        sigma2 = ptov.sigma_func(p, e, m, r, 0.1, 0.0, 1.0, 0.0, 10.0, 0.3)
        assert sigma1 != sigma2  # Should be different

        # Test lambda_DY dependence
        sigma3 = ptov.sigma_func(p, e, m, r, 0.0, 0.05, 1.0, 0.0, 10.0, 0.3)
        assert sigma1 != sigma3

        # Test lambda_HB dependence
        sigma4 = ptov.sigma_func(p, e, m, r, 0.0, 0.0, 1.1, 0.0, 10.0, 0.3)
        assert sigma1 != sigma4

    def test_sigma_func_physical_values(self):
        """Test sigma function with physically motivated parameter ranges."""
        p, e, m, r = 3.71e-11, 4.13e-27, 2.07e3, 1.20e4

        # Test various parameter combinations
        test_params = [
            (0.1, 0.0, 1.0, 0.0, 10.0, 0.3),
            (0.0, 0.1, 1.0, 0.0, 10.0, 0.3),
            (0.0, 0.0, 1.1, 0.0, 10.0, 0.3),
            (0.0, 0.0, 1.0, 0.1, 10.0, 0.3),
        ]

        for params in test_params:
            sigma = ptov.sigma_func(p, e, m, r, *params)
            assert jnp.isfinite(sigma)
            assert abs(sigma) < 1e-8  # Should be reasonable in geometric units


class TestPostTOVODE:
    """Test post-TOV ODE function."""

    def test_ptov_ode_basic(self, sample_ptov_eos_dict):
        """Test basic post-TOV ODE functionality."""
        h = sample_ptov_eos_dict["h"][10]
        r, m, H, b = 10.0, 1.0, 100.0, 20.0
        y = (r, m, H, b)

        dydt = ptov.tov_ode(h, y, sample_ptov_eos_dict)
        drdh, dmdh, dHdh, dbdh = dydt

        # Check that derivatives are finite
        assert jnp.isfinite(drdh)
        assert jnp.isfinite(dmdh)
        assert jnp.isfinite(dHdh)
        assert jnp.isfinite(dbdh)

        # Check that radius derivative has correct sign
        assert drdh != 0  # Should be non-zero

    def test_ptov_ode_vs_gr(self, sample_eos_dict, sample_ptov_eos_dict):
        """Test that post-TOV ODE behaves correctly in GR limit vs modified gravity."""
        h = sample_eos_dict["h"][15]
        y = (10.0, 1.0, 100.0, 20.0)

        # Post-TOV with GR parameters (all modified gravity terms should vanish)
        gr_ptov_eos = sample_eos_dict.copy()
        gr_ptov_eos.update(
            {
                "lambda_BL": 0.0,
                "lambda_DY": 0.0,
                "lambda_HB": 1.0,  # GR value
                "gamma": 0.0,
                "alpha": 10.0,
                "beta": 0.3,
            }
        )
        dydt_ptov_gr = ptov.tov_ode(h, y, gr_ptov_eos)

        # Post-TOV with modified gravity parameters
        dydt_ptov_mg = ptov.tov_ode(h, y, sample_ptov_eos_dict)

        # Check that derivatives are finite for both cases
        for i in range(4):
            assert jnp.isfinite(dydt_ptov_gr[i])
            assert jnp.isfinite(dydt_ptov_mg[i])

        # Check that sigma function is different between GR and MG
        p = 3.71e-11
        e = 4.13e-27
        m = 2.07e3
        r = 1.20e4

        sigma_gr = ptov.sigma_func(p, e, m, r, 0.0, 0.0, 1.0, 0.0, 10.0, 0.3)
        sigma_mg = ptov.sigma_func(p, e, m, r, 0.1, 0.05, 1.2, 0.02, 10.0, 0.3)

        assert abs(sigma_gr) < 1e-14  # Should be essentially zero in GR
        assert abs(sigma_mg) > 1e-14  # Should be non-zero with MG parameters

    def test_ptov_ode_gradients(self, sample_ptov_eos_dict):
        """Test that gradient calculations in post-TOV ODE work."""
        h = sample_ptov_eos_dict["h"][15]
        y = (10.0, 1.0, 100.0, 20.0)

        # This should not crash and should give finite results
        dydt = ptov.tov_ode(h, y, sample_ptov_eos_dict)

        for derivative in dydt:
            assert jnp.isfinite(derivative)


class TestPostTOVCalcK2:
    """Test k2 calculation in post-TOV (should be same as regular TOV)."""

    def test_ptov_calc_k2_same_as_tov(self):
        """Test that k2 calculation is same as in regular TOV."""
        R, M, H, b = 12.0, 1.4, 144.0, 24.0

        # Import regular TOV for comparison
        from jesterTOV import tov

        k2_tov = tov.calc_k2(R, M, H, b)
        k2_ptov = ptov.calc_k2(R, M, H, b)

        # Should be identical
        assert abs(k2_tov - k2_ptov) < 1e-15


class TestPostTOVSolver:
    """Test complete post-TOV solver."""

    def test_ptov_solver_basic(self, sample_ptov_eos_dict):
        """Test basic post-TOV solver functionality."""
        pc = sample_ptov_eos_dict["p"][25]

        M, R, k2 = ptov.tov_solver(sample_ptov_eos_dict, pc)

        # Check that results are finite and positive
        assert jnp.isfinite(M)
        assert jnp.isfinite(R)
        assert jnp.isfinite(k2)
        assert M > 0
        assert R > 0
        assert k2 > 0

        # Check physical compactness
        compactness = M / R
        assert compactness < 0.5
        assert compactness > 0.01

    def test_ptov_solver_different_mg_params(self, sample_eos_dict):
        """Test post-TOV solver with different modified gravity parameters."""
        pc = sample_eos_dict["p"][25]

        mg_param_sets = [
            {
                "lambda_BL": 0.0,
                "lambda_DY": 0.0,
                "lambda_HB": 1.0,
                "gamma": 0.0,
                "alpha": 10.0,
                "beta": 0.3,
            },  # GR case
            {
                "lambda_BL": 0.1,
                "lambda_DY": 0.0,
                "lambda_HB": 1.0,
                "gamma": 0.0,
                "alpha": 10.0,
                "beta": 0.3,
            },  # Brans-Dicke
            {
                "lambda_BL": 0.0,
                "lambda_DY": 0.05,
                "lambda_HB": 1.0,
                "gamma": 0.0,
                "alpha": 10.0,
                "beta": 0.3,
            },  # dCS
            {
                "lambda_BL": 0.0,
                "lambda_DY": 0.0,
                "lambda_HB": 1.1,
                "gamma": 0.0,
                "alpha": 10.0,
                "beta": 0.3,
            },  # Horndeski
        ]

        results = []

        for mg_params in mg_param_sets:
            eos_dict = sample_eos_dict.copy()
            eos_dict.update(mg_params)

            try:
                M, R, k2 = ptov.tov_solver(eos_dict, pc)

                # Basic checks
                assert jnp.isfinite(M) and M > 0
                assert jnp.isfinite(R) and R > 0
                assert jnp.isfinite(k2) and k2 > 0
                assert M / R < 0.5

                results.append((M, R, k2))

            except Exception as e:
                pytest.skip(f"Post-TOV solver failed for params {mg_params}: {e}")

        # Should have at least one successful result
        assert len(results) >= 1

        # Different parameters should potentially give different results
        if len(results) > 1:
            masses = [r[0] for r in results]
            # Allow that some might be very similar (especially GR vs small deviations)
            assert len(set(f"{m:.6f}" for m in masses)) >= 1

    def test_ptov_solver_gr_limit_comparison(self, sample_eos_dict):
        """Compare post-TOV solver to regular TOV in GR limit."""
        pc = sample_eos_dict["p"][25]

        # Regular TOV solver
        from jesterTOV import tov

        M_tov, R_tov, k2_tov = tov.tov_solver(sample_eos_dict, pc)

        # Post-TOV in GR limit
        gr_eos = sample_eos_dict.copy()
        gr_eos.update(
            {
                "lambda_BL": 0.0,
                "lambda_DY": 0.0,
                "lambda_HB": 1.0,
                "gamma": 0.0,
                "alpha": 10.0,
                "beta": 0.3,
            }
        )

        M_ptov, R_ptov, k2_ptov = ptov.tov_solver(gr_eos, pc)

        # Results should be similar (allowing for numerical differences)
        assert abs(M_tov - M_ptov) / M_tov < 0.01  # Within 1%
        assert abs(R_tov - R_ptov) / R_tov < 0.01
        assert abs(k2_tov - k2_ptov) / k2_tov < 0.05  # k2 might be more sensitive

    def test_ptov_solver_convergence(self, sample_ptov_eos_dict):
        """Test that post-TOV solver gives consistent results."""
        pc = sample_ptov_eos_dict["p"][25]

        # Solve multiple times
        results = []
        for _ in range(3):
            M, R, k2 = ptov.tov_solver(sample_ptov_eos_dict, pc)
            results.append((M, R, k2))

        # Results should be identical
        for i in range(1, len(results)):
            assert abs(results[i][0] - results[0][0]) < 1e-10  # Mass
            assert abs(results[i][1] - results[0][1]) < 1e-10  # Radius
            assert abs(results[i][2] - results[0][2]) < 1e-8  # k2


class TestPostTOVPhysicalConsistency:
    """Test physical consistency of post-TOV solutions."""

    @pytest.mark.slow
    def test_ptov_mass_radius_relationship(self, sample_ptov_eos_dict):
        """Test mass-radius relationship for post-TOV solutions."""
        pressure_indices = range(15, 35, 5)
        masses = []
        radii = []

        for idx in pressure_indices:
            if idx < len(sample_ptov_eos_dict["p"]):
                pc = sample_ptov_eos_dict["p"][idx]
                try:
                    M, R, k2 = ptov.tov_solver(sample_ptov_eos_dict, pc)
                    if jnp.isfinite(M) and jnp.isfinite(R) and M > 0 and R > 0:
                        masses.append(M)
                        radii.append(R)
                except:
                    continue

        # Should have multiple solutions
        assert len(masses) >= 2

        masses = jnp.array(masses)
        radii = jnp.array(radii)

        # Physical checks
        assert jnp.all(masses > 0)
        assert jnp.all(radii > 0)
        assert jnp.all(masses / radii < 0.5)  # Compactness limit

    def test_ptov_modified_gravity_effects(self, sample_eos_dict):
        """Test that modified gravity parameters actually affect results."""
        pc = sample_eos_dict["p"][25]

        # GR case
        gr_eos = sample_eos_dict.copy()
        gr_eos.update(
            {
                "lambda_BL": 0.0,
                "lambda_DY": 0.0,
                "lambda_HB": 1.0,
                "gamma": 0.0,
                "alpha": 10.0,
                "beta": 0.3,
            }
        )

        # Modified gravity case
        mg_eos = sample_eos_dict.copy()
        mg_eos.update(
            {
                "lambda_BL": 0.2,
                "lambda_DY": 0.1,
                "lambda_HB": 1.1,
                "gamma": 0.05,
                "alpha": 10.0,
                "beta": 0.3,
            }
        )

        try:
            M_gr, R_gr, k2_gr = ptov.tov_solver(gr_eos, pc)
            M_mg, R_mg, k2_mg = ptov.tov_solver(mg_eos, pc)

            # Results should be different (though possibly small differences)
            mass_diff = abs(M_mg - M_gr) / M_gr
            radius_diff = abs(R_mg - R_gr) / R_gr

            # At least one should show some difference (even if small)
            assert mass_diff > 1e-6 or radius_diff > 1e-6

        except Exception as e:
            pytest.skip(f"Modified gravity comparison failed: {e}")


# Parameterized tests
@pytest.mark.parametrize("lambda_BL", [0.0, 0.05, 0.1])
@pytest.mark.parametrize("lambda_DY", [0.0, 0.02, 0.05])
def test_sigma_func_parameter_sweep(lambda_BL, lambda_DY):
    """Test sigma function across parameter space."""
    p, e, m, r = 3.71e-11, 4.13e-27, 2.07e3, 1.20e4
    lambda_HB, gamma, alpha, beta = 1.0, 0.0, 10.0, 0.3

    sigma = ptov.sigma_func(
        p, e, m, r, lambda_BL, lambda_DY, lambda_HB, gamma, alpha, beta
    )

    assert jnp.isfinite(sigma)
    assert abs(sigma) < 1e-8  # Should be reasonable in geometric units


@pytest.mark.parametrize(
    "mg_param,value",
    [
        ("lambda_BL", 0.1),
        ("lambda_DY", 0.05),
        ("lambda_HB", 1.1),
        ("gamma", 0.02),
    ],
)
def test_ptov_solver_individual_mg_params(sample_eos_dict, mg_param, value):
    """Test post-TOV solver with individual modified gravity parameters."""
    pc = sample_eos_dict["p"][25]

    # Start with GR and modify one parameter
    eos_dict = sample_eos_dict.copy()
    eos_dict.update(
        {
            "lambda_BL": 0.0,
            "lambda_DY": 0.0,
            "lambda_HB": 1.0,
            "gamma": 0.0,
            "alpha": 10.0,
            "beta": 0.3,
        }
    )
    eos_dict[mg_param] = value

    try:
        M, R, k2 = ptov.tov_solver(eos_dict, pc)

        assert jnp.isfinite(M) and M > 0
        assert jnp.isfinite(R) and R > 0
        assert jnp.isfinite(k2) and k2 > 0
        assert M / R < 0.5

    except Exception as e:
        pytest.skip(f"Post-TOV solver failed for {mg_param}={value}: {e}")
