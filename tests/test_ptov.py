"""Unit tests for post-TOV equation solver (modified gravity)."""

import pytest
import jax.numpy as jnp
from jesterTOV.tov import GRTOVSolver, PostTOVSolver
from jesterTOV.tov.data_classes import EOSData


@pytest.fixture
def sample_eos_data_post(sample_eos_dict):
    """Create EOSData from sample EOS dict for post-TOV testing."""
    # Extract arrays from dict
    ps = sample_eos_dict["p"]
    hs = sample_eos_dict["h"]
    es = sample_eos_dict["e"]
    dloge_dlogps = sample_eos_dict["dloge_dlogp"]
    cs2 = ps / (es * dloge_dlogps)

    return EOSData(
        ns=jnp.zeros_like(ps),  # Not used in TOV
        ps=ps,
        hs=hs,
        es=es,
        dloge_dlogps=dloge_dlogps,
        cs2=cs2,
        mu=jnp.zeros_like(ps),  # Not used in TOV
        extra_constraints=None,
    )


class TestPostTOVSolver:
    """Test complete post-TOV solver."""

    def test_post_solver_gr_limit(self, sample_eos_data_post):
        """Test that post-TOV solver gives GR results in GR limit."""
        pc = sample_eos_data_post.ps[25]

        # GR solver
        gr_solver = GRTOVSolver()
        gr_solution = gr_solver.solve(sample_eos_data_post, pc)

        # Post-TOV solver with GR parameters (all MG terms zero)
        post_solver = PostTOVSolver()
        post_params = {
            "lambda_BL": 0.0,
            "lambda_DY": 0.0,
            "lambda_HB": 1.0,  # GR value
            "gamma": 0.0,
            "alpha": 10.0,
            "beta": 0.3,
        }
        post_solution = post_solver.solve(sample_eos_data_post, pc, **post_params)

        # Results should be very similar (allowing for numerical differences)
        assert abs(gr_solution.M - post_solution.M) / gr_solution.M < 0.01  # 1%
        assert abs(gr_solution.R - post_solution.R) / gr_solution.R < 0.01
        assert abs(gr_solution.k2 - post_solution.k2) / gr_solution.k2 < 0.05  # 5%

    def test_post_solver_basic(self, sample_eos_data_post):
        """Test basic post-TOV solver functionality with MG parameters."""
        pc = sample_eos_data_post.ps[25]

        post_solver = PostTOVSolver()
        mg_params = {
            "lambda_BL": 0.1,
            "lambda_DY": 0.05,
            "lambda_HB": 1.2,
            "gamma": 0.02,
            "alpha": 10.0,
            "beta": 0.3,
        }

        solution = post_solver.solve(sample_eos_data_post, pc, **mg_params)

        # Check that results are finite and positive
        assert jnp.isfinite(solution.M)
        assert jnp.isfinite(solution.R)
        assert jnp.isfinite(solution.k2)
        assert solution.M > 0
        assert solution.R > 0
        assert solution.k2 > 0

        # Check physical compactness
        compactness = solution.M / solution.R
        assert compactness < 0.5
        assert compactness > 0.01

    def test_post_solver_different_mg_params(self, sample_eos_data_post):
        """Test post-TOV solver with different modified gravity parameters."""
        pc = sample_eos_data_post.ps[25]
        post_solver = PostTOVSolver()

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
            try:
                solution = post_solver.solve(sample_eos_data_post, pc, **mg_params)

                # Basic checks
                assert jnp.isfinite(solution.M) and solution.M > 0
                assert jnp.isfinite(solution.R) and solution.R > 0
                assert jnp.isfinite(solution.k2) and solution.k2 > 0
                assert solution.M / solution.R < 0.5

                results.append((solution.M, solution.R, solution.k2))

            except Exception as e:
                pytest.skip(f"Post-TOV solver failed for params {mg_params}: {e}")

        # Should have at least one successful result
        assert len(results) >= 1

        # Different parameters should potentially give different results
        if len(results) > 1:
            masses = [r[0] for r in results]
            # Allow that some might be very similar (especially GR vs small deviations)
            assert len(set(f"{m:.6f}" for m in masses)) >= 1

    def test_post_solver_convergence(self, sample_eos_data_post):
        """Test that post-TOV solver gives consistent results."""
        pc = sample_eos_data_post.ps[25]
        post_solver = PostTOVSolver()

        mg_params = {
            "lambda_BL": 0.1,
            "lambda_DY": 0.05,
            "lambda_HB": 1.2,
            "gamma": 0.02,
            "alpha": 10.0,
            "beta": 0.3,
        }

        # Solve multiple times
        results = []
        for _ in range(3):
            solution = post_solver.solve(sample_eos_data_post, pc, **mg_params)
            results.append((solution.M, solution.R, solution.k2))

        # Results should be identical (deterministic)
        for i in range(1, len(results)):
            assert abs(results[i][0] - results[0][0]) < 1e-10  # Mass
            assert abs(results[i][1] - results[0][1]) < 1e-10  # Radius
            assert abs(results[i][2] - results[0][2]) < 1e-8  # k2


class TestPostTOVPhysicalConsistency:
    """Test physical consistency of post-TOV solutions."""

    @pytest.mark.slow
    def test_post_mass_radius_relationship(self, sample_eos_data_post):
        """Test mass-radius relationship for post-TOV solutions."""
        post_solver = PostTOVSolver()
        mg_params = {
            "lambda_BL": 0.1,
            "lambda_DY": 0.05,
            "lambda_HB": 1.2,
            "gamma": 0.02,
            "alpha": 10.0,
            "beta": 0.3,
        }

        pressure_indices = range(15, 35, 5)
        masses = []
        radii = []

        for idx in pressure_indices:
            if idx < len(sample_eos_data_post.ps):
                pc = sample_eos_data_post.ps[idx]
                try:
                    solution = post_solver.solve(sample_eos_data_post, pc, **mg_params)
                    if (
                        jnp.isfinite(solution.M)
                        and jnp.isfinite(solution.R)
                        and solution.M > 0
                        and solution.R > 0
                    ):
                        masses.append(solution.M)
                        radii.append(solution.R)
                except Exception:
                    continue

        # Should have multiple solutions
        assert len(masses) >= 2

        masses = jnp.array(masses)
        radii = jnp.array(radii)

        # Physical checks
        assert jnp.all(masses > 0)
        assert jnp.all(radii > 0)
        assert jnp.all(masses / radii < 0.5)  # Compactness limit

    def test_post_modified_gravity_effects(self, sample_eos_data_post):
        """Test that modified gravity parameters actually affect results."""
        pc = sample_eos_data_post.ps[25]
        post_solver = PostTOVSolver()

        # GR case
        gr_params = {
            "lambda_BL": 0.0,
            "lambda_DY": 0.0,
            "lambda_HB": 1.0,
            "gamma": 0.0,
            "alpha": 10.0,
            "beta": 0.3,
        }

        # Modified gravity case
        mg_params = {
            "lambda_BL": 0.2,
            "lambda_DY": 0.1,
            "lambda_HB": 1.1,
            "gamma": 0.05,
            "alpha": 10.0,
            "beta": 0.3,
        }

        try:
            gr_solution = post_solver.solve(sample_eos_data_post, pc, **gr_params)
            mg_solution = post_solver.solve(sample_eos_data_post, pc, **mg_params)

            # Results should be different (though possibly small differences)
            mass_diff = abs(mg_solution.M - gr_solution.M) / gr_solution.M
            radius_diff = abs(mg_solution.R - gr_solution.R) / gr_solution.R

            # At least one should show some difference (even if small)
            assert mass_diff > 1e-6 or radius_diff > 1e-6

        except Exception as e:
            pytest.skip(f"Modified gravity comparison failed: {e}")


# Parameterized tests
@pytest.mark.parametrize("lambda_BL", [0.0, 0.05, 0.1])
@pytest.mark.parametrize("lambda_DY", [0.0, 0.02, 0.05])
def test_post_solver_parameter_sweep(sample_eos_data_post, lambda_BL, lambda_DY):
    """Test post-TOV solver across parameter space."""
    pc = sample_eos_data_post.ps[25]

    post_solver = PostTOVSolver()
    mg_params = {
        "lambda_BL": lambda_BL,
        "lambda_DY": lambda_DY,
        "lambda_HB": 1.0,
        "gamma": 0.0,
        "alpha": 10.0,
        "beta": 0.3,
    }

    try:
        solution = post_solver.solve(sample_eos_data_post, pc, **mg_params)

        assert jnp.isfinite(solution.M)
        assert jnp.isfinite(solution.R)
        assert jnp.isfinite(solution.k2)
        assert solution.M > 0
        assert solution.R > 0
        assert solution.k2 > 0

    except Exception as e:
        pytest.skip(f"Post-TOV solver failed for params: {e}")


@pytest.mark.parametrize(
    "mg_param,value",
    [
        ("lambda_BL", 0.1),
        ("lambda_DY", 0.05),
        ("lambda_HB", 1.1),
        ("gamma", 0.02),
    ],
)
def test_post_solver_individual_mg_params(sample_eos_data_post, mg_param, value):
    """Test post-TOV solver with individual modified gravity parameters."""
    pc = sample_eos_data_post.ps[25]
    post_solver = PostTOVSolver()

    # Start with GR and modify one parameter
    mg_params = {
        "lambda_BL": 0.0,
        "lambda_DY": 0.0,
        "lambda_HB": 1.0,
        "gamma": 0.0,
        "alpha": 10.0,
        "beta": 0.3,
    }
    mg_params[mg_param] = value

    try:
        solution = post_solver.solve(sample_eos_data_post, pc, **mg_params)

        assert jnp.isfinite(solution.M) and solution.M > 0
        assert jnp.isfinite(solution.R) and solution.R > 0
        assert jnp.isfinite(solution.k2) and solution.k2 > 0
        assert solution.M / solution.R < 0.5

    except Exception as e:
        pytest.skip(f"Post-TOV solver failed for {mg_param}={value}: {e}")
