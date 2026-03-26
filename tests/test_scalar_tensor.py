"""Unit tests for Scalar-Tensor TOV equation solver."""

import pytest
import jax.numpy as jnp
from jesterTOV.tov.scalar_tensor import ScalarTensorTOVSolver
from jesterTOV.tov.gr import GRTOVSolver


class TestScalarTensorTOVSolver:
    """Test suite for ScalarTensorTOVSolver."""

    def test_solve_single_star(self, sample_eos_data):
        """Test single star solution with Scalar-Tensor solver."""
        # Choose a central pressure (in geometric units)
        pc = float(sample_eos_data.ps[25])

        solver = ScalarTensorTOVSolver()
        # Use values similar to notebook for stable test
        params = {"beta_ST": -4.5, "phi_inf_tgt": 1e-3, "phi_c": 1.0}

        solution = solver.solve(sample_eos_data, pc, tov_params=params)

        # Check that results are finite and positive
        assert jnp.isfinite(solution.M)
        assert jnp.isfinite(solution.R)
        assert jnp.isfinite(solution.k2)
        assert solution.M > 0
        assert solution.R > 0
        assert solution.k2 > 0

        # Check compactness
        compactness = solution.M / solution.R
        assert compactness < 0.5
        assert compactness > 0.01

    def test_construct_family(self, sample_eos_data):
        """Test M-R-Λ family construction for Scalar-Tensor gravity."""
        solver = ScalarTensorTOVSolver()
        params = {"beta_ST": -4.5, "phi_inf_tgt": 1e-3, "phi_c": 1.0}

        family_data = solver.construct_family(
            sample_eos_data, ndat=10, min_nsat=0.75, tov_params=params
        )

        # Check shapes
        assert len(family_data.log10pcs) == 10
        assert len(family_data.masses) == 10
        assert len(family_data.radii) == 10
        assert len(family_data.lambdas) == 10

        # Check that results are finite
        assert jnp.all(jnp.isfinite(family_data.masses))
        assert jnp.all(jnp.isfinite(family_data.radii))
        assert jnp.all(jnp.isfinite(family_data.lambdas))

        # Check extra fields are populated
        assert family_data.extra is not None
        assert "lambda_S" in family_data.extra
        assert "lambda_ST1" in family_data.extra
        assert "q" in family_data.extra

        # Check that extra fields have some finite values
        assert jnp.any(jnp.isfinite(family_data.extra["lambda_S"]))
        assert jnp.any(jnp.isfinite(family_data.extra["q"]))

    def test_gr_limit(self, sample_eos_data):
        """Test that Scalar-Tensor solver reduces to GR when beta_ST = 0."""
        pc = float(sample_eos_data.ps[25])

        st_solver = ScalarTensorTOVSolver()
        gr_solver = GRTOVSolver()

        # beta_ST = 0 with default notebook values for others should reduce to GR
        params = {"beta_ST": 0.0, "phi_inf_tgt": 1e-3, "phi_c": 1.0}

        sol_st = st_solver.solve(sample_eos_data, pc, tov_params=params)
        sol_gr = gr_solver.solve(sample_eos_data, pc, tov_params={})

        # Results should match GR
        assert jnp.allclose(sol_st.M, sol_gr.M, rtol=1e-3)
        assert jnp.allclose(sol_st.R, sol_gr.R, rtol=1e-3)
        # k2 might have small differences due to different ODE systems and matching
        assert jnp.allclose(sol_st.k2, sol_gr.k2, rtol=1e-2)

    def test_required_parameters(self):
        """Test that required parameters are correctly reported."""
        solver = ScalarTensorTOVSolver()
        required = solver.get_required_parameters()

        assert "beta_ST" in required
        assert "phi_inf_tgt" in required
        assert "phi_c" in required

    @pytest.mark.parametrize("beta_ST", [-5.0, -4.5, 0.0])
    def test_parameter_sensitivity(self, sample_eos_data, beta_ST):
        """Test sensitivity to beta_ST parameter."""
        pc = float(sample_eos_data.ps[25])
        solver = ScalarTensorTOVSolver()

        params = {"beta_ST": beta_ST, "phi_inf_tgt": 1e-3, "phi_c": 1.0}

        solution = solver.solve(sample_eos_data, pc, tov_params=params)
        assert jnp.isfinite(solution.M)
        assert solution.M > 0

    def test_calculate_tidal_false(self, sample_eos_data):
        """Test that calculate_tidal=False skips tidal calculations."""
        pc = float(sample_eos_data.ps[25])
        solver = ScalarTensorTOVSolver(calculate_tidal=False)
        params = {"beta_ST": -4.5, "phi_inf_tgt": 1e-3, "phi_c": 1.0}
        solution_without = solver.solve(sample_eos_data, pc, tov_params=params)
        # Mass and radius should be finite
        assert jnp.isfinite(solution_without.M)
        assert jnp.isfinite(solution_without.R)
        # k2 should be nan
        assert jnp.isnan(solution_without.k2)
        # extra dict should have nan values for tidal quantities
        assert solution_without.extra is not None
        assert jnp.isnan(solution_without.extra["lambda_S"])
        assert jnp.isnan(solution_without.extra["lambda_ST1"])
        assert jnp.isnan(solution_without.extra["lambda_ST2"])
        # q may be finite (needed for Jordan mass)
        # Compare with tidal calculation (default True)
        solver_tidal = ScalarTensorTOVSolver(calculate_tidal=True)
        solution_with = solver_tidal.solve(sample_eos_data, pc, tov_params=params)
        # Mass and radius should match within tolerance
        assert jnp.allclose(solution_with.M, solution_without.M, rtol=1e-5)
        assert jnp.allclose(solution_with.R, solution_without.R, rtol=1e-5)

    def test_construct_family_without_tidal(self, sample_eos_data):
        """Test M-R-Λ family construction with calculate_tidal=False."""
        solver = ScalarTensorTOVSolver(calculate_tidal=False)
        params = {"beta_ST": -4.5, "phi_inf_tgt": 1e-3, "phi_c": 1.0}
        family_data = solver.construct_family(
            sample_eos_data, ndat=10, min_nsat=0.75, tov_params=params
        )
        # Check shapes
        assert len(family_data.log10pcs) == 10
        assert len(family_data.masses) == 10
        assert len(family_data.radii) == 10
        assert len(family_data.lambdas) == 10
        # Lambdas should all be nan
        assert jnp.all(jnp.isnan(family_data.lambdas))
        # Extra fields should exist but be nan
        assert family_data.extra is not None
        assert "lambda_S" in family_data.extra
        assert jnp.all(jnp.isnan(family_data.extra["lambda_S"]))
        assert "q" in family_data.extra
        # q may be finite (still computed)
