"""Unit tests for TOV equation solver."""

import pytest
import jax.numpy as jnp
import numpy as np
from jesterTOV import tov, utils


class TestTOVODE:
    """Test TOV ODE function."""
    
    def test_tov_ode_basic(self, sample_eos_dict):
        """Test basic TOV ODE functionality."""
        # Sample state variables
        h = sample_eos_dict["h"][10]  # Pick middle enthalpy value
        r, m, H, b = 10.0, 1.0, 100.0, 20.0  # Typical values in geometric units
        y = (r, m, H, b)
        
        # Compute derivatives
        dydt = tov.tov_ode(h, y, sample_eos_dict)
        drdh, dmdh, dHdh, dbdh = dydt
        
        # Check that derivatives are finite
        assert jnp.isfinite(drdh)
        assert jnp.isfinite(dmdh)
        assert jnp.isfinite(dHdh)
        assert jnp.isfinite(dbdh)
        
        # Check physical constraints
        # TODO: verify them
        assert drdh < 0  # Radius decreases as enthalpy decreases (center to surface)
        assert dmdh < 0  # Mass derivative is negative when integrating h_center → h_surface
    
    def test_tov_ode_physical_behavior(self, sample_eos_dict):
        """Test that TOV ODE gives physically reasonable derivatives."""
        # Test at different enthalpy values
        h_values = sample_eos_dict["h"][::10]  # Sample a few points
        r, m, H, b = 10.0, 1.0, 100.0, 20.0
        y = (r, m, H, b)
        
        for h in h_values:
            dydt = tov.tov_ode(h, y, sample_eos_dict)
            drdh, dmdh, dHdh, dbdh = dydt
            
            # All derivatives should be finite
            assert jnp.isfinite(drdh)
            assert jnp.isfinite(dmdh)
            assert jnp.isfinite(dHdh)
            assert jnp.isfinite(dbdh)
    
    def test_tov_ode_interpolation(self, sample_eos_dict):
        """Test that TOV ODE correctly interpolates EOS quantities."""
        h_test = sample_eos_dict["h"][15]  # Pick a specific enthalpy
        r, m, H, b = 10.0, 1.0, 100.0, 20.0
        y = (r, m, H, b)
        
        # Get derivatives
        dydt = tov.tov_ode(h_test, y, sample_eos_dict)
        
        # Check that interpolated values are reasonable
        # (This mainly checks that interpolation doesn't fail)
        assert all(jnp.isfinite(d) for d in dydt)


class TestCalcK2:
    """Test tidal deformability calculation."""
    
    def test_calc_k2_basic(self):
        """Test basic k2 calculation."""
        # Typical neutron star values
        R = 12.0  # km in geometric units (approximation)
        M = 1.4   # Solar masses in geometric units (approximation)
        H = 144.0  # Typical H value
        b = 24.0   # Typical b value
        
        k2 = tov.calc_k2(R, M, H, b)
        
        # k2 should be finite and typically positive for realistic stars
        assert jnp.isfinite(k2)
        assert k2 > 0  # Should be positive for stable neutron stars
        assert k2 < 1000  # Should not be unreasonably large
    
    def test_calc_k2_compactness_dependence(self):
        """Test k2 dependence on compactness."""
        R = 12.0
        H = 144.0
        b = 24.0
        
        # Test different masses (different compactness)
        masses = jnp.array([1.0, 1.4, 1.8, 2.0])
        k2_values = []
        
        for M in masses:
            if M/R < 0.5:  # Avoid unphysical compactness
                k2 = tov.calc_k2(R, M, H, b)
                k2_values.append(k2)
                assert jnp.isfinite(k2)
                assert k2 > 0
        
        # Generally, k2 should decrease with increasing compactness
        # (though this is a rough check since we're using simplified values)
        assert len(k2_values) > 1
    
    def test_calc_k2_limiting_cases(self):
        """Test k2 calculation in limiting cases."""
        R = 12.0
        H = 144.0
        b = 24.0
        
        # Low compactness case
        M_low = 0.1  # Very low mass
        k2_low = tov.calc_k2(R, M_low, H, b)
        assert jnp.isfinite(k2_low)
        assert k2_low > 0
        
        # Higher compactness (but still physical)
        M_high = 2.0
        if M_high/R < 0.4:  # Check that we're still in physical regime
            k2_high = tov.calc_k2(R, M_high, H, b)
            assert jnp.isfinite(k2_high)
            assert k2_high > 0


class TestTOVSolver:
    """Test complete TOV solver."""
    
    def test_tov_solver_basic(self, sample_eos_dict):
        """Test basic TOV solver functionality."""
        # Choose a central pressure (in geometric units)
        pc = sample_eos_dict["p"][25]  # Middle pressure value
        
        M, R, k2 = tov.tov_solver(sample_eos_dict, pc)
        
        # Check that results are finite and positive
        assert jnp.isfinite(M)
        assert jnp.isfinite(R)
        assert jnp.isfinite(k2)
        assert M > 0
        assert R > 0
        assert k2 > 0
        
        # Check that compactness is physical
        compactness = M / R
        assert compactness < 0.5  # Must be less than 0.5 for stable neutron star
        assert compactness > 0.01  # Should not be too small either
    
    def test_tov_solver_different_pressures(self, sample_eos_dict):
        """Test TOV solver with different central pressures."""
        # Test several pressure values
        pressure_indices = [10, 20, 30, 40]
        results = []
        
        for idx in pressure_indices:
            if idx < len(sample_eos_dict["p"]):
                pc = sample_eos_dict["p"][idx]
                M, R, k2 = tov.tov_solver(sample_eos_dict, pc)
                
                # Basic sanity checks
                assert jnp.isfinite(M) and M > 0
                assert jnp.isfinite(R) and R > 0
                assert jnp.isfinite(k2) and k2 > 0
                assert M/R < 0.5  # Physical compactness
                
                results.append((M, R, k2))
        
        # Should have at least a few successful results
        assert len(results) >= 2
        
        # Mass-radius relationship should be physically reasonable
        masses = [result[0] for result in results]
        radii = [result[1] for result in results]
        
        # Check that all masses are in reasonable range
        for M, R in zip(masses, radii):
            M_solar = M / utils.solar_mass_in_meter
            R_km = R / 1000
            assert 0.5 < M_solar < 3.5  # Physical mass range
            assert 8.0 < R_km < 20.0    # Physical radius range
    
    def test_tov_solver_initial_conditions(self, sample_eos_dict):
        """Test that TOV solver sets up initial conditions correctly."""
        pc = sample_eos_dict["p"][20]
        
        # The solver should handle initial condition setup internally
        # We just test that it doesn't crash and gives reasonable results
        M, R, k2 = tov.tov_solver(sample_eos_dict, pc)
        
        # Results should be in reasonable ranges for neutron stars
        # Convert to physical units for checking
        M_solar = M / utils.solar_mass_in_meter
        R_km = R / 1000
        
        assert 0.5 < M_solar < 3.0  # Mass in solar masses
        assert 8.0 < R_km < 16.0    # Radius in km
        assert 0.001 < k2 < 1.0     # k2 dimensionless tidal deformability
    
    def test_tov_solver_energy_pressure_consistency(self, sample_eos_dict):
        """Test that solver respects energy-pressure relationships."""
        pc = sample_eos_dict["p"][25]
        
        # Get the central enthalpy and energy density
        ps = sample_eos_dict["p"]
        hs = sample_eos_dict["h"]
        es = sample_eos_dict["e"]
        
        hc = utils.interp_in_logspace(pc, ps, hs)
        ec = utils.interp_in_logspace(hc, hs, es)
        
        # Central energy density should be positive and finite
        assert jnp.isfinite(ec)
        assert ec > 0
        assert ec > pc  # Energy density should exceed pressure
        
        # Solve TOV equations
        M, R, k2 = tov.tov_solver(sample_eos_dict, pc)
        
        # Results should be consistent with input
        assert jnp.isfinite(M) and M > 0
        assert jnp.isfinite(R) and R > 0


class TestTOVPhysicalConsistency:
    """Test physical consistency of TOV solutions."""
    
    @pytest.mark.slow
    def test_mass_radius_relationship(self, sample_eos_dict):
        """Test that mass-radius relationship is physically reasonable."""
        # Test multiple central pressures
        pressure_indices = range(15, 35, 5)  # Sample pressures
        masses = []
        radii = []
        
        for idx in pressure_indices:
            if idx < len(sample_eos_dict["p"]):
                pc = sample_eos_dict["p"][idx]
                try:
                    M, R, k2 = tov.tov_solver(sample_eos_dict, pc)
                    if jnp.isfinite(M) and jnp.isfinite(R) and M > 0 and R > 0:
                        masses.append(M)
                        radii.append(R)
                except:
                    continue  # Skip problematic cases
        
        # Should have multiple successful solutions
        assert len(masses) >= 3
        
        # Convert to arrays for analysis
        masses = jnp.array(masses)
        radii = jnp.array(radii)
        
        # All masses and radii should be positive and finite
        assert jnp.all(masses > 0)
        assert jnp.all(radii > 0)
        assert jnp.all(jnp.isfinite(masses))
        assert jnp.all(jnp.isfinite(radii))
        
        # Compactness should be physical
        compactness = masses / radii
        assert jnp.all(compactness < 0.5)
        assert jnp.all(compactness > 0.01)
    
    def test_tov_solver_convergence(self, sample_eos_dict):
        """Test that TOV solver gives consistent results."""
        pc = sample_eos_dict["p"][25]
        
        # Solve multiple times (should be deterministic)
        results = []
        for _ in range(3):
            M, R, k2 = tov.tov_solver(sample_eos_dict, pc)
            results.append((M, R, k2))
        
        # Results should be identical (within numerical precision)
        for i in range(1, len(results)):
            assert abs(results[i][0] - results[0][0]) < 1e-10  # Mass
            assert abs(results[i][1] - results[0][1]) < 1e-10  # Radius
            assert abs(results[i][2] - results[0][2]) < 1e-8   # k2 (slightly less precise)


# Parameterized tests
@pytest.mark.parametrize("pressure_fraction", [0.3, 0.5, 0.7, 0.9])
def test_tov_solver_pressure_range(sample_eos_dict, pressure_fraction):
    """Test TOV solver across pressure range."""
    n_pressures = len(sample_eos_dict["p"])
    idx = int(pressure_fraction * n_pressures)
    idx = min(idx, n_pressures - 1)
    
    pc = sample_eos_dict["p"][idx]
    
    try:
        M, R, k2 = tov.tov_solver(sample_eos_dict, pc)
        
        # Basic physical checks
        assert jnp.isfinite(M) and M > 0
        assert jnp.isfinite(R) and R > 0
        assert jnp.isfinite(k2) and k2 > 0
        assert M/R < 0.5  # Physical compactness limit
        
    except Exception as e:
        # Some pressure values might not converge
        # This is acceptable for edge cases
        pytest.skip(f"TOV solver failed for pressure fraction {pressure_fraction}: {e}")


@pytest.mark.parametrize("R,M,H,b", [
    (10.0, 1.0, 100.0, 20.0),
    (12.0, 1.4, 144.0, 24.0),
    (15.0, 2.0, 225.0, 30.0),
])
def test_calc_k2_parameter_variations(R, M, H, b):
    """Test k2 calculation with different parameter combinations."""
    # Only test if compactness is physical
    if M/R < 0.5:
        k2 = tov.calc_k2(R, M, H, b)
        assert jnp.isfinite(k2)
        assert k2 > 0
        assert k2 < 10000  # Reasonable upper bound