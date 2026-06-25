"""Tests for inference prior system (parser and library)."""

import pytest
import jax
import jax.numpy as jnp

from jesterTOV.inference.priors import parser
from jesterTOV.inference.priors.parser import ParsedPrior
from jesterTOV.inference.base import CombinePrior, UniformPrior
from jesterTOV.inference.base.prior import Fixed


class TestPriorParser:
    """Test prior file parsing functionality."""

    def test_parse_basic_nep_prior(self, sample_prior_file):
        """Test parsing basic NEP parameters without CSE."""
        result = parser.parse_prior_file(sample_prior_file, nb_CSE=0)

        assert isinstance(result, ParsedPrior)
        prior = result.prior
        assert isinstance(prior, CombinePrior)
        # Should have 9 NEP parameters (E_sat + 4 _sat + 4 _sym)
        assert prior.n_dim == 9

        # Check parameter names
        param_names = prior.parameter_names
        assert "E_sat" in param_names
        assert "K_sat" in param_names
        assert "Q_sat" in param_names
        assert "Z_sat" in param_names
        assert "E_sym" in param_names
        assert "L_sym" in param_names
        assert "K_sym" in param_names
        assert "Q_sym" in param_names
        assert "Z_sym" in param_names

        # nbreak should NOT be included when nb_CSE=0
        assert "nbreak" not in param_names

        # No fixed params in a plain NEP prior
        assert result.fixed_params == {}

    def test_parse_nep_prior_with_cse(self, sample_prior_file_with_cse):
        """Test parsing NEP parameters with CSE."""
        nb_CSE = 8
        result = parser.parse_prior_file(sample_prior_file_with_cse, nb_CSE=nb_CSE)

        assert isinstance(result, ParsedPrior)
        prior = result.prior
        assert isinstance(prior, CombinePrior)
        # Should have 9 NEP + 1 nbreak + 8*2 CSE grid params + 1 final cs2
        # = 9 + 1 + 16 + 1 = 27
        expected_dim = 9 + 1 + (nb_CSE * 2) + 1
        assert prior.n_dim == expected_dim

        # Check parameter names
        param_names = prior.parameter_names
        assert "nbreak" in param_names

        # Check CSE grid parameters
        for i in range(nb_CSE):
            assert f"n_CSE_{i}_u" in param_names
            assert f"cs2_CSE_{i}" in param_names

        # Check final cs2 parameter
        assert f"cs2_CSE_{nb_CSE}" in param_names

    def test_parse_cse_parameter_count(self, sample_prior_file_with_cse):
        """Test that CSE parameter count is correct for different nb_CSE values."""
        test_cases = [
            (0, 9),  # No CSE: 9 NEP only
            (4, 9 + 1 + 4 * 2 + 1),  # 9 NEP + nbreak + 4*2 grid + 1 final = 19
            (8, 9 + 1 + 8 * 2 + 1),  # 9 NEP + nbreak + 8*2 grid + 1 final = 27
            (16, 9 + 1 + 16 * 2 + 1),  # 9 NEP + nbreak + 16*2 grid + 1 final = 43
        ]

        for nb_CSE, expected_dim in test_cases:
            result = parser.parse_prior_file(sample_prior_file_with_cse, nb_CSE=nb_CSE)
            prior = result.prior
            assert (
                prior.n_dim == expected_dim
            ), f"Expected {expected_dim} for nb_CSE={nb_CSE}, got {prior.n_dim}"

    def test_parse_nonexistent_file_fails(self):
        """Test that parsing nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parser.parse_prior_file("nonexistent.prior", nb_CSE=0)

    def test_parse_invalid_python_syntax_fails(self, temp_dir):
        """Test that invalid Python syntax raises error."""
        invalid_prior = temp_dir / "invalid.prior"
        invalid_prior.write_text("this is not valid python syntax !!!")

        with pytest.raises(ValueError, match="Error executing prior file"):
            parser.parse_prior_file(invalid_prior, nb_CSE=0)

    def test_parse_empty_prior_file_fails(self, temp_dir):
        """Test that empty prior file raises error."""
        empty_prior = temp_dir / "empty.prior"
        empty_prior.write_text("# Just a comment, no priors defined")

        with pytest.raises(ValueError, match="No sampled priors found"):
            parser.parse_prior_file(empty_prior, nb_CSE=0)

    def test_parse_prior_without_nep_params(self, temp_dir):
        """Test parsing prior with non-NEP parameters.

        This tests that parameters not ending in _sat or _sym are still included.
        """
        custom_prior = temp_dir / "custom.prior"
        custom_prior.write_text(
            """
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
custom_param = UniformPrior(0.0, 1.0, parameter_names=["custom_param"])
"""
        )

        result = parser.parse_prior_file(custom_prior, nb_CSE=0)
        prior = result.prior

        # Should have K_sat and custom_param
        assert prior.n_dim == 2
        assert "K_sat" in prior.parameter_names
        assert "custom_param" in prior.parameter_names

    def test_cse_priors_have_correct_bounds(self, sample_prior_file_with_cse):
        """Test that automatically generated CSE priors have correct bounds [0, 1]."""
        result = parser.parse_prior_file(sample_prior_file_with_cse, nb_CSE=8)
        prior = result.prior

        # Sample from CSE parameters and check they're in [0, 1]
        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=10)

        # Get CSE parameter values from samples dict (arrays of shape [10])
        for i in range(8):
            n_cse_vals = samples[f"n_CSE_{i}_u"]
            cs2_cse_vals = samples[f"cs2_CSE_{i}"]

            assert jnp.all(n_cse_vals >= 0.0) and jnp.all(
                n_cse_vals <= 1.0
            ), f"n_CSE_{i}_u out of bounds"
            assert jnp.all(cs2_cse_vals >= 0.0) and jnp.all(
                cs2_cse_vals <= 1.0
            ), f"cs2_CSE_{i} out of bounds"

        # Check final cs2 parameter
        cs2_final_vals = samples["cs2_CSE_8"]
        assert jnp.all(cs2_final_vals >= 0.0) and jnp.all(cs2_final_vals <= 1.0)


class TestFixedPrior:
    """Tests for the Fixed prior class."""

    def test_fixed_stores_value(self):
        """Test that Fixed stores the given value."""
        f = Fixed(3.14, parameter_names=["my_param"])
        assert f.value == pytest.approx(3.14)
        assert f.parameter_names == ["my_param"]
        assert f.n_dim == 1

    def test_fixed_accepts_int_value(self):
        """Test that Fixed converts int values to float."""
        f = Fixed(0, parameter_names=["x"])
        assert isinstance(f.value, float)

    def test_fixed_sample_raises(self):
        """Fixed.sample() must not be called — it raises NotImplementedError."""
        f = Fixed(1.0, parameter_names=["x"])
        with pytest.raises(NotImplementedError):
            f.sample(jax.random.PRNGKey(0), 1)

    def test_fixed_log_prob_raises(self):
        """Fixed.log_prob() must not be called — it raises NotImplementedError."""
        f = Fixed(1.0, parameter_names=["x"])
        with pytest.raises(NotImplementedError):
            f.log_prob({"x": 1.0})

    def test_fixed_repr(self):
        """Test Fixed string representation."""
        f = Fixed(2.5, parameter_names=["lambda_BL"])
        assert "Fixed" in repr(f)
        assert "2.5" in repr(f)
        assert "lambda_BL" in repr(f)

    def test_fixed_must_be_1d(self):
        """Fixed must have exactly one parameter name."""
        with pytest.raises(AssertionError):
            Fixed(1.0, parameter_names=["a", "b"])


class TestParserFixedParams:
    """Tests for Fixed parameter handling in the parser."""

    def test_parse_prior_with_fixed_param(self, temp_dir):
        """Fixed parameters are extracted to fixed_params, not sampled prior."""
        prior_file = temp_dir / "with_fixed.prior"
        prior_file.write_text(
            """
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
lambda_BL = Fixed(0.0, parameter_names=["lambda_BL"])
"""
        )

        result = parser.parse_prior_file(prior_file, nb_CSE=0)

        # lambda_BL should NOT be in the sampled prior
        assert "lambda_BL" not in result.prior.parameter_names
        assert result.prior.n_dim == 1  # only K_sat

        # lambda_BL should be in fixed_params with correct value
        assert "lambda_BL" in result.fixed_params
        assert result.fixed_params["lambda_BL"] == pytest.approx(0.0)

    def test_parse_prior_multiple_fixed_params(self, temp_dir):
        """Multiple Fixed parameters are all collected into fixed_params."""
        prior_file = temp_dir / "multi_fixed.prior"
        prior_file.write_text(
            """
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
lambda_BL = Fixed(0.5, parameter_names=["lambda_BL"])
alpha = Fixed(-1.0, parameter_names=["alpha"])
"""
        )

        result = parser.parse_prior_file(prior_file, nb_CSE=0)

        assert result.prior.n_dim == 2
        assert set(result.prior.parameter_names) == {"K_sat", "L_sym"}
        assert result.fixed_params == {
            "lambda_BL": pytest.approx(0.5),
            "alpha": pytest.approx(-1.0),
        }

    def test_parse_prior_only_fixed_params_raises(self, temp_dir):
        """A prior file with only Fixed entries has no sampled space — should raise."""
        prior_file = temp_dir / "only_fixed.prior"
        prior_file.write_text(
            """
lambda_BL = Fixed(0.0, parameter_names=["lambda_BL"])
"""
        )

        with pytest.raises(ValueError, match="No sampled priors found"):
            parser.parse_prior_file(prior_file, nb_CSE=0)

    def test_parse_prior_no_fixed_gives_empty_dict(self, sample_prior_file):
        """When no Fixed params are present, fixed_params is an empty dict."""
        result = parser.parse_prior_file(sample_prior_file, nb_CSE=0)
        assert result.fixed_params == {}


class TestCSEFixedParams:
    """Tests for Fixed and custom-bounded CSE parameters in the prior parser.

    When a user specifies a CSE grid parameter in the prior file (either as
    ``Fixed`` or with custom bounds), the auto-generation logic must respect
    that specification and not overwrite it with a default UniformPrior(0, 1).
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    NEP_AND_NBREAK = """
E_sat = UniformPrior(-16.1, -15.9, parameter_names=["E_sat"])
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])
nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])
"""

    def _make_prior_file(self, temp_dir, extra: str, name: str = "test.prior"):
        p = temp_dir / name
        p.write_text(self.NEP_AND_NBREAK + extra)
        return p

    # ------------------------------------------------------------------
    # Fixed CSE parameters
    # ------------------------------------------------------------------

    def test_single_fixed_cse_density_param(self, temp_dir):
        """Fix one density grid point: it goes to fixed_params, not the sampled prior."""
        nb_CSE = 4
        prior_file = self._make_prior_file(
            temp_dir,
            'n_CSE_0_u = Fixed(0.5, parameter_names=["n_CSE_0_u"])\n',
        )
        result = parser.parse_prior_file(prior_file, nb_CSE=nb_CSE)

        # n_CSE_0_u is fixed, not sampled
        assert "n_CSE_0_u" not in result.prior.parameter_names
        assert "n_CSE_0_u" in result.fixed_params
        assert result.fixed_params["n_CSE_0_u"] == pytest.approx(0.5)

        # All other CSE density params are still sampled
        for i in range(1, nb_CSE):
            assert f"n_CSE_{i}_u" in result.prior.parameter_names

        # Total dimension: 9 NEP + 1 nbreak + (nb_CSE*2+1) CSE − 1 fixed
        expected_dim = 9 + 1 + (nb_CSE * 2 + 1) - 1
        assert result.prior.n_dim == expected_dim

    def test_single_fixed_cse_cs2_param(self, temp_dir):
        """Fix one cs2 grid point: it goes to fixed_params, not the sampled prior."""
        nb_CSE = 4
        prior_file = self._make_prior_file(
            temp_dir,
            'cs2_CSE_2 = Fixed(0.3, parameter_names=["cs2_CSE_2"])\n',
        )
        result = parser.parse_prior_file(prior_file, nb_CSE=nb_CSE)

        assert "cs2_CSE_2" not in result.prior.parameter_names
        assert result.fixed_params["cs2_CSE_2"] == pytest.approx(0.3)

        expected_dim = 9 + 1 + (nb_CSE * 2 + 1) - 1
        assert result.prior.n_dim == expected_dim

    def test_fixed_final_cse_cs2_param(self, temp_dir):
        """Fix the final cs2 parameter (cs2_CSE_{nb_CSE}) at nmax."""
        nb_CSE = 4
        prior_file = self._make_prior_file(
            temp_dir,
            f'cs2_CSE_{nb_CSE} = Fixed(0.8, parameter_names=["cs2_CSE_{nb_CSE}"])\n',
        )
        result = parser.parse_prior_file(prior_file, nb_CSE=nb_CSE)

        assert f"cs2_CSE_{nb_CSE}" not in result.prior.parameter_names
        assert result.fixed_params[f"cs2_CSE_{nb_CSE}"] == pytest.approx(0.8)

        expected_dim = 9 + 1 + (nb_CSE * 2 + 1) - 1
        assert result.prior.n_dim == expected_dim

    def test_all_density_cse_params_fixed(self, temp_dir):
        """Fix all n_CSE_i_u params; cs2 parameters are still sampled freely."""
        nb_CSE = 3
        fixed_lines = "".join(
            f'n_CSE_{i}_u = Fixed({0.1 * (i + 1)}, parameter_names=["n_CSE_{i}_u"])\n'
            for i in range(nb_CSE)
        )
        prior_file = self._make_prior_file(temp_dir, fixed_lines)
        result = parser.parse_prior_file(prior_file, nb_CSE=nb_CSE)

        for i in range(nb_CSE):
            assert f"n_CSE_{i}_u" not in result.prior.parameter_names
            assert f"n_CSE_{i}_u" in result.fixed_params

        # cs2 params still sampled (nb_CSE + 1 of them)
        for i in range(nb_CSE + 1):
            assert f"cs2_CSE_{i}" in result.prior.parameter_names

        # Dimension: removed nb_CSE density params
        expected_dim = 9 + 1 + (nb_CSE * 2 + 1) - nb_CSE
        assert result.prior.n_dim == expected_dim

    def test_all_cse_params_fixed(self, temp_dir):
        """Fix every CSE parameter: only NEP + nbreak are sampled."""
        nb_CSE = 2
        fixed_lines = ""
        for i in range(nb_CSE):
            fixed_lines += (
                f'n_CSE_{i}_u = Fixed(0.5, parameter_names=["n_CSE_{i}_u"])\n'
                f'cs2_CSE_{i} = Fixed(0.5, parameter_names=["cs2_CSE_{i}"])\n'
            )
        fixed_lines += (
            f'cs2_CSE_{nb_CSE} = Fixed(0.5, parameter_names=["cs2_CSE_{nb_CSE}"])\n'
        )
        prior_file = self._make_prior_file(temp_dir, fixed_lines)
        result = parser.parse_prior_file(prior_file, nb_CSE=nb_CSE)

        # All CSE params fixed
        for i in range(nb_CSE):
            assert f"n_CSE_{i}_u" in result.fixed_params
            assert f"cs2_CSE_{i}" in result.fixed_params
        assert f"cs2_CSE_{nb_CSE}" in result.fixed_params

        # Sampled space is just 9 NEP + nbreak
        assert result.prior.n_dim == 10
        assert "nbreak" in result.prior.parameter_names

    # ------------------------------------------------------------------
    # Custom-bounded CSE parameters (not Fixed, but user-specified prior)
    # ------------------------------------------------------------------

    def test_custom_bounds_for_cse_density_param(self, temp_dir):
        """Custom UniformPrior for a CSE density param replaces the default [0, 1]."""
        nb_CSE = 4
        prior_file = self._make_prior_file(
            temp_dir,
            'n_CSE_1_u = UniformPrior(0.2, 0.8, parameter_names=["n_CSE_1_u"])\n',
        )
        result = parser.parse_prior_file(prior_file, nb_CSE=nb_CSE)

        # n_CSE_1_u must appear exactly once
        assert result.prior.parameter_names.count("n_CSE_1_u") == 1
        assert "n_CSE_1_u" in result.prior.parameter_names

        # Find that prior and check its bounds are the custom ones
        import jax

        samples = result.prior.sample(jax.random.PRNGKey(0), n_samples=500)
        vals = samples["n_CSE_1_u"]
        assert float(vals.min()) >= 0.2 - 1e-6
        assert float(vals.max()) <= 0.8 + 1e-6

        # Dimension unchanged (user-supplied prior replaces auto one, not added on top)
        expected_dim = 9 + 1 + (nb_CSE * 2 + 1)
        assert result.prior.n_dim == expected_dim

    def test_custom_bounds_for_final_cs2_param(self, temp_dir):
        """Custom UniformPrior for the final cs2_CSE_{nb_CSE} replaces the default."""
        nb_CSE = 4
        prior_file = self._make_prior_file(
            temp_dir,
            f'cs2_CSE_{nb_CSE} = UniformPrior(0.1, 0.6, parameter_names=["cs2_CSE_{nb_CSE}"])\n',
        )
        result = parser.parse_prior_file(prior_file, nb_CSE=nb_CSE)

        assert result.prior.parameter_names.count(f"cs2_CSE_{nb_CSE}") == 1
        expected_dim = 9 + 1 + (nb_CSE * 2 + 1)
        assert result.prior.n_dim == expected_dim

    # ------------------------------------------------------------------
    # Mixed: some fixed, some custom, some auto-generated
    # ------------------------------------------------------------------

    def test_partial_cse_fixed_modular(self, temp_dir):
        """Mix: fix density grid points, custom cs2 for one, rest auto-generated."""
        nb_CSE = 3
        extra = (
            'n_CSE_0_u = Fixed(0.2, parameter_names=["n_CSE_0_u"])\n'
            'n_CSE_1_u = Fixed(0.5, parameter_names=["n_CSE_1_u"])\n'
            'cs2_CSE_0 = UniformPrior(0.1, 0.5, parameter_names=["cs2_CSE_0"])\n'
        )
        prior_file = self._make_prior_file(temp_dir, extra)
        result = parser.parse_prior_file(prior_file, nb_CSE=nb_CSE)

        # Two density params fixed
        assert "n_CSE_0_u" in result.fixed_params
        assert "n_CSE_1_u" in result.fixed_params
        assert "n_CSE_0_u" not in result.prior.parameter_names
        assert "n_CSE_1_u" not in result.prior.parameter_names

        # Remaining density param auto-generated
        assert "n_CSE_2_u" in result.prior.parameter_names

        # cs2_CSE_0 present (user custom bounds) - appears exactly once
        assert result.prior.parameter_names.count("cs2_CSE_0") == 1

        # Total: 9 NEP + 1 nbreak + 3*2+1 CSE − 2 fixed density
        expected_dim = 9 + 1 + (nb_CSE * 2 + 1) - 2
        assert result.prior.n_dim == expected_dim

    def test_no_duplicate_params_with_fixed_cse(self, temp_dir):
        """Fixing CSE params must not produce duplicate entries in the prior."""
        nb_CSE = 4
        extra = "".join(
            f'n_CSE_{i}_u = Fixed(0.25 * {i + 1}, parameter_names=["n_CSE_{i}_u"])\n'
            for i in range(nb_CSE)
        )
        prior_file = self._make_prior_file(temp_dir, extra)
        result = parser.parse_prior_file(prior_file, nb_CSE=nb_CSE)

        names = result.prior.parameter_names
        assert len(names) == len(set(names)), "Duplicate parameter names found in prior"


class TestCombinePrior:
    """Test CombinePrior functionality."""

    def test_combine_prior_basic(self):
        """Test basic CombinePrior creation and properties."""
        prior1 = UniformPrior(0.0, 1.0, parameter_names=["param1"])
        prior2 = UniformPrior(10.0, 20.0, parameter_names=["param2"])

        combined = CombinePrior([prior1, prior2])

        assert combined.n_dim == 2
        assert combined.parameter_names == ["param1", "param2"]

    def test_combine_prior_sample(self):
        """Test sampling from CombinePrior."""
        prior1 = UniformPrior(0.0, 1.0, parameter_names=["param1"])
        prior2 = UniformPrior(10.0, 20.0, parameter_names=["param2"])

        combined = CombinePrior([prior1, prior2])

        # Sample using JAX PRNGKey
        rng_key = jax.random.PRNGKey(42)
        samples = combined.sample(rng_key, n_samples=10)

        assert isinstance(samples, dict)
        assert "param1" in samples
        assert "param2" in samples

        # Check values are in expected ranges (arrays of shape [10])
        assert jnp.all(samples["param1"] >= 0.0) and jnp.all(samples["param1"] <= 1.0)
        assert jnp.all(samples["param2"] >= 10.0) and jnp.all(samples["param2"] <= 20.0)

    def test_combine_prior_log_prob(self):
        """Test log probability calculation for CombinePrior.

        NOTE: Current implementation returns NaN for out-of-bounds samples
        instead of -inf. This may be a design choice from Jim/jimgw.
        """
        prior1 = UniformPrior(0.0, 1.0, parameter_names=["param1"])
        prior2 = UniformPrior(10.0, 20.0, parameter_names=["param2"])

        combined = CombinePrior([prior1, prior2])

        # Valid sample - should have finite log prob
        valid_sample = {"param1": 0.5, "param2": 15.0}
        log_prob = combined.log_prob(valid_sample)
        assert jnp.isfinite(log_prob)

        # Invalid sample (out of bounds) - returns NaN in current implementation
        invalid_sample = {"param1": 2.0, "param2": 15.0}  # param1 out of bounds
        log_prob_invalid = combined.log_prob(invalid_sample)
        # Accept either NaN or -inf as both indicate invalid
        assert jnp.isnan(log_prob_invalid) or log_prob_invalid == -jnp.inf


class TestUniformPrior:
    """Test UniformPrior functionality."""

    def test_uniform_prior_basic(self):
        """Test basic UniformPrior creation and properties."""
        prior = UniformPrior(0.0, 10.0, parameter_names=["test_param"])

        assert prior.parameter_names == ["test_param"]
        assert prior.n_dim == 1

    def test_uniform_prior_sample(self):
        """Test sampling from UniformPrior."""
        prior = UniformPrior(0.0, 10.0, parameter_names=["test_param"])

        # Sample using JAX PRNGKey
        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=100)

        # Check that samples are dict with correct key
        assert isinstance(samples, dict)
        assert "test_param" in samples

        # Check values are in expected range [0, 10]
        values = samples["test_param"]
        assert jnp.all(values >= 0.0) and jnp.all(values <= 10.0)

        # Check that we get reasonable coverage (not all same value)
        assert jnp.std(values) > 0.1  # Should have some variance

    def test_uniform_prior_log_prob(self):
        """Test log probability for UniformPrior.

        NOTE: Current implementation returns NaN for out-of-bounds samples
        instead of -inf. This may be a design choice from Jim/jimgw.
        """
        prior = UniformPrior(0.0, 10.0, parameter_names=["test_param"])

        # In bounds - should be log(1/10) = -log(10)
        in_bounds = {"test_param": 5.0}
        log_prob = prior.log_prob(in_bounds)
        assert jnp.isfinite(log_prob)
        assert log_prob == pytest.approx(-jnp.log(10.0), abs=1e-6)

        # Out of bounds - returns NaN in current implementation
        out_of_bounds_low = {"test_param": -1.0}
        out_of_bounds_high = {"test_param": 11.0}

        # Accept either NaN or -inf as both indicate invalid
        log_prob_low = prior.log_prob(out_of_bounds_low)
        log_prob_high = prior.log_prob(out_of_bounds_high)
        assert jnp.isnan(log_prob_low) or log_prob_low == -jnp.inf
        assert jnp.isnan(log_prob_high) or log_prob_high == -jnp.inf

    def test_uniform_prior_boundaries(self):
        """Test that boundary values are handled correctly.

        NOTE: Exact boundary values cause numerical issues with the logistic transform:
        - xmin (0.0) returns NaN
        - xmax (10.0) causes ZeroDivisionError in logit transform
        This is a known issue - always use values strictly inside the boundaries.
        """
        prior = UniformPrior(0.0, 10.0, parameter_names=["test_param"])

        # xmin boundary returns NaN
        at_min = {"test_param": 0.0}
        assert jnp.isnan(prior.log_prob(at_min))

        # xmax boundary causes ZeroDivisionError in logit transform
        at_max = {"test_param": 10.0}
        with pytest.raises(ZeroDivisionError):
            prior.log_prob(at_max)

        # Slightly inside boundaries should be valid
        slightly_above_min = {"test_param": 0.01}
        slightly_below_max = {"test_param": 9.99}

        assert jnp.isfinite(prior.log_prob(slightly_above_min))
        assert jnp.isfinite(prior.log_prob(slightly_below_max))


class TestPriorIntegration:
    """Integration tests for prior system."""

    def test_prior_sampling_deterministic(self, sample_prior_file):
        """Test that prior sampling is deterministic given same PRNGKey."""
        result = parser.parse_prior_file(sample_prior_file, nb_CSE=0)
        prior = result.prior

        # Same RNG key should give same output
        rng_key = jax.random.PRNGKey(42)
        samples1 = prior.sample(rng_key, n_samples=5)
        samples2 = prior.sample(rng_key, n_samples=5)

        # Check all parameters are identical
        for key in samples1.keys():
            assert jnp.allclose(samples1[key], samples2[key], atol=1e-10)

    def test_prior_samples_cover_full_range(self, sample_prior_file):
        """Test that sampling covers the parameter range."""
        result = parser.parse_prior_file(sample_prior_file, nb_CSE=0)
        prior = result.prior

        # Sample multiple times to get coverage
        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=100)

        # Check that each parameter has some variance (not all same value)
        for key in samples.keys():
            values = samples[key]
            assert jnp.std(values) > 0.01, f"Parameter {key} not varying enough"

    def test_parsed_prior_log_prob_valid_samples(self, sample_prior_file):
        """Test that log_prob is finite for valid samples from parsed prior."""
        result = parser.parse_prior_file(sample_prior_file, nb_CSE=0)
        prior = result.prior

        # Generate valid samples
        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=5)

        # Convert first sample to dict format for log_prob
        # (samples is dict of arrays, need dict of scalars)
        first_sample = {key: values[0] for key, values in samples.items()}

        log_prob = prior.log_prob(first_sample)
        assert jnp.isfinite(log_prob)

    def test_realistic_nep_prior_ranges(self, temp_dir):
        """Test realistic NEP parameter ranges match physical expectations.

        This test ensures that prior specifications for NEP parameters
        are physically reasonable for neutron star EOS inference.
        """
        realistic_prior = temp_dir / "realistic.prior"
        realistic_prior.write_text(
            """
# Saturation density parameters (MeV)
E_sat = UniformPrior(-17.0, -15.0, parameter_names=["E_sat"])
K_sat = UniformPrior(200.0, 280.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 500.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-1000.0, 1000.0, parameter_names=["Z_sat"])

# Symmetry energy parameters (MeV)
E_sym = UniformPrior(28.0, 36.0, parameter_names=["E_sym"])
L_sym = UniformPrior(40.0, 120.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-200.0, 100.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-500.0, 500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-1000.0, 1000.0, parameter_names=["Z_sym"])
"""
        )

        result = parser.parse_prior_file(realistic_prior, nb_CSE=0)
        prior = result.prior

        # Sample and check values are in physically reasonable ranges
        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=10)

        # Get first sample as dict of scalars
        first_sample = {key: values[0] for key, values in samples.items()}

        # Check saturation energy is negative (bound nucleus)
        assert first_sample["E_sat"] < 0

        # Check incompressibility is positive (stable matter)
        assert first_sample["K_sat"] > 0

        # Check symmetry energy is positive
        assert first_sample["E_sym"] > 0
