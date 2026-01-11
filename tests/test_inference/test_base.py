"""Tests for inference base classes (priors, transforms, likelihoods)."""

import pytest
import jax
import jax.numpy as jnp

from jesterTOV.inference.base import (
    Prior,
    UniformPrior,
    CombinePrior,
    LikelihoodBase,
)
from jesterTOV.inference.base.prior import (
    LogisticDistribution,
    SequentialTransformPrior,
)
from jesterTOV.inference.base.transform import (
    Transform,
    NtoMTransform,
    ScaleTransform,
    OffsetTransform,
    LogitTransform,
)


class TestPriorBase:
    """Test Prior base class functionality."""

    def test_prior_initialization(self):
        """Test Prior base class initialization."""
        prior = Prior(parameter_names=["param1", "param2"])

        assert prior.parameter_names == ["param1", "param2"]
        assert prior.n_dim == 2
        assert prior.composite is False

    def test_prior_add_name(self):
        """Test Prior.add_name converts array to dict."""
        prior = Prior(parameter_names=["mass", "radius"])

        # Create array of values
        values = jnp.array([1.4, 12.0])

        # Convert to dict
        result = prior.add_name(values)

        assert isinstance(result, dict)
        assert result["mass"] == 1.4
        assert result["radius"] == 12.0

    def test_prior_sample_not_implemented(self):
        """Test that Prior.sample raises NotImplementedError."""
        prior = Prior(parameter_names=["param1"])

        with pytest.raises(NotImplementedError):
            prior.sample(jax.random.PRNGKey(42), 10)

    def test_prior_log_prob_not_implemented(self):
        """Test that Prior.log_prob raises NotImplementedError."""
        prior = Prior(parameter_names=["param1"])

        with pytest.raises(NotImplementedError):
            prior.log_prob({"param1": 0.5})


class TestLogisticDistribution:
    """Test LogisticDistribution prior."""

    def test_logistic_initialization(self):
        """Test LogisticDistribution initializes correctly."""
        prior = LogisticDistribution(parameter_names=["x"])

        assert prior.n_dim == 1
        assert prior.composite is False

    def test_logistic_only_1d(self):
        """Test that LogisticDistribution requires 1D."""
        with pytest.raises(AssertionError, match="needs to be 1D"):
            LogisticDistribution(parameter_names=["x", "y"])

    def test_logistic_sample(self):
        """Test LogisticDistribution sampling."""
        prior = LogisticDistribution(parameter_names=["x"])

        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=100)

        # Should return dict with key "x"
        assert isinstance(samples, dict)
        assert "x" in samples

        # Samples should be from logistic distribution (unbounded)
        # Most samples should be in reasonable range (say, -10 to 10)
        # but some can be extreme
        assert jnp.isfinite(samples["x"]).all()

    def test_logistic_log_prob(self):
        """Test LogisticDistribution log probability."""
        prior = LogisticDistribution(parameter_names=["x"])

        # Evaluate at x=0 (mode of logistic distribution)
        log_prob = prior.log_prob({"x": 0.0})

        # Should be finite
        assert jnp.isfinite(log_prob)

        # Log prob at x=0: -0 - 2*log(1 + exp(0)) = -2*log(2) ≈ -1.386
        assert log_prob == pytest.approx(-2 * jnp.log(2.0), abs=1e-6)


class TestTransformBase:
    """Test Transform base class functionality."""

    def test_transform_initialization(self):
        """Test Transform initialization with name mapping."""
        name_mapping = (["x"], ["y"])
        transform = Transform(name_mapping)

        assert transform.name_mapping == name_mapping

    def test_transform_propagate_name_simple(self):
        """Test propagate_name with simple transform."""
        name_mapping = (["x"], ["y"])
        transform = Transform(name_mapping)

        # Start with ["x", "z"]
        input_names = ["x", "z"]

        # After transform: remove "x", add "y" → ["y", "z"]
        output_names = transform.propagate_name(input_names)

        assert set(output_names) == {"y", "z"}

    def test_transform_propagate_name_multiple(self):
        """Test propagate_name with multiple parameters."""
        name_mapping = (["x", "y"], ["u", "v"])
        transform = Transform(name_mapping)

        # Start with ["x", "y", "z"]
        input_names = ["x", "y", "z"]

        # After transform: remove ["x", "y"], add ["u", "v"] → ["u", "v", "z"]
        output_names = transform.propagate_name(input_names)

        assert set(output_names) == {"u", "v", "z"}


class TestScaleTransform:
    """Test ScaleTransform functionality."""

    def test_scale_transform_forward(self):
        """Test ScaleTransform forward transformation."""
        name_mapping = (["x"], ["y"])
        scale = 2.0
        transform = ScaleTransform(name_mapping, scale)

        # Forward: y = x * scale
        params = {"x": 5.0}
        result = transform.forward(params)

        assert "y" in result
        assert result["y"] == 10.0
        assert "x" not in result  # Original param removed

    def test_scale_transform_inverse(self):
        """Test ScaleTransform inverse transformation."""
        name_mapping = (["x"], ["y"])
        scale = 2.0
        transform = ScaleTransform(name_mapping, scale)

        # Inverse: x = y / scale
        params = {"y": 10.0}
        result, log_jacobian = transform.inverse(params)

        assert "x" in result
        assert result["x"] == 5.0
        assert "y" not in result  # Transformed param removed

        # Jacobian for scale transform: log|1/scale|
        assert log_jacobian == pytest.approx(jnp.log(1.0 / scale), abs=1e-6)

    def test_scale_transform_roundtrip(self):
        """Test that forward + inverse gives original value."""
        name_mapping = (["x"], ["y"])
        scale = 3.5
        transform = ScaleTransform(name_mapping, scale)

        # Start with x=7.0
        original = {"x": 7.0}

        # Forward
        forward_result = transform.forward(original)

        # Inverse
        inverse_result, _ = transform.inverse(forward_result)

        # Should get back original value
        assert inverse_result["x"] == pytest.approx(7.0, abs=1e-6)


class TestOffsetTransform:
    """Test OffsetTransform functionality."""

    def test_offset_transform_forward(self):
        """Test OffsetTransform forward transformation."""
        name_mapping = (["x"], ["y"])
        offset = 3.0
        transform = OffsetTransform(name_mapping, offset)

        # Forward: y = x + offset
        params = {"x": 5.0}
        result = transform.forward(params)

        assert "y" in result
        assert result["y"] == 8.0

    def test_offset_transform_inverse(self):
        """Test OffsetTransform inverse transformation."""
        name_mapping = (["x"], ["y"])
        offset = 3.0
        transform = OffsetTransform(name_mapping, offset)

        # Inverse: x = y - offset
        params = {"y": 8.0}
        result, log_jacobian = transform.inverse(params)

        assert "x" in result
        assert result["x"] == 5.0

        # Jacobian for offset transform: log|1| = 0
        assert log_jacobian == 0.0

    def test_offset_transform_roundtrip(self):
        """Test that forward + inverse gives original value."""
        name_mapping = (["x"], ["y"])
        offset = -2.5
        transform = OffsetTransform(name_mapping, offset)

        original = {"x": 10.0}
        forward_result = transform.forward(original)
        inverse_result, _ = transform.inverse(forward_result)

        assert inverse_result["x"] == pytest.approx(10.0, abs=1e-6)


class TestLogitTransform:
    """Test LogitTransform functionality."""

    def test_logit_transform_forward(self):
        """Test LogitTransform forward: y = 1 / (1 + exp(-x))."""
        name_mapping = (["x"], ["y"])
        transform = LogitTransform(name_mapping)

        # At x=0, y = 1/(1+1) = 0.5
        params = {"x": 0.0}
        result = transform.forward(params)

        assert "y" in result
        assert result["y"] == pytest.approx(0.5, abs=1e-6)

    def test_logit_transform_inverse(self):
        """Test LogitTransform inverse: x = log(y / (1-y))."""
        name_mapping = (["x"], ["y"])
        transform = LogitTransform(name_mapping)

        # At y=0.5, x = log(0.5/0.5) = 0
        params = {"y": 0.5}
        result, log_jacobian = transform.inverse(params)

        assert "x" in result
        assert result["x"] == pytest.approx(0.0, abs=1e-6)

        # Jacobian should be finite
        assert jnp.isfinite(log_jacobian)

    def test_logit_transform_boundary_issues(self):
        """Test LogitTransform at boundaries (known issue).

        NOTE: This documents the boundary issue found in prior tests.
        LogitTransform is undefined at y=0 and y=1 (division by zero).
        """
        name_mapping = (["x"], ["y"])
        transform = LogitTransform(name_mapping)

        # At y=1, inverse transform: log(1/(1-1)) = log(1/0) → division by zero
        params_at_max = {"y": 1.0}
        with pytest.raises(ZeroDivisionError):
            transform.inverse(params_at_max)

        # At y=0, inverse transform: log(0/1) = log(0) → -inf
        params_at_min = {"y": 0.0}
        result, _ = transform.inverse(params_at_min)
        # log(0/(1-0)) = -inf
        assert result["x"] == -jnp.inf

    def test_logit_transform_roundtrip(self):
        """Test that forward + inverse gives original value (away from boundaries)."""
        name_mapping = (["x"], ["y"])
        transform = LogitTransform(name_mapping)

        # Use value away from boundaries
        original = {"x": 2.5}
        forward_result = transform.forward(original)
        inverse_result, _ = transform.inverse(forward_result)

        assert inverse_result["x"] == pytest.approx(2.5, abs=1e-6)


class TestSequentialTransformPrior:
    """Test SequentialTransformPrior for chaining transforms."""

    def test_sequential_transform_initialization(self):
        """Test SequentialTransformPrior initialization."""
        base_prior = LogisticDistribution(parameter_names=["x_base"])
        transform = ScaleTransform((["x_base"], ["x"]), scale=2.0)

        sequential = SequentialTransformPrior(base_prior, [transform])

        assert sequential.composite is True
        assert "x" in sequential.parameter_names
        assert "x_base" not in sequential.parameter_names

    def test_sequential_transform_sample(self):
        """Test sampling from SequentialTransformPrior."""
        base_prior = LogisticDistribution(parameter_names=["x_base"])
        transform = ScaleTransform((["x_base"], ["x"]), scale=2.0)

        sequential = SequentialTransformPrior(base_prior, [transform])

        rng_key = jax.random.PRNGKey(42)
        samples = sequential.sample(rng_key, n_samples=10)

        # Should have transformed parameter "x", not "x_base"
        assert "x" in samples
        assert "x_base" not in samples

    def test_sequential_transform_log_prob(self):
        """Test log_prob with Jacobian correction."""
        base_prior = LogisticDistribution(parameter_names=["x_base"])
        transform = ScaleTransform((["x_base"], ["x"]), scale=2.0)

        sequential = SequentialTransformPrior(base_prior, [transform])

        # Evaluate at transformed value
        params = {"x": 4.0}  # x_base would be 2.0
        log_prob = sequential.log_prob(params)

        # Should include Jacobian correction
        assert jnp.isfinite(log_prob)

    def test_sequential_multiple_transforms(self):
        """Test chaining multiple transforms."""
        base_prior = LogisticDistribution(parameter_names=["z"])

        # Chain: z → y (scale by 2) → x (offset by 3)
        transforms = [
            ScaleTransform((["z"], ["y"]), scale=2.0),
            OffsetTransform((["y"], ["x"]), offset=3.0),
        ]

        sequential = SequentialTransformPrior(base_prior, transforms)

        # Final parameter should be "x"
        assert "x" in sequential.parameter_names
        assert "y" not in sequential.parameter_names
        assert "z" not in sequential.parameter_names


class TestUniformPriorComposition:
    """Test UniformPrior internal composition of transforms."""

    def test_uniform_prior_is_sequential_transform(self):
        """Test that UniformPrior is implemented as SequentialTransformPrior."""
        prior = UniformPrior(0.0, 10.0, parameter_names=["x"])

        # Should be a SequentialTransformPrior
        assert isinstance(prior, SequentialTransformPrior)
        assert prior.composite is True

    def test_uniform_prior_has_three_transforms(self):
        """Test that UniformPrior uses Logit + Scale + Offset transforms."""
        prior = UniformPrior(0.0, 10.0, parameter_names=["x"])

        # Should have 3 transforms: Logit, Scale, Offset
        assert len(prior.transforms) == 3
        assert isinstance(prior.transforms[0], LogitTransform)
        assert isinstance(prior.transforms[1], ScaleTransform)
        assert isinstance(prior.transforms[2], OffsetTransform)

    def test_uniform_prior_transform_scales_correctly(self):
        """Test that UniformPrior scale transform has correct scale."""
        prior = UniformPrior(5.0, 15.0, parameter_names=["x"])

        # Scale should be xmax - xmin = 10.0
        scale_transform = prior.transforms[1]
        assert scale_transform.scale == 10.0

    def test_uniform_prior_transform_offsets_correctly(self):
        """Test that UniformPrior offset transform has correct offset."""
        prior = UniformPrior(5.0, 15.0, parameter_names=["x"])

        # Offset should be xmin = 5.0
        offset_transform = prior.transforms[2]
        assert offset_transform.offset == 5.0


class TestCombinePriorBase:
    """Test CombinePrior base functionality (already tested in test_priors.py, but verify base properties)."""

    def test_combine_prior_is_composite(self):
        """Test that CombinePrior has composite=True."""
        prior1 = UniformPrior(0.0, 1.0, parameter_names=["x"])
        prior2 = UniformPrior(10.0, 20.0, parameter_names=["y"])

        combined = CombinePrior([prior1, prior2])

        assert combined.composite is True

    def test_combine_prior_concatenates_names(self):
        """Test that CombinePrior concatenates parameter names."""
        prior1 = UniformPrior(0.0, 1.0, parameter_names=["x"])
        prior2 = UniformPrior(10.0, 20.0, parameter_names=["y"])
        prior3 = UniformPrior(-5.0, 5.0, parameter_names=["z"])

        combined = CombinePrior([prior1, prior2, prior3])

        assert combined.parameter_names == ["x", "y", "z"]
        assert combined.n_dim == 3


class TestLikelihoodBase:
    """Test LikelihoodBase abstract interface."""

    def test_likelihood_base_is_abstract(self):
        """Test that LikelihoodBase cannot be instantiated directly."""
        # LikelihoodBase is abstract - should not be able to create instance
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LikelihoodBase()

    def test_likelihood_base_requires_evaluate(self):
        """Test that subclasses must implement evaluate method."""

        # Create a minimal subclass without evaluate
        class IncompleteLikelihood(LikelihoodBase):
            pass

        # Should fail to instantiate because evaluate is not implemented
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteLikelihood()

    def test_likelihood_base_with_evaluate(self):
        """Test that LikelihoodBase can be subclassed with evaluate."""

        # Create a complete subclass
        class CompleteLikelihood(LikelihoodBase):
            def evaluate(self, params):
                return 0.0

        # Should succeed
        likelihood = CompleteLikelihood()
        assert likelihood.evaluate({}) == 0.0


class TestNtoMTransform:
    """Test NtoMTransform (N-to-M parameter mapping)."""

    def test_ntom_transform_forward(self):
        """Test NtoMTransform forward transformation."""
        # Create transform that doubles x and adds constant
        name_mapping = (["x"], ["y", "z"])

        def transform_func(params):
            return {"y": params["x"] * 2, "z": params["x"] + 1}

        transform = NtoMTransform(name_mapping)
        transform.transform_func = transform_func

        # Test forward
        params = {"x": 5.0, "other": 10.0}
        result = transform.forward(params)

        # Should have y and z, not x
        assert "y" in result
        assert "z" in result
        assert "x" not in result
        assert "other" in result  # Other params preserved

        assert result["y"] == 10.0
        assert result["z"] == 6.0

    def test_ntom_transform_preserves_unrelated_params(self):
        """Test that NtoMTransform preserves parameters not in name_mapping."""
        name_mapping = (["x"], ["y"])

        def transform_func(params):
            return {"y": params["x"] * 2}

        transform = NtoMTransform(name_mapping)
        transform.transform_func = transform_func

        # Include extra parameters
        params = {"x": 5.0, "a": 1.0, "b": 2.0}
        result = transform.forward(params)

        # a and b should be preserved
        assert result["a"] == 1.0
        assert result["b"] == 2.0


class TestBijectiveTransform:
    """Test BijectiveTransform (invertible N-to-N mapping)."""

    def test_bijective_transform_has_inverse(self):
        """Test that BijectiveTransform requires inverse_transform_func."""
        # ScaleTransform is a BijectiveTransform
        transform = ScaleTransform((["x"], ["y"]), scale=2.0)

        # Should have both forward and inverse functions
        assert hasattr(transform, "transform_func")
        assert hasattr(transform, "inverse_transform_func")

    def test_bijective_transform_backward(self):
        """Test BijectiveTransform.backward (inverse without Jacobian)."""
        transform = ScaleTransform((["x"], ["y"]), scale=2.0)

        # Backward should invert without Jacobian
        params = {"y": 10.0}
        result = transform.backward(params)

        assert "x" in result
        assert result["x"] == 5.0
        # backward returns dict, not tuple


class TestTransformIntegration:
    """Integration tests for transform system."""

    def test_logit_scale_offset_chain(self):
        """Test the full chain used by UniformPrior: Logit → Scale → Offset."""
        # This is what UniformPrior does internally for [0, 10]
        base_param = "x_base"
        final_param = "x"

        # Logit: x_base → u (in [0, 1])
        logit = LogitTransform(([base_param], ["u"]))

        # Scale: u → v (in [0, 10])
        scale = ScaleTransform((["u"], ["v"]), scale=10.0)

        # Offset: v → x (in [0, 10] but could be shifted)
        offset = OffsetTransform((["v"], [final_param]), offset=0.0)

        # Chain them
        transforms = [logit, scale, offset]

        # Start with base value (logistic)
        base_value = 0.0  # logit(0) = 0.5

        params = {base_param: base_value}

        # Apply transforms
        for t in transforms:
            params = t.forward(params)

        # Should end up at x ≈ 5.0 (since logit(0) = 0.5, scale by 10 = 5.0)
        assert params[final_param] == pytest.approx(5.0, abs=1e-6)

    def test_uniform_prior_samples_in_bounds(self):
        """Test that UniformPrior samples respect bounds (integration test)."""
        prior = UniformPrior(2.0, 8.0, parameter_names=["x"])

        rng_key = jax.random.PRNGKey(42)
        samples = prior.sample(rng_key, n_samples=1000)

        # All samples should be in [2, 8]
        assert jnp.all(samples["x"] >= 2.0)
        assert jnp.all(samples["x"] <= 8.0)

        # Mean should be approximately 5.0 (center of [2, 8])
        mean = jnp.mean(samples["x"])
        assert 4.5 < mean < 5.5  # Allow some variance

        # Standard deviation for uniform on [a, b]: (b-a)/sqrt(12) ≈ 1.73
        std = jnp.std(samples["x"])
        expected_std = (8.0 - 2.0) / jnp.sqrt(12.0)
        assert 0.8 * expected_std < std < 1.2 * expected_std
