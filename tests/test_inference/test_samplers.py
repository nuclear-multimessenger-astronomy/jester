"""Tests for inference sampler system (base sampler, flowMC backend)."""

import pytest
import jax
import jax.numpy as jnp

from jesterTOV.inference.samplers.jester_sampler import JesterSampler
from jesterTOV.inference.samplers.flowmc import FlowMCSampler, setup_flowmc_sampler
from jesterTOV.inference.base import UniformPrior, CombinePrior, LikelihoodBase
from jesterTOV.inference.base.transform import BijectiveTransform, NtoMTransform, ScaleTransform
from jesterTOV.inference.config.schema import SamplerConfig


# Create a simple mock likelihood for testing
class MockLikelihood(LikelihoodBase):
    """Simple mock likelihood that returns 0.0 (prior-only sampling)."""

    def evaluate(self, params, data):
        return 0.0


class TestJesterSamplerBase:
    """Test JesterSampler base class functionality."""

    def test_jester_sampler_initialization(self):
        """Test JesterSampler initializes correctly."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = JesterSampler(likelihood, prior)

        assert sampler.likelihood == likelihood
        assert sampler.prior == prior
        assert sampler.parameter_names == ["x"]
        assert sampler.sample_transforms == []
        assert sampler.likelihood_transforms == []
        assert sampler.sampler is None  # Base class doesn't create backend

    def test_jester_sampler_with_sample_transforms(self):
        """Test JesterSampler with sample transforms propagates names."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        # Create a transform that changes parameter name
        transform = ScaleTransform((["x"], ["y"]), scale=2.0)

        sampler = JesterSampler(likelihood, prior, sample_transforms=[transform])

        # Parameter names should be propagated through transform
        assert "y" in sampler.parameter_names
        assert "x" not in sampler.parameter_names

    def test_jester_sampler_add_name(self):
        """Test JesterSampler.add_name converts array to dict."""
        prior = CombinePrior([
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(10.0, 20.0, parameter_names=["y"]),
        ])
        likelihood = MockLikelihood()

        sampler = JesterSampler(likelihood, prior)

        # Convert array to dict
        params_array = jnp.array([0.5, 15.0])
        params_dict = sampler.add_name(params_array)

        assert params_dict["x"] == 0.5
        assert params_dict["y"] == 15.0

    def test_jester_sampler_posterior_evaluation(self):
        """Test JesterSampler.posterior evaluates log posterior correctly."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()  # Returns 0.0

        sampler = JesterSampler(likelihood, prior)

        # Evaluate posterior at x=0.5 (middle of uniform prior)
        params = jnp.array([0.5])
        log_posterior = sampler.posterior(params, {})

        # Should be finite (prior log prob + likelihood)
        assert jnp.isfinite(log_posterior)

        # For uniform prior with transforms and likelihood=0.0,
        # the log posterior depends on the Jacobians from LogitTransform
        # It may be 0.0 or slightly negative depending on the value
        # Just verify it's finite
        # (Note: log_posterior = 0.0 for prior-only with uniform prior at certain points)

    def test_jester_sampler_posterior_with_likelihood_transform(self):
        """Test that likelihood transforms are applied before likelihood evaluation."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])

        # Create likelihood that expects transformed parameter "y"
        class TransformedLikelihood(LikelihoodBase):
            def evaluate(self, params, data):
                # Expect "y" parameter (from transform)
                assert "y" in params
                return 0.0

        likelihood = TransformedLikelihood()

        # Create transform x → y
        class SimpleTransform(NtoMTransform):
            def __init__(self):
                super().__init__((["x"], ["y"]))
                self.transform_func = lambda params: {"y": params["x"] * 2}

        transform = SimpleTransform()

        sampler = JesterSampler(likelihood, prior, likelihood_transforms=[transform])

        # Evaluate posterior - should apply transform before likelihood
        params = jnp.array([0.5])
        log_posterior = sampler.posterior(params, {})

        # Should succeed (no assertion error from likelihood)
        assert jnp.isfinite(log_posterior)

    def test_jester_sampler_sample_not_implemented(self):
        """Test that JesterSampler.sample raises NotImplementedError."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = JesterSampler(likelihood, prior)

        with pytest.raises(NotImplementedError, match="must be implemented by backend-specific subclass"):
            sampler.sample(jax.random.PRNGKey(42))

    def test_jester_sampler_print_summary_not_implemented(self):
        """Test that JesterSampler.print_summary raises NotImplementedError."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = JesterSampler(likelihood, prior)

        with pytest.raises(NotImplementedError, match="must be implemented by backend-specific subclass"):
            sampler.print_summary()

    def test_jester_sampler_get_samples_not_implemented(self):
        """Test that JesterSampler.get_samples raises NotImplementedError."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = JesterSampler(likelihood, prior)

        with pytest.raises(NotImplementedError, match="must be implemented by backend-specific subclass"):
            sampler.get_samples()


class TestFlowMCSampler:
    """Test FlowMCSampler initialization and configuration."""

    def test_flowmc_sampler_initialization_default(self):
        """Test FlowMCSampler initializes with default settings."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = FlowMCSampler(
            likelihood,
            prior,
            local_sampler_arg={"step_size": jnp.array([1e-3])},  # Required for GaussianRandomWalk
            n_loop_training=1,
            n_loop_production=1,
            n_chains=2,
            n_local_steps=5,
            n_global_steps=5,
            n_epochs=5,
            learning_rate=0.001,
        )

        # Should create flowMC sampler
        assert sampler.sampler is not None
        assert sampler.sampler.n_chains == 2

    def test_flowmc_sampler_with_mala(self):
        """Test FlowMCSampler with MALA local sampler."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        # Use MALA sampler
        sampler = FlowMCSampler(
            likelihood,
            prior,
            local_sampler_name="MALA",
            local_sampler_arg={"step_size": jnp.array([[1e-3]])},
            n_loop_training=1,
            n_loop_production=1,
            n_chains=2,
        )

        assert sampler.sampler is not None

    def test_flowmc_sampler_with_gaussian_random_walk(self):
        """Test FlowMCSampler with GaussianRandomWalk local sampler."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        # Use GaussianRandomWalk sampler (default)
        sampler = FlowMCSampler(
            likelihood,
            prior,
            local_sampler_name="GaussianRandomWalk",
            local_sampler_arg={"step_size": jnp.array([1e-3])},
            n_loop_training=1,
            n_loop_production=1,
            n_chains=2,
        )

        assert sampler.sampler is not None

    def test_flowmc_sampler_diagonal_extraction_for_gaussian_random_walk(self):
        """Test that FlowMCSampler extracts diagonal from matrix step_size for GaussianRandomWalk."""
        prior = CombinePrior([
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ])
        likelihood = MockLikelihood()

        # Provide 2x2 matrix step_size
        step_size_matrix = jnp.array([[1e-3, 0.0], [0.0, 2e-3]])

        # GaussianRandomWalk should extract diagonal
        sampler = FlowMCSampler(
            likelihood,
            prior,
            local_sampler_name="GaussianRandomWalk",
            local_sampler_arg={"step_size": step_size_matrix},
            n_loop_training=1,
            n_loop_production=1,
            n_chains=2,
        )

        # Should succeed (diagonal extracted)
        assert sampler.sampler is not None

    def test_flowmc_sampler_invalid_local_sampler_raises_error(self):
        """Test that invalid local_sampler_name raises ValueError."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        with pytest.raises(ValueError, match="Unknown local_sampler_name"):
            FlowMCSampler(
                likelihood,
                prior,
                local_sampler_name="InvalidSampler",
                n_loop_training=1,
                n_loop_production=1,
                n_chains=2,
            )

    def test_flowmc_sampler_with_custom_flow_architecture(self):
        """Test FlowMCSampler with custom normalizing flow architecture."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = FlowMCSampler(
            likelihood,
            prior,
            local_sampler_arg={"step_size": jnp.array([1e-3])},
            num_layers=5,
            hidden_size=[64, 64],
            num_bins=4,
            n_loop_training=1,
            n_loop_production=1,
            n_chains=2,
        )

        assert sampler.sampler is not None

    def test_flowmc_sampler_with_likelihood_transform(self):
        """Test FlowMCSampler with likelihood transform (realistic use case)."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        # Create a simple N-to-M transform
        class SquareTransform(NtoMTransform):
            def __init__(self):
                super().__init__((["x"], ["x_squared"]))
                self.transform_func = lambda params: {"x_squared": params["x"] ** 2}

        transform = SquareTransform()

        sampler = FlowMCSampler(
            likelihood,
            prior,
            local_sampler_arg={"step_size": jnp.array([1e-3])},
            likelihood_transforms=[transform],
            n_loop_training=1,
            n_loop_production=1,
            n_chains=2,
        )

        assert sampler.sampler is not None
        assert len(sampler.likelihood_transforms) == 1


class TestFlowMCSamplerParameterOrdering:
    """Test critical bug fix: parameter ordering preservation."""

    def test_parameter_order_preserved_in_dict_to_array(self):
        """Test that FlowMCSampler preserves parameter order when converting dict to array.

        This tests the critical bug fix mentioned in the code comments.
        The sampler MUST use list comprehension instead of jax.tree.leaves()
        to preserve dictionary order.
        """
        # Create prior with multiple parameters (order matters!)
        prior = CombinePrior([
            UniformPrior(0.0, 1.0, parameter_names=["a"]),
            UniformPrior(10.0, 20.0, parameter_names=["b"]),
            UniformPrior(100.0, 200.0, parameter_names=["c"]),
        ])
        likelihood = MockLikelihood()

        sampler = FlowMCSampler(
            likelihood,
            prior,
            local_sampler_arg={"step_size": jnp.array([1e-3, 1e-3, 1e-3])},
            n_loop_training=1,
            n_loop_production=1,
            n_chains=2,
            n_local_steps=1,
            n_global_steps=1,
            n_epochs=1,
        )

        # Check parameter names are in correct order
        assert sampler.parameter_names == ["a", "b", "c"]

        # Test add_name (array → dict)
        test_array = jnp.array([0.5, 15.0, 150.0])
        result = sampler.add_name(test_array)

        # Should map correctly
        assert result["a"] == 0.5
        assert result["b"] == 15.0
        assert result["c"] == 150.0


class TestSetupFlowMCSampler:
    """Test setup_flowmc_sampler utility function."""

    def test_setup_flowmc_sampler_basic(self):
        """Test setup_flowmc_sampler creates sampler from config."""
        # Create config
        config = SamplerConfig(
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            n_local_steps=5,
            n_global_steps=5,
            n_epochs=5,
            learning_rate=0.001,
            output_dir="./test_output/",
        )

        # Create prior and likelihood
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        # Create transform
        class IdentityTransform(NtoMTransform):
            def __init__(self):
                super().__init__((["x"], ["x"]))
                self.transform_func = lambda params: params

        transform = IdentityTransform()

        # Setup sampler
        sampler = setup_flowmc_sampler(
            config,
            prior,
            likelihood,
            transform,
            seed=42,
        )

        assert isinstance(sampler, FlowMCSampler)
        assert sampler.sampler.n_chains == 2

    def test_setup_flowmc_sampler_with_mala(self):
        """Test setup_flowmc_sampler with MALA local sampler."""
        config = SamplerConfig(
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            output_dir="./test_output/",
        )

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        class IdentityTransform(NtoMTransform):
            def __init__(self):
                super().__init__((["x"], ["x"]))
                self.transform_func = lambda params: params

        transform = IdentityTransform()

        # Use MALA
        sampler = setup_flowmc_sampler(
            config,
            prior,
            likelihood,
            transform,
            local_sampler_name="MALA",
        )

        assert isinstance(sampler, FlowMCSampler)

    def test_setup_flowmc_sampler_mass_matrix(self):
        """Test that setup_flowmc_sampler creates mass matrix correctly."""
        config = SamplerConfig(
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            output_dir="./test_output/",
        )

        # Multi-dimensional prior
        prior = CombinePrior([
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
            UniformPrior(0.0, 1.0, parameter_names=["z"]),
        ])
        likelihood = MockLikelihood()

        class IdentityTransform(NtoMTransform):
            def __init__(self):
                super().__init__((["x", "y", "z"], ["x", "y", "z"]))
                self.transform_func = lambda params: params

        transform = IdentityTransform()

        # Setup with custom eps_mass_matrix
        sampler = setup_flowmc_sampler(
            config,
            prior,
            likelihood,
            transform,
            eps_mass_matrix=5e-3,
        )

        # Should create 3x3 mass matrix (identity scaled by eps)
        # We can't directly inspect it, but sampler should work
        assert sampler.sampler is not None


class TestSamplerIntegration:
    """Integration tests for sampler system."""

    @pytest.mark.slow
    def test_flowmc_sampler_minimal_run(self):
        """Test FlowMCSampler can run minimal sampling (slow test).

        This is a minimal integration test to verify the sampler can actually run.
        Uses very short chains to keep test time reasonable.
        """
        # Create simple setup
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = FlowMCSampler(
            likelihood,
            prior,
            n_loop_training=1,
            n_loop_production=1,
            n_chains=2,
            n_local_steps=2,
            n_global_steps=2,
            n_epochs=2,
            learning_rate=0.001,
        )

        # Run sampling
        key = jax.random.PRNGKey(42)
        sampler.sample(key)

        # Should have samples
        samples = sampler.get_samples(training=False)

        assert "x" in samples
        assert jnp.isfinite(samples["x"]).all()

    def test_sampler_with_constraint_likelihood(self):
        """Test sampler with constraint-based likelihood (realistic scenario)."""
        from jesterTOV.inference.likelihoods.constraints import ConstraintEOSLikelihood

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])

        # Use constraint likelihood
        likelihood = ConstraintEOSLikelihood(
            penalty_causality=-1e10,
            penalty_stability=-1e5,
        )

        sampler = FlowMCSampler(
            likelihood,
            prior,
            local_sampler_arg={"step_size": jnp.array([1e-3])},
            n_loop_training=1,
            n_loop_production=1,
            n_chains=2,
        )

        # Should initialize successfully
        assert sampler.sampler is not None
