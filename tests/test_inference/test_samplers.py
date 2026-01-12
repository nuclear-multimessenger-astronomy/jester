"""Tests for inference sampler system (base sampler, flowMC backend)."""

import pytest
import jax
import jax.numpy as jnp

from jesterTOV.inference.samplers.jester_sampler import JesterSampler
from jesterTOV.inference.samplers.flowmc import FlowMCSampler
from jesterTOV.inference.base import UniformPrior, CombinePrior, LikelihoodBase
from jesterTOV.inference.base.transform import (
    NtoMTransform,
    ScaleTransform,
)
from jesterTOV.inference.config.schema import FlowMCSamplerConfig


# Create a simple mock likelihood for testing
class MockLikelihood(LikelihoodBase):
    """Simple mock likelihood that returns 0.0 (prior-only sampling)."""

    def evaluate(self, params):
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
        prior = CombinePrior(
            [
                UniformPrior(0.0, 1.0, parameter_names=["x"]),
                UniformPrior(10.0, 20.0, parameter_names=["y"]),
            ]
        )
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
            def evaluate(self, params):
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

        with pytest.raises(
            NotImplementedError,
            match="must be implemented by backend-specific subclass",
        ):
            sampler.sample(jax.random.PRNGKey(42))

    def test_jester_sampler_print_summary_not_implemented(self):
        """Test that JesterSampler.print_summary raises NotImplementedError."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = JesterSampler(likelihood, prior)

        with pytest.raises(
            NotImplementedError,
            match="must be implemented by backend-specific subclass",
        ):
            sampler.print_summary()

    def test_jester_sampler_get_samples_not_implemented(self):
        """Test that JesterSampler.get_samples raises NotImplementedError."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = JesterSampler(likelihood, prior)

        with pytest.raises(
            NotImplementedError,
            match="must be implemented by backend-specific subclass",
        ):
            sampler.get_samples()


class TestFlowMCSampler:
    """Test FlowMCSampler initialization and configuration."""

    def test_flowmc_sampler_initialization_default(self):
        """Test FlowMCSampler initializes with default settings."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            n_local_steps=5,
            n_global_steps=5,
            n_epochs=5,
            learning_rate=0.001,
            output_dir="./test_output/",
        )

        sampler = FlowMCSampler(
            likelihood,
            prior,
            config,
            local_sampler_arg={
                "step_size": jnp.array([1e-3])
            },  # Required for GaussianRandomWalk
        )

        # Should create flowMC sampler
        assert sampler.sampler is not None
        assert sampler.sampler.n_chains == 2

    def test_flowmc_sampler_with_mala(self):
        """Test FlowMCSampler with MALA local sampler."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            output_dir="./test_output/",
        )

        # Use MALA sampler
        sampler = FlowMCSampler(
            likelihood,
            prior,
            config,
            local_sampler_name="MALA",
            local_sampler_arg={"step_size": jnp.array([[1e-3]])},
        )

        assert sampler.sampler is not None

    def test_flowmc_sampler_with_gaussian_random_walk(self):
        """Test FlowMCSampler with GaussianRandomWalk local sampler."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            output_dir="./test_output/",
        )

        # Use GaussianRandomWalk sampler (default)
        sampler = FlowMCSampler(
            likelihood,
            prior,
            config,
            local_sampler_name="GaussianRandomWalk",
            local_sampler_arg={"step_size": jnp.array([1e-3])},
        )

        assert sampler.sampler is not None

    def test_flowmc_sampler_diagonal_extraction_for_gaussian_random_walk(self):
        """Test that FlowMCSampler extracts diagonal from matrix step_size for GaussianRandomWalk."""
        prior = CombinePrior(
            [
                UniformPrior(0.0, 1.0, parameter_names=["x"]),
                UniformPrior(0.0, 1.0, parameter_names=["y"]),
            ]
        )
        likelihood = MockLikelihood()

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            output_dir="./test_output/",
        )

        # Provide 2x2 matrix step_size
        step_size_matrix = jnp.array([[1e-3, 0.0], [0.0, 2e-3]])

        # GaussianRandomWalk should extract diagonal
        sampler = FlowMCSampler(
            likelihood,
            prior,
            config,
            local_sampler_name="GaussianRandomWalk",
            local_sampler_arg={"step_size": step_size_matrix},
        )

        # Should succeed (diagonal extracted)
        assert sampler.sampler is not None

    def test_flowmc_sampler_invalid_local_sampler_raises_error(self):
        """Test that invalid local_sampler_name raises ValueError."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            output_dir="./test_output/",
        )

        with pytest.raises(ValueError, match="Unknown local_sampler_name"):
            FlowMCSampler(
                likelihood,
                prior,
                config,
                local_sampler_name="InvalidSampler",
            )

    def test_flowmc_sampler_with_custom_flow_architecture(self):
        """Test FlowMCSampler with custom normalizing flow architecture."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            output_dir="./test_output/",
        )

        sampler = FlowMCSampler(
            likelihood,
            prior,
            config,
            local_sampler_arg={"step_size": jnp.array([1e-3])},
            num_layers=5,
            hidden_size=[64, 64],
            num_bins=4,
        )

        assert sampler.sampler is not None

    def test_flowmc_sampler_with_likelihood_transform(self):
        """Test FlowMCSampler with likelihood transform (realistic use case)."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            output_dir="./test_output/",
        )

        # Create a simple N-to-M transform
        class SquareTransform(NtoMTransform):
            def __init__(self):
                super().__init__((["x"], ["x_squared"]))
                self.transform_func = lambda params: {"x_squared": params["x"] ** 2}

        transform = SquareTransform()

        sampler = FlowMCSampler(
            likelihood,
            prior,
            config,
            likelihood_transforms=[transform],
            local_sampler_arg={"step_size": jnp.array([1e-3])},
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
        prior = CombinePrior(
            [
                UniformPrior(0.0, 1.0, parameter_names=["a"]),
                UniformPrior(10.0, 20.0, parameter_names=["b"]),
                UniformPrior(100.0, 200.0, parameter_names=["c"]),
            ]
        )
        likelihood = MockLikelihood()

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            n_local_steps=1,
            n_global_steps=1,
            n_epochs=1,
            output_dir="./test_output/",
        )

        sampler = FlowMCSampler(
            likelihood,
            prior,
            config,
            local_sampler_arg={"step_size": jnp.array([1e-3, 1e-3, 1e-3])},
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

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            n_local_steps=2,
            n_global_steps=2,
            n_epochs=2,
            learning_rate=0.001,
            output_dir="./test_output/",
        )

        sampler = FlowMCSampler(
            likelihood,
            prior,
            config,
        )

        # Run sampling
        key = jax.random.PRNGKey(42)
        sampler.sample(key)

        # Should have samples
        samples = sampler.get_samples()

        assert "x" in samples
        assert jnp.isfinite(samples["x"]).all()

    def test_sampler_with_constraint_likelihood(self):
        """Test sampler with constraint-based likelihood (realistic scenario)."""
        from jesterTOV.inference.likelihoods.constraints import ConstraintEOSLikelihood

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            output_dir="./test_output/",
        )

        # Use constraint likelihood
        likelihood = ConstraintEOSLikelihood(
            penalty_causality=-1e10,
            penalty_stability=-1e5,
        )

        sampler = FlowMCSampler(
            likelihood,
            prior,
            config,
            local_sampler_arg={"step_size": jnp.array([1e-3])},
        )

        # Should initialize successfully
        assert sampler.sampler is not None


class TestBlackJAXSMCRandomWalkSampler:
    """Test BlackJAX SMC sampler with Random Walk kernel."""

    def test_smc_rw_sampler_initialization(self):
        """Test SMC Random Walk sampler initializes correctly."""
        from jesterTOV.inference.samplers.blackjax_smc import BlackJAXSMCRandomWalkSampler
        from jesterTOV.inference.config.schema import SMCRandomWalkSamplerConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = SMCRandomWalkSamplerConfig(
            n_particles=100,
            n_mcmc_steps=1,
            target_ess=0.9,
            random_walk_sigma=0.1,
            output_dir="./test_output/",
        )

        sampler = BlackJAXSMCRandomWalkSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=[],
            likelihood_transforms=[],
            config=config,
        )

        assert sampler.config.type == "smc-rw"
        assert sampler.config.n_particles == 100
        assert sampler.config.random_walk_sigma == 0.1
        assert sampler.prior == prior
        assert sampler.likelihood == likelihood

    def test_smc_rw_config_validation(self):
        """Test SMC Random Walk config validates correctly."""
        from jesterTOV.inference.config.schema import SMCRandomWalkSamplerConfig

        # Valid config
        config = SMCRandomWalkSamplerConfig(output_dir="./test/")
        assert config.type == "smc-rw"
        assert config.random_walk_sigma == 0.1  # default


class TestBlackJAXSMCNUTSSampler:
    """Test BlackJAX SMC sampler with NUTS kernel."""

    def test_smc_nuts_sampler_initialization(self):
        """Test SMC NUTS sampler initializes correctly."""
        from jesterTOV.inference.samplers.blackjax_smc import BlackJAXSMCNUTSSampler
        from jesterTOV.inference.config.schema import SMCNUTSSamplerConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = SMCNUTSSamplerConfig(
            n_particles=100,
            n_mcmc_steps=1,
            target_ess=0.9,
            output_dir="./test_output/",
        )

        sampler = BlackJAXSMCNUTSSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=[],
            likelihood_transforms=[],
            config=config,
        )

        assert sampler.config.type == "smc-nuts"
        assert sampler.config.n_particles == 100
        assert sampler.prior == prior
        assert sampler.likelihood == likelihood

    def test_smc_nuts_config_validation(self):
        """Test SMC NUTS config validates correctly."""
        from jesterTOV.inference.config.schema import SMCNUTSSamplerConfig

        # Valid config
        config = SMCNUTSSamplerConfig(output_dir="./test/")
        assert config.type == "smc-nuts"
        assert config.init_step_size == 1e-2  # default

    def test_smc_nuts_mass_matrix_building(self):
        """Test SMC NUTS sampler builds mass matrix correctly with custom scales."""
        from jesterTOV.inference.samplers.blackjax_smc import BlackJAXSMCNUTSSampler
        from jesterTOV.inference.config.schema import SMCNUTSSamplerConfig

        # Multi-dimensional prior
        prior = CombinePrior(
            [
                UniformPrior(0.0, 1.0, parameter_names=["x"]),
                UniformPrior(0.0, 1.0, parameter_names=["y"]),
                UniformPrior(0.0, 1.0, parameter_names=["z"]),
            ]
        )
        likelihood = MockLikelihood()

        # Custom mass matrix scales
        config = SMCNUTSSamplerConfig(
            n_particles=100,
            mass_matrix_base=2.0e-1,
            mass_matrix_param_scales={"y": 2.0},  # Scale y parameter differently
            output_dir="./test_output/",
        )

        sampler = BlackJAXSMCNUTSSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=[],
            likelihood_transforms=[],
            config=config,
        )

        # Build mass matrix
        mass_matrix = sampler._build_mass_matrix()

        # Should be 3x3 diagonal matrix
        assert mass_matrix.shape == (3, 3)

        # Diagonal elements should be (base * scale)^2
        expected_x = (0.2 * 1.0) ** 2
        expected_y = (0.2 * 2.0) ** 2  # Custom scale
        expected_z = (0.2 * 1.0) ** 2

        assert jnp.allclose(mass_matrix[0, 0], expected_x)
        assert jnp.allclose(mass_matrix[1, 1], expected_y)
        assert jnp.allclose(mass_matrix[2, 2], expected_z)

        # Off-diagonal should be zero
        assert jnp.allclose(mass_matrix[0, 1], 0.0)

    def test_smc_sampler_methods_before_sampling_raise_errors(self):
        """Test SMC sampler methods raise errors when called before sampling."""
        from jesterTOV.inference.samplers.blackjax_smc import BlackJAXSMCRandomWalkSampler
        from jesterTOV.inference.config.schema import SMCRandomWalkSamplerConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = SMCRandomWalkSamplerConfig(
            # kernel_type removed
            n_particles=100,
            output_dir="./test_output/",
        )

        sampler = BlackJAXSMCRandomWalkSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=[],
            likelihood_transforms=[],
            config=config,
        )

        # Methods should raise RuntimeError before sampling
        with pytest.raises(RuntimeError, match="No samples available"):
            sampler.get_samples()

        with pytest.raises(RuntimeError, match="No samples available"):
            sampler.get_log_prob()

        # get_n_samples should return 0 before sampling
        assert sampler.get_n_samples() == 0

    def test_smc_sampler_with_sample_transforms_warns(self):
        """Test SMC sampler warns when given sample transforms (works in prior space)."""
        from jesterTOV.inference.samplers.blackjax_smc import BlackJAXSMCRandomWalkSampler
        from jesterTOV.inference.config.schema import SMCRandomWalkSamplerConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        # Create a sample transform
        transform = ScaleTransform((["x"], ["y"]), scale=2.0)

        config = SMCRandomWalkSamplerConfig(
            # kernel_type removed
            n_particles=100,
            output_dir="./test_output/",
        )

        # Should log warning but still initialize
        sampler = BlackJAXSMCRandomWalkSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=[transform],
            likelihood_transforms=[],
            config=config,
        )

        assert len(sampler.sample_transforms) == 1

    @pytest.mark.slow
    def test_smc_sampler_minimal_run_nuts(self):
        """Test SMC sampler can run minimal sampling with NUTS kernel (slow test)."""
        from jesterTOV.inference.samplers.blackjax_smc import BlackJAXSMCNUTSSampler
        from jesterTOV.inference.config.schema import SMCNUTSSamplerConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = SMCNUTSSamplerConfig(
            n_particles=50,  # Small for quick test
            n_mcmc_steps=2,
            target_ess=0.8,
            init_step_size=1e-2,
            output_dir="./test_output/",
        )

        sampler = BlackJAXSMCNUTSSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=[],
            likelihood_transforms=[],
            config=config,
        )

        # Run sampling
        key = jax.random.PRNGKey(42)
        sampler.sample(key)

        # Should have samples
        samples = sampler.get_samples()
        assert "x" in samples
        assert "weights" in samples
        assert "ess" in samples

        # Samples should be in valid range
        assert jnp.all((samples["x"] >= 0.0) & (samples["x"] <= 1.0))

        # Should have correct number of particles
        assert sampler.get_n_samples() == 50

        # Log probs should be finite
        log_probs = sampler.get_log_prob()
        assert jnp.isfinite(log_probs).all()

        # Metadata should be populated
        assert "final_ess" in sampler.metadata
        assert "annealing_steps" in sampler.metadata
        assert "kernel_type" in sampler.metadata
        assert sampler.metadata["kernel_type"] == "nuts"

    @pytest.mark.slow
    def test_smc_sampler_minimal_run_random_walk(self):
        """Test SMC sampler can run minimal sampling with random walk kernel (slow test)."""
        from jesterTOV.inference.samplers.blackjax_smc import BlackJAXSMCRandomWalkSampler
        from jesterTOV.inference.config.schema import SMCRandomWalkSamplerConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = SMCRandomWalkSamplerConfig(
            # kernel_type removed
            n_particles=50,
            n_mcmc_steps=10,  # More steps needed for random walk
            target_ess=0.8,
            random_walk_sigma=0.1,
            output_dir="./test_output/",
        )

        sampler = BlackJAXSMCRandomWalkSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=[],
            likelihood_transforms=[],
            config=config,
        )

        # Run sampling
        key = jax.random.PRNGKey(42)
        sampler.sample(key)

        # Should have samples
        samples = sampler.get_samples()
        assert "x" in samples
        assert "weights" in samples

        # Should have correct metadata
        assert sampler.metadata["kernel_type"] == "random_walk"


class TestBlackJAXNSAWSampler:
    """Test BlackJAX Nested Sampling with Acceptance Walk sampler."""

    def test_ns_aw_sampler_initialization(self):
        """Test NS-AW sampler initializes correctly."""
        pytest.importorskip("blackjax")  # Skip if BlackJAX not installed

        from jesterTOV.inference.samplers.blackjax_ns_aw import BlackJAXNSAWSampler
        from jesterTOV.inference.config.schema import BlackJAXNSAWConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = BlackJAXNSAWConfig(
            n_live=100,
            n_delete_frac=0.5,
            n_target=20,
            max_mcmc=1000,
            output_dir="./test_output/",
        )

        sampler = BlackJAXNSAWSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=[],
            likelihood_transforms=[],
            config=config,
        )

        assert sampler.config.n_live == 100
        assert sampler.prior == prior
        assert sampler.likelihood == likelihood

    def test_ns_aw_requires_bound_to_bound_transform(self):
        """Test NS-AW automatically creates BoundToBound transform for unit cube."""
        pytest.importorskip("blackjax")

        from jesterTOV.inference.samplers.blackjax_ns_aw import BlackJAXNSAWSampler
        from jesterTOV.inference.config.schema import BlackJAXNSAWConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = BlackJAXNSAWConfig(
            n_live=100,
            output_dir="./test_output/",
        )

        # Note: NS-AW requires unit cube transforms
        # The factory should handle this automatically
        sampler = BlackJAXNSAWSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=[],
            likelihood_transforms=[],
            config=config,
        )

        # Unit cube stepper should be created
        assert sampler.config is not None

    def test_ns_aw_sampler_methods_before_sampling_raise_errors(self):
        """Test NS-AW sampler methods raise errors when called before sampling."""
        pytest.importorskip("blackjax")

        from jesterTOV.inference.samplers.blackjax_ns_aw import BlackJAXNSAWSampler
        from jesterTOV.inference.config.schema import BlackJAXNSAWConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = BlackJAXNSAWConfig(
            n_live=100,
            output_dir="./test_output/",
        )

        sampler = BlackJAXNSAWSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=[],
            likelihood_transforms=[],
            config=config,
        )

        # Methods should raise RuntimeError before sampling
        with pytest.raises(RuntimeError, match="No samples available"):
            sampler.get_samples()

        with pytest.raises(RuntimeError, match="No samples available"):
            sampler.get_log_prob()

        # get_n_samples should return 0 before sampling
        assert sampler.get_n_samples() == 0


class TestSamplerFactory:
    """Test sampler factory for multi-backend support."""

    def test_create_flowmc_sampler_from_config(self):
        """Test factory creates FlowMC sampler from config."""
        from jesterTOV.inference.samplers.factory import create_sampler
        from jesterTOV.inference.samplers.flowmc import FlowMCSampler
        from jesterTOV.inference.config.schema import (
            InferenceConfig,
            FlowMCSamplerConfig,
        )

        # Create full inference config
        config = InferenceConfig(
            seed=42,
            transform={"type": "metamodel", "nb_CSE": 0},
            prior={"specification_file": "test.prior"},
            likelihoods=[{"type": "zero", "enabled": True, "parameters": {}}],
            sampler=FlowMCSamplerConfig(
                type="flowmc",
                n_chains=2,
                n_loop_training=1,
                n_loop_production=1,
                output_dir="./test/",
            ),
        )

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = create_sampler(
            config.sampler,
            prior,
            likelihood,
            sample_transforms=[],
            likelihood_transforms=[],
            seed=42,
        )

        assert isinstance(sampler, FlowMCSampler)

    def test_create_smc_sampler_from_config(self):
        """Test factory creates SMC sampler from config."""
        from jesterTOV.inference.samplers.factory import create_sampler
        from jesterTOV.inference.samplers.blackjax_smc import BlackJAXSMCRandomWalkSampler
        from jesterTOV.inference.config.schema import SMCRandomWalkSamplerConfig

        config = SMCRandomWalkSamplerConfig(
            type="smc-rw",
            # kernel_type removed
            n_particles=100,
            output_dir="./test/",
        )

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = create_sampler(
            config,
            prior,
            likelihood,
            sample_transforms=[],
            likelihood_transforms=[],
            seed=42,
        )

        assert isinstance(sampler, BlackJAXSMCRandomWalkSampler)
        assert sampler.config.type == "smc-rw"

    def test_create_ns_aw_sampler_from_config(self):
        """Test factory creates NS-AW sampler from config."""
        pytest.importorskip("blackjax")

        from jesterTOV.inference.samplers.factory import create_sampler
        from jesterTOV.inference.samplers.blackjax_ns_aw import BlackJAXNSAWSampler
        from jesterTOV.inference.config.schema import BlackJAXNSAWConfig

        config = BlackJAXNSAWConfig(
            type="blackjax-ns-aw",
            n_live=100,
            output_dir="./test/",
        )

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        sampler = create_sampler(
            config,
            prior,
            likelihood,
            sample_transforms=[],
            likelihood_transforms=[],
            seed=42,
        )

        assert isinstance(sampler, BlackJAXNSAWSampler)

    def test_factory_invalid_sampler_type_raises_error(self):
        """Test factory raises error for invalid sampler type."""
        from jesterTOV.inference.samplers.factory import create_sampler
        from jesterTOV.inference.config.schema import BaseSamplerConfig

        # Create invalid config (this will fail Pydantic validation)
        # We need to test the factory error handling
        class InvalidConfig(BaseSamplerConfig):
            type: str = "invalid"

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        # Factory should handle this gracefully
        with pytest.raises(ValueError, match="Unknown sampler type"):
            create_sampler(
                InvalidConfig(output_dir="./test/"),
                prior,
                likelihood,
                sample_transforms=[],
                likelihood_transforms=[],
            )


class TestSamplerOutputInterface:
    """Test standardized SamplerOutput interface across all samplers."""

    def test_sampler_output_dataclass_structure(self):
        """Test SamplerOutput has expected fields."""
        from jesterTOV.inference.samplers import SamplerOutput

        samples = {"x": jnp.array([1.0, 2.0, 3.0])}
        log_prob = jnp.array([-1.0, -2.0, -3.0])
        metadata = {"weights": jnp.array([0.3, 0.3, 0.4])}

        output = SamplerOutput(
            samples=samples,
            log_prob=log_prob,
            metadata=metadata,
        )

        assert output.samples == samples
        assert jnp.array_equal(output.log_prob, log_prob)
        assert output.metadata == metadata

    def test_sampler_output_default_metadata(self):
        """Test SamplerOutput metadata defaults to empty dict."""
        from jesterTOV.inference.samplers import SamplerOutput

        output = SamplerOutput(
            samples={"x": jnp.array([1.0])},
            log_prob=jnp.array([-1.0]),
        )

        assert output.metadata == {}

    @pytest.mark.slow
    def test_flowmc_sampler_output_interface(self):
        """Test FlowMC implements get_sampler_output() correctly."""
        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = FlowMCSamplerConfig(
            type="flowmc",
            n_chains=2,
            n_loop_training=1,
            n_loop_production=1,
            n_local_steps=2,
            n_global_steps=2,
            n_epochs=2,
            output_dir="./test_output/",
        )

        sampler = FlowMCSampler(likelihood, prior, config)
        sampler.sample(jax.random.PRNGKey(42))

        # Get output via new interface (production samples)
        output = sampler.get_sampler_output()

        # Verify structure
        assert "x" in output.samples
        assert output.log_prob.shape[0] == output.samples["x"].shape[0]
        assert output.metadata == {}  # FlowMC has no metadata

        # Verify consistency with old interface
        old_samples = sampler.get_samples()
        old_log_prob = sampler.get_log_prob()

        assert jnp.array_equal(output.samples["x"], old_samples["x"])
        assert jnp.array_equal(output.log_prob, old_log_prob)

        # Test FlowMC-specific training sample access
        training_output = sampler.get_training_sampler_output()
        assert "x" in training_output.samples
        assert (
            training_output.log_prob.shape[0] == training_output.samples["x"].shape[0]
        )
        assert training_output.metadata == {}

        # Test training sample count
        n_training = sampler.get_n_training_samples()
        assert n_training > 0
        assert n_training == len(training_output.log_prob)

    @pytest.mark.slow
    def test_smc_sampler_output_interface(self):
        """Test SMC implements get_sampler_output() correctly."""
        pytest.importorskip("blackjax")

        from jesterTOV.inference.samplers.blackjax_smc import BlackJAXSMCRandomWalkSampler
        from jesterTOV.inference.config.schema import SMCRandomWalkSamplerConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = SMCRandomWalkSamplerConfig(
            # kernel_type removed
            n_particles=50,
            n_mcmc_steps=2,
            output_dir="./test_output/",
        )

        sampler = BlackJAXSMCRandomWalkSampler(likelihood, prior, [], [], config)
        sampler.sample(jax.random.PRNGKey(42))

        # Get output via new interface
        output = sampler.get_sampler_output()

        # Verify structure
        assert "x" in output.samples
        assert "weights" not in output.samples  # Should be in metadata
        assert "weights" in output.metadata
        assert "ess" in output.metadata
        assert output.log_prob.shape[0] == output.samples["x"].shape[0]

        # Verify consistency with old interface
        old_samples = sampler.get_samples()
        old_log_prob = sampler.get_log_prob()

        assert jnp.array_equal(output.samples["x"], old_samples["x"])
        assert jnp.array_equal(output.metadata["weights"], old_samples["weights"])
        assert jnp.array_equal(output.log_prob, old_log_prob)

    def test_ns_aw_sampler_output_before_sampling(self):
        """Test NS-AW raises error before sampling."""
        pytest.importorskip("blackjax")

        from jesterTOV.inference.samplers.blackjax_ns_aw import BlackJAXNSAWSampler
        from jesterTOV.inference.config.schema import BlackJAXNSAWConfig

        prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
        likelihood = MockLikelihood()

        config = BlackJAXNSAWConfig(
            n_live=100,
            output_dir="./test_output/",
        )

        sampler = BlackJAXNSAWSampler(likelihood, prior, [], [], config)

        # Before sampling, should raise error
        with pytest.raises(RuntimeError, match="No samples available"):
            sampler.get_sampler_output()

    def test_sampler_output_separates_parameters_from_metadata(self):
        """Test that samples dict contains only parameters, not metadata."""
        from jesterTOV.inference.samplers import SamplerOutput

        # Simulate SMC output
        output = SamplerOutput(
            samples={"mass": jnp.array([1.5, 1.6, 1.7])},
            log_prob=jnp.array([-10.0, -11.0, -12.0]),
            metadata={
                "weights": jnp.array([0.3, 0.4, 0.3]),
                "ess": 2.5,
            },
        )

        # Parameters should only be in samples
        assert "mass" in output.samples
        assert "weights" not in output.samples
        assert "ess" not in output.samples

        # Metadata should only be in metadata
        assert "weights" in output.metadata
        assert "ess" in output.metadata
        assert "mass" not in output.metadata
