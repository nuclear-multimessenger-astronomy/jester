r"""flowMC sampler implementation and setup"""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx

from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.states import State
from flowMC.resource.logPDF import LogPDF
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.local_kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.strategy.lambda_function import Lambda
from flowMC.strategy.take_steps import TakeSerialSteps, TakeGroupSteps
from flowMC.strategy.train_model import TrainModel
from flowMC.strategy.update_state import UpdateState
from flowMC.Sampler import Sampler

from .jester_sampler import JesterSampler, SamplerOutput
from ..config.schema import FlowMCSamplerConfig
from ..base import LikelihoodBase, Prior, BijectiveTransform, NtoMTransform
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class FlowMCSampler(JesterSampler):
    """
    FlowMC-specific sampler implementation.

    This class inherits from JesterSampler and adds flowMC-specific
    initialization and configuration. It creates a flowMC Sampler with:
    - Local sampler (MALA or GaussianRandomWalk)
    - Normalizing flow model (MaskedCouplingRQSpline)
    - Training and production sampling loops

    Parameters
    ----------
    likelihood : LikelihoodBase
        Likelihood object with evaluate(params, data) method
    prior : Prior
        Prior object with sample() and log_prob() methods
    config : FlowMCSamplerConfig
        Configuration object from YAML (contains n_chains, n_loop_training, learning_rate, etc.)
    sample_transforms : list[BijectiveTransform], optional
        Bijective transforms applied during sampling (with Jacobians)
    likelihood_transforms : list[NtoMTransform], optional
        N-to-M transforms applied before likelihood evaluation
    seed : int, optional
        Random seed (default: 0)
    local_sampler_name : str, optional
        Name of the local sampler: "MALA" or "GaussianRandomWalk" (default: "GaussianRandomWalk")
    local_sampler_arg : dict[str, Any], optional
        Arguments for local sampler (e.g., {"step_size": ...})
    num_layers : int, optional
        Number of coupling layers in normalizing flow (default: 10)
    hidden_size : list[int], optional
        Hidden layer sizes for normalizing flow (default: [128, 128])
    num_bins : int, optional
        Number of bins for rational quadratic splines (default: 8)

    Attributes
    ----------
    sampler : Sampler
        FlowMC sampler instance
    config : FlowMCSamplerConfig
        Configuration object
    """

    sampler: Sampler

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        config: FlowMCSamplerConfig,
        sample_transforms: list[BijectiveTransform] | None = None,
        likelihood_transforms: list[NtoMTransform] | None = None,
        seed: int = 0,
        local_sampler_name: str = "GaussianRandomWalk",
        local_sampler_arg: dict[str, Any] | None = None,
        num_layers: int = 10,
        hidden_size: list[int] | None = None,
        num_bins: int = 8,
    ) -> None:
        # Handle None defaults
        if sample_transforms is None:
            sample_transforms = []
        if likelihood_transforms is None:
            likelihood_transforms = []
        if local_sampler_arg is None:
            local_sampler_arg = {}
        if hidden_size is None:
            hidden_size = [128, 128]

        # Initialize base class (sets up transforms and parameter names)
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)

        # Store config
        self.config = config

        # FlowMC-specific initialization
        rng_key = jax.random.PRNGKey(seed)

        # Create logpdf wrapper that matches new flowMC API
        def logpdf_func(x: Float[Array, " n_dim"], data: dict) -> Float:
            """Log PDF function for flowMC 0.4.5 API."""
            # Convert array to dict with parameter names
            # NOTE: Do NOT use float() on JAX traced values - causes ConcretizationTypeError
            params_dict = {name: x[i] for i, name in enumerate(self.parameter_names)}
            return self.posterior_from_dict(params_dict, data)

        # Build the custom bundle for JESTER
        n_dims = self.prior.n_dim
        n_chains = config.n_chains
        n_local_steps = config.n_local_steps
        n_global_steps = config.n_global_steps
        n_training_loops = config.n_loop_training
        n_production_loops = config.n_loop_production
        n_epochs = config.n_epochs
        learning_rate = config.learning_rate
        local_thinning = config.train_thinning
        global_thinning = config.train_thinning
        output_local_thinning = config.output_thinning
        output_global_thinning = config.output_thinning

        # Validate thinning values to prevent zero-length buffers
        thinning_errors = []
        if local_thinning > n_local_steps:
            thinning_errors.append(
                f"train_thinning ({local_thinning}) exceeds n_local_steps ({n_local_steps})"
            )
        if global_thinning > n_global_steps:
            thinning_errors.append(
                f"train_thinning ({global_thinning}) exceeds n_global_steps ({n_global_steps})"
            )
        if output_local_thinning > n_local_steps:
            thinning_errors.append(
                f"output_thinning ({output_local_thinning}) exceeds n_local_steps ({n_local_steps})"
            )
        if output_global_thinning > n_global_steps:
            thinning_errors.append(
                f"output_thinning ({output_global_thinning}) exceeds n_global_steps ({n_global_steps})"
            )
        if thinning_errors:
            error_msg = (
                "Thinning values exceed step counts, which would produce zero-length buffers:\n  "
                + "\n  ".join(thinning_errors)
                + "\nPlease reduce thinning values or increase step counts in your config."
            )
            raise ValueError(error_msg)

        # Calculate buffer sizes (guaranteed non-zero after validation)
        n_training_steps = (
            n_local_steps // local_thinning * n_training_loops
            + n_global_steps // global_thinning * n_training_loops
        )
        n_production_steps = (
            n_local_steps // output_local_thinning * n_production_loops
            + n_global_steps // output_global_thinning * n_production_loops
        )
        n_total_epochs = n_training_loops * n_epochs

        # Create buffers
        positions_training = Buffer(
            "positions_training", (n_chains, n_training_steps, n_dims), 1
        )
        log_prob_training = Buffer("log_prob_training", (n_chains, n_training_steps), 1)
        local_accs_training = Buffer(
            "local_accs_training", (n_chains, n_training_steps), 1
        )
        global_accs_training = Buffer(
            "global_accs_training", (n_chains, n_training_steps), 1
        )
        loss_buffer = Buffer("loss_buffer", (n_total_epochs,), 0)

        position_production = Buffer(
            "positions_production", (n_chains, n_production_steps, n_dims), 1
        )
        log_prob_production = Buffer(
            "log_prob_production", (n_chains, n_production_steps), 1
        )
        local_accs_production = Buffer(
            "local_accs_production", (n_chains, n_production_steps), 1
        )
        global_accs_production = Buffer(
            "global_accs_production", (n_chains, n_production_steps), 1
        )

        # Select and create local sampler
        step_size = local_sampler_arg.get("step_size")
        if step_size is None:
            # Provide default step_size
            if local_sampler_name == "MALA":
                step_size = 1e-1
            else:  # GaussianRandomWalk
                step_size = jnp.ones(n_dims) * 1e-3
        elif isinstance(step_size, jnp.ndarray) and step_size.ndim == 2:
            # Extract diagonal from DxD matrix for GaussianRandomWalk
            step_size = jnp.diag(step_size)

        if local_sampler_name == "MALA":
            local_sampler = MALA(step_size=step_size)
        elif local_sampler_name == "GaussianRandomWalk":
            local_sampler = GaussianRandomWalk(step_size=step_size)
        else:
            raise ValueError(
                f"Unknown local_sampler_name: {local_sampler_name}. "
                f"Supported options: 'MALA', 'GaussianRandomWalk'"
            )

        # Create normalizing flow model
        rng_key, subkey = jax.random.split(rng_key)
        model = MaskedCouplingRQSpline(
            n_dims, num_layers, hidden_size, num_bins, subkey
        )
        global_sampler = NFProposal(model, n_NFproposal_batch_size=10000)
        optimizer = Optimizer(model=model, learning_rate=learning_rate)
        logpdf_resource = LogPDF(logpdf_func, n_dims=n_dims)

        # Create sampler state
        sampler_state = State(
            {
                "target_positions": "positions_training",
                "target_log_prob": "log_prob_training",
                "target_local_accs": "local_accs_training",
                "target_global_accs": "global_accs_training",
                "training": True,
            },
            name="sampler_state",
        )

        # Build resources dict
        resources = {
            "logpdf": logpdf_resource,
            "positions_training": positions_training,
            "log_prob_training": log_prob_training,
            "local_accs_training": local_accs_training,
            "global_accs_training": global_accs_training,
            "loss_buffer": loss_buffer,
            "positions_production": position_production,
            "log_prob_production": log_prob_production,
            "local_accs_production": local_accs_production,
            "global_accs_production": global_accs_production,
            "local_sampler": local_sampler,
            "global_sampler": global_sampler,
            "model": model,
            "optimizer": optimizer,
            "sampler_state": sampler_state,
        }

        # Create strategies
        local_stepper = TakeSerialSteps(
            "logpdf",
            "local_sampler",
            "sampler_state",
            ["target_positions", "target_log_prob", "target_local_accs"],
            n_local_steps,
            thinning=local_thinning,
            chain_batch_size=0,
            verbose=False,
        )

        global_stepper = TakeGroupSteps(
            "logpdf",
            "global_sampler",
            "sampler_state",
            ["target_positions", "target_log_prob", "target_global_accs"],
            n_global_steps,
            thinning=global_thinning,
            chain_batch_size=0,
            verbose=False,
        )

        model_trainer = TrainModel(
            "model",
            "positions_training",
            "optimizer",
            loss_buffer_name="loss_buffer",
            n_epochs=n_epochs,
            batch_size=10000,
            n_max_examples=10000,
            verbose=False,
        )

        update_state = UpdateState(
            "sampler_state",
            [
                "target_positions",
                "target_log_prob",
                "target_local_accs",
                "target_global_accs",
                "training",
            ],
            [
                "positions_production",
                "log_prob_production",
                "local_accs_production",
                "global_accs_production",
                False,
            ],
        )

        # Update production phase thinning
        def update_production_thinning(
            rng_key: PRNGKeyArray,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            PRNGKeyArray,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Update thinning for production phase."""
            local_stepper.thinning = output_local_thinning
            global_stepper.thinning = output_global_thinning
            return rng_key, resources, initial_position

        update_production_thinning_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: update_production_thinning(
                rng_key, resources, initial_position, data
            )
        )

        def reset_steppers(
            rng_key: PRNGKeyArray,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            PRNGKeyArray,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Reset the steppers to the initial position."""
            local_stepper.set_current_position(0)
            global_stepper.set_current_position(0)
            return rng_key, resources, initial_position

        reset_steppers_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: reset_steppers(
                rng_key, resources, initial_position, data
            )
        )

        update_global_step = Lambda(
            lambda rng_key, resources, initial_position, data: global_stepper.set_current_position(
                local_stepper.current_position
            )
        )
        update_local_step = Lambda(
            lambda rng_key, resources, initial_position, data: local_stepper.set_current_position(
                global_stepper.current_position
            )
        )

        def update_model(
            rng_key: PRNGKeyArray,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            PRNGKeyArray,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Update the model."""
            model = resources["model"]
            resources["global_sampler"] = eqx.tree_at(
                lambda x: x.model,
                resources["global_sampler"],
                model,
            )
            return rng_key, resources, initial_position

        update_model_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: update_model(
                rng_key, resources, initial_position, data
            )
        )

        strategies = {
            "local_stepper": local_stepper,
            "global_stepper": global_stepper,
            "model_trainer": model_trainer,
            "update_state": update_state,
            "update_global_step": update_global_step,
            "update_local_step": update_local_step,
            "reset_steppers": reset_steppers_lambda,
            "update_model": update_model_lambda,
            "update_production_thinning": update_production_thinning_lambda,
        }

        # Build strategy order
        training_phase = [
            "local_stepper",
            "update_global_step",
            "model_trainer",
            "update_model",
            "global_stepper",
            "update_local_step",
        ]
        production_phase = [
            "local_stepper",
            "update_global_step",
            "global_stepper",
            "update_local_step",
        ]
        strategy_order = []
        for _ in range(n_training_loops):
            strategy_order.extend(training_phase)

        strategy_order.append("reset_steppers")
        strategy_order.append("update_state")
        strategy_order.append("update_production_thinning")
        for _ in range(n_production_loops):
            strategy_order.extend(production_phase)

        # Create flowMC sampler
        self.sampler = Sampler(
            n_dim=n_dims,
            n_chains=n_chains,
            rng_key=rng_key,
            resources=resources,
            strategies=strategies,
            strategy_order=strategy_order,
        )

    def sample(self, key):
        """
        Run flowMC sampling.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key

        Notes
        -----
        This method includes a critical bug fix: parameter ordering is preserved
        when converting from dictionary to array using a list comprehension instead
        of jax.tree.leaves().
        """
        # Sample initial positions from prior
        # Use jnp.inf instead of jnp.nan for initialization
        initial_position = (
            jnp.zeros((self.sampler.n_chains, self.prior.n_dim)) + jnp.inf
        )

        while not jax.tree.reduce(
            jnp.logical_and,
            jax.tree.map(lambda x: jnp.isfinite(x), initial_position),
        ).all():
            non_finite_index = jnp.where(
                jnp.any(
                    ~jax.tree.reduce(
                        jnp.logical_and,
                        jax.tree.map(lambda x: jnp.isfinite(x), initial_position),
                    ),
                    axis=1,
                )
            )[0]

            key, subkey = jax.random.split(key)
            guess = self.prior.sample(subkey, self.sampler.n_chains)
            for transform in self.sample_transforms:
                guess = jax.vmap(transform.forward)(guess)

            # CRITICAL FIX: Preserve parameter order when converting dict to array
            # Do NOT use jax.tree.leaves() as it doesn't preserve dictionary order
            guess = jnp.array(
                [guess[param_name] for param_name in self.parameter_names]
            ).T

            finite_guess = jnp.where(
                jnp.all(jax.tree.map(lambda x: jnp.isfinite(x), guess), axis=1)
            )[0]
            common_length = min(len(finite_guess), len(non_finite_index))
            initial_position = initial_position.at[
                non_finite_index[:common_length]
            ].set(guess[:common_length])
        self.sampler.sample(initial_position, {})  # Empty data dict

    def get_samples(self) -> dict:
        """
        Get production samples from the flowMC sampler.

        Returns
        -------
        dict
            Dictionary of production samples
        """
        # Access production buffer
        from flowMC.resource.buffers import Buffer

        positions_buffer = self.sampler.resources["positions_production"]
        assert isinstance(positions_buffer, Buffer)
        chains = positions_buffer.data  # Access the actual buffer data
        chains = chains.reshape(-1, self.prior.n_dim)
        chains = jax.vmap(self.add_name)(chains)
        for sample_transform in reversed(self.sample_transforms):
            chains = jax.vmap(sample_transform.backward)(chains)
        return chains

    def get_log_prob(self) -> Array:
        """
        Get log probabilities from flowMC sampler (production samples only).

        Returns
        -------
        Array
            Log posterior probability values (1D array, flattened across chains)
        """
        from flowMC.resource.buffers import Buffer

        log_prob_buffer = self.sampler.resources["log_prob_production"]
        assert isinstance(log_prob_buffer, Buffer)
        return log_prob_buffer.data.flatten()

    def get_n_samples(self) -> int:
        """
        Get number of production samples from flowMC sampler.

        Returns
        -------
        int
            Number of production samples (total across all chains)
        """
        log_prob = self.get_log_prob()
        return len(log_prob)

    def get_n_training_samples(self) -> int:
        """
        Get number of training samples from flowMC sampler.

        This is a FlowMC-specific method for diagnostic purposes.

        Returns
        -------
        int
            Number of training samples (total across all chains)
        """
        from flowMC.resource.buffers import Buffer

        log_prob_buffer = self.sampler.resources["log_prob_training"]
        assert isinstance(log_prob_buffer, Buffer)
        return len(log_prob_buffer.data.flatten())

    def get_sampler_output(self) -> SamplerOutput:
        """
        Get standardized sampler output (production samples only).

        Returns
        -------
        SamplerOutput
            - samples: Parameter samples (dict of arrays)
            - log_prob: Log posterior probability
            - metadata: {} (empty, MCMC has equal weights)
        """
        # Get production samples
        samples = self.get_samples()
        log_prob = self.get_log_prob()

        # FlowMC has no metadata (equal weights)
        metadata: dict[str, Any] = {}

        return SamplerOutput(
            samples=samples,
            log_prob=log_prob,
            metadata=metadata,
        )

    def get_training_sampler_output(self) -> SamplerOutput:
        """
        Get standardized sampler output for training samples.

        This is a FlowMC-specific method for diagnostic purposes.

        Returns
        -------
        SamplerOutput
            - samples: Parameter samples from training phase (dict of arrays)
            - log_prob: Log posterior probability from training phase
            - metadata: {} (empty, MCMC has equal weights)
        """
        # Get training samples directly from buffer
        from flowMC.resource.buffers import Buffer

        positions_buffer = self.sampler.resources["positions_training"]
        assert isinstance(positions_buffer, Buffer)
        chains = positions_buffer.data
        chains = chains.reshape(-1, self.prior.n_dim)
        chains = jax.vmap(self.add_name)(chains)
        for sample_transform in reversed(self.sample_transforms):
            chains = jax.vmap(sample_transform.backward)(chains)
        samples = chains

        # Get training log_prob
        from flowMC.resource.buffers import Buffer

        log_prob_buffer = self.sampler.resources["log_prob_training"]
        assert isinstance(log_prob_buffer, Buffer)
        log_prob = log_prob_buffer.data.flatten()

        # FlowMC has no metadata (equal weights)
        metadata: dict[str, Any] = {}

        return SamplerOutput(
            samples=samples,
            log_prob=log_prob,
            metadata=metadata,
        )

    def print_summary(self, transform: bool = True):
        """
        Generate summary of the flowMC run.

        Parameters
        ----------
        transform : bool, optional
            Whether to apply inverse sample transforms to results (default: True)
        """
        # Access training data
        from flowMC.resource.buffers import Buffer

        positions_training_buf = self.sampler.resources["positions_training"]
        log_prob_training_buf = self.sampler.resources["log_prob_training"]
        local_accs_training_buf = self.sampler.resources["local_accs_training"]
        global_accs_training_buf = self.sampler.resources["global_accs_training"]
        loss_buf = self.sampler.resources["loss_buffer"]
        assert isinstance(positions_training_buf, Buffer)
        assert isinstance(log_prob_training_buf, Buffer)
        assert isinstance(local_accs_training_buf, Buffer)
        assert isinstance(global_accs_training_buf, Buffer)
        assert isinstance(loss_buf, Buffer)
        positions_training = positions_training_buf.data
        log_prob_training = log_prob_training_buf.data
        local_accs_training = local_accs_training_buf.data
        global_accs_training = global_accs_training_buf.data
        loss_vals = loss_buf.data

        training_chain = positions_training.reshape(-1, self.prior.n_dim).T
        training_chain = self.add_name(training_chain)
        if transform:
            for sample_transform in reversed(self.sample_transforms):
                training_chain = jax.vmap(sample_transform.backward)(training_chain)
        training_log_prob = log_prob_training.flatten()
        training_local_acceptance = local_accs_training.flatten()
        training_global_acceptance = global_accs_training.flatten()

        # Access production data
        positions_production_buf = self.sampler.resources["positions_production"]
        log_prob_production_buf = self.sampler.resources["log_prob_production"]
        local_accs_production_buf = self.sampler.resources["local_accs_production"]
        global_accs_production_buf = self.sampler.resources["global_accs_production"]
        assert isinstance(positions_production_buf, Buffer)
        assert isinstance(log_prob_production_buf, Buffer)
        assert isinstance(local_accs_production_buf, Buffer)
        assert isinstance(global_accs_production_buf, Buffer)
        positions_production = positions_production_buf.data
        log_prob_production = log_prob_production_buf.data
        local_accs_production = local_accs_production_buf.data
        global_accs_production = global_accs_production_buf.data

        production_chain = positions_production.reshape(-1, self.prior.n_dim).T
        production_chain = self.add_name(production_chain)
        if transform:
            for sample_transform in reversed(self.sample_transforms):
                production_chain = jax.vmap(sample_transform.backward)(production_chain)
        production_log_prob = log_prob_production.flatten()
        production_local_acceptance = local_accs_production.flatten()
        production_global_acceptance = global_accs_production.flatten()

        # flowMC 0.4.5: Buffer is initialized with -inf; local and global acceptance
        # buffers are only half-filled (local/global steppers share current_position
        # but write to separate buffers). Filter -inf slots before computing stats.
        valid_training_local = training_local_acceptance[
            training_local_acceptance > -jnp.inf
        ]
        valid_training_global = training_global_acceptance[
            training_global_acceptance > -jnp.inf
        ]

        logger.info("Training summary")
        logger.info("=" * 10)
        for key, value in training_chain.items():
            logger.info(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")
        logger.info(
            f"Log probability: {training_log_prob.mean():.3f} +/- {training_log_prob.std():.3f}"
        )
        logger.info(
            f"Local acceptance: {valid_training_local.mean():.3f} +/- {valid_training_local.std():.3f}"
        )
        logger.info(
            f"Global acceptance: {valid_training_global.mean():.3f} +/- {valid_training_global.std():.3f}"
        )
        logger.info(f"Max loss: {loss_vals.max():.3f}, Min loss: {loss_vals.min():.3f}")

        valid_production_local = production_local_acceptance[
            production_local_acceptance > -jnp.inf
        ]
        valid_production_global = production_global_acceptance[
            production_global_acceptance > -jnp.inf
        ]

        logger.info("Production summary")
        logger.info("=" * 10)
        for key, value in production_chain.items():
            logger.info(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")
        logger.info(
            f"Log probability: {production_log_prob.mean():.3f} +/- {production_log_prob.std():.3f}"
        )
        logger.info(
            f"Local acceptance: {valid_production_local.mean():.3f} +/- {valid_production_local.std():.3f}"
        )
        logger.info(
            f"Global acceptance: {valid_production_global.mean():.3f} +/- {valid_production_global.std():.3f}"
        )
