r"""flowMC sampler implementation and setup"""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.proposal.MALA import MALA
from flowMC.proposal.Gaussian_random_walk import GaussianRandomWalk
from flowMC.Sampler import Sampler

from .jester_sampler import JesterSampler
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
        local_sampler_name: str = "GaussianRandomWalk", # TODO: use literal here, MALA and GaussianRandomWalk for now
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

        # Select local sampler based on name
        if local_sampler_name == "MALA":
            # MALA uses matrix-vector multiplication, so it can handle full matrices
            # Provide default step_size if not given
            if "step_size" not in local_sampler_arg:
                local_sampler_arg = {**local_sampler_arg, "step_size": jnp.ones((self.prior.n_dim, self.prior.n_dim)) * 1e-3}
            local_sampler = MALA(self.posterior, True, **local_sampler_arg)
        elif local_sampler_name == "GaussianRandomWalk":
            # GaussianRandomWalk uses element-wise multiplication, so convert matrix to diagonal
            step_size: Array | None = local_sampler_arg.get("step_size")  # Can be 1D or 2D array
            if step_size is None:
                # Provide default step_size if not given
                step_size = jnp.ones(self.prior.n_dim) * 1e-3
                local_sampler_arg = {**local_sampler_arg, "step_size": step_size}
            elif step_size.ndim == 2:
                # Extract diagonal from DxD matrix
                local_sampler_arg = {**local_sampler_arg, "step_size": jnp.diag(step_size)}
            local_sampler = GaussianRandomWalk(self.posterior, True, **local_sampler_arg)
        else:
            raise ValueError(
                f"Unknown local_sampler_name: {local_sampler_name}. "
                f"Supported options: 'MALA', 'GaussianRandomWalk'"
            )

        # Create normalizing flow model
        rng_key, subkey = jax.random.split(rng_key)
        model = MaskedCouplingRQSpline(
            self.prior.n_dim, num_layers, hidden_size, num_bins, subkey
        )

        # Create flowMC sampler with config parameters, we do not use data dict (therefore, None)
        # TODO: in the future, we need to pass along all kwargs and ensure the kwarg names are correct, etc.
        self.sampler = Sampler(
            self.prior.n_dim,
            rng_key,
            None,  # type: ignore
            local_sampler,
            model,
            n_loop_training=config.n_loop_training,
            n_loop_production=config.n_loop_production,
            n_chains=config.n_chains,
            n_local_steps=config.n_local_steps,
            n_global_steps=config.n_global_steps,
            n_epochs=config.n_epochs,
            learning_rate=config.learning_rate,
            train_thinning=config.train_thinning,
            output_thinning=config.output_thinning,
        )

    def sample(self, key, initial_position=jnp.array([])):
        """
        Run flowMC sampling.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key
        initial_position : Array, optional
            Initial positions for chains. If not provided, samples from prior.

        Notes
        -----
        This method includes a critical bug fix: parameter ordering is preserved
        when converting from dictionary to array using a list comprehension instead
        of jax.tree.leaves().
        """
        if initial_position.size == 0:
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
        self.sampler.sample(initial_position, None)  # type: ignore

    def print_summary(self, transform: bool = True):
        """
        Generate summary of the flowMC run.

        Parameters
        ----------
        transform : bool, optional
            Whether to apply inverse sample transforms to results (default: True)
        """
        train_summary = self.sampler.get_sampler_state(training=True)
        production_summary = self.sampler.get_sampler_state(training=False)

        training_chain = train_summary["chains"].reshape(-1, self.prior.n_dim).T
        training_chain = self.add_name(training_chain)
        if transform:
            for sample_transform in reversed(self.sample_transforms):
                training_chain = jax.vmap(sample_transform.backward)(training_chain)
        training_log_prob = train_summary["log_prob"]
        training_local_acceptance = train_summary["local_accs"]
        training_global_acceptance = train_summary["global_accs"]
        training_loss = train_summary["loss_vals"]

        production_chain = production_summary["chains"].reshape(-1, self.prior.n_dim).T
        production_chain = self.add_name(production_chain)
        if transform:
            for sample_transform in reversed(self.sample_transforms):
                production_chain = jax.vmap(sample_transform.backward)(production_chain)
        production_log_prob = production_summary["log_prob"]
        production_local_acceptance = production_summary["local_accs"]
        production_global_acceptance = production_summary["global_accs"]

        logger.info("Training summary")
        logger.info("=" * 10)
        for key, value in training_chain.items():
            logger.info(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")
        logger.info(
            f"Log probability: {training_log_prob.mean():.3f} +/- {training_log_prob.std():.3f}"
        )
        logger.info(
            f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}"
        )
        logger.info(
            f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}"
        )
        logger.info(
            f"Max loss: {training_loss.max():.3f}, Min loss: {training_loss.min():.3f}"
        )

        logger.info("Production summary")
        logger.info("=" * 10)
        for key, value in production_chain.items():
            logger.info(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")
        logger.info(
            f"Log probability: {production_log_prob.mean():.3f} +/- {production_log_prob.std():.3f}"
        )
        logger.info(
            f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}"
        )
        logger.info(
            f"Global acceptance: {production_global_acceptance.mean():.3f} +/- {production_global_acceptance.std():.3f}"
        )

    def get_samples(self, training: bool = False) -> dict:
        """
        Get the samples from the flowMC sampler.

        Parameters
        ----------
        training : bool, optional
            Whether to get the training samples or the production samples, by default False

        Returns
        -------
        dict
            Dictionary of samples
        """
        if training:
            chains = self.sampler.get_sampler_state(training=True)["chains"]
        else:
            chains = self.sampler.get_sampler_state(training=False)["chains"]

        chains = chains.reshape(-1, self.prior.n_dim)
        chains = jax.vmap(self.add_name)(chains)
        for sample_transform in reversed(self.sample_transforms):
            chains = jax.vmap(sample_transform.backward)(chains)
        return chains

    def get_log_prob(self, training: bool = False) -> Array:
        """
        Get log probabilities from flowMC sampler.

        Parameters
        ----------
        training : bool, optional
            Whether to get training or production log probs (default: False)

        Returns
        -------
        Array
            Log posterior probability values (1D array, flattened across chains)
        """
        if training:
            sampler_state = self.sampler.get_sampler_state(training=True)
        else:
            sampler_state = self.sampler.get_sampler_state(training=False)

        return sampler_state["log_prob"].flatten()

    def get_n_samples(self, training: bool = False) -> int:
        """
        Get number of samples from flowMC sampler.

        Parameters
        ----------
        training : bool, optional
            Whether to count training or production samples (default: False)

        Returns
        -------
        int
            Number of samples (total across all chains)
        """
        log_prob = self.get_log_prob(training=training)
        return len(log_prob)
