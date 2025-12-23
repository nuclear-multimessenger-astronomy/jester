"""flowMC sampler implementation and setup"""

import jax
import jax.numpy as jnp
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.proposal.MALA import MALA
from flowMC.proposal.Gaussian_random_walk import GaussianRandomWalk
from flowMC.Sampler import Sampler

from .jester_sampler import JesterSampler
from ..config.schema import SamplerConfig
from ..base import LikelihoodBase, Prior, BijectiveTransform, NtoMTransform


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
    sample_transforms : list[BijectiveTransform], optional
        Bijective transforms applied during sampling (with Jacobians)
    likelihood_transforms : list[NtoMTransform], optional
        N-to-M transforms applied before likelihood evaluation
    local_sampler_name : str, optional
        Name of the local sampler: "MALA" or "GaussianRandomWalk" (default: "GaussianRandomWalk")
    seed : int, optional
        Random seed (default: 0)
    local_sampler_arg : dict, optional
        Arguments for local sampler (e.g., {"step_size": ...})
    num_layers : int, optional
        Number of coupling layers in normalizing flow (default: 10)
    hidden_size : list[int], optional
        Hidden layer sizes for normalizing flow (default: [128, 128])
    num_bins : int, optional
        Number of bins for rational quadratic splines (default: 8)
    **kwargs
        Additional arguments passed to flowMC Sampler:
        - n_loop_training : int
        - n_loop_production : int
        - n_chains : int
        - n_local_steps : int
        - n_global_steps : int
        - n_epochs : int
        - learning_rate : float or optax schedule
        - train_thinning : int
        - output_thinning : int
        - use_global : bool

    Attributes
    ----------
    sampler : flowMC.Sampler
        FlowMC sampler instance
    """

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform] = [],
        likelihood_transforms: list[NtoMTransform] = [],
        local_sampler_name: str = "GaussianRandomWalk",
        seed: int = 0,
        local_sampler_arg: dict = {},
        num_layers: int = 10,
        hidden_size: list[int] = [128, 128],
        num_bins: int = 8,
        **kwargs,
    ):
        # Initialize base class (sets up transforms and parameter names)
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)

        # FlowMC-specific initialization
        rng_key = jax.random.PRNGKey(seed)

        # Select local sampler based on name
        if local_sampler_name == "MALA":
            # MALA uses matrix-vector multiplication, so it can handle full matrices
            local_sampler = MALA(self.posterior, True, **local_sampler_arg)
        elif local_sampler_name == "GaussianRandomWalk":
            # GaussianRandomWalk uses element-wise multiplication, so convert matrix to diagonal
            step_size = local_sampler_arg.get("step_size")
            if step_size is not None and step_size.ndim == 2:
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

        # Create flowMC sampler
        self.sampler = Sampler(
            self.prior.n_dim,
            rng_key,
            None,  # type: ignore
            local_sampler,
            model,
            **kwargs,
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
        This method includes the critical bug fix: parameter ordering is preserved
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

        print("Training summary")
        print("=" * 10)
        for key, value in training_chain.items():
            print(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")
        print(
            f"Log probability: {training_log_prob.mean():.3f} +/- {training_log_prob.std():.3f}"
        )
        print(
            f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}"
        )
        print(
            f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}"
        )
        print(
            f"Max loss: {training_loss.max():.3f}, Min loss: {training_loss.min():.3f}"
        )

        print("Production summary")
        print("=" * 10)
        for key, value in production_chain.items():
            print(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")
        print(
            f"Log probability: {production_log_prob.mean():.3f} +/- {production_log_prob.std():.3f}"
        )
        print(
            f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}"
        )
        print(
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


def setup_flowmc_sampler(
    config: SamplerConfig,
    prior,
    likelihood,
    transform,
    seed: int = 0,
    eps_mass_matrix: float = 1e-3,
    local_sampler_name: str = "GaussianRandomWalk",
):
    """
    Setup flowMC sampler from configuration

    Parameters
    ----------
    config : SamplerConfig
        Sampler configuration
    prior : CombinePrior
        Combined prior object
    likelihood : LikelihoodBase
        Likelihood object (possibly combined)
    transform : JesterTransformBase
        Transform for micro to macro parameters
    seed : int, optional
        Random seed for sampler initialization
    eps_mass_matrix : float, optional
        Overall scaling factor for step size matrix
    local_sampler_name : str, optional
        Name of the local sampler to use: "MALA" or "GaussianRandomWalk" (default: "GaussianRandomWalk")

    Returns
    -------
    FlowMCSampler
        Configured FlowMC sampler instance
    """
    # Setup mass matrix
    mass_matrix = jnp.eye(prior.n_dim)
    local_sampler_arg = {"step_size": mass_matrix * eps_mass_matrix}

    # Create FlowMC sampler
    sampler = FlowMCSampler(
        likelihood,
        prior,
        sample_transforms=[],
        likelihood_transforms=[transform],
        seed=seed,
        local_sampler_name=local_sampler_name,
        local_sampler_arg=local_sampler_arg,
        num_layers=10,
        hidden_size=[128, 128],
        num_bins=8,
        n_loop_training=config.n_loop_training,
        n_loop_production=config.n_loop_production,
        n_chains=config.n_chains,
        n_local_steps=config.n_local_steps,
        n_global_steps=config.n_global_steps,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        train_thinning=config.train_thinning,
        output_thinning=config.output_thinning,
        use_global=True,  # Enable global (NF) proposals
    )

    return sampler
