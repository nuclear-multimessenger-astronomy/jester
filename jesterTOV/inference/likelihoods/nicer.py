r"""NICER X-ray timing likelihood implementation"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.logging_config import get_logger

from jesterTOV.inference.flows.flow import Flow

logger = get_logger("jester")


class NICERLikelihood(LikelihoodBase):
    """
    NICER likelihood using normalizing flows.

    This likelihood implementation uses pre-trained normalizing flows on
    M-R posteriors for efficient and deterministic likelihood evaluation.

    The likelihood loads pre-trained flow models for one or both of the Amsterdam
    and Maryland analysis groups, and evaluates the likelihood by:
    1. Pre-sampling masses ONCE at initialization (deterministic with seed)
    2. During evaluation: interpolating radius from the EOS for pre-sampled masses
    3. Evaluating the flow log probability at (mass, radius)
    4. Averaging over all samples, then averaging over available groups

    At least one of ``amsterdam_model_dir`` or ``maryland_model_dir`` must be provided.
    If only one group is provided, the likelihood uses only that group.

    Parameters
    ----------
    psr_name : str
        Pulsar name (e.g., "J0030", "J0740", "J0437", "J0614")
    amsterdam_model_dir : str | None
        Path to directory containing Amsterdam flow model
        (flow_weights.eqx, metadata.json, flow_kwargs.json).
        If None, Amsterdam group is omitted.
    maryland_model_dir : str | None
        Path to directory containing Maryland flow model.
        If None, Maryland group is omitted.
    penalty_value : float, optional
        Penalty value for samples where mass exceeds Mtov (default: -99999.0)
    N_masses_evaluation : int, optional
        Number of mass samples per likelihood evaluation (default: 20)
    N_masses_batch_size : int, optional
        Batch size for processing mass samples (default: 10)
    seed : int, optional
        Random seed for pre-sampling masses (default: 42)

    Attributes
    ----------
    psr_name : str
        Pulsar name
    penalty_value : float
        Penalty value for samples where mass exceeds Mtov
    N_masses_evaluation : int
        Number of mass samples per likelihood evaluation
    N_masses_batch_size : int
        Batch size for processing mass samples
    seed : int
        Random seed for deterministic pre-sampling
    amsterdam_flow : Flow | None
        Normalizing flow for Amsterdam M-R posterior, or None if not provided
    maryland_flow : Flow | None
        Normalizing flow for Maryland M-R posterior, or None if not provided
    amsterdam_fixed_mass_samples : Float[Array, "n_samples"] | None
        Pre-sampled mass values from Amsterdam flow (fixed at initialization), or None
    maryland_fixed_mass_samples : Float[Array, "n_samples"] | None
        Pre-sampled mass values from Maryland flow (fixed at initialization), or None
    """

    psr_name: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    seed: int

    def __init__(
        self,
        psr_name: str,
        amsterdam_model_dir: str | None = None,
        maryland_model_dir: str | None = None,
        penalty_value: float = -99999.0,
        N_masses_evaluation: int = 20,
        N_masses_batch_size: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.psr_name = psr_name
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size
        self.seed = seed

        if amsterdam_model_dir is None and maryland_model_dir is None:
            raise ValueError(
                f"At least one of amsterdam_model_dir or maryland_model_dir must be "
                f"provided for {psr_name}."
            )

        key = jax.random.key(seed)
        key_amsterdam, key_maryland = jax.random.split(key)

        if amsterdam_model_dir is not None:
            self.amsterdam_flow, self.amsterdam_fixed_mass_samples = (
                self._load_flow_and_presample(
                    amsterdam_model_dir, key_amsterdam, "Amsterdam"
                )
            )
            print(
                f"Amsterdam flow loaded for {psr_name}. Pre-sampled mass range: "
                f"[{jnp.min(self.amsterdam_fixed_mass_samples):.3f}, "
                f"{jnp.max(self.amsterdam_fixed_mass_samples):.3f}] Msun"
            )
        else:
            self.amsterdam_flow = None
            self.amsterdam_fixed_mass_samples = None

        if maryland_model_dir is not None:
            self.maryland_flow, self.maryland_fixed_mass_samples = (
                self._load_flow_and_presample(
                    maryland_model_dir, key_maryland, "Maryland"
                )
            )
            print(
                f"Maryland flow loaded for {psr_name}. Pre-sampled mass range: "
                f"[{jnp.min(self.maryland_fixed_mass_samples):.3f}, "
                f"{jnp.max(self.maryland_fixed_mass_samples):.3f}] Msun"
            )
        else:
            self.maryland_flow = None
            self.maryland_fixed_mass_samples = None

        self.active_groups: list[tuple[Flow, Float[Array, "n_samples"]]] = [
            (flow, samples)
            for flow, samples in [
                (self.amsterdam_flow, self.amsterdam_fixed_mass_samples),
                (self.maryland_flow, self.maryland_fixed_mass_samples),
            ]
            if flow is not None and samples is not None
        ]

        logger.info(
            f"Loaded {len(self.active_groups)} normalizing flow(s) for {psr_name}"
        )

    def _load_flow_and_presample(
        self,
        model_dir: str,
        key: Array,
        group_name: str,
    ) -> tuple[Flow, Float[Array, "n_samples"]]:
        from jesterTOV.inference.flows.flow import Flow

        logger.info(f"Loading {group_name} flow for {self.psr_name} from {model_dir}")
        flow = Flow.from_directory(model_dir)
        mass_samples: Float[Array, "n_samples"] = flow.sample(
            key, (self.N_masses_evaluation,)
        )[:, 0]
        logger.info(
            f"Pre-sampled {group_name} mass range: "
            f"[{jnp.min(mass_samples):.3f}, {jnp.max(mass_samples):.3f}] Msun"
        )
        return flow, mass_samples

    def _get_preset_model_path(self, psr_name: str, group: str) -> str:
        """
        Get preset model path for a pulsar and analysis group.

        Parameters
        ----------
        psr_name : str
            Pulsar name (e.g., "J0030", "J0740")
        group : str
            Analysis group ("amsterdam" or "maryland")

        Returns
        -------
        str
            Path to preset model directory

        Raises
        ------
        ValueError
            If no preset exists for this pulsar/group combination
        """
        # TODO: Define preset paths once NICER flow models are trained
        # For now, this is a placeholder that will be updated in Phase 3

        # Example preset structure (to be implemented):
        # base_dir = Path(__file__).parent.parent / "flows" / "models" / "nicer_maf"
        # model_dir = base_dir / psr_name / f"{psr_name}_{group}_NICER_model"

        raise NotImplementedError(
            f"Preset model paths for {psr_name} {group} not yet implemented. "
            "Please provide explicit model_dir paths or train NICER flows first "
            "(see TODO_FLOW_TRAINING.md Phase 3)."
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters.

        Uses pre-sampled masses from initialization (deterministic evaluation).

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'radii_EOS': Array of neutron star radii from EOS

        Returns
        -------
        Float
            Log likelihood value for this NICER observation
        """
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        radii_EOS: Float[Array, " n_points"] = params["radii_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        def compute_group_logL(
            flow: Flow, mass_samples: Float[Array, "n_samples"]
        ) -> Float:
            def process_sample(mass: Float) -> Float:
                radius = jnp.interp(mass, masses_EOS, radii_EOS, right=0.0)
                mr_point = jnp.array([[mass, radius]])  # Shape: (1, 2)
                logpdf = flow.log_prob(mr_point)
                return logpdf + jnp.where(mass > mtov, self.penalty_value, 0.0)

            logprobs = jax.lax.map(
                process_sample, mass_samples, batch_size=self.N_masses_batch_size
            )
            return logsumexp(logprobs) - jnp.log(logprobs.shape[0])

        group_logLs = jnp.stack(
            [compute_group_logL(flow, samples) for flow, samples in self.active_groups]
        )
        return logsumexp(group_logLs) - jnp.log(float(group_logLs.shape[0]))
