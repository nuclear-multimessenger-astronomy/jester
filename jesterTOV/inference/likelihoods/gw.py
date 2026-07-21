r"""Gravitational wave event likelihood implementations"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.flows.flow import Flow, disable_x64
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class GWLikelihoodResampled(LikelihoodBase):
    """
    Gravitational wave likelihood for a single GW event using normalizing flow posteriors

    This likelihood evaluates the GW posterior by:
    1. Sampling masses (m1, m2) from the trained normalizing flow
    2. Interpolating tidal deformabilities (Λ1, Λ2) from the EOS
    3. Evaluating the NF log probability on (m1, m2, Λ1, Λ2)

    Parameters
    ----------
    event_name : str
        Name of the GW event (e.g., "GW170817")
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float, optional
        Penalty value for samples where masses exceed Mtov (default: 0.0, i.e. no penalty)
    N_masses_evaluation : int, optional
        Number of mass samples per likelihood evaluation (default: 20)
    N_masses_batch_size : int, optional
        Batch size passed to ``jax.lax.map`` for processing mass samples
        (default: 1, i.e. a plain ``jax.lax.scan`` with no inner batching).
        See the ``GWLikelihood.N_masses_batch_size`` docstring below for the
        speed/memory tradeoff this controls - the same reasoning applies here.

    Attributes
    ----------
    event_name : str
        Name of the GW event
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float
        Penalty value for samples where masses exceed Mtov
    N_masses_evaluation : int
        Number of mass samples per likelihood evaluation
    N_masses_batch_size : int
        Batch size for processing mass samples
    flow : Flow
        Normalizing flow model for this GW event
    """

    event_name: str
    model_dir: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    flow: Flow

    def __init__(
        self,
        event_name: str,
        model_dir: str,
        penalty_value: float = 0.0,
        N_masses_evaluation: int = 20,
        N_masses_batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.event_name = event_name
        self.model_dir = model_dir
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size

        # Load Flow model for this event
        logger.info(f"Loading NF model for {event_name} from {model_dir}")
        self.flow = Flow.from_directory(model_dir)
        logger.info(f"Loaded NF model for {event_name}")

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - '_random_key': Random seed for mass sampling (cast to int64)
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS

        Returns
        -------
        Float
            Log likelihood value for this GW event
        """
        # Extract parameters
        sampled_key = params["_random_key"].astype("int64")
        key = jax.random.key(sampled_key)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        # Sample all N_masses_evaluation samples from NF in one go
        all_nf_samples: Float[Array, "n_samples 2"] = self.flow.sample(
            key, (self.N_masses_evaluation,)
        )

        def process_sample(sample: Float[Array, " 2"]) -> Float:
            """
            Process a single NF sample

            Note: jax.lax.map with batch_size still applies the function to individual
            elements, not batches. The batch_size parameter is for compilation optimization.

            Parameters
            ----------
            sample : Float[Array, " 2"]
                Single sample with [m1, m2]

            Returns
            -------
            Float
                Log probability including penalties for this sample
            """
            m1 = sample[0]
            m2 = sample[1]

            # Interpolate lambdas
            lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
            lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)

            # Evaluate log_prob on single sample
            ml_sample = jnp.array([m1, m2, lambda_1, lambda_2])
            logpdf = self.flow.log_prob(ml_sample)

            # Penalties for masses exceeding Mtov
            penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
            penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)

            # Return log prob + penalties for this sample
            return logpdf + penalty_m1 + penalty_m2

        # Use jax.lax.map with batching for memory-efficient processing
        # batch_size helps with compilation memory, not runtime batching
        all_logprobs = jax.lax.map(
            process_sample, all_nf_samples, batch_size=self.N_masses_batch_size
        )

        # Average over all samples for this event
        log_likelihood = jnp.mean(all_logprobs)

        return log_likelihood


class GWLikelihood(LikelihoodBase):
    """
    Gravitational wave likelihood using pre-sampled masses for deterministic evaluation

    This likelihood improves upon GWLikelihoodResampled by pre-sampling mass pairs once at
    initialization, eliminating the need for the _random_key parameter and providing
    deterministic likelihood evaluations critical for sampler convergence.

    Key improvements over GWLikelihoodResampled:
    1. Deterministic: Same EOS parameters → same likelihood value
    2. No _random_key hack: Uses fixed seed at initialization
    3. Scalable: Can use N=10,000+ samples efficiently on GPU
    4. Fair comparison: All EOS evaluated at identical mass points
    5. Better convergence: Smooth likelihood surface for MCMC/SMC

    The likelihood works by:

    1. Pre-sampling (m1, m2) pairs from the trained flow at initialization
    2. For each EOS evaluation: interpolate Λ1, Λ2 from the candidate EOS at
       the fixed mass points, evaluate flow log_prob on (m1, m2, Λ1_EOS, Λ2_EOS),
       apply penalties for masses exceeding Mtov, and average over all
       pre-sampled mass pairs

    Parameters
    ----------
    event_name : str
        Name of the GW event (e.g., "GW170817")
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float, optional
        Penalty value for samples where masses exceed Mtov (default: 0.0, i.e. no penalty)
    N_masses_evaluation : int, optional
        Number of mass samples to pre-sample (default: 500). This sets the
        size of a Monte Carlo sum over the flow's own mass posterior, so
        larger values reduce estimator noise at the cost of proportionally
        more ``flow.log_prob`` evaluations.
    N_masses_batch_size : int, optional
        Batch size passed to ``jax.lax.map`` for processing the pre-sampled
        mass grid (default: 1). This controls a speed/memory tradeoff that
        matters a lot once this likelihood is evaluated inside an outer
        ``jax.vmap`` over sampler particles/walkers (e.g. SMC).
    seed : int, optional
        Random seed for mass pre-sampling (default: 42)
        Fixed seed ensures reproducibility across runs

    Attributes
    ----------
    event_name : str
        Name of the GW event
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float
        Penalty value for samples where masses exceed Mtov
    N_masses_evaluation : int
        Number of pre-sampled mass pairs
    N_masses_batch_size : int
        Batch size passed to jax.lax.map for processing the mass grid
    seed : int
        Random seed used for pre-sampling
    flow : Flow
        Normalizing flow model for this GW event
    fixed_mass_samples : Float[Array, "n_samples 2"]
        Pre-sampled (m1, m2) pairs from the flow, shape [N, 2]

    Notes
    -----
    This class does NOT require _random_key in the parameter dictionary,
    unlike GWLikelihoodResampled. The seed is only used once at initialization.

    N_masses_batch_size controls how the mass grid is pushed through
    jax.lax.map. The default (1) keeps memory flat as N_masses_evaluation and
    the number of combined GW events grow, at the cost of standalone
    (non-vmapped) evaluations being a few ms slower than the largest-batch
    alternative - a good trade for production runs.

    Note: the ``type: "gw"`` YAML config (``GWLikelihoodConfig``) does not
    construct this class directly for multi-event runs - it builds one
    ``StackedGWLikelihood`` covering all configured events instead (see that
    class's docstring). ``GWLikelihood`` remains directly importable/usable
    on its own, e.g. for a single event or outside the config system.

    Examples
    --------
    Configure in YAML::

        likelihoods:
          - type: "gw"
            enabled: true
            parameters:
              events:
                - name: "GW170817"
              N_masses_evaluation: 500   # Default value
              N_masses_batch_size: 1     # Default value
              seed: 42
    """

    event_name: str
    model_dir: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    seed: int
    flow: Flow
    fixed_mass_samples: Float[Array, "n_samples 2"]

    def __init__(
        self,
        event_name: str,
        model_dir: str,
        penalty_value: float = 0.0,
        N_masses_evaluation: int = 500,
        N_masses_batch_size: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.event_name = event_name
        self.model_dir = model_dir
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size
        self.seed = seed

        # Load Flow model for this event
        logger.info(f"Loading NF model for {event_name} from {model_dir}")
        self.flow = Flow.from_directory(model_dir)
        logger.info(f"Loaded NF model for {event_name}")

        # Pre-sample masses ONCE at initialization
        logger.info(
            f"Pre-sampling {N_masses_evaluation} mass pairs with seed={seed} for {event_name}"
        )
        key = jax.random.key(seed)
        samples = self.flow.sample(key, (N_masses_evaluation,))
        # Extract only (m1, m2), discard Lambda values from flow
        self.fixed_mass_samples = samples[:, :2]  # Shape: [N, 2]
        logger.info(
            f"Pre-sampled mass range: m1=[{jnp.min(self.fixed_mass_samples[:, 0]):.3f}, "
            f"{jnp.max(self.fixed_mass_samples[:, 0]):.3f}] Msun, "
            f"m2=[{jnp.min(self.fixed_mass_samples[:, 1]):.3f}, "
            f"{jnp.max(self.fixed_mass_samples[:, 1]):.3f}] Msun"
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS

            Note: Does NOT require '_random_key' (unlike GWLikelihood)

        Returns
        -------
        Float
            Log likelihood value for this GW event
        """
        # Extract EOS parameters (no _random_key needed!)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        def process_sample(sample: Float[Array, " 2"]) -> Float:
            """
            Process a single pre-sampled mass pair

            Note: jax.lax.map with batch_size applies function to individual
            elements. The batch_size parameter is for compilation optimization.

            Parameters
            ----------
            sample : Float[Array, " 2"]
                Pre-sampled mass pair [m1, m2]

            Returns
            -------
            Float
                Log probability including penalties for this sample
            """
            m1 = sample[0]
            m2 = sample[1]

            # Interpolate lambdas from candidate EOS
            lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
            lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)

            # Evaluate log_prob on single sample
            ml_sample = jnp.array([m1, m2, lambda_1, lambda_2])
            logpdf = self.flow.log_prob(ml_sample)

            # Penalties for masses exceeding Mtov
            penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
            penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)

            # Return log prob + penalties for this sample
            return logpdf + penalty_m1 + penalty_m2

        # Use jax.lax.map with batching for memory-efficient processing
        # Process all pre-sampled mass pairs
        all_logprobs = jax.lax.map(
            process_sample, self.fixed_mass_samples, batch_size=self.N_masses_batch_size
        )

        # Take logsumexp over all pre-sampled mass pairs
        log_likelihood = logsumexp(all_logprobs) - jnp.log(self.N_masses_evaluation)

        return log_likelihood


_FLOW_ARCHITECTURE_KEYS = (
    "flow_type",
    "nn_depth",
    "nn_block_dim",
    "nn_width",
    "flow_layers",
    "invert",
    "cond_dim",
    "transformer_type",
    "transformer_knots",
    "transformer_interval",
)


def _flow_architecture_signature(flow: Flow) -> tuple:
    """Hashable summary of a Flow's architecture, for cross-event compatibility checks.

    Only includes the ``flow_kwargs`` entries that ``create_flow`` (flows/flow.py)
    consumes to build the architecture, plus the data dimensionality and
    standardization method -- everything that determines the pytree *structure* of
    ``flow.flow`` and thus whether its weights can be stacked with another event's.
    Excludes 'seed' and per-event data statistics ('standardize', 'data_mean',
    'data_std', ...) that legitimately differ across events and are stored in the
    same ``flow_kwargs.json`` file but do not affect stackability.
    """
    kwargs = {
        k: flow.flow_kwargs.get(k)
        for k in _FLOW_ARCHITECTURE_KEYS
        if k in flow.flow_kwargs
    }
    return (
        tuple(sorted(kwargs.items())),
        flow.flow.shape[0],
        flow.standardization_method,
    )


class StackedGWLikelihood(LikelihoodBase):
    """
    Gravitational wave likelihood for many events, evaluated as one batched
    computation instead of one per event.

    Motivation
    ----------
    ``CombinedLikelihood.evaluate`` (combined.py) sums one ``GWLikelihood.evaluate()``
    call per event via a plain Python list comprehension.
    This class replaces the per-event Python loop with a single ``jax.lax.map`` over
    a *stacked* pytree of per-event flow weights (all events must share the same flow
    architecture -- see ``_flow_architecture_signature``, checked eagerly at
    construction time with a clear error otherwise).

    Numerically, this computes exactly ``sum(GWLikelihood(...).evaluate(params) for
    each event)`` (i.e. a drop-in replacement for combining N ``GWLikelihood``
    instances via ``CombinedLikelihood``).

    Parameters
    ----------
    event_names : list[str]
        Names of the GW events (for error messages/logging only).
    model_dirs : list[str]
        Paths to each event's trained normalizing flow model directory, same
        order as ``event_names``.
    penalty_value : float, optional
        Penalty value for samples where masses exceed Mtov (default: 0.0).
    N_masses_evaluation : int, optional
        Number of pre-sampled mass pairs per event (default: 500). See the
        same parameter on ``GWLikelihood`` for the accuracy/cost tradeoff.
    N_masses_batch_size : int, optional
        Batch size for ``jax.lax.map`` over mass samples, per event (default: 1,
        matching ``GWLikelihood``'s default - see its docstring for the tradeoff).
    event_batch_size : int | None, optional
        Batch size for ``jax.lax.map`` over events (default: 1, i.e. a plain scan
        over events - the safe default for production SMC runs with many
        particles and/or many events.
    seed : int, optional
        Random seed for mass pre-sampling, same seed used for every event (matches
        ``GWLikelihood``'s default behaviour; events still get distinct samples
        because their flows differ).
    use_float32 : bool, optional
        Evaluate the flows in float32 instead of the default float64 (default:
        False). Training is unaffected either way: this only changes how the already
        trained flow is loaded and evaluated.

    Raises
    ------
    ValueError
        If any two events' flows have a different architecture (flow type, layer
        widths/depths, dimensionality, or standardization method), since their
        weight pytrees then cannot be stacked into one array. Also raised if
        ``use_float32=True`` is requested for an architecture that hasn't been
        validated for float32 evaluation.

    Examples
    --------
    Configure in YAML (built automatically for every ``type: "gw"`` config -
    not constructed directly)::

        likelihoods:
          - type: "gw"
            enabled: true
            parameters:
              events:
                - name: "GW170817"
                - name: "GW190425"
              N_masses_evaluation: 500   # Default value
              N_masses_batch_size: 1     # Default value
              event_batch_size: 1        # Default value
              seed: 42
              use_float32: false         # Default value
    """

    event_names: list[str]
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    event_batch_size: int
    seed: int
    standardization_method: str
    use_float32: bool

    def __init__(
        self,
        event_names: list[str],
        model_dirs: list[str],
        penalty_value: float = 0.0,
        N_masses_evaluation: int = 500,
        N_masses_batch_size: int = 1,
        event_batch_size: int | None = 1,
        seed: int = 42,
        use_float32: bool = False,
    ) -> None:
        super().__init__()
        if len(event_names) != len(model_dirs):
            raise ValueError(
                f"event_names ({len(event_names)}) and model_dirs "
                f"({len(model_dirs)}) must have the same length"
            )

        self.event_names = event_names
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size
        self.event_batch_size = event_batch_size or len(event_names)
        self.seed = seed
        self.use_float32 = use_float32

        logger.info(
            f"Loading NF models for {len(event_names)} GW events "
            f"(stacked/batched evaluation, event_batch_size={self.event_batch_size}, "
            f"use_float32={self.use_float32})"
        )
        _dtype = "float32" if use_float32 else "float64"
        flows = [Flow.from_directory(d, dtype=_dtype) for d in model_dirs]

        # Fail fast with a clear message if architectures don't match, rather than
        # a confusing jax.tree_util error from jnp.stack deep inside __init__.
        ref_signature = _flow_architecture_signature(flows[0])
        mismatched = [
            name
            for name, flow in zip(event_names[1:], flows[1:])
            if _flow_architecture_signature(flow) != ref_signature
        ]
        if mismatched:
            raise ValueError(
                "StackedGWLikelihood requires all events to share the same flow "
                "architecture (flow_type, nn_width/nn_depth/flow_layers/..., data "
                f"dimensionality, and standardization method) as '{event_names[0]}' "
                f"so their weights can be stacked into one pytree. Events with a "
                f"different architecture: {mismatched}. If these events genuinely "
                "use different architectures, evaluate them via separate "
                "GWLikelihood instances combined through CombinedLikelihood instead."
            )

        # Split each event's flowjax model into (weights, architecture) and stack
        # the weights along a new leading "event" axis. `static` (the architecture)
        # is identical across events by the check above, so any one copy works for
        # eqx.combine when reconstructing a per-event flow inside evaluate().
        dynamic_list, static_list = zip(
            *(eqx.partition(flow.flow, eqx.is_array) for flow in flows)
        )
        self._stacked_dynamic = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves), *dynamic_list
        )
        self._static = static_list[0]

        # Stack the per-event (de)standardization arrays the same way -- these are
        # plain jnp arrays already, not part of the flowjax pytree.
        self.standardization_method = flows[0].standardization_method
        if self.standardization_method == "zscore":
            self._loc = jnp.stack([flow.data_mean for flow in flows])
            self._scale = jnp.stack([flow.data_std for flow in flows])
        else:
            # "minmax" or "none" (identity: data_min=0, data_range=1)
            self._loc = jnp.stack([flow.data_min for flow in flows])
            self._scale = jnp.stack([flow.data_range for flow in flows])

        # Pre-sample masses ONCE at initialization, per event, mirroring
        # GWLikelihood's behaviour of using the same fixed seed for every event.
        key = jax.random.key(seed)

        def sample_one_event(dynamic_leaf, loc, scale):
            flow = eqx.combine(dynamic_leaf, self._static)
            std_samples = flow.sample(key, (N_masses_evaluation,))
            samples = std_samples * scale + loc
            return samples[:, :2]  # (m1, m2) only

        # For float32, this pre-sampling must run inside the same disable_x64()
        # scope as the flow construction above. Doing only the weight-loading
        # under disable_x64() but not this vmap silently raises to float64
        vmapped_sample = jax.vmap(sample_one_event)
        sample_args = (self._stacked_dynamic, self._loc, self._scale)
        if self.use_float32:
            with disable_x64():
                fixed_mass_samples = vmapped_sample(*sample_args)
        else:
            fixed_mass_samples = vmapped_sample(*sample_args)
        self._fixed_mass_samples: Float[Array, "n_events n_samples 2"] = (
            fixed_mass_samples
        )

        logger.info(f"Pre-sampled and stacked flows for {len(event_names)} GW events")

    def _log_prob_one_event(
        self, dynamic_leaf, loc, scale, ml_sample: Float[Array, " 4"]
    ) -> Float:
        """Mirrors Flow.log_prob for one event's (already-combined) flow."""
        flow = eqx.combine(dynamic_leaf, self._static)
        x_std = (ml_sample - loc) / scale
        log_p = flow.log_prob(x_std)
        log_det_jacobian = -jnp.sum(jnp.log(scale))
        return log_p + log_det_jacobian

    def evaluate_per_event(
        self, params: dict[str, Float | Array]
    ) -> Float[Array, " n_events"]:
        """
        Evaluate the log likelihood of each event separately, without summing.

        Used by :class:`~jesterTOV.inference.samplers.blackjax.smc.partial_posteriors.BlackJAXPartialPosteriorsRandomWalkSampler`,
        which needs to mask individual events in and out during data
        tempering -- ``evaluate()`` only exposes their sum.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS

        Returns
        -------
        Float[Array, "n_events"]
            Log likelihood of each event, in ``self.event_names`` order.
        """
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)
        n_masses_evaluation = self.N_masses_evaluation

        def process_one_event(carry) -> Float:
            dynamic_leaf, loc, scale, mass_samples = carry

            def process_sample(sample: Float[Array, " 2"]) -> Float:
                m1, m2 = sample[0], sample[1]
                lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
                lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)
                ml_sample = jnp.array([m1, m2, lambda_1, lambda_2])
                logpdf = self._log_prob_one_event(dynamic_leaf, loc, scale, ml_sample)
                penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
                penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)
                return logpdf + penalty_m1 + penalty_m2

            # For float32, disable_x64() must wrap the ENTIRE per-event mass-sample
            # map, not just the innermost flow.log_prob call. Wrapping only the
            # innermost call breaks once this is embedded under the sampler's own
            # outer vmap over particles (which every SMC production run has).
            if self.use_float32:
                with disable_x64():
                    all_logprobs = jax.lax.map(
                        process_sample,
                        mass_samples,
                        batch_size=self.N_masses_batch_size,
                    )
            else:
                all_logprobs = jax.lax.map(
                    process_sample, mass_samples, batch_size=self.N_masses_batch_size
                )
            return logsumexp(all_logprobs) - jnp.log(n_masses_evaluation)

        per_event_loglike = jax.lax.map(
            process_one_event,
            (self._stacked_dynamic, self._loc, self._scale, self._fixed_mass_samples),
            batch_size=self.event_batch_size,
        )
        return per_event_loglike

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate summed log likelihood over all events for given EOS parameters.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS

        Returns
        -------
        Float
            Sum of log likelihoods over all events (matches summing individual
            GWLikelihood.evaluate() calls through CombinedLikelihood).
        """
        return jnp.sum(self.evaluate_per_event(params))
