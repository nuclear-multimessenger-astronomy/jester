"""Pydantic models for sampler configuration."""

from typing import Literal, Union, Annotated
from pydantic import Field, field_validator, model_validator, ConfigDict, Discriminator

from ._base import JesterBaseModel


class BaseSamplerConfig(JesterBaseModel):
    """Base configuration for all samplers.

    This base class provides common fields shared by all sampler types.
    Each subclass must define its own 'type' field with a specific literal value
    for use as a discriminator in the SamplerConfig union.

    Attributes
    ----------
    output_dir : str
        Directory to save results
    n_eos_samples : int
        Number of EOS samples to generate after inference (default: 10000)
    log_prob_batch_size : int
        Batch size for computing log probabilities and generating EOS samples (default: 1000)
    """

    model_config = ConfigDict(extra="forbid")

    output_dir: str = "./outdir/"
    n_eos_samples: int = 10_000
    log_prob_batch_size: int = 1000

    @field_validator("n_eos_samples", "log_prob_batch_size")
    @classmethod
    def _validate_base_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v


class FlowMCSamplerConfig(BaseSamplerConfig):
    """Configuration for FlowMC sampler (normalizing flow-enhanced MCMC).

    Attributes
    ----------
    type : Literal["flowmc"]
        Sampler type identifier
    n_chains : int
        Number of parallel chains
    n_loop_training : int
        Number of training loops
    n_loop_production : int
        Number of production loops
    n_local_steps : int
        Number of local MCMC steps per loop
    n_global_steps : int
        Number of global steps per loop
    n_epochs : int
        Number of training epochs for normalizing flow
    learning_rate : float
        Learning rate for flow training
    train_thinning : int
        Thinning factor for training samples (default: 1)
    output_thinning : int
        Thinning factor for output samples (default: 5)
    output_dir : str
        Directory to save results
    n_eos_samples : int
        Number of EOS samples to generate after inference (default: 10000)
    """

    type: Literal["flowmc"] = "flowmc"
    n_chains: int = 20
    n_loop_training: int = 3
    n_loop_production: int = 3
    n_local_steps: int = 100
    n_global_steps: int = 100
    n_epochs: int = 30
    learning_rate: float = 0.001
    train_thinning: int = 1
    output_thinning: int = 5

    @field_validator(
        "n_chains",
        "n_loop_training",
        "n_loop_production",
        "n_local_steps",
        "n_global_steps",
        "n_epochs",
        "train_thinning",
        "output_thinning",
    )
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def _validate_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v


class BlackJAXNSAWConfig(BaseSamplerConfig):
    """Configuration for BlackJAX Nested Sampling with Acceptance Walk.

    Attributes
    ----------
    type : Literal["blackjax-ns-aw"]
        Sampler type identifier
    n_live : int
        Number of live points (default: 1000)
    n_delete_frac : float
        Fraction of live points to delete per iteration (default: 0.5)
    n_target : int
        Target number of accepted MCMC steps (default: 60)
    max_mcmc : int
        Maximum MCMC steps per iteration (default: 5000)
    max_proposals : int
        Maximum proposal attempts per MCMC step (default: 1000)
    termination_dlogz : float
        Evidence convergence criterion (default: 0.1)
    output_dir : str
        Directory to save results
    n_eos_samples : int
        Number of EOS samples to generate after inference (default: 10000)
    """

    type: Literal["blackjax-ns-aw"] = "blackjax-ns-aw"
    n_live: int = 1000
    n_delete_frac: float = 0.5
    n_target: int = 60
    max_mcmc: int = 5000
    max_proposals: int = 1000
    termination_dlogz: float = 0.1

    @field_validator("n_delete_frac")
    @classmethod
    def _validate_delete_frac(cls, v: float) -> float:
        if v <= 0 or v > 1:
            raise ValueError(f"n_delete_frac must be in (0, 1], got: {v}")
        return v

    @field_validator("n_live", "n_target", "max_mcmc", "max_proposals")
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v


class SMCRandomWalkParamsMixin(JesterBaseModel):
    """Shared knob set for the SMC-RW kernel: particles, MCMC steps, target ESS,
    random-walk sigma, and the optional adaptive step-size machinery.

    Mixed into both :class:`SMCRandomWalkSamplerConfig` (the standalone sampler) and
    :class:`InnerSMCRandomWalkConfig` (the per-event ramp-in loop inside
    :class:`SMCPartialPosteriorsRandomWalkSamplerConfig`), so that any new SMC-RW knob
    only needs to be added here to be available in both places.

    Attributes
    ----------
    n_particles : int
        Number of particles (default: 10000)
    n_mcmc_steps : int
        Number of MCMC steps per tempering level (default: 1)
    target_ess : float
        Target effective sample size for adaptive tempering (default: 0.9)
    random_walk_sigma : float
        Fixed sigma scaling for Gaussian random walk kernel (default: 1.0).
        The proposal covariance is computed from particles and scaled by sigma^2.
        Default of 1.0 uses the empirical covariance directly.
        When `adaptive_step_size` is enabled, this is only the *starting* value: it is
        pretuned before annealing begins and then continuously adapted per particle.
    adaptive_step_size : bool
        Enable per-particle adaptive step size targeting `target_acceptance_rate` (default: False).
        Recommended for high-SNR signals (e.g. ET), where the posterior narrows quickly during
        annealing and a fixed sigma causes the acceptance rate to collapse. Uses BlackJAX's
        Robbins-Monro update (`update_scale_from_acceptance_rate`), which is the standard
        Roberts-Rosenthal optimal-scaling approach for random-walk Metropolis.
    target_acceptance_rate : float
        Target acceptance rate for adaptive step size (default: 0.234, the standard
        Roberts-Rosenthal optimal value for random-walk Metropolis in high dimensions).
        Only used when `adaptive_step_size` is True.
    n_pretune_steps : int
        Number of pilot Metropolis steps run on the initial (prior) particles, before annealing
        starts, to calibrate a good initial step size regardless of how `random_walk_sigma` was
        set (default: 20). Only used when `adaptive_step_size` is True. Set to 0 to disable
        pretuning and start annealing directly from `random_walk_sigma`.
    """

    model_config = ConfigDict(extra="forbid")

    n_particles: int = 10000
    n_mcmc_steps: int = 1
    target_ess: float = 0.9
    random_walk_sigma: float = 1.0
    adaptive_step_size: bool = False
    target_acceptance_rate: float = 0.234
    n_pretune_steps: int = 20

    @field_validator("n_particles", "n_mcmc_steps")
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v

    @field_validator("n_pretune_steps")
    @classmethod
    def _validate_nonnegative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Value must be non-negative, got: {v}")
        return v

    @field_validator("target_ess", "target_acceptance_rate")
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if v <= 0 or v > 1:
            raise ValueError(f"Value must be in (0, 1], got: {v}")
        return v

    @field_validator("random_walk_sigma")
    @classmethod
    def _validate_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v


class SMCRandomWalkSamplerConfig(BaseSamplerConfig, SMCRandomWalkParamsMixin):
    """Configuration for Sequential Monte Carlo with Random Walk kernel.

    See :class:`SMCRandomWalkParamsMixin` for the full knob set (particles, MCMC
    steps, target ESS, random-walk sigma, adaptive step size).

    Attributes
    ----------
    type : Literal["smc-rw"]
        Sampler type identifier
    """

    type: Literal["smc-rw"] = "smc-rw"


class SMCNUTSSamplerConfig(BaseSamplerConfig):
    """Configuration for Sequential Monte Carlo with NUTS kernel (EXPERIMENTAL).

    WARNING: This sampler is experimental and should be used with caution.

    Attributes
    ----------
    type : Literal["smc-nuts"]
        Sampler type identifier
    n_particles : int
        Number of particles (default: 10000)
    n_mcmc_steps : int
        Number of MCMC steps per tempering level (default: 1)
    target_ess : float
        Target effective sample size for adaptive tempering (default: 0.9)
    init_step_size : float
        Initial NUTS step size (default: 1e-2)
    mass_matrix_base : float
        Base value for diagonal mass matrix (default: 2e-1)
    mass_matrix_param_scales : dict[str, float]
        Per-parameter scaling for mass matrix (default: {})
    target_acceptance : float
        Target acceptance rate (default: 0.7)
    adaptation_rate : float
        Adaptation rate for step size tuning (default: 0.3)
    """

    type: Literal["smc-nuts"] = "smc-nuts"
    n_particles: int = 10000
    n_mcmc_steps: int = 1
    target_ess: float = 0.9
    init_step_size: float = 1e-2
    mass_matrix_base: float = 2e-1
    mass_matrix_param_scales: dict[str, float] = Field(default_factory=dict)
    target_acceptance: float = 0.7
    adaptation_rate: float = 0.3

    @field_validator("n_particles", "n_mcmc_steps")
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v

    @field_validator("target_ess", "target_acceptance", "adaptation_rate")
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if v <= 0 or v > 1:
            raise ValueError(f"Value must be in (0, 1], got: {v}")
        return v

    @field_validator("init_step_size", "mass_matrix_base")
    @classmethod
    def _validate_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v


class InnerSMCRandomWalkConfig(SMCRandomWalkParamsMixin):
    """Configuration for the inner adaptive-tempering SMC-RW loop that ramps
    in a single event within :class:`SMCPartialPosteriorsRandomWalkSamplerConfig`.

    Identical knob set to :class:`SMCRandomWalkSamplerConfig` (both inherit from
    :class:`SMCRandomWalkParamsMixin`: particles, MCMC steps, target ESS, random-walk
    sigma, adaptive step size), but scoped to the per-event ramp-in loop rather than
    the outer event-assimilation orchestration -- see ``inner`` on the
    partial-posteriors config. Any new field added to the mixin is automatically
    available here too.
    """


class SMCPartialPosteriorsRandomWalkSamplerConfig(BaseSamplerConfig):
    r"""Configuration for SMC on the "path of partial posteriors" (data
    tempering / IBIS, Chopin 2002; see Dai, Heng, Jacob & Whiteley,
    "An invitation to sequential Monte Carlo samplers", arXiv:2007.11936,
    "Path of partial posteriors" section) with a Random Walk kernel.

    Unlike :class:`SMCRandomWalkSamplerConfig`, which tempers a single
    inverse-temperature :math:`\lambda` from 0 to 1 over the full combined
    likelihood, this sampler tempers by *number of GW events included*:
    each configured GW event's likelihood term is turned on one at a time,
    exposing more of the combined likelihood as sampling progresses. This
    enables both (a) a single run that gradually assimilates events, and
    (b) warm-starting a new run from a previous run's converged posterior
    when a new event becomes available (see the sampler class docstring).

    Turning an event's mask on in a single SMC step is measurably biased
    for informative events (verified against a closed-form conjugate
    Gaussian model): the underlying ``blackjax.smc.base.step`` reweights
    *after* the MCMC rejuvenation move, which is only a good approximation
    for small target-to-target jumps. Each event is therefore ramped in
    over several small fractional mask increments rather than a single
    jump, matching the source paper's own recommendation of "a geometric
    path between successive partial posteriors": an ESS-targeting bisection
    search (``blackjax.smc.ess.ess_solver`` + ``blackjax.smc.solver.dichotomy``),
    the same machinery used by ``smc-rw``'s adaptive :math:`\lambda` schedule,
    applied to the mask fraction instead. Because the mask-weighted
    logposterior is linear in the fraction of a single event being ramped
    in, the same solver machinery applies unmodified. The number of
    sub-steps is uncapped: the search runs until the mask fraction reaches
    1.0, however many sub-steps that takes.

    This config is deliberately split into two levels: the top level only
    orchestrates the event-assimilation procedure (which events, in what
    order, warm-starting), while ``inner`` fully specifies the adaptive
    SMC-RW loop used to ramp in each individual event.

    Attributes
    ----------
    type : Literal["smc-partial-posteriors-rw"]
        Sampler type identifier
    event_order : list[str] | None
        Order in which GW events are assimilated. ``None`` (default) uses
        the order the events appear in the likelihood configuration. This
        order is recorded in the result metadata and must match (as a
        strict prefix) between an initial run and any later warm-started
        run that adds more events.
    warm_start_from : str | None
        Path to a previous run's ``InferenceResult`` HDF5 file. When set,
        the initial particles are resampled from that run's posterior
        (instead of the prior) and its ``metadata["event_order"]`` is
        required to be a strict prefix of this run's ``event_order``: the
        already-covered events are skipped entirely (their mask entries
        start at 1, not ramped again) and only the newly added event(s) are
        assimilated. The source run may have zero GW events (e.g. a radio-
        or ChiEFT-only run): the empty list is trivially a strict prefix,
        so its posterior only seeds the initial particles and every
        configured GW event is assimilated from scratch. ``None`` (default)
        starts from the prior, assimilating every configured event as in a
        single from-scratch run.
    cadence : int | list[int] | Literal["auto"]
        Controls how many *new* events (i.e. events not already covered by
        ``warm_start_from``) are turned on together per data-tempering
        step, instead of the default one-event-at-a-time. All events
        within a group are ramped in jointly, sharing a single mask
        fraction that is bisected up from 0 to 1 exactly like the
        single-event case (the group's combined log-likelihood is still
        linear in that shared fraction, so the same ESS-targeting
        machinery applies unmodified).

        - An ``int`` (default: ``1``) chunks the new events into
          fixed-size groups of that many events, in order; the final group
          takes the remainder if the new-event count doesn't divide evenly.
        - A ``list[int]`` gives explicit group sizes, e.g. ``[10, 20, 30,
          50]`` to process 100 new events as four steps of that size. The
          list must sum to exactly the number of new events for this run;
          this is checked at sample time (once ``warm_start_from`` has been
          resolved), not at config-parse time.
        - ``"auto"`` (or ``"automatic"``) builds each group dynamically
          instead of using a fixed schedule: new events are added to a
          pending queue one at a time, and after each addition a *full-jump*
          ESS is computed from the queue's combined log-likelihood against
          the *current* particle cloud, with no annealing and no MCMC move
          (see ``_predict_full_jump_ess`` in
          ``samplers/blackjax/smc/partial_posteriors.py`` -- this is exactly
          the diagnostic ``ess_solver``/``dichotomy`` already evaluate as
          their first probe at the start of any sub-step bisection, just run
          standalone before committing to an update). While that ESS stays
          at or above ``auto_ess_threshold``, the queued events are
          quantitatively "not surprising" and are left queued rather than
          triggering an update; once it drops below the threshold (or the
          last configured event is reached), the whole queue is assimilated
          together via the normal, unbiased sub-step bisection, exactly as
          for a fixed-size group. Only reads ``auto_ess_threshold`` when set
          to this value.
    auto_ess_threshold : float | None
        Full-jump ESS threshold (fraction of ``n_particles``, in (0, 1])
        used only when ``cadence == "auto"``; ignored otherwise. Defaults to
        ``None``, which falls back to ``inner.target_ess`` at sample time --
        the same threshold the sub-step bisection itself targets, since the
        auto-cadence check reuses that bisection's own first-probe
        computation as a pre-check (see ``cadence``'s docstring). Setting
        this explicitly is only accepted when ``cadence == "auto"``; setting
        it with any other ``cadence`` raises a validation error, since it
        would otherwise silently do nothing.
    inner : InnerSMCRandomWalkConfig
        Configuration of the adaptive SMC-RW loop used to ramp in each
        event group's mask fraction from 0 to 1 (particles, MCMC steps per
        sub-step, target ESS, random-walk sigma).
    save_intermediate_results : bool
        When ``True``, save a full ``InferenceResult`` HDF5 (posterior
        samples, derived EOS quantities via the TOV solver, and metadata)
        after each data-tempering step, to
        ``outdir/substep_results/results_<n>.h5`` where ``<n>`` is the
        1-indexed position of the step's first event within the full
        configured ``event_order`` (consistent across warm-started runs).
        This lets users inspect how the posterior evolves as GW events are
        assimilated one batch at a time, e.g. to identify which events are
        most informative. Default ``True``; adds a TOV solve per step, so
        set to ``False`` to skip it for runs that don't need the
        intermediate posteriors.
    """

    type: Literal["smc-partial-posteriors-rw"] = "smc-partial-posteriors-rw"
    event_order: list[str] | None = None
    warm_start_from: str | None = None
    cadence: int | list[int] | Literal["auto", "automatic"] = 1
    auto_ess_threshold: float | None = None
    inner: InnerSMCRandomWalkConfig = Field(default_factory=InnerSMCRandomWalkConfig)
    save_intermediate_results: bool = True

    @field_validator("cadence")
    @classmethod
    def _validate_cadence(
        cls, v: int | list[int] | Literal["auto", "automatic"]
    ) -> int | list[int] | Literal["auto", "automatic"]:
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("cadence list must not be empty")
            if any(n <= 0 for n in v):
                raise ValueError(f"All cadence list entries must be positive, got: {v}")
        elif isinstance(v, int) and v <= 0:
            raise ValueError(f"cadence must be positive, got: {v}")
        return v

    @field_validator("auto_ess_threshold")
    @classmethod
    def _validate_auto_ess_threshold(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 < v <= 1.0):
            raise ValueError(f"auto_ess_threshold must be in (0, 1], got: {v}")
        return v

    @model_validator(mode="after")
    def _check_auto_ess_threshold_requires_auto_cadence(self):
        is_auto_cadence = isinstance(self.cadence, str) and self.cadence in (
            "auto",
            "automatic",
        )
        if self.auto_ess_threshold is not None and not is_auto_cadence:
            raise ValueError(
                "auto_ess_threshold is only meaningful when cadence is "
                f'"auto"/"automatic" (got cadence={self.cadence!r}) -- it '
                "would otherwise silently have no effect. Remove it or set "
                'cadence to "auto".'
            )
        return self

    @property
    def is_auto_cadence(self) -> bool:
        """Whether ``cadence`` requests dynamic, ESS-triggered grouping."""
        return isinstance(self.cadence, str) and self.cadence in ("auto", "automatic")

    @property
    def resolved_auto_ess_threshold(self) -> float:
        """The full-jump ESS threshold to use in auto-cadence mode.

        Falls back to ``inner.target_ess`` when ``auto_ess_threshold`` isn't
        explicitly set (see the class docstring).
        """
        if self.auto_ess_threshold is not None:
            return self.auto_ess_threshold
        return self.inner.target_ess

    def __getattr__(self, name: str):
        """Delegate any :class:`SMCRandomWalkParamsMixin` field (``n_particles``,
        ``random_walk_sigma``, ``adaptive_step_size``, ...) to ``inner``.

        ``BlackJAXSMCRandomWalkSampler._setup_mcmc_kernel`` (reused by the
        partial-posteriors sampler for kernel construction) reads these directly off
        ``self.config``, which here is ``self`` rather than ``inner``. Delegating via
        ``__getattr__`` over the mixin's field names (rather than one hand-written
        ``@property`` per field) means a new field added to
        :class:`SMCRandomWalkParamsMixin` is automatically readable here too, with no
        separate edit required.
        """
        if name in SMCRandomWalkParamsMixin.model_fields:
            return getattr(self.inner, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class EOSReweightingConfig(BaseSamplerConfig):
    r"""Configuration for EOS reweighting sampler.

    This sampler evaluates jester's GPU-accelerated likelihoods on a discrete
    set of tabulated EOS curves (M, :math:`\Lambda`, R tables), not necessarily
    produced by jester itself, rather than sampling a parametric EOS model.
    It computes the marginal log-likelihood per EOS and the Bayesian evidence :math:`\log Z`.

    EOS tables must be NPZ files with keys:

    - ``masses``: 1D float64 array in :math:`M_\odot`, monotone increasing
    - ``lambdas``: 1D float64 array, dimensionless tidal deformability
    - ``radii``: 1D float64 array in km (required)

    For a file containing N EOS curves: arrays shaped ``[N, n_points]``.

    Attributes
    ----------
    type : Literal["eos-reweighting"]
        Sampler type identifier
    eos_file : str
        Path to an NPZ file containing EOS curves
    n_grid : int
        Number of mass grid points for common interpolation grid (default: 200)
    m_min : float
        Minimum mass for interpolation grid in :math:`M_\odot` (default: 1.0)
    m_max : float | None
        Maximum mass for grid in :math:`M_\odot`. None → use max(M_TOV) across
        all curves, capped at 3.0 :math:`M_\odot`.
    batch_size : int
        Number of EOS curves processed simultaneously by JAX (default: 1000).
        Progress is logged after each batch. Choose a value that fits in
        memory; there is no automatic OOM recovery.
    n_resample : int | None
        Number of posterior EOS samples to draw (with replacement, weighted
        by the normalised posterior weights) after evidence computation.
        ``None`` (default) uses the Kish effective sample size
        :math:`N_\mathrm{eff}`, rounded to the nearest integer.
    """

    type: Literal["eos-reweighting"] = "eos-reweighting"

    eos_file: str

    n_grid: int = Field(default=200, gt=10)
    m_min: float = Field(default=1.0, gt=0.0)
    m_max: float | None = None

    batch_size: int = Field(
        default=1000,
        gt=0,
        description="Number of EOS curves processed simultaneously by JAX. Progress is logged after each batch.",
    )

    n_resample: int | None = Field(
        default=None,
        description=(
            "Number of posterior EOS samples to draw via weighted resampling "
            "after evidence computation. None -> use N_eff (Kish effective "
            "sample size, rounded)."
        ),
    )

    @field_validator("n_resample")
    @classmethod
    def _validate_n_resample(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"n_resample must be positive, got: {v}")
        return v


# Discriminated union for sampler configurations
SamplerConfig = Annotated[
    Union[
        FlowMCSamplerConfig,
        BlackJAXNSAWConfig,
        SMCRandomWalkSamplerConfig,
        SMCNUTSSamplerConfig,
        SMCPartialPosteriorsRandomWalkSamplerConfig,
        EOSReweightingConfig,
    ],
    Discriminator("type"),
]
