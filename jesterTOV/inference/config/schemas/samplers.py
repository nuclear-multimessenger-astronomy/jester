"""Pydantic models for sampler configuration."""

from typing import Literal, Union, Annotated
from pydantic import Field, field_validator, ConfigDict, Discriminator

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


class SMCRandomWalkSamplerConfig(BaseSamplerConfig):
    """Configuration for Sequential Monte Carlo with Random Walk kernel.

    Attributes
    ----------
    type : Literal["smc-rw"]
        Sampler type identifier
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

    type: Literal["smc-rw"] = "smc-rw"
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


class SMCPartialPosteriorsSamplerConfig(BaseSamplerConfig):
    """Configuration for the IBIS-hybridized-with-likelihood-tempering sampler
    ("partial posteriors" / smc-pp), which assimilates GW events one at a time.

    See ``samplers/blackjax/smc/ibis.py``'s module docstring for the full
    algorithm. The user-facing "partial posteriors" naming here is
    deliberately distinct from the internal "IBIS" class/module naming --
    see that module's docstring for why.

    Attributes
    ----------
    type : Literal["smc-pp"]
        Sampler type identifier
    ess_threshold : float
        Alpha in "if ESS >= alpha * N, keep cheaply reweighting; else run a
        full SMC batch" (default: 0.5). Distinct from ``inner.target_ess``,
        which governs the *within-batch* adaptive-lambda annealing schedule
        -- do not conflate the two knobs.
    inner : SMCRandomWalkSamplerConfig
        Full SMC-RW configuration for the batch-annealing fallback (RW
        kernel only for now). ``inner.n_particles`` is authoritative for the
        total particle count -- there is no separate top-level n_particles
        field.
    n_final_rejuvenation_steps : int
        Number of fixed-target MCMC steps run on the terminal particles
        after the last SMC batch's resample-to-uniform-weights step, to
        de-duplicate the exact-duplicate particles that resampling produces
        (see ``samplers/blackjax/smc/rejuvenation.py``). Default: 10.
    particle_batch_size : int
        Batch size for ``jax.lax.map`` when computing the per-particle,
        per-event log-likelihood matrix used for cheap ESS-threshold
        reweighting (default: 1000). Distinct from ``log_prob_batch_size``
        (used only for the post-hoc ``get_log_prob()``/EOS-sample pass); the
        two are often set equal in practice but govern different, separately
        tunable computations.
    """

    type: Literal["smc-pp"] = "smc-pp"
    ess_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    inner: SMCRandomWalkSamplerConfig = Field(default_factory=SMCRandomWalkSamplerConfig)
    n_final_rejuvenation_steps: int = Field(default=10, ge=0)
    particle_batch_size: int = Field(default=1000, gt=0)


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
        SMCPartialPosteriorsSamplerConfig,
        EOSReweightingConfig,
    ],
    Discriminator("type"),
]
