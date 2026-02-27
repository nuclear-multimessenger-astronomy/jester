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
    def validate_base_positive(cls, v: int) -> int:
        """Validate that value is positive."""
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
    def validate_positive(cls, v: int) -> int:
        """Validate that value is positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """Validate that learning rate is positive."""
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
    def validate_delete_frac(cls, v: float) -> float:
        """Validate that deletion fraction is in (0, 1]."""
        if v <= 0 or v > 1:
            raise ValueError(f"n_delete_frac must be in (0, 1], got: {v}")
        return v

    @field_validator("n_live", "n_target", "max_mcmc", "max_proposals")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate that value is positive."""
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
    """

    type: Literal["smc-rw"] = "smc-rw"
    n_particles: int = 10000
    n_mcmc_steps: int = 1
    target_ess: float = 0.9
    random_walk_sigma: float = 1.0

    @field_validator("n_particles", "n_mcmc_steps")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate that value is positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v

    @field_validator("target_ess")
    @classmethod
    def validate_fraction(cls, v: float) -> float:
        """Validate that value is in (0, 1]."""
        if v <= 0 or v > 1:
            raise ValueError(f"Value must be in (0, 1], got: {v}")
        return v

    @field_validator("random_walk_sigma")
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """Validate that value is positive."""
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
    def validate_positive(cls, v: int) -> int:
        """Validate that value is positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v

    @field_validator("target_ess", "target_acceptance", "adaptation_rate")
    @classmethod
    def validate_fraction(cls, v: float) -> float:
        """Validate that value is in (0, 1]."""
        if v <= 0 or v > 1:
            raise ValueError(f"Value must be in (0, 1], got: {v}")
        return v

    @field_validator("init_step_size", "mass_matrix_base")
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """Validate that value is positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v


# Discriminated union for sampler configurations
SamplerConfig = Annotated[
    Union[
        FlowMCSamplerConfig,
        BlackJAXNSAWConfig,
        SMCRandomWalkSamplerConfig,
        SMCNUTSSamplerConfig,
    ],
    Discriminator("type"),
]
