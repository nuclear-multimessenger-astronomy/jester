r"""Pydantic models for inference configuration validation.

IMPORTANT: When you modify these schemas, regenerate the YAML reference documentation:

    uv run python -m jesterTOV.inference.config.generate_yaml_reference

TODO: make this automatic in CI/CD, so this note can be removed and user is not burdened with it

This ensures the user documentation stays in sync with the actual validation rules.
"""

from pydantic import BaseModel, Field, field_validator, ValidationInfo, ConfigDict
from typing import Literal, Union, Annotated
from pydantic import Discriminator


class TransformConfig(BaseModel):
    """Configuration for EOS parameter transforms.

    Attributes
    ----------
    type : Literal["metamodel", "metamodel_cse", "spectral"]
        Type of transform to use
    ndat_metamodel : int
        Number of data points for MetaModel EOS
    nmax_nsat : float
        Maximum density in units of saturation density
    nb_CSE : int
        Number of CSE parameters (only for metamodel_cse)
    n_points_high : int
        Number of high-density points for spectral EOS (only for spectral)
    nmin_MM_nsat : float
        Starting density for metamodel grid as fraction of nsat (default: 0.75)
    min_nsat_TOV : float
        Minimum central density for TOV integration (units of nsat)
    ndat_TOV : int
        Number of data points for TOV integration
    nb_masses : int
        Number of masses to sample
    crust_name : Literal["DH", "BPS", "DH_fixed", "SLy"]
        Name of crust model to use
    tov_solver : Literal["gr", "post", "scalar_tensor"]
        TOV solver type to use (default: "gr")
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["metamodel", "metamodel_cse", "spectral"]
    ndat_metamodel: int = 100
    nmax_nsat: float = 25.0
    nb_CSE: int = 8  # Only for metamodel_cse
    n_points_high: int = 500  # Only for spectral
    nmin_MM_nsat: float = 0.75  # Starting density for metamodel grid
    min_nsat_TOV: float = 0.75  # Minimum density for TOV integration
    ndat_TOV: int = 100
    nb_masses: int = 100
    crust_name: Literal["DH", "BPS", "DH_fixed", "SLy"] = (
        "DH"  # TODO: this should be done in the crust source code, not here, and here just fetch from there
    )
    tov_solver: Literal["gr", "post", "scalar_tensor"] = "gr"

    @field_validator("nb_CSE")
    @classmethod
    def validate_nb_cse(cls, v: int, info: ValidationInfo) -> int:
        """Validate that nb_CSE is only used with metamodel_cse."""
        if (
            "type" in info.data
            and info.data["type"] in ["metamodel", "spectral"]
            and v != 0
        ):
            raise ValueError(
                "nb_CSE must be 0 for type='metamodel' or type='spectral'. "
                "Use type='metamodel_cse' for CSE extension."
            )
        return v

    @field_validator("crust_name")
    @classmethod
    def validate_crust_name(cls, v: str, info: ValidationInfo) -> str:
        """Validate crust name is appropriate for the transform type."""
        if "type" in info.data and info.data["type"] == "spectral" and v != "SLy":
            raise ValueError(
                "Spectral transform requires crust_name='SLy' for LALSuite compatibility. "
                f"Got: {v}"
            )
        return v


class PriorConfig(BaseModel):
    """Configuration for priors.

    Attributes
    ----------
    specification_file : str
        Path to .prior file specifying prior distributions
    """

    model_config = ConfigDict(extra="forbid")

    specification_file: str

    @field_validator("specification_file")
    @classmethod
    def validate_file_extension(cls, v: str) -> str:
        """Validate that specification file has .prior extension."""
        if not v.endswith(".prior"):
            raise ValueError(
                f"Prior specification file must have .prior extension, got: {v}"
            )
        return v


# Base likelihood configuration
class BaseLikelihoodConfig(BaseModel):
    """Base configuration for all likelihood types."""

    enabled: bool = Field(
        default=True, description="Whether this likelihood is enabled in the analysis"
    )


# GW Likelihood Configs
class GWLikelihoodConfig(BaseLikelihoodConfig):
    """Gravitational wave likelihood configuration (presampled version).

    This is the default GW likelihood that pre-samples masses from the
    GW posterior for efficient evaluation during MCMC sampling.

    Examples
    --------
    .. code-block:: yaml

        - type: "gw"
          enabled: true
          events:
            - name: "GW170817"
              model_dir: "./NFs/GW170817"
          penalty_value: -99999.0
          N_masses_evaluation: 2000
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["gw"] = Field(default="gw", description="Likelihood type identifier")

    events: list[dict[str, str]] = Field(
        description=(
            "List of GW events to include. Each event must have 'name' key. "
            "Optional 'model_dir' key specifies path to normalizing flow model. "
            "If omitted, uses preset paths based on event name."
        ),
        min_length=1,
    )

    penalty_value: float = Field(
        default=-99999.0,
        description="Log-likelihood penalty returned when M > M_TOV",
    )

    N_masses_evaluation: int = Field(
        default=2000,
        gt=0,
        description="Number of mass samples to pre-sample from GW posterior",
    )

    N_masses_batch_size: int = Field(
        default=1000,
        gt=0,
        description="Batch size for jax.lax.map processing of mass grid",
    )

    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducible mass sampling from GW posterior",
    )

    @field_validator("events")
    @classmethod
    def validate_events(cls, v: list[dict[str, str]]) -> list[dict[str, str]]:
        """Validate event structure."""
        for i, event in enumerate(v):
            if "name" not in event:
                raise ValueError(f"Event {i} missing required 'name' field")
            if not isinstance(event["name"], str):
                raise ValueError(f"Event {i} 'name' must be a string")
        return v


class GWResampledLikelihoodConfig(BaseLikelihoodConfig):
    """Gravitational wave likelihood configuration (legacy resampled version).

    Legacy version that resamples masses from GW posterior on-the-fly
    during each likelihood evaluation. Slower than presampled version.

    Examples
    --------
    .. code-block:: yaml

        - type: "gw_resampled"
          enabled: true
          events:
            - name: "GW170817"
          N_masses_evaluation: 20
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["gw_resampled"] = Field(
        default="gw_resampled", description="Likelihood type identifier"
    )

    events: list[dict[str, str]] = Field(
        description="List of GW events (see GWLikelihoodConfig for format)",
        min_length=1,
    )

    penalty_value: float = Field(default=-99999.0)
    N_masses_evaluation: int = Field(default=20, gt=0)
    N_masses_batch_size: int = Field(default=10, gt=0)

    @field_validator("events")
    @classmethod
    def validate_events(cls, v: list[dict[str, str]]) -> list[dict[str, str]]:
        """Validate event structure."""
        for i, event in enumerate(v):
            if "name" not in event:
                raise ValueError(f"Event {i} missing required 'name' field")
        return v


# NICER Likelihood Config
class NICERLikelihoodConfig(BaseLikelihoodConfig):
    """NICER X-ray timing likelihood configuration.

    Constrains mass-radius relation using NICER observations of
    millisecond pulsars. Marginalizes over pulsar mass using
    M-R posterior samples from analysis teams.

    Examples
    --------
    .. code-block:: yaml

        - type: "nicer"
          enabled: true
          pulsars:
            - name: "J0030"
              amsterdam_samples_file: "./data/J0030_amsterdam.txt"
              maryland_samples_file: "./data/J0030_maryland.txt"
            - name: "J0740"
          N_masses_evaluation: 100
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["nicer"] = Field(
        default="nicer", description="Likelihood type identifier"
    )

    pulsars: list[dict[str, str]] = Field(
        description=(
            "List of pulsars to include. Each pulsar must have 'name' key. "
            "Optional 'amsterdam_samples_file' and 'maryland_samples_file' keys "
            "specify paths to M-R posterior samples. If omitted, uses preset paths."
        ),
        min_length=1,
    )

    N_masses_evaluation: int = Field(
        default=100,
        gt=0,
        description="Number of mass grid points for marginalization over pulsar mass",
    )

    N_masses_batch_size: int = Field(
        default=20,
        gt=0,
        description="Batch size for processing mass grid points",
    )

    @field_validator("pulsars")
    @classmethod
    def validate_pulsars(cls, v: list[dict[str, str]]) -> list[dict[str, str]]:
        """Validate pulsar structure."""
        for i, pulsar in enumerate(v):
            if "name" not in pulsar:
                raise ValueError(f"Pulsar {i} missing required 'name' field")
            # Currently both sample files are required (preset paths not implemented yet)
            if "amsterdam_samples_file" not in pulsar:
                raise ValueError(
                    f"Pulsar {i} missing required 'amsterdam_samples_file' field"
                )
            if "maryland_samples_file" not in pulsar:
                raise ValueError(
                    f"Pulsar {i} missing required 'maryland_samples_file' field"
                )
        return v


# Radio Likelihood Config
class RadioLikelihoodConfig(BaseLikelihoodConfig):
    """Radio pulsar timing likelihood configuration.

    Constrains neutron star masses using radio timing measurements.
    Applies Gaussian mass constraints from pulsar timing observations.

    Examples
    --------
    .. code-block:: yaml

        - type: "radio"
          enabled: true
          pulsars:
            - name: "J0740+6620"
              mass_mean: 2.08
              mass_std: 0.07
          penalty_value: -1e5
          nb_masses: 100
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["radio"] = Field(
        default="radio", description="Likelihood type identifier"
    )

    pulsars: list[dict[str, str | float]] = Field(
        description=(
            "List of pulsars with timing mass measurements. Each pulsar must have "
            "'name', 'mass_mean' (solar masses), and 'mass_std' (1-sigma, solar masses) keys."
        ),
        min_length=1,
    )

    penalty_value: float = Field(
        default=-1e5,
        description="Log-likelihood penalty for invalid TOV solutions (M_TOV ≤ m_min)",
    )

    nb_masses: int = Field(
        default=100,
        gt=0,
        description="Number of mass points for numerical integration of Gaussian constraint",
    )

    @field_validator("pulsars")
    @classmethod
    def validate_pulsars(cls, v: list[dict]) -> list[dict]:
        """Validate pulsar structure."""
        for i, pulsar in enumerate(v):
            required = {"name", "mass_mean", "mass_std"}
            missing = required - set(pulsar.keys())
            if missing:
                raise ValueError(f"Pulsar {i} missing required fields: {missing}")
            if not isinstance(pulsar["mass_mean"], (int, float)):
                raise ValueError(f"Pulsar {i} 'mass_mean' must be a number")
            if (
                not isinstance(pulsar["mass_std"], (int, float))
                or pulsar["mass_std"] <= 0
            ):
                raise ValueError(f"Pulsar {i} 'mass_std' must be a positive number")
        return v


# ChiEFT Likelihood Config
class ChiEFTLikelihoodConfig(BaseLikelihoodConfig):
    """Chiral effective field theory likelihood configuration.

    Constrains EOS at low densities using chiral EFT uncertainty bands.
    Checks that predicted pressure-density relation falls within bands.

    Examples
    --------
    .. code-block:: yaml

        - type: "chieft"
          enabled: true
          low_filename: "./data/chiEFT/low.dat"
          high_filename: "./data/chiEFT/high.dat"
          nb_n: 100
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["chieft"] = Field(
        default="chieft", description="Likelihood type identifier"
    )

    low_filename: str | None = Field(
        default=None,
        description=(
            "Path to lower bound ChiEFT data file. "
            "If None, uses default: data/chiEFT/2402.04172/low.dat"
        ),
    )

    high_filename: str | None = Field(
        default=None,
        description=(
            "Path to upper bound ChiEFT data file. "
            "If None, uses default: data/chiEFT/2402.04172/high.dat"
        ),
    )

    nb_n: int = Field(
        default=100,
        gt=0,
        description="Number of density points to evaluate against ChiEFT bands",
    )


# Constraint Likelihood Configs
class EOSConstraintsLikelihoodConfig(BaseLikelihoodConfig):
    """EOS constraint likelihood configuration.

    Applies physics-motivated constraints on equation of state properties:
    causality (cs² ≤ 1), thermodynamic stability (cs² ≥ 0), and
    monotonic pressure.

    Examples
    --------
    .. code-block:: yaml

        - type: "constraints_eos"
          enabled: true
          penalty_causality: -1e10
          penalty_stability: -1e5
          penalty_pressure: -1e5
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["constraints_eos"] = Field(
        default="constraints_eos", description="Likelihood type identifier"
    )

    penalty_causality: float = Field(
        default=-1e10,
        description="Log-likelihood penalty for causality violation (cs² > 1)",
    )

    penalty_stability: float = Field(
        default=-1e5,
        description="Log-likelihood penalty for thermodynamic instability (cs² < 0)",
    )

    penalty_pressure: float = Field(
        default=-1e5,
        description="Log-likelihood penalty for non-monotonic pressure",
    )


class TOVConstraintsLikelihoodConfig(BaseLikelihoodConfig):
    """TOV constraint likelihood configuration.

    Applies penalty for TOV integration failures.

    Examples
    --------
    .. code-block:: yaml

        - type: "constraints_tov"
          enabled: true
          penalty_tov: -1e10
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["constraints_tov"] = Field(
        default="constraints_tov", description="Likelihood type identifier"
    )

    penalty_tov: float = Field(
        default=-1e10,
        description="Log-likelihood penalty for TOV integration failure",
    )


class GammaConstraintsLikelihoodConfig(BaseLikelihoodConfig):
    """Gamma constraint likelihood configuration (spectral EOS only).

    Applies bounds on spectral decomposition Gamma parameters.
    Only applicable when using spectral EOS parametrization.

    Examples
    --------
    .. code-block:: yaml

        - type: "constraints_gamma"
          enabled: true
          penalty_gamma: -1e10
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["constraints_gamma"] = Field(
        default="constraints_gamma", description="Likelihood type identifier"
    )

    penalty_gamma: float = Field(
        default=-1e10,
        description=(
            "Log-likelihood penalty for Gamma bound violation. "
            "Applies bounds Γ ∈ [0.6, 4.5] for spectral decomposition EOS."
        ),
    )


class DeprecatedConstraintsLikelihoodConfig(BaseLikelihoodConfig):
    """Deprecated combined constraint likelihood configuration.

    DEPRECATED: Use constraints_eos + constraints_tov instead.
    Combined EOS and TOV constraints in a single likelihood.

    Examples
    --------
    .. code-block:: yaml

        - type: "constraints"
          enabled: true
          penalty_tov: -1e10
          penalty_causality: -1e10
          penalty_stability: -1e5
          penalty_pressure: -1e5
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["constraints"] = Field(
        default="constraints", description="Likelihood type identifier"
    )

    penalty_tov: float = Field(
        default=-1e10,
        description="Log-likelihood penalty for TOV integration failure",
    )

    penalty_causality: float = Field(
        default=-1e10,
        description="Log-likelihood penalty for causality violation (cs² > 1)",
    )

    penalty_stability: float = Field(
        default=-1e5,
        description="Log-likelihood penalty for thermodynamic instability (cs² < 0)",
    )

    penalty_pressure: float = Field(
        default=-1e5,
        description="Log-likelihood penalty for non-monotonic pressure",
    )


class REXLikelihoodConfig(BaseLikelihoodConfig):
    """REX (PREX/CREX) likelihood configuration.

    NOT IMPLEMENTED YET - placeholder for future development.

    Examples
    --------
    .. code-block:: yaml

        - type: "rex"
          enabled: true
          experiment_name: "PREX"
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["rex"] = Field(
        default="rex", description="Likelihood type identifier"
    )

    experiment_name: str = Field(
        default="PREX",
        description="Name of REX experiment (PREX or CREX)",
    )


class ZeroLikelihoodConfig(BaseLikelihoodConfig):
    """Zero likelihood configuration for prior-only sampling.

    Returns zero log-likelihood (uniform likelihood) for all EOS configurations.
    Use this for prior-only sampling without observational constraints.

    Examples
    --------
    .. code-block:: yaml

        - type: "zero"
          enabled: true
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["zero"] = Field(
        default="zero", description="Likelihood type identifier"
    )


# Discriminated union of all likelihood types
LikelihoodConfig = Annotated[
    Union[
        GWLikelihoodConfig,
        GWResampledLikelihoodConfig,
        NICERLikelihoodConfig,
        RadioLikelihoodConfig,
        ChiEFTLikelihoodConfig,
        EOSConstraintsLikelihoodConfig,
        TOVConstraintsLikelihoodConfig,
        GammaConstraintsLikelihoodConfig,
        DeprecatedConstraintsLikelihoodConfig,
        REXLikelihoodConfig,
        ZeroLikelihoodConfig,
    ],
    Discriminator("type"),
]


class BaseSamplerConfig(BaseModel):
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
        "learning_rate",
        "train_thinning",
        "output_thinning",
    )
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate that value is positive."""
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

    type: Literal["smc-rw"] = "smc-rw"  # Discriminator for Pydantic union
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

    type: Literal["smc-nuts"] = "smc-nuts"  # Discriminator for Pydantic union
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
# This allows Pydantic to automatically select the correct config class based on the 'type' field
SamplerConfig = Annotated[
    Union[
        FlowMCSamplerConfig,
        BlackJAXNSAWConfig,
        SMCRandomWalkSamplerConfig,
        SMCNUTSSamplerConfig,
    ],
    Discriminator("type"),
]


class PostprocessingConfig(BaseModel):
    r"""Configuration for postprocessing plots.

    Attributes
    ----------
    enabled : bool
        Whether to run postprocessing after inference (default: True)
    make_cornerplot : bool
        Generate cornerplot of EOS parameters (default: True)
    make_massradius : bool
        Generate mass-radius plot (default: True)
    make_masslambda : bool
        Generate mass-Lambda plot (default: True)
    make_pressuredensity : bool
        Generate pressure-density plot (default: True)
    make_histograms : bool
        Generate parameter histograms (default: True)
    make_cs2 : bool
        Generate cs2-density plot (default: True)
    prior_dir : str | None
        Directory containing prior samples for comparison (default: None)
    injection_eos_path : str | None
        Path to NPZ file containing injection EOS data for plotting (default: None).
        The NPZ file should contain arrays in geometric units:
        - masses_EOS: Solar masses :math:`M_{\odot}`
        - radii_EOS: :math:`\mathrm{km}`
        - Lambda_EOS: dimensionless tidal deformability
        - n: geometric units :math:`m^{-2}`
        - p: geometric units :math:`m^{-2}`
        - e: geometric units :math:`m^{-2}`
        - cs2: dimensionless
        This matches LALSuite EOS format and JESTER HDF5 output. Missing keys handled gracefully.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    make_cornerplot: bool = True
    make_massradius: bool = True
    make_masslambda: bool = True
    make_pressuredensity: bool = True
    make_histograms: bool = True
    make_cs2: bool = True
    prior_dir: str | None = None
    injection_eos_path: str | None = None


class InferenceConfig(BaseModel):
    """Top-level inference configuration.

    Attributes
    ----------
    seed : int
        Random seed for reproducibility
    transform : TransformConfig
        Transform configuration
    prior : PriorConfig
        Prior configuration
    likelihoods : list[LikelihoodConfig]
        List of likelihood configurations
    sampler : SamplerConfig
        Sampler configuration
    postprocessing : PostprocessingConfig
        Postprocessing configuration
    data_paths : dict[str, str]
        Override default data paths
    dry_run : bool
        Setup everything but don't run sampler (default: False)
    validate_only : bool
        Only validate configuration, don't run inference (default: False)
    debug_nans : bool
        Enable JAX NaN debugging for catching numerical issues (default: False)
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = 43
    transform: TransformConfig
    prior: PriorConfig
    likelihoods: list[LikelihoodConfig]
    sampler: SamplerConfig
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)
    data_paths: dict[str, str] = Field(default_factory=dict)
    dry_run: bool = False
    validate_only: bool = False
    debug_nans: bool = Field(
        default=False,
        description="Enable JAX NaN debugging for catching numerical issues during inference",
    )

    @field_validator("likelihoods")
    @classmethod
    def validate_likelihoods(cls, v: list[LikelihoodConfig]) -> list[LikelihoodConfig]:
        """Validate that at least one likelihood is enabled."""
        if not any(lk.enabled for lk in v):
            raise ValueError("At least one likelihood must be enabled")
        return v

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int) -> int:
        """Validate that seed is non-negative."""
        if v < 0:
            raise ValueError(f"Seed must be non-negative, got: {v}")
        return v
