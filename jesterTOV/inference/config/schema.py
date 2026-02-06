r"""Pydantic models for inference configuration validation.

IMPORTANT: When you modify these schemas, regenerate the YAML reference documentation:

    uv run python -m jesterTOV.inference.config.generate_yaml_reference

TODO: make this automatic in CI/CD, so this note can be removed and user is not burdened with it

This ensures the user documentation stays in sync with the actual validation rules.
"""

from pydantic import BaseModel, Field, field_validator, ValidationInfo, ConfigDict
from typing import Literal, Dict, Any, Union, Annotated
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


class LikelihoodConfig(BaseModel):
    """Configuration for individual likelihood.

    Attributes
    ----------
    type : Literal["gw", "gw_resampled", "nicer", "radio", "chieft", "rex", "constraints", "zero"]
        Type of likelihood constraint
    enabled : bool
        Whether this likelihood is enabled
    parameters : dict
        Likelihood-specific parameters

        For GW likelihoods (type: "gw", presampled version - default):
            events : list[dict]
                List of GW events with 'name' and 'model_dir' keys
            penalty_value : float
                Penalty for masses exceeding Mtov (default: -99999.0)
            N_masses_evaluation : int
                Number of mass samples to pre-sample (default: 2000)
            N_masses_batch_size : int
                Batch size for jax.lax.map processing (default: 1000)
            seed : int
                Random seed for mass pre-sampling (default: 42)

        For GW resampled likelihoods (type: "gw_resampled", legacy on-the-fly resampling):
            events : list[dict]
                List of GW events with 'name' and 'model_dir' keys
            penalty_value : float
                Penalty for masses exceeding Mtov (default: -99999.0)
            N_masses_evaluation : int
                Number of mass samples per evaluation (default: 20)
            N_masses_batch_size : int
                Batch size for mass sampling (default: 10)

        For NICER likelihoods:
            pulsars : list[dict]
                List of pulsars with 'name', 'amsterdam_samples_file', and 'maryland_samples_file' keys
            N_masses_evaluation : int
                Number of mass grid points for marginalization (default: 100)
            N_masses_batch_size : int
                Batch size for processing mass grid points (default: 20)

        For radio timing likelihoods:
            pulsars : list[dict]
                List of pulsars with 'name', 'mass_mean', and 'mass_std' keys
            penalty_value : float
                Penalty for invalid TOV solutions (M_TOV ≤ m_min) (default: -1e5)
            nb_masses : int
                Number of mass points for numerical integration (default: 100)

        For chiEFT likelihoods:
            low_filename : str, optional
                Path to lower bound data file (default: data/chiEFT/2402.04172/low.dat)
            high_filename : str, optional
                Path to upper bound data file (default: data/chiEFT/2402.04172/high.dat)
            nb_n : int
                Number of density points for integration (default: 100)

        For constraint likelihoods (type: "constraints" - deprecated, use constraints_eos + constraints_tov):
            penalty_tov : float
                Log likelihood penalty for TOV integration failure (default: -1e10)
            penalty_causality : float
                Log likelihood penalty for causality violation (cs^2 > 1) (default: -1e10)
            penalty_stability : float
                Log likelihood penalty for thermodynamic instability (cs^2 < 0) (default: -1e5)
            penalty_pressure : float
                Log likelihood penalty for non-monotonic pressure (default: -1e5)

        For EOS constraint likelihoods (type: "constraints_eos"):
            penalty_causality : float
                Log likelihood penalty for causality violation (cs^2 > 1) (default: -1e10)
            penalty_stability : float
                Log likelihood penalty for thermodynamic instability (cs^2 < 0) (default: -1e5)
            penalty_pressure : float
                Log likelihood penalty for non-monotonic pressure (default: -1e5)

        For TOV constraint likelihoods (type: "constraints_tov"):
            penalty_tov : float
                Log likelihood penalty for TOV integration failure (default: -1e10)

        For Gamma constraint likelihoods (type: "constraints_gamma", spectral EOS only):
            penalty_gamma : float
                Log likelihood penalty for Gamma bound violation (default: -1e10)
                Only applies to spectral decomposition EOS (Γ ∈ [0.6, 4.5])
    """

    model_config = ConfigDict(extra="forbid")

    # TODO: deprecate rex for now: not implemented yet
    type: Literal[
        "gw",
        "gw_resampled",
        "nicer",
        "radio",
        "chieft",
        "rex",
        "constraints",
        "constraints_eos",
        "constraints_tov",
        "constraints_gamma",
        "zero",
    ]
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def validate_likelihood_parameters(
        cls, v: Dict[str, Any], info: ValidationInfo
    ) -> Dict[str, Any]:
        """Validate likelihood-specific parameters."""
        if "type" not in info.data:
            return v

        # Skip validation if likelihood is disabled
        if "enabled" in info.data and not info.data["enabled"]:
            return v

        likelihood_type = info.data["type"]

        # Validate GW likelihood parameters (presampled is now default)
        if likelihood_type == "gw":
            if "events" not in v:
                raise ValueError(
                    "GW likelihood requires 'events' parameter (list of dicts with 'name' and 'model_dir')"
                )

            events = v["events"]
            if not isinstance(events, list) or len(events) == 0:
                raise ValueError("GW likelihood 'events' must be a non-empty list")

            # Validate each event
            for i, event in enumerate(events):
                if not isinstance(event, dict):
                    raise ValueError(
                        f"Event {i} must be a dict with 'name' and optional 'model_dir' keys"
                    )
                if "name" not in event:
                    raise ValueError(f"Event {i} missing required 'name' field")
                # model_dir is now optional - will use presets if not provided

            # Set defaults for optional parameters (presampled version)
            v.setdefault("penalty_value", -99999.0)
            v.setdefault("N_masses_evaluation", 2000)  # Default for presampled
            v.setdefault("N_masses_batch_size", 1000)
            v.setdefault("seed", 42)

        # Validate GW resampled likelihood parameters (legacy behavior)
        elif likelihood_type == "gw_resampled":
            if "events" not in v:
                raise ValueError(
                    "GW resampled likelihood requires 'events' parameter (list of dicts with 'name' and 'model_dir')"
                )

            events = v["events"]
            if not isinstance(events, list) or len(events) == 0:
                raise ValueError(
                    "GW resampled likelihood 'events' must be a non-empty list"
                )

            # Validate each event
            for i, event in enumerate(events):
                if not isinstance(event, dict):
                    raise ValueError(
                        f"Event {i} must be a dict with 'name' and optional 'model_dir' keys"
                    )
                if "name" not in event:
                    raise ValueError(f"Event {i} missing required 'name' field")
                # model_dir is now optional - will use presets if not provided

            # Set defaults for optional parameters (resampled version)
            v.setdefault("penalty_value", -99999.0)
            v.setdefault("N_masses_evaluation", 20)
            v.setdefault("N_masses_batch_size", 10)

        # Validate NICER likelihood parameters
        elif likelihood_type == "nicer":
            if "pulsars" not in v:
                raise ValueError(
                    "NICER likelihood requires 'pulsars' parameter "
                    "(list of dicts with 'name', 'amsterdam_samples_file', and 'maryland_samples_file')"
                )

            pulsars = v["pulsars"]
            if not isinstance(pulsars, list) or len(pulsars) == 0:
                raise ValueError("NICER likelihood 'pulsars' must be a non-empty list")

            # Validate each pulsar
            for i, pulsar in enumerate(pulsars):
                if not isinstance(pulsar, dict):
                    raise ValueError(
                        f"Pulsar {i} must be a dict with 'name', 'amsterdam_samples_file', "
                        f"and 'maryland_samples_file' keys"
                    )
                if "name" not in pulsar:
                    raise ValueError(f"Pulsar {i} missing required 'name' field")
                if "amsterdam_samples_file" not in pulsar:
                    raise ValueError(
                        f"Pulsar {i} missing required 'amsterdam_samples_file' field"
                    )
                if "maryland_samples_file" not in pulsar:
                    raise ValueError(
                        f"Pulsar {i} missing required 'maryland_samples_file' field"
                    )

            # Set defaults for optional parameters
            v.setdefault("N_masses_evaluation", 100)
            v.setdefault("N_masses_batch_size", 20)

        # Validate radio timing likelihood parameters
        elif likelihood_type == "radio":
            if "pulsars" not in v:
                raise ValueError(
                    "Radio timing likelihood requires 'pulsars' parameter "
                    "(list of dicts with 'name', 'mass_mean', and 'mass_std')"
                )

            pulsars = v["pulsars"]
            if not isinstance(pulsars, list) or len(pulsars) == 0:
                raise ValueError(
                    "Radio timing likelihood 'pulsars' must be a non-empty list"
                )

            # Validate each pulsar
            for i, pulsar in enumerate(pulsars):
                if not isinstance(pulsar, dict):
                    raise ValueError(
                        f"Pulsar {i} must be a dict with 'name', 'mass_mean', and 'mass_std' keys"
                    )
                if "name" not in pulsar:
                    raise ValueError(f"Pulsar {i} missing required 'name' field")
                if "mass_mean" not in pulsar:
                    raise ValueError(f"Pulsar {i} missing required 'mass_mean' field")
                if "mass_std" not in pulsar:
                    raise ValueError(f"Pulsar {i} missing required 'mass_std' field")

                # Validate mass values
                if (
                    not isinstance(pulsar["mass_mean"], (int, float))
                    or pulsar["mass_mean"] <= 0
                ):
                    raise ValueError(
                        f"Pulsar {i} 'mass_mean' must be a positive number, got: {pulsar['mass_mean']}"
                    )
                if (
                    not isinstance(pulsar["mass_std"], (int, float))
                    or pulsar["mass_std"] <= 0
                ):
                    raise ValueError(
                        f"Pulsar {i} 'mass_std' must be a positive number, got: {pulsar['mass_std']}"
                    )

            # Set defaults for optional parameters
            v.setdefault("penalty_value", -1e5)
            v.setdefault("nb_masses", 100)

        # Validate constraint likelihood parameters (deprecated - use constraints_eos + constraints_tov)
        elif likelihood_type == "constraints":
            # Set defaults for optional parameters
            v.setdefault("penalty_tov", -1e10)
            v.setdefault("penalty_causality", -1e10)
            v.setdefault("penalty_stability", -1e5)
            v.setdefault("penalty_pressure", -1e5)

        # Validate EOS constraint likelihood parameters
        elif likelihood_type == "constraints_eos":
            # Set defaults for optional parameters
            v.setdefault("penalty_causality", -1e10)
            v.setdefault("penalty_stability", -1e5)
            v.setdefault("penalty_pressure", -1e5)

        # Validate TOV constraint likelihood parameters
        elif likelihood_type == "constraints_tov":
            # Set defaults for optional parameters
            v.setdefault("penalty_tov", -1e10)

        # Validate gamma constraint likelihood parameters
        elif likelihood_type == "constraints_gamma":
            # Set defaults for optional parameters
            v.setdefault("penalty_gamma", -1e10)

        return v


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
    mass_matrix_param_scales: Dict[str, float] = Field(default_factory=dict)
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
    data_paths: Dict[str, str] = Field(default_factory=dict)
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
