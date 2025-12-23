"""Pydantic models for inference configuration validation.

IMPORTANT: When you modify these schemas, regenerate the YAML reference documentation:
    uv run python -m jesterTOV.inference.config.generate_yaml_reference

This ensures the user documentation stays in sync with the actual validation rules.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Dict, Any


class TransformConfig(BaseModel):
    """Configuration for EOS parameter transforms.

    Attributes
    ----------
    type : Literal["metamodel", "metamodel_cse"]
        Type of transform to use
    ndat_metamodel : int
        Number of data points for MetaModel EOS
    nmax_nsat : float
        Maximum density in units of saturation density
    nb_CSE : int
        Number of CSE parameters (only for metamodel_cse)
    min_nsat_TOV : float
        Minimum density for TOV integration (units of nsat)
    ndat_TOV : int
        Number of data points for TOV integration
    nb_masses : int
        Number of masses to sample
    crust_name : Literal["DH", "BPS", "DH_fixed"]
        Name of crust model to use
    """

    type: Literal["metamodel", "metamodel_cse"]
    ndat_metamodel: int = 100
    nmax_nsat: float = 25.0
    nb_CSE: int = 8  # Only for metamodel_cse
    min_nsat_TOV: float = 0.75
    ndat_TOV: int = 100
    nb_masses: int = 100
    crust_name: Literal["DH", "BPS", "DH_fixed"] = "DH" # FIXME: this should be done in the crust source code, not here, and here just fetch from there

    @field_validator("nb_CSE")
    @classmethod
    def validate_nb_cse(cls, v: int, info) -> int:
        """Validate that nb_CSE is only used with metamodel_cse."""
        if "type" in info.data and info.data["type"] == "metamodel" and v != 0:
            raise ValueError(
                "nb_CSE must be 0 for type='metamodel'. "
                "Use type='metamodel_cse' for CSE extension."
            )
        return v


class PriorConfig(BaseModel):
    """Configuration for priors.

    Attributes
    ----------
    specification_file : str
        Path to .prior file specifying prior distributions
    """

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
    type : Literal["gw", "nicer", "radio", "chieft", "rex", "zero"]
        Type of likelihood constraint
    enabled : bool
        Whether this likelihood is enabled
    parameters : dict
        Likelihood-specific parameters
    """

    type: Literal["gw", "nicer", "radio", "chieft", "rex", "zero"]
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)


class SamplerConfig(BaseModel):
    """Configuration for MCMC sampler.

    Attributes
    ----------
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
    output_dir : str
        Directory to save results
    train_thinning : int
        Thinning factor for training samples (default: 1)
    output_thinning : int
        Thinning factor for output samples (default: 5)
    n_eos_samples : int
        Number of EOS samples to generate after inference (default: 10000)
    """

    n_chains: int = 20
    n_loop_training: int = 3
    n_loop_production: int = 3
    n_local_steps: int = 100
    n_global_steps: int = 100
    n_epochs: int = 30
    learning_rate: float = 0.001
    output_dir: str = "./outdir/"
    train_thinning: int = 1
    output_thinning: int = 5
    n_eos_samples: int = 10_000

    @field_validator("n_chains", "n_loop_training", "n_loop_production")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate that value is positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate that learning rate is reasonable."""
        if v <= 0:
            raise ValueError(f"Learning rate must be in (0, 1], got: {v}")
        return v


class PostprocessingConfig(BaseModel):
    """Configuration for postprocessing plots.

    Attributes
    ----------
    enabled : bool
        Whether to run postprocessing after inference (default: True)
    make_cornerplot : bool
        Generate cornerplot of EOS parameters (default: True)
    make_massradius : bool
        Generate mass-radius plot (default: True)
    make_pressuredensity : bool
        Generate pressure-density plot (default: True)
    make_histograms : bool
        Generate parameter histograms (default: True)
    make_contours : bool
        Generate contour plots (default: True)
    prior_dir : str | None
        Directory containing prior samples for comparison (default: None)
    """

    enabled: bool = True
    make_cornerplot: bool = True
    make_massradius: bool = True
    make_pressuredensity: bool = True
    make_histograms: bool = True
    make_contours: bool = True
    prior_dir: str | None = None


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
    """

    seed: int = 43
    transform: TransformConfig
    prior: PriorConfig
    likelihoods: list[LikelihoodConfig]
    sampler: SamplerConfig
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)
    data_paths: Dict[str, str] = Field(default_factory=dict)
    dry_run: bool = False
    validate_only: bool = False

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
