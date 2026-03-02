r"""Pydantic models for inference configuration validation.

IMPORTANT: When you modify these schemas, regenerate the YAML reference documentation:

    uv run python -m jesterTOV.inference.config.generate_yaml_reference

TODO: make this automatic in CI/CD, so this note can be removed and user is not burdened with it

This ensures the user documentation stays in sync with the actual validation rules.

Schema organisation
-------------------
Config models are split into domain-specific sub-modules under ``schemas/``:

- ``schemas/eos.py``          EOS configuration (BaseEOSConfig + concrete types)
- ``schemas/tov.py``          TOV solver configuration (BaseTOVConfig + concrete types)
- ``schemas/likelihoods.py``  Likelihood configurations
- ``schemas/samplers.py``     Sampler configurations

This module assembles them into the top-level :class:`InferenceConfig` and re-exports
every name so that existing imports (``from .schema import ...``) continue to work.
"""

from pydantic import Field, field_validator, ConfigDict

from .schemas._base import JesterBaseModel

# EOS schemas
from .schemas.eos import (
    BaseEOSConfig,
    BaseMetamodelEOSConfig,
    MetamodelEOSConfig,
    MetamodelCSEEOSConfig,
    SpectralEOSConfig,
    EOSConfig,
)

# TOV schemas
from .schemas.tov import (
    BaseTOVConfig,
    GRTOVConfig,
    AnisotropyTOVConfig,
    TOVConfig,
)

# Likelihood schemas
from .schemas.likelihoods import (
    BaseLikelihoodConfig,
    GWEventConfig,
    GWLikelihoodConfig,
    GWResampledLikelihoodConfig,
    NICERLikelihoodConfig,
    NICERKDELikelihoodConfig,
    RadioLikelihoodConfig,
    ChiEFTLikelihoodConfig,
    EOSConstraintsLikelihoodConfig,
    TOVConstraintsLikelihoodConfig,
    GammaConstraintsLikelihoodConfig,
    DeprecatedConstraintsLikelihoodConfig,
    REXLikelihoodConfig,
    ZeroLikelihoodConfig,
    LikelihoodConfig,
)

# Sampler schemas
from .schemas.samplers import (
    BaseSamplerConfig,
    FlowMCSamplerConfig,
    BlackJAXNSAWConfig,
    SMCRandomWalkSamplerConfig,
    SMCNUTSSamplerConfig,
    SamplerConfig,
)


# ============================================================================
# Prior Configuration
# ============================================================================


class PriorConfig(JesterBaseModel):
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


# ============================================================================
# Postprocessing Configuration
# ============================================================================


class PostprocessingConfig(JesterBaseModel):
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


# ============================================================================
# Top-level InferenceConfig
# ============================================================================


class InferenceConfig(JesterBaseModel):
    """Top-level inference configuration.

    Attributes
    ----------
    seed : int
        Random seed for reproducibility
    eos : EOSConfig
        EOS configuration (discriminated union by type)
    tov : TOVConfig
        TOV solver configuration (discriminated union by type)
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
    eos: EOSConfig
    tov: TOVConfig
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


__all__ = [
    # EOS
    "BaseEOSConfig",
    "BaseMetamodelEOSConfig",
    "MetamodelEOSConfig",
    "MetamodelCSEEOSConfig",
    "SpectralEOSConfig",
    "EOSConfig",
    # TOV
    "BaseTOVConfig",
    "GRTOVConfig",
    "AnisotropyTOVConfig",
    "TOVConfig",
    # Likelihoods
    "BaseLikelihoodConfig",
    "GWEventConfig",
    "GWLikelihoodConfig",
    "GWResampledLikelihoodConfig",
    "NICERLikelihoodConfig",
    "NICERKDELikelihoodConfig",
    "RadioLikelihoodConfig",
    "ChiEFTLikelihoodConfig",
    "EOSConstraintsLikelihoodConfig",
    "TOVConstraintsLikelihoodConfig",
    "GammaConstraintsLikelihoodConfig",
    "DeprecatedConstraintsLikelihoodConfig",
    "REXLikelihoodConfig",
    "ZeroLikelihoodConfig",
    "LikelihoodConfig",
    # Samplers
    "BaseSamplerConfig",
    "FlowMCSamplerConfig",
    "BlackJAXNSAWConfig",
    "SMCRandomWalkSamplerConfig",
    "SMCNUTSSamplerConfig",
    "SamplerConfig",
    # Other
    "PriorConfig",
    "PostprocessingConfig",
    "InferenceConfig",
]
