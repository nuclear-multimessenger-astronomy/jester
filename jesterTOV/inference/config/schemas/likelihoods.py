"""Pydantic models for likelihood configuration."""

import warnings
from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field, field_validator, ConfigDict, Discriminator

from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class BaseLikelihoodConfig(BaseModel):
    """Base configuration for all likelihood types."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True, description="Whether this likelihood is enabled in the analysis"
    )


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


class NICERLikelihoodConfig(BaseLikelihoodConfig):
    """NICER X-ray timing likelihood configuration using normalizing flows (DEFAULT).

    Constrains mass-radius relation using NICER observations of
    millisecond pulsars. Uses pre-trained normalizing flows on M-R
    posteriors for efficient likelihood evaluation.

    For the legacy KDE-based version, use type: "nicer_kde".

    Examples
    --------
    .. code-block:: yaml

        - type: "nicer"
          enabled: true
          pulsars:
            - name: "J0030"
              amsterdam_model_dir: "./flows/models/nicer_maf/J00300451/amsterdam_st_pst"
              maryland_model_dir: "./flows/models/nicer_maf/J00300451/maryland_2spot_rm"
            - name: "J0740"
              amsterdam_model_dir: "./flows/models/nicer_maf/J07406620/amsterdam_gamma_nicerxmm"
              maryland_model_dir: "./flows/models/nicer_maf/J07406620/maryland_unknown_nicerxmm_rm"
          N_masses_evaluation: 100

    Notes
    -----
    Both ``amsterdam_model_dir`` and ``maryland_model_dir`` are REQUIRED for each pulsar.
    The schema validator will issue warnings if omitted, but ``NICERLikelihood.__init__``
    will raise ``ValueError`` at runtime. Preset model paths are not yet implemented.
    """

    type: Literal["nicer"] = Field(
        default="nicer", description="Likelihood type identifier"
    )

    pulsars: list[dict[str, str]] = Field(
        description=(
            "List of pulsars to include. Each pulsar must have 'name' key. "
            "REQUIRED: 'amsterdam_model_dir' and 'maryland_model_dir' keys "
            "specify paths to trained flow model directories. "
            "NICERLikelihood.__init__ will raise ValueError if either is missing."
        ),
        min_length=1,
    )

    N_masses_evaluation: int = Field(
        default=100,
        gt=0,
        description="Number of mass samples for likelihood evaluation",
    )

    N_masses_batch_size: int = Field(
        default=20,
        gt=0,
        description="Batch size for processing mass samples",
    )

    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducible mass sampling in NICER likelihood",
    )

    @field_validator("pulsars")
    @classmethod
    def validate_pulsars(cls, v: list[dict[str, str]]) -> list[dict[str, str]]:
        """Validate pulsar structure."""
        for i, pulsar in enumerate(v):
            if "name" not in pulsar:
                raise ValueError(f"Pulsar {i} missing required 'name' field")

            # Warn if model directories not provided (will fail at runtime)
            if "amsterdam_model_dir" not in pulsar:
                logger.warning(
                    f"Pulsar {i} ({pulsar['name']}) missing 'amsterdam_model_dir'. "
                    "NICERLikelihood.__init__ will raise ValueError at runtime. "
                    "Preset model paths are not yet implemented."
                )
            if "maryland_model_dir" not in pulsar:
                logger.warning(
                    f"Pulsar {i} ({pulsar['name']}) missing 'maryland_model_dir'. "
                    "NICERLikelihood.__init__ will raise ValueError at runtime. "
                    "Preset model paths are not yet implemented."
                )
        return v


class NICERKDELikelihoodConfig(BaseLikelihoodConfig):
    """NICER X-ray timing likelihood configuration using KDE (LEGACY).

    This is the legacy KDE-based NICER likelihood. For the recommended
    flow-based version, use type: "nicer".

    Constrains mass-radius relation using NICER observations of
    millisecond pulsars. Uses kernel density estimation on M-R
    posterior samples from analysis teams.

    Examples
    --------
    .. code-block:: yaml

        - type: "nicer_kde"
          enabled: true
          pulsars:
            - name: "J0030"
              amsterdam_samples_file: "./data/J0030_amsterdam.npz"
              maryland_samples_file: "./data/J0030_maryland.npz"
            - name: "J0740"
              amsterdam_samples_file: "./data/J0740_amsterdam.npz"
              maryland_samples_file: "./data/J0740_maryland.npz"
          N_masses_evaluation: 100
    """

    type: Literal["nicer_kde"] = Field(
        default="nicer_kde", description="Likelihood type identifier"
    )

    pulsars: list[dict[str, str]] = Field(
        description=(
            "List of pulsars to include. Each pulsar must have 'name' key. "
            "'amsterdam_samples_file' and 'maryland_samples_file' keys "
            "specify paths to M-R posterior samples (.npz files)."
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
            # Both sample files are required for KDE approach
            if "amsterdam_samples_file" not in pulsar:
                raise ValueError(
                    f"Pulsar {i} missing required 'amsterdam_samples_file' field"
                )
            if "maryland_samples_file" not in pulsar:
                raise ValueError(
                    f"Pulsar {i} missing required 'maryland_samples_file' field"
                )
        return v


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
    def validate_pulsars(
        cls, v: list[dict[str, str | float]]
    ) -> list[dict[str, str | float]]:
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

    def model_post_init(self, __context: object) -> None:
        warnings.warn(
            "Deprecated: 'constraints' config is removed; use 'constraints_eos' + 'constraints_tov' instead",
            DeprecationWarning,
            stacklevel=2,
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

    type: Literal["rex"] = Field(
        default="rex", description="Likelihood type identifier"
    )

    experiment_name: str = Field(
        default="PREX",
        description="Name of REX experiment (PREX or CREX)",
    )

    def model_post_init(self, __context: object) -> None:
        raise NotImplementedError("REX likelihood is not implemented")


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

    type: Literal["zero"] = Field(
        default="zero", description="Likelihood type identifier"
    )


# Discriminated union of all likelihood types
LikelihoodConfig = Annotated[
    Union[
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
    ],
    Discriminator("type"),
]
