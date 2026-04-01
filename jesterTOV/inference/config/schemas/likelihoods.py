"""Pydantic models for likelihood configuration."""

import warnings
from typing import Literal, Union, Annotated
from pydantic import (
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    Discriminator,
)

from jesterTOV.logging_config import get_logger

from ._base import JesterBaseModel

logger = get_logger("jester")


class BaseLikelihoodConfig(JesterBaseModel):
    """Base configuration for all likelihood types."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True, description="Whether this likelihood is enabled in the analysis"
    )


class GWEventConfig(JesterBaseModel):
    r"""Configuration for a single GW event in the likelihood.

    Three modes are supported:

    **Mode 1 — pre-trained flow** (default):
      Provide ``nf_model_dir`` or omit it to use a built-in preset.

    **Mode 2 — from bilby result**:
      Provide ``from_bilby_result`` to extract samples from a bilby HDF5 file
      and train the flow automatically when running
      :func:`~jesterTOV.inference.run_inference.main`.

    **Mode 3 — from NPZ file**:
      Provide ``from_npz_file`` to train the flow directly from an existing
      NPZ file (e.g. one previously produced by the ``jester_extract_gw_posterior_bilby``
      CLI tool), skipping the bilby extraction step.

    ``nf_model_dir`` is mutually exclusive with both ``from_bilby_result`` and
    ``from_npz_file``.  ``from_bilby_result`` and ``from_npz_file`` are also
    mutually exclusive with each other.  ``flow_config`` and ``retrain_flow``
    are valid in both Mode 2 and Mode 3.

    Examples
    --------
    Pre-trained flow (using preset):

    .. code-block:: yaml

        events:
          - name: GW170817

    Pre-trained flow (custom path):

    .. code-block:: yaml

        events:
          - name: GW170817
            nf_model_dir: ./my_flow

    From bilby result:

    .. code-block:: yaml

        events:
          - name: GW170817
            from_bilby_result: ./GW170817_result.hdf5
            flow_config: ./flow_config.yaml    # optional
            retrain_flow: false                # optional

    From NPZ file:

    .. code-block:: yaml

        events:
          - name: GW170817
            from_npz_file: ./GW170817_posterior.npz
            flow_config: ./flow_config.yaml    # optional
            retrain_flow: false                # optional
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="GW event name (e.g. 'GW170817')")
    nf_model_dir: str | None = Field(
        default=None,
        description=(
            "Path to a pre-trained normalizing flow model directory. "
            "If omitted, uses a built-in preset for known events."
        ),
    )
    from_bilby_result: str | None = Field(
        default=None,
        description=(
            "Path to a bilby result HDF5 file.  When set, jester will extract "
            "posterior samples and train a normalizing flow automatically before "
            "running inference."
        ),
    )
    from_npz_file: str | None = Field(
        default=None,
        description=(
            "Path to an NPZ file with posterior samples "
            "(mass_1_source, mass_2_source, lambda_1, lambda_2).  "
            "When set, jester will train a normalizing flow directly from this "
            "file before running inference, skipping the bilby extraction step."
        ),
    )
    flow_config: str | None = Field(
        default=None,
        description=(
            "Path to a YAML file with FlowTrainingConfig overrides.  "
            "Only meaningful when 'from_bilby_result' or 'from_npz_file' is set."
        ),
    )
    retrain_flow: bool = Field(
        default=False,
        description=(
            "Force retraining even if a cached flow already exists.  "
            "Only meaningful when 'from_bilby_result' or 'from_npz_file' is set."
        ),
    )

    @model_validator(mode="after")
    def validate_mode_consistency(self) -> "GWEventConfig":
        """Ensure that the three source modes are not mixed."""
        has_pretrained = self.nf_model_dir is not None
        has_bilby = self.from_bilby_result is not None
        has_npz = self.from_npz_file is not None

        if has_pretrained and has_bilby:
            raise ValueError(
                "Cannot specify both 'from_bilby_result' and 'nf_model_dir'. "
                "Use 'from_bilby_result' to extract samples and train a flow, "
                "or 'nf_model_dir' to point to an already-trained flow."
            )
        if has_pretrained and has_npz:
            raise ValueError(
                "Cannot specify both 'from_npz_file' and 'nf_model_dir'. "
                "Use 'from_npz_file' to train a flow from an existing NPZ, "
                "or 'nf_model_dir' to point to an already-trained flow."
            )
        if has_bilby and has_npz:
            raise ValueError(
                "Cannot specify both 'from_bilby_result' and 'from_npz_file'. "
                "Use 'from_bilby_result' to extract samples from a bilby HDF5, "
                "or 'from_npz_file' to start directly from an existing NPZ file."
            )

        # flow_config and retrain_flow only make sense when training a flow
        needs_training = has_bilby or has_npz
        if not needs_training:
            if self.flow_config is not None:
                raise ValueError(
                    "'flow_config' is only valid when 'from_bilby_result' or "
                    "'from_npz_file' is provided."
                )
            if self.retrain_flow:
                raise ValueError(
                    "'retrain_flow' is only valid when 'from_bilby_result' or "
                    "'from_npz_file' is provided."
                )
        return self


class GWLikelihoodConfig(BaseLikelihoodConfig):
    r"""Gravitational wave likelihood configuration (presampled version).

    This is the default GW likelihood that pre-samples masses from the
    GW posterior for efficient evaluation during MCMC sampling.

    Each entry in ``events`` is a :class:`GWEventConfig` that supports three
    modes: using a pre-trained normalizing flow (default), automatically
    extracting samples from a bilby result file and training the flow, or
    training the flow directly from an existing NPZ file.

    Examples
    --------
    .. code-block:: yaml

        - type: "gw"
          enabled: true
          events:
            - name: "GW170817"
              nf_model_dir: "./NFs/GW170817"
          N_masses_evaluation: 2000
    """

    type: Literal["gw"] = Field(default="gw", description="Likelihood type identifier")

    events: list[GWEventConfig] = Field(
        description=(
            "List of GW events to include. Each event must have 'name'. "
            "Optional 'nf_model_dir' specifies path to normalizing flow model; "
            "if omitted, uses preset paths based on event name. "
            "Alternatively, set 'from_bilby_result' to train a flow automatically."
        ),
        min_length=1,
    )

    @field_validator("events")
    @classmethod
    def validate_unique_event_names(cls, v: list[GWEventConfig]) -> list[GWEventConfig]:
        """Ensure all GW event names are unique."""
        names = [event.name for event in v]
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                f"Duplicate GW event names found: {sorted(set(duplicates))}. "
                "Each event must have a unique name."
            )
        return v

    penalty_value: float = Field(
        default=0.0,
        description="Log-likelihood penalty returned when M > M_TOV (default: 0.0, i.e. no penalty)",
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

    penalty_value: float = Field(default=0.0)
    N_masses_evaluation: int = Field(default=20, gt=0)
    N_masses_batch_size: int = Field(default=10, gt=0)

    @field_validator("events")
    @classmethod
    def validate_events(cls, v: list[dict[str, str]]) -> list[dict[str, str]]:
        """Validate event structure and uniqueness."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for i, event in enumerate(v):
            if "name" not in event:
                raise ValueError(f"Event {i} missing required 'name' field")
            name = event["name"]
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                f"Duplicate GW event names found: {sorted(set(duplicates))}. "
                "Each event must have a unique name."
            )
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
        """Validate pulsar structure and uniqueness."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for i, pulsar in enumerate(v):
            if "name" not in pulsar:
                raise ValueError(f"Pulsar {i} missing required 'name' field")

            name = pulsar["name"]
            if name in seen:
                duplicates.append(name)
            seen.add(name)

            # Warn if model directories not provided (will fail at runtime)
            if "amsterdam_model_dir" not in pulsar:
                logger.warning(
                    f"Pulsar {i} ({name}) missing 'amsterdam_model_dir'. "
                    "NICERLikelihood.__init__ will raise ValueError at runtime. "
                    "Preset model paths are not yet implemented."
                )
            if "maryland_model_dir" not in pulsar:
                logger.warning(
                    f"Pulsar {i} ({name}) missing 'maryland_model_dir'. "
                    "NICERLikelihood.__init__ will raise ValueError at runtime. "
                    "Preset model paths are not yet implemented."
                )
        if duplicates:
            raise ValueError(
                f"Duplicate NICER pulsar names found: {sorted(set(duplicates))}. "
                "Each pulsar must have a unique name."
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
        """Validate pulsar structure and uniqueness."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for i, pulsar in enumerate(v):
            if "name" not in pulsar:
                raise ValueError(f"Pulsar {i} missing required 'name' field")
            name = pulsar["name"]
            if name in seen:
                duplicates.append(name)
            seen.add(name)
            # Both sample files are required for KDE approach
            if "amsterdam_samples_file" not in pulsar:
                raise ValueError(
                    f"Pulsar {i} missing required 'amsterdam_samples_file' field"
                )
            if "maryland_samples_file" not in pulsar:
                raise ValueError(
                    f"Pulsar {i} missing required 'maryland_samples_file' field"
                )
        if duplicates:
            raise ValueError(
                f"Duplicate NICER pulsar names found: {sorted(set(duplicates))}. "
                "Each pulsar must have a unique name."
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
          penalty_stability: -1e10
          penalty_pressure: -1e10
    """

    type: Literal["constraints_eos"] = Field(
        default="constraints_eos", description="Likelihood type identifier"
    )

    penalty_causality: float = Field(
        default=-1e10,
        description="Log-likelihood penalty for causality violation (cs² > 1)",
    )

    penalty_stability: float = Field(
        default=-1e10,
        description="Log-likelihood penalty for thermodynamic instability (cs² < 0)",
    )

    penalty_pressure: float = Field(
        default=-1e10,
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
          penalty_stability: -1e10
          penalty_pressure: -1e10
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
        default=-1e10,
        description="Log-likelihood penalty for thermodynamic instability (cs² < 0)",
    )

    penalty_pressure: float = Field(
        default=-1e10,
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
