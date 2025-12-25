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
    type : Literal["gw", "nicer", "radio", "chieft", "rex", "constraints", "zero"]
        Type of likelihood constraint
    enabled : bool
        Whether this likelihood is enabled
    parameters : dict
        Likelihood-specific parameters

        For GW likelihoods:
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
    """

    type: Literal["gw", "nicer", "radio", "chieft", "rex", "constraints", "constraints_eos", "constraints_tov", "zero"]
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def validate_likelihood_parameters(cls, v: Dict[str, Any], info) -> Dict[str, Any]:
        """Validate likelihood-specific parameters."""
        if "type" not in info.data:
            return v

        likelihood_type = info.data["type"]

        # Validate GW likelihood parameters
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
                    raise ValueError(f"Event {i} must be a dict with 'name' and 'model_dir' keys")
                if "name" not in event:
                    raise ValueError(f"Event {i} missing required 'name' field")
                if "model_dir" not in event:
                    raise ValueError(f"Event {i} missing required 'model_dir' field")

            # Set defaults for optional parameters
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
                    raise ValueError(f"Pulsar {i} missing required 'amsterdam_samples_file' field")
                if "maryland_samples_file" not in pulsar:
                    raise ValueError(f"Pulsar {i} missing required 'maryland_samples_file' field")

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
                raise ValueError("Radio timing likelihood 'pulsars' must be a non-empty list")

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
                if not isinstance(pulsar["mass_mean"], (int, float)) or pulsar["mass_mean"] <= 0:
                    raise ValueError(
                        f"Pulsar {i} 'mass_mean' must be a positive number, got: {pulsar['mass_mean']}"
                    )
                if not isinstance(pulsar["mass_std"], (int, float)) or pulsar["mass_std"] <= 0:
                    raise ValueError(
                        f"Pulsar {i} 'mass_std' must be a positive number, got: {pulsar['mass_std']}"
                    )

            # Set defaults for optional parameters
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

        return v


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
