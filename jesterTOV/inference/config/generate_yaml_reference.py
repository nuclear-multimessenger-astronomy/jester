#!/usr/bin/env python
"""
Generate comprehensive YAML configuration reference from Pydantic schemas.

This script extracts all fields, types, defaults, and descriptions from the
Pydantic models in schema.py and generates a complete YAML reference document
using Jinja2 templates.

Usage:
    uv run python -m jesterTOV.inference.config.generate_yaml_reference

Output:
    Writes to: docs/inference/yaml_reference.md
"""

from typing import Any
from pathlib import Path
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from jinja2 import Environment, FileSystemLoader

from .schema import (
    InferenceConfig,
)


def get_type_string(field_type: type | None) -> str:
    """Convert Python type annotation to readable string."""
    if field_type is None:
        return "Any"

    # Handle type strings directly
    type_str = str(field_type)

    # Clean up common patterns
    type_str = type_str.replace("typing.", "")
    type_str = type_str.replace("<class '", "").replace("'>", "")

    return type_str


def format_default_value(default: Any) -> str:
    """Format default value for display."""
    if default is None:
        return "null"
    elif isinstance(default, str):
        return f'"{default}"'
    elif isinstance(default, bool):
        return str(default).lower()
    elif isinstance(default, dict) and len(default) == 0:
        return "{}"
    elif isinstance(default, list) and len(default) == 0:
        return "[]"
    else:
        return str(default)


def extract_field_dict(
    field_name: str, field: Any, model: type[BaseModel]
) -> dict[str, Any]:
    """Extract field information into a dictionary."""
    # Handle default values
    if field.is_required():
        default = None
        required = True
    elif field.default is PydanticUndefined or field.default is None:
        if (
            field.default_factory is not None
            and field.default_factory is not PydanticUndefined
        ):
            try:
                default = (
                    field.default_factory() if callable(field.default_factory) else None
                )
            except Exception:
                default = None
        else:
            default = None
        required = False
    else:
        default = field.default
        required = False

    return {
        "name": field_name,
        "type": get_type_string(field.annotation),
        "required": required,
        "default": format_default_value(default),
        "description": field.description or "",
        "raw_default": default,
    }


def extract_run_options() -> list[dict[str, Any]]:
    """Extract run options fields from InferenceConfig."""
    config_fields = InferenceConfig.model_fields

    run_option_names = ["seed", "dry_run", "validate_only", "debug_nans"]
    fields = []

    for name in run_option_names:
        if name in config_fields:
            field_dict = extract_field_dict(name, config_fields[name], InferenceConfig)
            # Add example values and ensure descriptions are set
            if name == "seed":
                field_dict["example"] = "43"
                field_dict["inline_comment"] = "Random seed for reproducibility"
                field_dict["description"] = (
                    "Random seed for reproducibility across runs"
                )
            elif name == "dry_run":
                field_dict["example"] = "false"
                field_dict["inline_comment"] = (
                    "Validate configuration without running inference"
                )
                field_dict["description"] = (
                    "Parse and validate configuration without running inference"
                )
            elif name == "validate_only":
                field_dict["example"] = "false"
                field_dict["inline_comment"] = "Only validate configuration and exit"
                field_dict["description"] = (
                    "Validate configuration and prior file, then exit"
                )
            elif name == "debug_nans":
                field_dict["example"] = "false"
                field_dict["inline_comment"] = (
                    "Enable JAX NaN debugging for numerical issues"
                )
                if not field_dict["description"]:
                    field_dict["description"] = (
                        "Enable JAX NaN debugging to catch numerical issues during inference"
                    )
            else:
                field_dict["example"] = field_dict["default"]
                field_dict["inline_comment"] = None

            fields.append(field_dict)

    return fields


def extract_eos_configs() -> list[dict[str, Any]]:
    """Extract EOS configuration options."""
    eos_configs = [
        {
            "title": "Metamodel",
            "description": "Metamodel EOS parametrization",
            "open": False,
            "fields": [
                {
                    "name": "type",
                    "example": '"metamodel"',
                    "inline_comment": "Required: EOS parametrization type",
                },
                {
                    "name": "ndat_metamodel",
                    "example": "100",
                    "inline_comment": "Number of points for EOS table",
                },
                {
                    "name": "nmax_nsat",
                    "example": "25.0",
                    "inline_comment": "Maximum density (in units of saturation density)",
                },
                {
                    "name": "nmin_MM_nsat",
                    "example": "0.75",
                    "inline_comment": "Minimum density for metamodel (in units of n_sat)",
                },
                {
                    "name": "crust_name",
                    "example": '"DH"',
                    "inline_comment": 'Crust model: "DH", "BPS", "DH_fixed", or "SLy"',
                },
                {
                    "name": "nb_CSE",
                    "example": "0",
                    "inline_comment": "Must be 0 for standard metamodel",
                },
            ],
            "requirements": [
                "`nb_CSE` must be 0 (or omitted) for this parametrization"
            ],
            "recommended": None,
        },
        {
            "title": "Metamodel CSE",
            "description": "Metamodel EOS parametrization with speed-of-sound extension above a breakdown density",
            "open": False,
            "fields": [
                {
                    "name": "type",
                    "example": '"metamodel_cse"',
                    "inline_comment": "Required: EOS parametrization type",
                },
                {
                    "name": "nb_CSE",
                    "example": "8",
                    "inline_comment": "Number of CSE enforcement points (must be > 0)",
                },
                {
                    "name": "ndat_metamodel",
                    "example": "100",
                    "inline_comment": "Number of points for EOS table",
                },
                {
                    "name": "nmax_nsat",
                    "example": "25.0",
                    "inline_comment": "Maximum density (in units of saturation density)",
                },
                {
                    "name": "nmin_MM_nsat",
                    "example": "0.75",
                    "inline_comment": "Minimum density for metamodel (in units of n_sat)",
                },
                {
                    "name": "crust_name",
                    "example": '"DH"',
                    "inline_comment": 'Crust model: "DH", "BPS", "DH_fixed", or "SLy"',
                },
            ],
            "requirements": ["`nb_CSE` must be > 0 for this parametrization"],
            "recommended": None,
        },
        {
            "title": "Spectral (LALSuite-Compatible)",
            "description": "Spectral decomposition parametrization compatible with LALSimulation for GW analysis.",
            "open": False,
            "fields": [
                {
                    "name": "type",
                    "example": '"spectral"',
                    "inline_comment": "Required: EOS parametrization type",
                },
                {
                    "name": "n_points_high",
                    "example": "500",
                    "inline_comment": "Number of points for high-density spectral region",
                },
                {
                    "name": "crust_name",
                    "example": '"SLy"',
                    "inline_comment": 'Must be "SLy" for LALSuite compatibility',
                },
            ],
            "requirements": [
                '`crust_name` must be `"SLy"` (LALSuite compatibility requirement)',
                "`nb_CSE` must be 0 (or omitted)",
                "`n_points_high` defines high-density spectral region sampling (default: 500)",
            ],
            "recommended": [
                "Use `constraints_gamma` likelihood to bound Gamma parameters (optional but recommended)"
            ],
        },
    ]

    return eos_configs


def extract_tov_config() -> dict[str, Any]:
    """Extract TOV solver configuration fields."""
    return {
        "title": "TOV Solver Configuration",
        "description": "Configuration for the Tolman-Oppenheimer-Volkoff equation solver.",
        "fields": [
            {
                "name": "type",
                "example": '"gr"',
                "inline_comment": 'TOV solver: currently only "gr" is implemented',
            },
            {
                "name": "min_nsat_TOV",
                "example": "0.75",
                "inline_comment": "Minimum density for TOV solver (in units of n_sat)",
            },
            {
                "name": "ndat_TOV",
                "example": "100",
                "inline_comment": "Number of points for TOV integration",
            },
            {
                "name": "nb_masses",
                "example": "100",
                "inline_comment": "Number of masses for family construction",
            },
        ],
        "field_details": [
            {
                "name": "type",
                "type": "str",
                "default": '"gr"',
                "description": "TOV solver type. Supported values: 'gr' (General Relativity) and 'anisotropy' (post-TOV with beyond-GR corrections). 'scalar_tensor' is planned.",
            },
            {
                "name": "min_nsat_TOV",
                "type": "float",
                "default": "0.75",
                "description": "Minimum central density for TOV integration in units of saturation density",
            },
            {
                "name": "ndat_TOV",
                "type": "int",
                "default": "100",
                "description": "Number of data points for TOV integration",
            },
            {
                "name": "nb_masses",
                "type": "int",
                "default": "100",
                "description": "Number of masses to sample when constructing the M-R-Λ family",
            },
        ],
    }


def extract_prior_fields() -> list[dict[str, Any]]:
    """Extract prior configuration fields."""
    return [
        {
            "name": "specification_file",
            "type": "str",
            "required": True,
            "default": None,
            "example": '"prior.prior"',
            "inline_comment": "Path to prior specification file (required)",
            "description": "Path to prior specification file (must end with `.prior`)",
        }
    ]


def extract_likelihoods() -> list[dict[str, Any]]:
    """Extract likelihood configurations organized by category.

    WARNING: LikelihoodConfig uses a generic `parameters: dict` field documented
    in the docstring rather than typed Pydantic fields. This function manually
    defines parameter structures based on schema.py documentation and validation logic.

    IMPORTANT: Keep this in sync with:
    - jesterTOV/inference/config/schema.py LikelihoodConfig docstring
    - jesterTOV/inference/config/schema.py validate_likelihood_parameters()

    TODO: Refactor schema.py to use discriminated unions with typed parameter models
    for automatic introspection.
    """
    categories = [
        {
            "title": "Gravitational Wave Observations",
            "description": "Constrain the EOS using gravitational wave observations of binary neutron star mergers.",
            "likelihoods": [
                {
                    "title": "Standard GW Likelihood (Presampled)",
                    "type": "gw",
                    "parameters": [
                        {
                            "name": "events",
                            "example": '[{"name": "GW170817", "nf_model_dir": "./NFs/GW170817"}]',
                            "inline_comment": "List of GW events (see GWEventConfig below)",
                        },
                        {
                            "name": "N_masses_evaluation",
                            "example": "2000",
                            "inline_comment": "Number of mass samples to pre-sample (optional, default: 2000)",
                        },
                        {
                            "name": "N_masses_batch_size",
                            "example": "1000",
                            "inline_comment": "Batch size for processing (optional, default: 1000)",
                        },
                        {
                            "name": "seed",
                            "example": "42",
                            "inline_comment": "Random seed for mass sampling (optional, default: 42)",
                        },
                    ],
                    "field_details": [
                        {
                            "name": "events",
                            "type": "list[GWEventConfig]",
                            "default": None,
                            "description": (
                                "List of GW event configs (see **GWEventConfig** below). "
                                "Each entry must have `name`. Three modes are supported:\n"
                                "  - **Pre-trained flow**: set `nf_model_dir` to point to a trained flow, "
                                "or omit it to use a built-in preset.\n"
                                "  - **From bilby result**: set `from_bilby_result` to the path of a bilby "
                                "HDF5 result file; jester will extract posterior samples and train a flow "
                                "automatically before inference.\n"
                                "  - **From NPZ file**: set `from_npz_file` to an existing `.npz` file with "
                                "posterior samples; jester will train a flow directly from it, skipping the "
                                "bilby extraction step."
                            ),
                        },
                        {
                            "name": "penalty_value",
                            "type": "float",
                            "default": "0.0",
                            "description": "Log-likelihood penalty for masses exceeding TOV maximum mass (default: 0.0, i.e. no penalty)",
                        },
                        {
                            "name": "N_masses_evaluation",
                            "type": "int",
                            "default": "2000",
                            "description": "Number of mass samples to pre-sample from the GW posterior",
                        },
                        {
                            "name": "N_masses_batch_size",
                            "type": "int",
                            "default": "1000",
                            "description": "Batch size for jax.lax.map processing of mass grid",
                        },
                        {
                            "name": "seed",
                            "type": "int",
                            "default": "42",
                            "description": "Random seed for mass pre-sampling from GW posterior",
                        },
                    ],
                    "description_text": (
                        "**Default GW likelihood** (presampled version): pre-samples masses from "
                        "the GW posterior for efficient evaluation. Recommended for production use.\n\n"
                        "**GWEventConfig fields** (each entry in `events`):\n\n"
                        "| Field | Type | Default | Description |\n"
                        "|-------|------|---------|-------------|\n"
                        "| `name` | str | required | Event name, e.g. `GW170817` |\n"
                        "| `nf_model_dir` | str\\|null | null | Path to a pre-trained normalizing flow directory. Mutually exclusive with `from_bilby_result` and `from_npz_file`. |\n"
                        "| `from_bilby_result` | str\\|null | null | Path to a bilby result `.hdf5` file. jester will extract posterior samples and train a flow automatically. Mutually exclusive with `nf_model_dir` and `from_npz_file`. |\n"
                        "| `from_npz_file` | str\\|null | null | Path to an existing `.npz` file with posterior samples (`mass_1_source`, `mass_2_source`, `lambda_1`, `lambda_2`). jester will train a flow directly from this file, skipping bilby extraction. Mutually exclusive with `nf_model_dir` and `from_bilby_result`. |\n"
                        "| `flow_config` | str\\|null | null | Path to a `FlowTrainingConfig` YAML file for custom flow training (only valid with `from_bilby_result` or `from_npz_file`). |\n"
                        "| `retrain_flow` | bool | false | Force re-training even if a cached flow exists (only valid with `from_bilby_result` or `from_npz_file`). |\n\n"
                        "**Examples**:\n\n"
                        "```yaml\n"
                        "# Pre-trained flow (preset):\n"
                        "events:\n"
                        "  - name: GW170817\n\n"
                        "# Pre-trained flow (custom path):\n"
                        "events:\n"
                        "  - name: GW170817\n"
                        "    nf_model_dir: ./my_flow\n\n"
                        "# From bilby result (auto-train):\n"
                        "events:\n"
                        "  - name: GW170817\n"
                        "    from_bilby_result: ./GW170817_result.hdf5\n\n"
                        "# From existing NPZ file (skip bilby extraction):\n"
                        "events:\n"
                        "  - name: GW170817\n"
                        "    from_npz_file: ./GW170817_posterior.npz\n"
                        "```"
                    ),
                },
                {
                    "title": "Resampled GW Likelihood (Legacy)",
                    "type": "gw_resampled",
                    "parameters": [
                        {
                            "name": "events",
                            "example": '[{"name": "GW170817", "nf_model_dir": "./NFs/GW170817"}]',
                            "inline_comment": "List of GW events",
                        },
                        {
                            "name": "N_masses_evaluation",
                            "example": "20",
                            "inline_comment": "Number of masses per evaluation (optional, default: 20)",
                        },
                        {
                            "name": "N_masses_batch_size",
                            "example": "10",
                            "inline_comment": "Batch size for sampling (optional, default: 10)",
                        },
                    ],
                    "field_details": [
                        {
                            "name": "events",
                            "type": "list[dict]",
                            "default": None,
                            "description": "List of GW events with `name` and optional `nf_model_dir` keys",
                        },
                        {
                            "name": "penalty_value",
                            "type": "float",
                            "default": "0.0",
                            "description": "Log-likelihood penalty for masses exceeding TOV maximum mass (default: 0.0, i.e. no penalty)",
                        },
                        {
                            "name": "N_masses_evaluation",
                            "type": "int",
                            "default": "20",
                            "description": "Number of mass samples to draw on-the-fly per likelihood evaluation",
                        },
                        {
                            "name": "N_masses_batch_size",
                            "type": "int",
                            "default": "10",
                            "description": "Batch size for mass sampling and processing",
                        },
                    ],
                    "description_text": "**Legacy GW likelihood**: Resamples masses from GW posterior on-the-fly during each likelihood evaluation. Slower than presampled version.",
                },
            ],
        },
        {
            "title": "X-ray Observations",
            "description": "Constrain the mass-radius relation using NICER X-ray timing observations of millisecond pulsars.",
            "likelihoods": [
                {
                    "title": "NICER Flow Likelihood (DEFAULT)",
                    "type": "nicer",
                    "parameters": [
                        {
                            "name": "pulsars",
                            "example": '[{"name": "J0030", "amsterdam_model_dir": "./flows/models/nicer_maf/J0030/amsterdam", "maryland_model_dir": "./flows/models/nicer_maf/J0030/maryland"}]',
                            "inline_comment": "List of pulsars with flow model directories",
                        },
                        {
                            "name": "N_masses_evaluation",
                            "example": "100",
                            "inline_comment": "Number of mass samples (optional, default: 100)",
                        },
                        {
                            "name": "N_masses_batch_size",
                            "example": "20",
                            "inline_comment": "Batch size for processing (optional, default: 20)",
                        },
                        {
                            "name": "seed",
                            "example": "42",
                            "inline_comment": "Random seed for mass pre-sampling (optional, default: 42)",
                        },
                    ],
                    "field_details": [
                        {
                            "name": "pulsars",
                            "type": "list[dict]",
                            "default": None,
                            "description": "List of pulsars with `name`, `amsterdam_model_dir`, and `maryland_model_dir` keys. Model directories must point to trained normalizing flow models.",
                        },
                        {
                            "name": "N_masses_evaluation",
                            "type": "int",
                            "default": "100",
                            "description": "Number of mass samples to pre-sample from flow for deterministic evaluation",
                        },
                        {
                            "name": "N_masses_batch_size",
                            "type": "int",
                            "default": "20",
                            "description": "Batch size for processing mass samples with jax.lax.map",
                        },
                        {
                            "name": "seed",
                            "type": "int",
                            "default": "42",
                            "description": "Random seed for reproducible mass pre-sampling from flow",
                        },
                    ],
                    "description_text": "**Default NICER likelihood** using pre-trained normalizing flows on M-R posteriors. Pre-samples masses once at initialization for efficient, deterministic evaluation. Recommended for production use.",
                },
                {
                    "title": "NICER KDE Likelihood (LEGACY)",
                    "type": "nicer_kde",
                    "parameters": [
                        {
                            "name": "pulsars",
                            "example": '[{"name": "J0030", "amsterdam_samples_file": "./data/NICER/J0030/amsterdam.npz", "maryland_samples_file": "./data/NICER/J0030/maryland.npz"}]',
                            "inline_comment": "List of pulsars with sample files",
                        },
                        {
                            "name": "N_masses_evaluation",
                            "example": "100",
                            "inline_comment": "Number of masses per evaluation (optional, default: 100)",
                        },
                        {
                            "name": "N_masses_batch_size",
                            "example": "20",
                            "inline_comment": "Batch size for sampling (optional, default: 20)",
                        },
                    ],
                    "field_details": [
                        {
                            "name": "pulsars",
                            "type": "list[dict]",
                            "default": None,
                            "description": "List of pulsars with `name`, `amsterdam_samples_file`, and `maryland_samples_file` keys pointing to M-R posterior samples (npz format).",
                        },
                        {
                            "name": "N_masses_evaluation",
                            "type": "int",
                            "default": "100",
                            "description": "Number of mass samples to draw on-the-fly from posterior samples per evaluation",
                        },
                        {
                            "name": "N_masses_batch_size",
                            "type": "int",
                            "default": "20",
                            "description": "Batch size for mass sampling and KDE evaluation",
                        },
                    ],
                    "description_text": "**Legacy NICER likelihood** using kernel density estimation on M-R posterior samples. Resamples masses during each evaluation (slower, non-deterministic). For backward compatibility only - use flow-based version for new analyses.",
                },
            ],
        },
        {
            "title": "Radio Pulsar Observations",
            "description": "Constrain neutron star masses using radio pulsar timing measurements.",
            "likelihoods": [
                {
                    "title": "Radio Pulsar Likelihood",
                    "type": "radio",
                    "parameters": [
                        {
                            "name": "pulsars",
                            "example": '[{"name": "J0740+6620", "mass_mean": 2.08, "mass_std": 0.07}]',
                            "inline_comment": "List of pulsars",
                        },
                        {
                            "name": "penalty_value",
                            "example": "-1e5",
                            "inline_comment": "Penalty for M_TOV ≤ m_min (optional, default: -1e5)",
                        },
                        {
                            "name": "nb_masses",
                            "example": "100",
                            "inline_comment": "Number of mass points (optional, default: 100)",
                        },
                    ],
                    "field_details": [
                        {
                            "name": "pulsars",
                            "type": "list[dict]",
                            "default": None,
                            "description": "List of pulsars with `name`, `mass_mean`, and `mass_std` keys for Gaussian mass constraints",
                        },
                        {
                            "name": "penalty_value",
                            "type": "float",
                            "default": "-1e5",
                            "description": "Log-likelihood penalty for invalid TOV solutions (M_TOV ≤ m_min)",
                        },
                        {
                            "name": "nb_masses",
                            "type": "int",
                            "default": "100",
                            "description": "Number of mass points for numerical integration of Gaussian mass constraint",
                        },
                    ],
                },
            ],
        },
        {
            "title": "Nuclear Theory Constraints",
            "description": "Constrain the low-density EOS using nuclear theory calculations and laboratory measurements.",
            "likelihoods": [
                {
                    "title": "ChiEFT Likelihood",
                    "type": "chieft",
                    "parameters": [
                        {
                            "name": "nb_n",
                            "example": "100",
                            "inline_comment": "Number of density points to check against bands",
                        },
                    ],
                    "field_details": [
                        {
                            "name": "nb_n",
                            "type": "int",
                            "default": "100",
                            "description": "Number of density points to evaluate against ChiEFT uncertainty bands",
                        },
                    ],
                    "description_text": "Constrains the EOS at densities below ~2 n_sat using chiral effective field theory calculations. The likelihood checks that the predicted pressure-density relation falls within the ChiEFT uncertainty bands.",
                },
                {
                    "title": "REX Likelihood",
                    "type": "rex",
                    "parameters": [
                        {
                            "name": "experiment_name",
                            "example": '"PREX"',
                            "inline_comment": 'Experiment: "PREX" or "CREX"',
                        },
                    ],
                    "field_details": [
                        {
                            "name": "experiment_name",
                            "type": "str",
                            "default": None,
                            "description": 'Nuclear experiment identifier: `"PREX"` or `"CREX"`',
                        },
                    ],
                    "description_text": "Constrains the nuclear symmetry energy using neutron skin thickness measurements:\n- **PREX** - Lead Radius Experiment (²⁰⁸Pb)\n- **CREX** - Calcium Radius Experiment (⁴⁸Ca)",
                },
            ],
        },
        {
            "title": "Generic Constraints",
            "description": "Apply custom physics-motivated constraints on EOS and TOV observables.",
            "likelihoods": [
                {
                    "title": "EOS Constraints",
                    "type": "constraints_eos",
                    "parameters": None,
                    "description_text": "Apply custom constraints on equation of state properties (pressure, energy density, sound speed).",
                },
                {
                    "title": "TOV Constraints",
                    "type": "constraints_tov",
                    "parameters": None,
                    "description_text": "Apply custom constraints on TOV solution properties (maximum mass, radius bounds, etc.).",
                },
                {
                    "title": "Gamma Constraints",
                    "type": "constraints_gamma",
                    "parameters": None,
                    "description_text": 'Apply bounds on spectral decomposition Gamma parameters. Recommended when using `type: "spectral"` transform.',
                },
            ],
        },
        {
            "title": "Prior-Only Sampling",
            "description": "Sample from the prior without applying observational constraints.",
            "likelihoods": [
                {
                    "title": "Zero Likelihood",
                    "type": "zero",
                    "parameters": None,
                    "no_params_comment": "No parameters needed",
                    "description_text": "Returns zero log-likelihood (uniform likelihood) for all EOS configurations. Use this for prior-only sampling to explore the prior volume without observational constraints.",
                },
            ],
        },
    ]

    return categories


def extract_samplers() -> list[dict[str, Any]]:
    """Extract sampler configurations."""
    samplers = [
        {
            "title": "FlowMC (Normalizing Flow MCMC)",
            "description": "Normalizing flow-enhanced MCMC combining local MCMC proposals with global normalizing flow proposals.",
            "type": "flowmc",
            "type_comment": "Sampler type identifier",
            "fields": [
                {
                    "name": "type",
                    "example": '"flowmc"',
                    "inline_comment": "Sampler type identifier",
                    "blank_line_after": False,
                },
                {
                    "name": "output_dir",
                    "example": '"./outdir/"',
                    "inline_comment": "Output directory for results",
                    "blank_line_after": False,
                },
                {
                    "name": "n_eos_samples",
                    "example": "10000",
                    "inline_comment": "Number of final posterior samples",
                    "blank_line_after": False,
                },
                {
                    "name": "log_prob_batch_size",
                    "example": "1000",
                    "inline_comment": "Batch size for log-probability evaluation",
                    "blank_line_after": True,
                },
                {
                    "name": "n_chains",
                    "example": "20",
                    "inline_comment": "Number of parallel MCMC chains",
                    "blank_line_after": False,
                },
                {
                    "name": "n_loop_training",
                    "example": "3",
                    "inline_comment": "Number of training loops",
                    "blank_line_after": False,
                },
                {
                    "name": "n_local_steps",
                    "example": "100",
                    "inline_comment": "Local MCMC steps per training loop",
                    "blank_line_after": False,
                },
                {
                    "name": "n_epochs",
                    "example": "30",
                    "inline_comment": "NF training epochs per loop",
                    "blank_line_after": False,
                },
                {
                    "name": "learning_rate",
                    "example": "0.001",
                    "inline_comment": "NF optimizer learning rate",
                    "blank_line_after": False,
                },
                {
                    "name": "train_thinning",
                    "example": "1",
                    "inline_comment": "Thinning factor for training samples",
                    "blank_line_after": True,
                },
                {
                    "name": "n_loop_production",
                    "example": "3",
                    "inline_comment": "Number of production loops",
                    "blank_line_after": False,
                },
                {
                    "name": "n_global_steps",
                    "example": "100",
                    "inline_comment": "Global NF proposal steps per production loop",
                    "blank_line_after": False,
                },
                {
                    "name": "output_thinning",
                    "example": "5",
                    "inline_comment": "Thinning factor for output samples",
                    "blank_line_after": False,
                },
            ],
            "phases": [
                {
                    "title": "Training Phase",
                    "description": "`n_loop_training` loops of:",
                    "details": [
                        "`n_local_steps` MCMC steps using local proposals",
                        "Train normalizing flow for `n_epochs` on collected samples",
                    ],
                },
                {
                    "title": "Production Phase",
                    "description": "`n_loop_production` loops of:",
                    "details": [
                        "`n_local_steps` MCMC steps using local proposals",
                        "`n_global_steps` using normalizing flow proposals",
                    ],
                },
            ],
            "when_to_use": [
                "Multi-modal or high-dimensional posteriors",
                "Long production runs requiring efficient exploration",
                "When training overhead is acceptable",
            ],
        },
        {
            "title": "Sequential Monte Carlo with Random Walk",
            "description": "BlackJAX SMC with adaptive tempering and Gaussian Random Walk kernel. **Production-ready and recommended for most analyses.**",
            "type": "smc-rw",
            "type_comment": "Sampler type identifier",
            "fields": [
                {
                    "name": "type",
                    "example": '"smc-rw"',
                    "inline_comment": "Sampler type identifier",
                    "blank_line_after": False,
                },
                {
                    "name": "output_dir",
                    "example": '"./outdir/"',
                    "inline_comment": "Output directory for results",
                    "blank_line_after": False,
                },
                {
                    "name": "n_eos_samples",
                    "example": "10000",
                    "inline_comment": "Number of final posterior samples",
                    "blank_line_after": False,
                },
                {
                    "name": "log_prob_batch_size",
                    "example": "1000",
                    "inline_comment": "Batch size for log-probability evaluation",
                    "blank_line_after": True,
                },
                {
                    "name": "n_particles",
                    "example": "10000",
                    "inline_comment": "Number of SMC particles",
                    "blank_line_after": False,
                },
                {
                    "name": "n_mcmc_steps",
                    "example": "1",
                    "inline_comment": "MCMC steps per tempering stage",
                    "blank_line_after": False,
                },
                {
                    "name": "target_ess",
                    "example": "0.9",
                    "inline_comment": "Target effective sample size (ESS) fraction",
                    "blank_line_after": False,
                },
                {
                    "name": "random_walk_sigma",
                    "example": "1.0",
                    "inline_comment": "Gaussian random walk step size",
                    "blank_line_after": False,
                },
            ],
            "field_details": [
                {
                    "name": "n_particles",
                    "type": "int",
                    "default": "10000",
                    "description": "Number of particles for SMC",
                },
                {
                    "name": "n_mcmc_steps",
                    "type": "int",
                    "default": "1",
                    "description": "MCMC rejuvenation steps per tempering stage",
                },
                {
                    "name": "target_ess",
                    "type": "float",
                    "default": "0.9",
                    "description": "Target ESS fraction for adaptive tempering (0.0-1.0)",
                },
                {
                    "name": "random_walk_sigma",
                    "type": "float",
                    "default": "1.0",
                    "description": "Step size for Gaussian random walk kernel",
                },
            ],
            "output": [
                "Posterior samples with equal weights",
                "Effective sample size (ESS) statistics per tempering stage",
            ],
            "when_to_use": [
                "General-purpose Bayesian inference (**recommended default**)",
                "Fast inference on CPU or GPU",
                "When derivative information is unavailable or expensive",
            ],
        },
        {
            "title": "Sequential Monte Carlo with NUTS",
            "description": "BlackJAX SMC with adaptive tempering and No-U-Turn Sampler (NUTS) kernel. **EXPERIMENTAL - use with caution.**",
            "type": "smc-nuts",
            "type_comment": "Sampler type identifier (EXPERIMENTAL)",
            "fields": [
                {
                    "name": "type",
                    "example": '"smc-nuts"',
                    "inline_comment": "Sampler type identifier (EXPERIMENTAL)",
                    "blank_line_after": False,
                },
                {
                    "name": "output_dir",
                    "example": '"./outdir/"',
                    "inline_comment": "Output directory for results",
                    "blank_line_after": False,
                },
                {
                    "name": "n_eos_samples",
                    "example": "10000",
                    "inline_comment": "Number of final posterior samples",
                    "blank_line_after": False,
                },
                {
                    "name": "log_prob_batch_size",
                    "example": "1000",
                    "inline_comment": "Batch size for log-probability evaluation",
                    "blank_line_after": True,
                },
                {
                    "name": "n_particles",
                    "example": "10000",
                    "inline_comment": "Number of SMC particles",
                    "blank_line_after": False,
                },
                {
                    "name": "n_mcmc_steps",
                    "example": "1",
                    "inline_comment": "NUTS steps per tempering stage",
                    "blank_line_after": False,
                },
                {
                    "name": "target_ess",
                    "example": "0.9",
                    "inline_comment": "Target effective sample size (ESS) fraction",
                    "blank_line_after": True,
                },
                {
                    "name": "init_step_size",
                    "example": "0.01",
                    "inline_comment": "Initial NUTS step size",
                    "blank_line_after": False,
                },
                {
                    "name": "mass_matrix_base",
                    "example": "0.2",
                    "inline_comment": "Base value for mass matrix diagonal",
                    "blank_line_after": False,
                },
                {
                    "name": "mass_matrix_param_scales",
                    "example": "{}",
                    "inline_comment": "Per-parameter mass matrix scaling",
                    "blank_line_after": False,
                },
                {
                    "name": "target_acceptance",
                    "example": "0.7",
                    "inline_comment": "Target acceptance rate for step size adaptation",
                    "blank_line_after": False,
                },
                {
                    "name": "adaptation_rate",
                    "example": "0.3",
                    "inline_comment": "Rate of step size adaptation",
                    "blank_line_after": False,
                },
            ],
            "field_details": [
                {
                    "name": "init_step_size",
                    "type": "float",
                    "default": "0.01",
                    "description": "Initial step size for NUTS integrator",
                },
                {
                    "name": "mass_matrix_base",
                    "type": "float",
                    "default": "0.2",
                    "description": "Base diagonal value for mass matrix",
                },
                {
                    "name": "mass_matrix_param_scales",
                    "type": "dict",
                    "default": "{}",
                    "description": "Per-parameter scaling factors for mass matrix",
                },
                {
                    "name": "target_acceptance",
                    "type": "float",
                    "default": "0.7",
                    "description": "Target acceptance probability for step size tuning",
                },
                {
                    "name": "adaptation_rate",
                    "type": "float",
                    "default": "0.3",
                    "description": "Adaptation rate for step size controller",
                },
            ],
            "output": [
                "Posterior samples with equal weights",
                "Effective sample size (ESS) statistics per tempering stage",
            ],
            "when_to_use": [
                "**EXPERIMENTAL** - Not recommended for production use",
                "High-dimensional posteriors where gradient information helps",
                "When NUTS kernel stability can be verified",
            ],
            "warning": "This sampler is experimental. Use SMC Random Walk for production analyses.",
        },
        {
            "title": "Nested Sampling (BlackJAX NS-AW)",
            "description": "BlackJAX nested sampling with acceptance walk for Bayesian evidence estimation and posterior sampling.",
            "type": "blackjax-ns-aw",
            "type_comment": "Sampler type identifier",
            "fields": [
                {
                    "name": "type",
                    "example": '"blackjax-ns-aw"',
                    "inline_comment": "Sampler type identifier",
                    "blank_line_after": False,
                },
                {
                    "name": "output_dir",
                    "example": '"./outdir/"',
                    "inline_comment": "Output directory for results",
                    "blank_line_after": False,
                },
                {
                    "name": "n_eos_samples",
                    "example": "10000",
                    "inline_comment": "Number of final posterior samples",
                    "blank_line_after": False,
                },
                {
                    "name": "log_prob_batch_size",
                    "example": "1000",
                    "inline_comment": "Batch size for log-probability evaluation",
                    "blank_line_after": True,
                },
                {
                    "name": "n_live",
                    "example": "1000",
                    "inline_comment": "Number of live points",
                    "blank_line_after": False,
                },
                {
                    "name": "n_delete_frac",
                    "example": "0.5",
                    "inline_comment": "Fraction of live points to delete per iteration",
                    "blank_line_after": False,
                },
                {
                    "name": "n_target",
                    "example": "60",
                    "inline_comment": "Target number of MCMC steps",
                    "blank_line_after": False,
                },
                {
                    "name": "max_mcmc",
                    "example": "5000",
                    "inline_comment": "Maximum MCMC steps per iteration",
                    "blank_line_after": False,
                },
                {
                    "name": "max_proposals",
                    "example": "1000",
                    "inline_comment": "Maximum proposals per live point update",
                    "blank_line_after": False,
                },
                {
                    "name": "termination_dlogz",
                    "example": "0.1",
                    "inline_comment": "Termination criterion (log evidence uncertainty)",
                    "blank_line_after": False,
                },
            ],
            "field_details": [
                {
                    "name": "n_live",
                    "type": "int",
                    "default": "1000",
                    "description": "Number of live points for nested sampling",
                },
                {
                    "name": "n_delete_frac",
                    "type": "float",
                    "default": "0.5",
                    "description": "Fraction of live points to delete per iteration",
                },
                {
                    "name": "n_target",
                    "type": "int",
                    "default": "60",
                    "description": "Target number of MCMC steps for acceptance walk",
                },
                {
                    "name": "max_mcmc",
                    "type": "int",
                    "default": "5000",
                    "description": "Maximum MCMC steps per iteration",
                },
                {
                    "name": "max_proposals",
                    "type": "int",
                    "default": "1000",
                    "description": "Maximum proposal attempts per live point update",
                },
                {
                    "name": "termination_dlogz",
                    "type": "float",
                    "default": "0.1",
                    "description": "Terminate when log-evidence uncertainty < this value",
                },
            ],
            "output": [
                "Log-evidence (logZ) with uncertainty estimate",
                "Posterior samples with importance weights",
            ],
            "when_to_use": [
                "Model comparison requiring Bayesian evidence",
                "Exploring multi-modal posteriors",
                "When evidence estimation is primary goal",
            ],
        },
    ]

    return samplers


def extract_data_paths() -> list[dict[str, Any]]:
    """Extract data path sections."""
    sections = [
        {
            "comment": "NICER data files",
            "paths": [
                {
                    "key": "nicer_j0030_amsterdam",
                    "example": '"./data/NICER/J0030/amsterdam.txt"',
                },
                {
                    "key": "nicer_j0030_maryland",
                    "example": '"./data/NICER/J0030/maryland.txt"',
                },
                {
                    "key": "nicer_j0740_amsterdam",
                    "example": '"./data/NICER/J0740/amsterdam.dat"',
                },
                {
                    "key": "nicer_j0740_maryland",
                    "example": '"./data/NICER/J0740/maryland.txt"',
                },
            ],
        },
        {
            "comment": "ChiEFT uncertainty bands",
            "paths": [
                {"key": "chieft_low", "example": '"./data/chieft/low_density.txt"'},
                {"key": "chieft_high", "example": '"./data/chieft/high_density.txt"'},
            ],
        },
        {
            "comment": "Gravitational wave normalizing flow models",
            "paths": [
                {"key": "gw170817_model", "example": '"./NFs/GW170817/model.eqx"'},
            ],
        },
        {
            "comment": "REX posteriors",
            "paths": [
                {"key": "prex_posterior", "example": '"./data/REX/PREX_posterior.npz"'},
                {"key": "crex_posterior", "example": '"./data/REX/CREX_posterior.npz"'},
            ],
        },
    ]

    return sections


def extract_postprocessing() -> list[dict[str, Any]]:
    """Extract postprocessing configuration fields."""
    fields = [
        {
            "name": "enabled",
            "type": "bool",
            "default": "true",
            "example": "true",
            "inline_comment": "Enable postprocessing",
            "description": "Enable/disable all postprocessing",
        },
        {
            "name": "make_cornerplot",
            "type": "bool",
            "default": "true",
            "example": "true",
            "inline_comment": "Generate corner plot of posterior",
            "description": "Generate corner plot of EOS parameters",
        },
        {
            "name": "make_massradius",
            "type": "bool",
            "default": "true",
            "example": "true",
            "inline_comment": "Generate M-R diagram",
            "description": "Generate mass-radius diagram with posterior families",
        },
        {
            "name": "make_masslambda",
            "type": "bool",
            "default": "true",
            "example": "true",
            "inline_comment": "Generate M-Λ diagram",
            "description": "Generate mass-tidal deformability diagram",
        },
        {
            "name": "make_pressuredensity",
            "type": "bool",
            "default": "true",
            "example": "true",
            "inline_comment": "Generate P-ε diagram",
            "description": "Generate pressure-energy density relation",
        },
        {
            "name": "make_histograms",
            "type": "bool",
            "default": "true",
            "example": "true",
            "inline_comment": "Generate 1D posterior histograms",
            "description": "Generate 1D marginalized posterior histograms",
        },
        {
            "name": "make_cs2",
            "type": "bool",
            "default": "true",
            "example": "true",
            "inline_comment": "Generate speed-of-sound plot",
            "description": "Generate speed-of-sound as function of density",
        },
        {
            "name": "prior_dir",
            "type": "str | None",
            "default": "null",
            "example": "null",
            "inline_comment": "Optional: directory with prior samples",
            "description": "Directory containing prior samples for comparison",
        },
        {
            "name": "injection_eos_path",
            "type": "str | None",
            "default": "null",
            "example": "null",
            "inline_comment": "Optional: path to true EOS for injection studies",
            "description": "Path to true EOS for injection studies",
        },
    ]

    return fields


def extract_examples() -> list[dict[str, Any]]:
    """Extract complete configuration examples."""
    examples = [
        {
            "title": "Minimal Configuration (Prior-Only)",
            "description": "Sample from the prior distribution without observational constraints.",
            "yaml": """seed: 43

eos:
  type: "metamodel"

tov:
  type: "gr"

prior:
  specification_file: "prior.prior"

likelihoods:
  - type: "zero"
    enabled: true

sampler:
  type: "smc-rw"
  n_particles: 5000
  output_dir: "./outdir/\"""",
        },
        {
            "title": "Multi-Messenger Configuration",
            "description": "Combine gravitational wave, X-ray, radio, and nuclear theory constraints.",
            "yaml": """seed: 43

eos:
  type: "metamodel_cse"
  nb_CSE: 8
  ndat_metamodel: 100
  nmax_nsat: 25.0

tov:
  type: "gr"
  min_nsat_TOV: 0.75
  ndat_TOV: 100
  nb_masses: 100

prior:
  specification_file: "prior.prior"

likelihoods:
  - type: "gw"
    enabled: true
    parameters:
      event_name: "GW170817"

  - type: "nicer"
    enabled: true
    parameters:
      targets: ["J0030", "J0740"]
      analysis_groups: ["amsterdam", "maryland"]

  - type: "radio"
    enabled: true
    parameters:
      psr_name: "J0740+6620"
      mass_mean: 2.08
      mass_std: 0.07

  - type: "chieft"
    enabled: true

sampler:
  type: "smc-rw"
  n_particles: 10000
  n_mcmc_steps: 1
  target_ess: 0.9
  output_dir: "./outdir/"

postprocessing:
  enabled: true
  make_cornerplot: true
  make_massradius: true""",
        },
        {
            "title": "Spectral Parametrization (LALSuite-Compatible)",
            "description": "Configuration using spectral decomposition for GW analysis workflows.",
            "yaml": """seed: 43

eos:
  type: "spectral"
  crust_name: "SLy"               # Required for spectral
  n_points_high: 500

tov:
  type: "gr"
  min_nsat_TOV: 0.75
  ndat_TOV: 100
  nb_masses: 100

prior:
  specification_file: "spectral_prior.prior"

likelihoods:
  - type: "gw"
    enabled: true
    parameters:
      event_name: "GW170817"

  - type: "constraints_gamma"     # Recommended for spectral
    enabled: true

sampler:
  type: "flowmc"
  n_chains: 20
  n_loop_training: 3
  n_loop_production: 5
  output_dir: "./outdir/\"""",
        },
    ]

    return examples


def extract_validation_rules() -> list[dict[str, Any]]:
    """Extract validation rules."""
    rules = [
        {
            "title": "EOS Type Consistency",
            "entries": [
                '`type: "metamodel"` requires `nb_CSE: 0` (or omit the field entirely)',
                '`type: "metamodel_cse"` requires `nb_CSE > 0`',
                '`type: "spectral"` requires:',
                '  - `crust_name: "SLy"` (LALSuite compatibility)',
                "  - `nb_CSE: 0` (or omit the field)",
                "  - Recommended: Include `constraints_gamma` likelihood",
            ],
        },
        {
            "title": "TOV Configuration",
            "entries": [
                '`type` must be `"gr"` or `"anisotropy"`; `"scalar_tensor"` is planned but not yet available',
                "`min_nsat_TOV`, `ndat_TOV`, and `nb_masses` must be positive",
            ],
        },
        {
            "title": "Prior File",
            "entries": [
                "`specification_file` must end with `.prior` extension",
            ],
        },
        {
            "title": "Likelihood Requirements",
            "entries": [
                "At least one likelihood must have `enabled: true`",
            ],
        },
        {
            "title": "Positive Value Constraints",
            "entries": [
                "`n_chains`, `n_loop_training`, `n_loop_production` must be > 0",
                "`learning_rate` must be in range (0, 1]",
                "`n_particles`, `n_live` must be > 0",
            ],
        },
        {
            "title": "Crust Models",
            "entries": [
                '`crust_name` must be one of: `"DH"`, `"BPS"`, `"DH_fixed"`, or `"SLy"`',
                'Spectral EOS specifically requires `"SLy"`',
            ],
        },
    ]

    return rules


def generate_documentation() -> str:
    """Generate complete YAML reference documentation using Jinja2 template."""

    # Find template directory
    script_path = Path(__file__).resolve()
    template_dir = script_path.parent / "templates"

    # Set up Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Load template
    template = env.get_template("yaml_reference.md.j2")

    # Extract all data
    data = {
        "run_options": extract_run_options(),
        "eos_configs": extract_eos_configs(),
        "tov_config": extract_tov_config(),
        "prior_fields": extract_prior_fields(),
        "likelihood_categories": extract_likelihoods(),
        "samplers": extract_samplers(),
        "data_path_sections": extract_data_paths(),
        "postprocessing_fields": extract_postprocessing(),
        "examples": extract_examples(),
        "validation_rules": extract_validation_rules(),
    }

    # Render template
    return template.render(**data)


def main():
    """Generate YAML reference and write to docs/inference/"""

    # Find repository root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent.parent  # jester/
    docs_dir = repo_root / "docs" / "inference"

    # Ensure docs directory exists
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Generate reference
    print("Generating YAML configuration reference from Pydantic schemas...")
    reference = generate_documentation()

    # Write to file
    output_path = docs_dir / "yaml_reference.md"
    with open(output_path, "w") as f:
        f.write(reference)

    print(f"✓ Reference written to: {output_path}")
    print(f"  Total lines: {len(reference.splitlines())}")
    print("\nTo view the reference:")
    print(f"  cat {output_path}")


if __name__ == "__main__":
    main()
