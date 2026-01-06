#!/usr/bin/env python
"""
TODO: might be better to relocate this script to docs/scripts/ or similar?
TODO: Automate running this as part of CI/CD?

Generate comprehensive YAML configuration reference from Pydantic schemas. Script procuced by Claude.

This script extracts all fields, types, defaults, and descriptions from the
Pydantic models in schema.py and generates a complete YAML reference document.

Usage:
    uv run python -m jesterTOV.inference.config.generate_yaml_reference

Output:
    Writes to: docs/inference_yaml_reference.md
"""

from typing import Any, get_args, get_origin
from pathlib import Path
from pydantic import BaseModel

from .schema import (
    InferenceConfig,
    TransformConfig,
    PriorConfig,
    LikelihoodConfig,
    FlowMCSamplerConfig,
    BlackJAXNSAWConfig,
    SMCSamplerConfig,
)


def get_type_string(field_type: type | None) -> str:
    """Convert Python type to readable string"""
    if field_type is None:
        return "Any"
    origin = get_origin(field_type)

    # Handle Literal types
    if origin is None:
        if hasattr(field_type, '__name__'):
            return field_type.__name__
        return str(field_type)

    # Handle generic types
    if origin is list:
        args = get_args(field_type)
        if args:
            return f"list[{get_type_string(args[0])}]"
        return "list"

    if origin is dict:
        args = get_args(field_type)
        if args:
            return f"dict[{get_type_string(args[0])}, {get_type_string(args[1])}]"
        return "dict"

    # Handle Literal
    if hasattr(origin, '__name__') and origin.__name__ == 'Literal':
        args = get_args(field_type)
        return " | ".join(f'"{arg}"' if isinstance(arg, str) else str(arg) for arg in args)

    return str(field_type)


def extract_field_info(model: type[BaseModel]) -> list[dict[str, Any]]:
    """Extract all field information from a Pydantic model"""
    from pydantic_core import PydanticUndefined

    fields_info = []

    for field_name, field in model.model_fields.items():
        # Handle default values properly
        if field.is_required():
            default = None
        elif field.default is PydanticUndefined or field.default is None:
            # Check if there's a default_factory
            if field.default_factory is not None and field.default_factory is not PydanticUndefined:
                try:
                    # Try calling with no arguments; default_factory is typically a callable
                    if callable(field.default_factory):
                        default = field.default_factory()  # type: ignore[call-arg]
                    else:
                        default = None
                except Exception:
                    default = None
            else:
                default = None
        else:
            default = field.default

        field_info = {
            "name": field_name,
            "type": get_type_string(field.annotation),
            "required": field.is_required(),
            "default": default,
            "description": field.description or "",
        }

        # Get additional info from Field() if available
        if hasattr(field, 'json_schema_extra') and field.json_schema_extra:
            # json_schema_extra can be dict or callable - only handle dict case
            if isinstance(field.json_schema_extra, dict):
                field_info.update(field.json_schema_extra)

        fields_info.append(field_info)

    return fields_info


def generate_field_docs(fields: list[dict[str, Any]], indent: int = 0) -> str:
    """Generate markdown documentation for fields"""
    lines = []
    indent_str = "  " * indent

    for field in fields:
        name = field["name"]
        ftype = field["type"]
        required = field["required"]
        default = field["default"]
        desc = field["description"]

        # Field header
        req_str = "**required**" if required else "optional"
        lines.append(f"{indent_str}- `{name}`: `{ftype}` ({req_str})")

        # Default value
        if not required and default is not None:
            if isinstance(default, str):
                lines.append(f"{indent_str}  - Default: `\"{default}\"`")
            elif isinstance(default, dict) and len(default) == 0:
                lines.append(f"{indent_str}  - Default: `{{}}`")
            else:
                lines.append(f"{indent_str}  - Default: `{default}`")

        # Description
        if desc:
            lines.append(f"{indent_str}  - {desc}")

        lines.append("")

    return "\n".join(lines)

def generate_example_yaml(fields: list[dict[str, Any]]) -> str:
    """Generate example YAML for a model"""
    lines = []

    for field in fields:
        name = field["name"]
        ftype = field["type"]
        default = field["default"]
        required = field["required"]

        # Generate example value
        if not required and default is not None:
            if isinstance(default, str):
                value = f'"{default}"'
            elif isinstance(default, dict):
                value = "{}" if len(default) == 0 else str(default)
            else:
                value = str(default)
        elif "int" in ftype:
            value = "100"
        elif "float" in ftype:
            value = "1.0"
        elif "str" in ftype or '"' in ftype:
            # Extract first option from Literal if present
            if '"' in ftype:
                value = ftype.split('"')[1]
            else:
                value = "example"
            value = f'"{value}"'
        elif "bool" in ftype:
            value = "true"
        elif "list" in ftype:
            value = "[]"
        elif "dict" in ftype:
            value = "{}"
        else:
            value = "..."

        comment = " # required" if required else ""
        lines.append(f"{name}: {value}{comment}")

    return "\n".join(lines)


def generate_complete_reference() -> str:
    """Generate complete YAML reference documentation"""

    doc = """# JESTER Inference YAML Configuration Reference

**Auto-generated from Pydantic schemas** - This document is the authoritative reference for all supported YAML configuration options.

> **Note**: This file is automatically generated from `jesterTOV/inference/config/schema.py`.
> To update this reference, run:
> ```bash
> uv run python -m jesterTOV.inference.config.generate_yaml_reference
> ```

## Overview

The JESTER inference system uses YAML configuration files validated by Pydantic models. This reference documents every supported field, its type, default value, and purpose.

---

## Complete Configuration Template

```yaml
# Complete JESTER inference configuration with all available options

"""

    # Generate InferenceConfig section
    doc += "# Top-level configuration\n"
    top_fields = extract_field_info(InferenceConfig)
    # Filter out nested models for the template
    simple_fields = [f for f in top_fields if f["name"] not in ["transform", "prior", "likelihoods", "sampler"]]
    for field in simple_fields:
        if field['name'] == 'data_paths':
            continue  # Handle separately at the end
        default_val = field['default'] if field['default'] is not None else '...'
        doc += f"{field['name']}: {default_val}\n"

    doc += "\n# Transform configuration\n"
    doc += "transform:\n"
    transform_fields = extract_field_info(TransformConfig)
    for field in transform_fields:
        default = field['default'] if field['default'] is not None else '...'
        if isinstance(default, str):
            default = f'"{default}"'
        doc += f"  {field['name']}: {default}\n"

    doc += "\n# Prior configuration\n"
    doc += "prior:\n"
    prior_fields = extract_field_info(PriorConfig)
    for field in prior_fields:
        doc += f"  {field['name']}: \"{field['default'] if field['default'] is not None else 'prior.prior'}\"\n"

    doc += "\n# Likelihoods (list of likelihood configurations)\n"
    doc += "likelihoods:\n"
    doc += "  - type: \"gw\"  # or \"nicer\", \"radio\", \"chieft\", \"rex\", \"zero\"\n"
    doc += "    enabled: true\n"
    doc += "    parameters:\n"
    doc += "      # Likelihood-specific parameters (see section below)\n"

    doc += "\n# Sampler configuration (choose one type)\n"
    doc += "sampler:\n"
    doc += "  type: \"flowmc\"  # or \"nested_sampling\", \"smc\"\n"
    doc += "  # See sampler-specific fields below\n"

    doc += "\n# Data paths (optional overrides)\n"
    doc += "data_paths: {}\n"

    doc += "```\n\n---\n\n"

    # Document each section in detail
    doc += "## Field Reference\n\n"

    # Top-level fields
    doc += "### Top-Level Configuration\n\n"
    doc += generate_field_docs(top_fields)

    # Transform fields
    doc += "### Transform Configuration (`transform:`)\n\n"
    doc += "Defines how EOS parameters are transformed to observables.\n\n"
    doc += generate_field_docs(transform_fields)

    # Add validation note for transform
    doc += "**Validation Rules**:\n"
    doc += "- If `type: \"metamodel\"`, then `nb_CSE` must be 0 (or omitted)\n"
    doc += "- If `type: \"metamodel_cse\"`, then `nb_CSE` must be > 0\n"
    doc += "- `crust_name` must be one of: `\"DH\"`, `\"BPS\"`, or `\"DH_fixed\"`\n\n"

    # Prior fields
    doc += "### Prior Configuration (`prior:`)\n\n"
    doc += "Specifies prior distributions for parameters.\n\n"
    doc += generate_field_docs(prior_fields)

    # Likelihood fields
    doc += "### Likelihood Configuration (`likelihoods:`)\n\n"
    doc += "List of observational constraints. Each likelihood has:\n\n"
    likelihood_fields = extract_field_info(LikelihoodConfig)
    doc += generate_field_docs(likelihood_fields)

    # Document likelihood-specific parameters
    doc += "**Likelihood-Specific Parameters** (`parameters:`):\n\n"

    doc += "#### Gravitational Wave (`type: \"gw\"`)\n\n"
    doc += "```yaml\n"
    doc += "- type: \"gw\"\n"
    doc += "  enabled: true\n"
    doc += "  parameters:\n"
    doc += "    event_name: \"GW170817\"          # GW event name\n"
    doc += "    model_path: \"./NFs/model.eqx\"   # Path to normalizing flow model\n"
    doc += "    very_negative_value: -9999999.0  # Return for invalid M-R (optional)\n"
    doc += "```\n\n"

    doc += "#### NICER X-ray Timing (`type: \"nicer\"`)\n\n"
    doc += "```yaml\n"
    doc += "- type: \"nicer\"\n"
    doc += "  enabled: true\n"
    doc += "  parameters:\n"
    doc += "    targets: [\"J0030\", \"J0740\"]              # Pulsar names\n"
    doc += "    analysis_groups: [\"amsterdam\", \"maryland\"]  # Analysis groups to use\n"
    doc += "    m_min: 1.0                                # Min mass for marginalization\n"
    doc += "    m_max: 2.5                                # Max mass for marginalization\n"
    doc += "    nb_masses: 100                            # Mass grid size\n"
    doc += "```\n\n"

    doc += "#### Radio Pulsar Timing (`type: \"radio\"`)\n\n"
    doc += "```yaml\n"
    doc += "- type: \"radio\"\n"
    doc += "  enabled: true\n"
    doc += "  parameters:\n"
    doc += "    psr_name: \"J0740+6620\"  # Pulsar name (for labeling)\n"
    doc += "    mass_mean: 2.08         # Mean mass (solar masses)\n"
    doc += "    mass_std: 0.07          # Mass uncertainty (1-sigma)\n"
    doc += "    nb_masses: 100          # Mass grid size for marginalization\n"
    doc += "```\n\n"

    doc += "#### Chiral Effective Field Theory (`type: \"chieft\"`)\n\n"
    doc += "```yaml\n"
    doc += "- type: \"chieft\"\n"
    doc += "  enabled: true\n"
    doc += "  parameters:\n"
    doc += "    nb_n: 100  # Number of density points to check against bands\n"
    doc += "```\n\n"

    doc += "#### PREX/CREX (`type: \"rex\"`)\n\n"
    doc += "```yaml\n"
    doc += "- type: \"rex\"\n"
    doc += "  enabled: true\n"
    doc += "  parameters:\n"
    doc += "    experiment_name: \"PREX\"  # \"PREX\" or \"CREX\"\n"
    doc += "```\n\n"

    doc += "#### Zero Likelihood (`type: \"zero\"`)\n\n"
    doc += "```yaml\n"
    doc += "- type: \"zero\"\n"
    doc += "  enabled: true\n"
    doc += "  parameters: {}  # No parameters needed\n"
    doc += "```\n\n"

    # Sampler fields - discriminated union
    doc += "### Sampler Configuration (`sampler:`)\n\n"
    doc += "The sampler configuration uses a discriminated union based on the `type` field. Choose one of:\n\n"

    # FlowMC
    doc += "#### FlowMC Sampler (`type: \"flowmc\"`)\n\n"
    doc += "Normalizing flow-enhanced MCMC with local and global sampling phases.\n\n"
    flowmc_fields = extract_field_info(FlowMCSamplerConfig)
    doc += generate_field_docs(flowmc_fields)
    doc += "\n**Sampling Phases**:\n"
    doc += "- **Training**: `n_loop_training` loops of `n_local_steps` MCMC + NF training for `n_epochs`\n"
    doc += "- **Production**: `n_loop_production` loops of `n_local_steps` MCMC + `n_global_steps` NF proposals\n\n"

    # Nested Sampling
    doc += "#### Nested Sampling (`type: \"nested_sampling\"`)\n\n"
    doc += "BlackJAX nested sampling with acceptance walk for Bayesian evidence estimation.\n\n"
    ns_fields = extract_field_info(BlackJAXNSAWConfig)
    doc += generate_field_docs(ns_fields)
    doc += "\n**Output**: Evidence (logZ ± error) and posterior samples with importance weights.\n\n"

    # SMC
    doc += "#### Sequential Monte Carlo (`type: \"smc\"`)\n\n"
    doc += "BlackJAX SMC with adaptive tempering and NUTS kernel.\n\n"
    smc_fields = extract_field_info(SMCSamplerConfig)
    doc += generate_field_docs(smc_fields)
    doc += "\n**Output**: Posterior samples and effective sample size (ESS) statistics.\n\n"

    # Data paths
    doc += "### Data Paths (`data_paths:`)\n\n"
    doc += "Optional dictionary to override default data file locations.\n\n"
    doc += "**Supported keys**:\n\n"
    doc += "```yaml\n"
    doc += "data_paths:\n"
    doc += "  # NICER data\n"
    doc += "  nicer_j0030_amsterdam: \"./data/NICER/J0030/amsterdam.txt\"\n"
    doc += "  nicer_j0030_maryland: \"./data/NICER/J0030/maryland.txt\"\n"
    doc += "  nicer_j0740_amsterdam: \"./data/NICER/J0740/amsterdam.dat\"\n"
    doc += "  nicer_j0740_maryland: \"./data/NICER/J0740/maryland.txt\"\n"
    doc += "  \n"
    doc += "  # ChiEFT bands\n"
    doc += "  chieft_low: \"./data/chieft/low_density.txt\"\n"
    doc += "  chieft_high: \"./data/chieft/high_density.txt\"\n"
    doc += "  \n"
    doc += "  # GW normalizing flow models\n"
    doc += "  gw170817_model: \"./NFs/GW170817/model.eqx\"\n"
    doc += "  \n"
    doc += "  # REX posteriors\n"
    doc += "  prex_posterior: \"./data/REX/PREX_posterior.npz\"\n"
    doc += "  crex_posterior: \"./data/REX/CREX_posterior.npz\"\n"
    doc += "```\n\n"

    # Validation summary
    doc += "## Validation Rules\n\n"
    doc += "The configuration is validated using Pydantic. Common validation rules:\n\n"
    doc += "1. **Transform type consistency**:\n"
    doc += "   - `type: \"metamodel\"` requires `nb_CSE: 0`\n"
    doc += "   - `type: \"metamodel_cse\"` requires `nb_CSE > 0`\n\n"
    doc += "2. **Prior file extension**:\n"
    doc += "   - Must end with `.prior`\n\n"
    doc += "3. **Likelihood requirements**:\n"
    doc += "   - At least one likelihood must have `enabled: true`\n\n"
    doc += "4. **Positive values**:\n"
    doc += "   - `n_chains`, `n_loop_training`, `n_loop_production` must be > 0\n"
    doc += "   - `learning_rate` must be in (0, 1]\n\n"
    doc += "5. **Valid crust models**:\n"
    doc += "   - `crust_name` must be `\"DH\"`, `\"BPS\"`, or `\"DH_fixed\"`\n\n"

    # Examples
    doc += "## Complete Examples\n\n"

    doc += "### Minimal Configuration (Prior-only)\n\n"
    doc += "```yaml\n"
    doc += "seed: 43\n\n"
    doc += "transform:\n"
    doc += "  type: \"metamodel\"\n\n"
    doc += "prior:\n"
    doc += "  specification_file: \"prior.prior\"\n\n"
    doc += "likelihoods:\n"
    doc += "  - type: \"zero\"\n"
    doc += "    enabled: true\n\n"
    doc += "sampler:\n"
    doc += "  n_chains: 10\n"
    doc += "  n_loop_training: 2\n"
    doc += "  n_loop_production: 2\n"
    doc += "  output_dir: \"./outdir/\"\n"
    doc += "```\n\n"

    doc += "### Full Multi-Messenger Configuration\n\n"
    doc += "```yaml\n"
    doc += "seed: 43\n\n"
    doc += "transform:\n"
    doc += "  type: \"metamodel_cse\"\n"
    doc += "  nb_CSE: 8\n"
    doc += "  ndat_metamodel: 100\n"
    doc += "  nmax_nsat: 25.0\n\n"
    doc += "prior:\n"
    doc += "  specification_file: \"prior.prior\"\n\n"
    doc += "likelihoods:\n"
    doc += "  - type: \"gw\"\n"
    doc += "    enabled: true\n"
    doc += "    parameters:\n"
    doc += "      event_name: \"GW170817\"\n"
    doc += "  \n"
    doc += "  - type: \"nicer\"\n"
    doc += "    enabled: true\n"
    doc += "    parameters:\n"
    doc += "      targets: [\"J0030\", \"J0740\"]\n"
    doc += "  \n"
    doc += "  - type: \"radio\"\n"
    doc += "    enabled: true\n"
    doc += "    parameters:\n"
    doc += "      mass_mean: 2.08\n"
    doc += "      mass_std: 0.07\n"
    doc += "  \n"
    doc += "  - type: \"chieft\"\n"
    doc += "    enabled: true\n\n"
    doc += "sampler:\n"
    doc += "  n_chains: 20\n"
    doc += "  n_loop_training: 3\n"
    doc += "  n_loop_production: 5\n"
    doc += "  n_local_steps: 200\n"
    doc += "  n_global_steps: 200\n"
    doc += "  output_dir: \"./outdir/\"\n"
    doc += "```\n\n"

    # Footer
    doc += "---\n\n"
    doc += "**Document Status**: Auto-generated\n"
    doc += "**Source**: `jesterTOV/inference/config/schema.py`\n"
    doc += "**Generator**: `jesterTOV/inference/config/generate_yaml_reference.py`\n\n"
    doc += "To regenerate this reference:\n"
    doc += "```bash\n"
    doc += "uv run python -m jesterTOV.inference.config.generate_yaml_reference\n"
    doc += "```\n"

    return doc


def main():
    """Generate YAML reference and write to docs/"""

    # Find repository root (go up from this file until we find jester/)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent.parent  # jester/
    docs_dir = repo_root / "docs"

    # Ensure docs directory exists
    docs_dir.mkdir(exist_ok=True)

    # Generate reference
    print("Generating YAML configuration reference from Pydantic schemas...")
    reference = generate_complete_reference()

    # Write to file
    output_path = docs_dir / "inference_yaml_reference.md"
    with open(output_path, "w") as f:
        f.write(reference)

    print(f"✓ Reference written to: {output_path}")
    print(f"  Total lines: {len(reference.splitlines())}")
    print("\nTo view the reference:")
    print(f"  cat {output_path}")


if __name__ == "__main__":
    main()
