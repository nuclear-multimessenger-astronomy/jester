r"""Configuration file parser for jesterTOV inference system."""

import yaml
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from .schema import InferenceConfig
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


def load_config(config_path: Union[str, Path]) -> InferenceConfig:
    """Load and validate inference configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file

    Returns
    -------
    InferenceConfig
        Validated inference configuration object

    Raises
    ------
    FileNotFoundError
        If config file does not exist
    yaml.YAMLError
        If YAML parsing fails
    pydantic.ValidationError
        If configuration validation fails

    Examples
    --------
    >>> config = load_config("config.yaml")
    >>> print(config.eos.type)
    metamodel_cse
    >>> print(config.tov.type)
    gr
    """
    config_path = Path(config_path).resolve()
    logger.debug(f"Loading configuration from: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Error parsing YAML configuration file {config_path}: {e}"
            ) from e

    if config_dict is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    logger.debug(f"Raw configuration keys: {list(config_dict.keys())}")

    # Resolve relative paths in prior specification file
    # Make them relative to the config file directory, not CWD
    if "prior" in config_dict and "specification_file" in config_dict["prior"]:
        spec_file = Path(config_dict["prior"]["specification_file"])
        if not spec_file.is_absolute():
            # Resolve relative to config file directory
            spec_file = (config_path.parent / spec_file).resolve()
            config_dict["prior"]["specification_file"] = str(spec_file)

    try:
        config = InferenceConfig(**config_dict)
        logger.debug("Configuration validation successful")
        logger.debug(f"  Seed: {config.seed}")
        logger.debug(f"  EOS type: {config.eos.type}")
        logger.debug(f"  TOV solver: {config.tov.type}")
        logger.debug(f"  Prior file: {config.prior.specification_file}")
        logger.debug(
            f"  Enabled likelihoods: {[lk.type for lk in config.likelihoods if lk.enabled]}"
        )
        logger.debug(f"  Sampler type: {config.sampler.type}")
        logger.debug(f"  Output directory: {config.sampler.output_dir}")
        return config
    except ValidationError as e:
        raise ValueError(_format_validation_error(e, config_path)) from e
    except Exception as e:
        raise ValueError(
            f"Error validating configuration from {config_path}: {e}"
        ) from e


def _format_validation_error(e: ValidationError, config_path: Path) -> str:
    """Format a Pydantic ValidationError into a clean, actionable message.

    Strips Pydantic's auto-appended metadata (``[type=value_error, input_value=...]``)
    so that multi-line messages from JesterBaseModel's extra-field validator are
    displayed without noise.
    """
    _YAML_REFERENCE_URL = "https://nuclear-multimessenger-astronomy.github.io/jester/inference/yaml_reference.html"
    blocks: list[str] = [
        f"Configuration error in {config_path}:",
    ]
    for error in e.errors():
        loc_parts = [str(p) for p in error["loc"] if p != "__root__"]
        loc = " → ".join(loc_parts) if loc_parts else "(top level)"
        msg: str = error["msg"]
        # Pydantic prefixes our ValueError messages with "Value error, " — strip it.
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        blocks.append(f"\n[{loc}]\n{msg}")
    blocks.append(f"\nFor all available config options, see: {_YAML_REFERENCE_URL}")
    return "\n".join(blocks)
