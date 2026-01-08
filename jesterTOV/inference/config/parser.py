r"""Configuration file parser for jesterTOV inference system."""

import yaml
from pathlib import Path
from typing import Union

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
    >>> print(config.transform.type)
    'metamodel_cse'
    >>> print(config.sampler.n_chains)
    20
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
        logger.debug(f"  Transform type: {config.transform.type}")
        logger.debug(f"  Prior file: {config.prior.specification_file}")
        logger.debug(
            f"  Enabled likelihoods: {[lk.type for lk in config.likelihoods if lk.enabled]}"
        )
        logger.debug(f"  Sampler type: {config.sampler.type}")
        logger.debug(f"  Output directory: {config.sampler.output_dir}")
        return config
    except Exception as e:
        raise ValueError(
            f"Error validating configuration from {config_path}: {e}"
        ) from e
