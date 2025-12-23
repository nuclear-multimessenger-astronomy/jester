"""Configuration file parser for jesterTOV inference system."""

# FIXME: need to add jester logger and then put debug where we show ALL of the settings that are passed, so we can check the config parsing is working correctly.

import yaml
from pathlib import Path
from typing import Union

from .schema import InferenceConfig


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

    # Resolve relative paths in prior specification file
    # Make them relative to the config file directory, not CWD
    if "prior" in config_dict and "specification_file" in config_dict["prior"]:
        spec_file = Path(config_dict["prior"]["specification_file"])
        if not spec_file.is_absolute():
            # Resolve relative to config file directory
            spec_file = (config_path.parent / spec_file).resolve()
            config_dict["prior"]["specification_file"] = str(spec_file)

    try:
        return InferenceConfig(**config_dict)
    except Exception as e:
        raise ValueError(
            f"Error validating configuration from {config_path}: {e}"
        ) from e
