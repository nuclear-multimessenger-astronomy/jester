"""Pydantic configuration schema for normalizing flow training.

This module provides type-safe configuration for training normalizing flows
on gravitational wave posterior samples, replacing the argparse interface
in train_flow.py.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator


class FlowTrainingConfig(BaseModel):
    """Configuration for training normalizing flows on posterior samples.

    Attributes
    ----------
    posterior_file : str
        Path to .npz file with posterior samples
    output_dir : str
        Directory to save model weights, kwargs, and plots
    parameter_names : List[str] | None
        List of parameter names to extract from posterior file.
        If None, defaults to GW parameters: mass_1_source, mass_2_source, lambda_1, lambda_2
        Cannot be an empty list.
        Examples:
            - GW: ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]
            - NICER: ["mass", "radius"]
            - EOS: ["log_p1", "gamma1", "gamma2", "gamma3"]
    num_epochs : int
        Number of training epochs (default: 600)
    learning_rate : float
        Learning rate for training (default: 1e-3)
    max_patience : int
        Early stopping patience (default: 50)
    nn_depth : int
        Depth of neural network blocks (default: 5)
    nn_block_dim : int
        Dimension of neural network blocks (default: 8)
    flow_layers : int
        Number of flow layers (default: 1)
    invert : bool
        Whether to invert the flow (default: True)
    cond_dim : int | None
        Conditional dimension for conditional flows (default: None)
    max_samples : int
        Maximum number of samples to use for training (default: 50,000)
    seed : int
        Random seed for reproducibility (default: 0)
    plot_corner : bool
        Generate corner plot comparison (default: True)
    plot_losses : bool
        Plot training and validation losses (default: True)
    flow_type : Literal["block_neural_autoregressive_flow", "masked_autoregressive_flow", "coupling_flow"]
        Type of normalizing flow to use (default: masked_autoregressive_flow)
    nn_width : int
        Width of neural network hidden layers (default: 50)
    standardize : bool
        Standardize input data to [0,1] domain using min-max scaling (default: False)
    transformer : Literal["affine", "rational_quadratic_spline"]
        Transformer type for masked_autoregressive_flow and coupling_flow (default: affine)
    transformer_knots : int
        Number of knots for RationalQuadraticSpline transformer (default: 8)
    transformer_interval : float
        Interval for RationalQuadraticSpline transformer (default: 4.0)
    val_prop : float
        Proportion of data to use for validation (default: 0.2)
    batch_size : int
        Batch size for training (default: 128)
    """

    posterior_file: str
    output_dir: str
    parameter_names: List[str] | None = None
    num_epochs: int = 600
    learning_rate: float = 1e-3
    max_patience: int = 50
    nn_depth: int = 5
    nn_block_dim: int = 8
    flow_layers: int = 1
    invert: bool = True
    cond_dim: int | None = None
    max_samples: int = 50_000
    seed: int = 0
    plot_corner: bool = True
    plot_losses: bool = True
    flow_type: Literal[
        "block_neural_autoregressive_flow",
        "masked_autoregressive_flow",
        "coupling_flow",
    ] = "masked_autoregressive_flow"
    nn_width: int = 50
    standardize: bool = False
    transformer: Literal["affine", "rational_quadratic_spline"] = "affine"
    transformer_knots: int = 8
    transformer_interval: float = 4.0
    val_prop: float = 0.2
    batch_size: int = 128

    @field_validator(
        "num_epochs",
        "max_patience",
        "nn_depth",
        "nn_block_dim",
        "flow_layers",
        "max_samples",
        "nn_width",
        "transformer_knots",
        "batch_size",
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that integer value is positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v

    @field_validator("learning_rate", "val_prop", "transformer_interval")
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """Validate that float value is positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got: {v}")
        return v

    @field_validator("val_prop")
    @classmethod
    def validate_val_prop_range(cls, v: float) -> float:
        """Validate that validation proportion is in (0, 1)."""
        if v <= 0 or v >= 1:
            raise ValueError(f"val_prop must be in (0, 1), got: {v}")
        return v

    @field_validator("parameter_names")
    @classmethod
    def validate_parameter_names(cls, v: List[str] | None) -> List[str] | None:
        """Validate that parameter_names is not an empty list."""
        if v is not None and len(v) == 0:
            raise ValueError(
                "parameter_names cannot be an empty list. "
                "Either provide parameter names or set to None to use defaults."
            )
        return v

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "FlowTrainingConfig":
        """
        Load configuration from a YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            FlowTrainingConfig instance with loaded configuration

        Example:
            >>> config = FlowTrainingConfig.from_yaml("config.yaml")
        """
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
