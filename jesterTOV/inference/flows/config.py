"""Pydantic configuration schema for normalizing flow training.

This module provides type-safe configuration for training normalizing flows
on posterior samples (GW, NICER, EOS parameters, etc.), replacing the argparse
interface in train_flow.py.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator


class FlowTrainingConfig(BaseModel):
    """Configuration for training normalizing flows on posterior samples.

    Attributes
    ----------
    posterior_file : str
        Path to .npz file with posterior samples
    output_dir : str
        Directory to save model weights, kwargs, and plots
    parameter_names : list[str]
        List of parameter names to extract from posterior file.
        Examples: GW parameters ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
        NICER ["mass", "radius"]
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
    condition_names : list[str] | None
        Names of conditioning variables to extract from the posterior file,
        e.g. ["mass_1_source", "mass_2_source"] to train a conditional flow
        p(parameter_names | condition_names), such as p(lambda_1, lambda_2 |
        mass_1_source, mass_2_source). Must be disjoint from parameter_names.
        Defaults to None (unconditional flow).
    cond_dim : int | None
        Conditional dimension for conditional flows. When condition_names is
        set, this is derived automatically as len(condition_names) and does not
        need to be set explicitly; if both are set they must agree.
        Defaults to None (unconditional flow).
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
        Whether to standardize input data (default: True, changed from False)
    standardization_method : Literal["zscore", "minmax"]
        Method for standardizing input data (default: zscore).
        - "zscore": Standardize to mean=0, std=1 (recommended for most cases)
        - "minmax": Standardize to [0, 1] range (legacy, for backward compatibility)
        Only used if standardize=True.
    transformer : Literal["affine", "rational_quadratic_spline"]
        Transformer type for masked_autoregressive_flow and coupling_flow
        (default: rational_quadratic_spline, changed from affine)
    transformer_knots : int
        Number of knots for RationalQuadraticSpline transformer (default: 10, changed from 8)
    transformer_interval : float
        Interval for RationalQuadraticSpline transformer (default: 5.0, changed from 4.0)
    val_prop : float
        Proportion of data to use for validation (default: 0.2)
    batch_size : int
        Batch size for training (default: 128)
    """

    posterior_file: str
    output_dir: str
    parameter_names: list[str]
    num_epochs: int = 600
    learning_rate: float = 1e-3
    max_patience: int = 50
    nn_depth: int = 5
    nn_block_dim: int = 8
    flow_layers: int = 1
    invert: bool = True
    condition_names: list[str] | None = None
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
    standardize: bool = True
    standardization_method: Literal["zscore", "minmax"] = "zscore"
    transformer: Literal["affine", "rational_quadratic_spline"] = (
        "rational_quadratic_spline"
    )
    transformer_knots: int = 10
    transformer_interval: float = 5.0
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
    def validate_parameter_names(cls, v: list[str]) -> list[str]:
        """Validate that parameter_names is a non-empty list."""
        if len(v) == 0:
            raise ValueError("parameter_names cannot be an empty list.")
        return v

    @field_validator("condition_names")
    @classmethod
    def validate_condition_names(cls, v: list[str] | None) -> list[str] | None:
        """Validate that condition_names, if given, is a non-empty list."""
        if v is not None and len(v) == 0:
            raise ValueError(
                "condition_names cannot be an empty list; use None for an "
                "unconditional flow."
            )
        return v

    @model_validator(mode="after")
    def validate_condition_dim_consistency(self) -> "FlowTrainingConfig":
        """Cross-check condition_names, cond_dim, and parameter_names.

        - condition_names and parameter_names must be disjoint (a variable can't
          be both modelled and conditioned on).
        - If both condition_names and cond_dim are given, they must agree; this
          catches stale/inconsistent configs early rather than at flow-build time.
        """
        if self.condition_names is not None:
            overlap = set(self.condition_names) & set(self.parameter_names)
            if overlap:
                raise ValueError(
                    "condition_names and parameter_names must be disjoint; "
                    f"found overlap: {sorted(overlap)}"
                )
            if self.cond_dim is not None and self.cond_dim != len(self.condition_names):
                raise ValueError(
                    f"cond_dim ({self.cond_dim}) does not match "
                    f"len(condition_names) ({len(self.condition_names)}). "
                    "Leave cond_dim unset to derive it automatically."
                )
        return self

    @property
    def effective_cond_dim(self) -> int | None:
        """The conditioning dimension actually used to build the flow.

        Derived from ``len(condition_names)`` when set (the usual path for a
        conditional flow), otherwise falls back to the explicit ``cond_dim``
        (or None, for an unconditional flow). Consistency between the two, when
        both are given, is already enforced by ``validate_condition_dim_consistency``.
        """
        if self.condition_names is not None:
            return len(self.condition_names)
        return self.cond_dim

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
