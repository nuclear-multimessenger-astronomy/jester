r"""
Wrapper for trained normalizing flows with automatic data preprocessing.

This module provides a high-level interface for loading and using pre-trained
normalizing flow models for gravitational wave inference. The Flow class handles
the complexities of data standardization and model loading, allowing users to
sample from or evaluate trained flows with a simple API.

Normalizing flows trained on gravitational wave posterior samples can be used
for importance sampling in EOS inference, providing efficient proposals that
capture the correlations between binary component masses and tidal deformabilities.

Key Features
------------
- Automatic min-max standardization and inverse transformation
- Simple save/load interface compatible with flowjax models
- JAX-accelerated sampling and probability evaluation

Typical Workflow
----------------
1. Train a flow on GW posterior samples using train_flow.py
2. Load the trained flow: flow = Flow.from_directory("path/to/model/")
3. Sample or evaluate: samples = flow.sample(key, (1000,))

See Also
--------
train_flow : Module for training normalizing flows on GW posteriors

Examples
--------
Load a trained flow and generate samples:

>>> from jesterTOV.inference.flows import Flow
>>> import jax
>>> flow = Flow.from_directory("./models/gw170817/")
>>> samples = flow.sample(jax.random.key(0), (1000,))
>>> print(samples.shape)  # (1000, 4) for (m1, m2, λ1, λ2)

Evaluate log-probability of data points:

>>> data = jnp.array([[1.4, 1.3, 100, 200]])
>>> log_prob = flow.log_prob(data)
"""

import json
import os
from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.flows import (
    block_neural_autoregressive_flow,
    coupling_flow,
    masked_autoregressive_flow,
    triangular_spline_flow,
)
from flowjax.bijections import (
    RationalQuadraticSpline,
    Affine,
)


class Flow:
    """
    Wrapper class for flowjax normalizing flows with automatic standardization handling.

    This class encapsulates a trained normalizing flow and handles data standardization
    transparently. When sampling, it automatically converts samples back to the original
    scale if standardization was used during training.

    Attributes:
        flow: The underlying flowjax flow model
        metadata: Training metadata dictionary
        flow_kwargs: Flow architecture kwargs
        standardize: Whether standardization was used during training
        data_bounds: Min/max bounds for each feature (if standardization was used)

    Example:
        >>> # Load a trained flow
        >>> flow = Flow.from_directory("./models/gw170817/")
        >>>
        >>> # Sample in original scale (standardization handled automatically)
        >>> samples = flow.sample(jax.random.key(0), (1000,))
        >>>
        >>> # Access metadata
        >>> print(f"Flow type: {flow.metadata['flow_type']}")
        >>> print(f"Standardized: {flow.standardize}")
    """

    def __init__(
        self,
        flow: AbstractDistribution,
        metadata: Dict[str, Any],
        flow_kwargs: Dict[str, Any],
    ):
        """
        Initialize Flow wrapper.

        Args:
            flow: Trained flowjax flow model
            metadata: Training metadata
            flow_kwargs: Flow architecture kwargs
        """
        self.flow = flow
        self.metadata = metadata
        self.flow_kwargs = flow_kwargs
        self.standardize = metadata[
            "standardize"
        ]  # TODO: this is a bit redunant, only used below, simplify

        # Always store bounds as JAX arrays
        # If standardization is disabled, use trivial bounds (min=0, range=1)
        # that make standardization operations identity transformations
        if self.standardize:
            self.data_min = jnp.array(metadata["data_bounds_min"])
            self.data_max = jnp.array(metadata["data_bounds_max"])
        else:
            # Trivial bounds: min=0, max=1 → operations become identity
            # Assume 4D flow (m1, m2, lambda1, lambda2)
            # TODO: this might have to be changed in the future, but keep for now
            n_features = 4
            self.data_min = jnp.zeros(n_features)
            self.data_max = jnp.ones(n_features)

        # Precompute data range (max - min) to avoid repeated computation
        self.data_range = self.data_max - self.data_min
        # Avoid division by zero (though this shouldn't happen with our bounds)
        self.data_range = jnp.where(self.data_range == 0, 1.0, self.data_range)

    @classmethod
    def from_directory(cls, output_dir: str) -> "Flow":
        """
        Load a trained flow from a directory.

        Args:
            output_dir: Directory containing flow_weights.eqx, flow_kwargs.json, metadata.json

        Returns:
            Flow instance with loaded model and metadata

        Example:
            >>> flow = Flow.from_directory("./models/gw170817/")
        """
        # Load the flow model and metadata
        flow_model, metadata = load_model(output_dir)

        # Load kwargs
        kwargs_path = os.path.join(output_dir, "flow_kwargs.json")
        with open(kwargs_path, "r") as f:
            flow_kwargs = json.load(f)

        return cls(flow_model, metadata, flow_kwargs)

    def sample(self, key: Array, shape: Tuple[int, ...]) -> Array:
        """
        Sample from the flow and return in original scale.

        If standardization was used during training, samples are automatically
        converted back to the original scale. If not, the transformation is
        identity (no-op).

        Args:
            key: JAX random key (jax.Array)
            shape: Shape of samples to generate (e.g., (1000,) for 1000 samples)

        Returns:
            Samples in original scale as JAX array of shape (*shape, n_features)

        Example:
            >>> samples = flow.sample(jax.random.key(0), (1000,))
            >>> print(samples.shape)  # (1000, 4) for 4D flow
        """
        samples = self.flow.sample(key, shape)

        # Inverse standardization: [0,1] -> original scale
        # If standardization was disabled, this is identity (min=0, range=1)
        samples = samples * self.data_range + self.data_min

        return samples

    def standardize_input(self, data: Array) -> Array:
        """
        Standardize input data to [0, 1] domain using training bounds.

        If standardization was disabled, this is identity (no-op).

        Args:
            data: Input data in original scale (JAX array)

        Returns:
            Data scaled to [0, 1] (or unchanged if standardization not used)

        Example:
            >>> original_data = jnp.array([[1.4, 1.3, 100, 200]])
            >>> standardized = flow.standardize_input(original_data)
        """
        # Standardization: original scale -> [0,1]
        # If standardization was disabled, this is identity (min=0, range=1)
        return (data - self.data_min) / self.data_range

    def destandardize_output(self, data: Array) -> Array:
        """
        Convert standardized data back to original scale.

        If standardization was disabled, this is identity (no-op).

        Args:
            data: Data in [0, 1] domain (JAX array)

        Returns:
            Data in original scale (or unchanged if standardization not used)

        Example:
            >>> standardized_data = jnp.array([[0.5, 0.5, 0.5, 0.5]])
            >>> original = flow.destandardize_output(standardized_data)
        """
        # Inverse standardization: [0,1] -> original scale
        # If standardization was disabled, this is identity (min=0, range=1)
        return data * self.data_range + self.data_min

    def log_prob(self, x: Array) -> Array:
        """
        Evaluate log probability of data under the flow.

        If standardization was used, input data is automatically standardized
        before evaluation and Jacobian correction is applied. If not, operations
        are identity (no-op).

        Args:
            x: Data in original scale, shape (n_samples, n_features).
               JAX array.

        Returns:
            Log probabilities as JAX array, shape (n_samples,)

        Example:
            >>> data = jnp.array([[1.4, 1.3, 100, 200]])
            >>> log_prob = flow.log_prob(data)
        """
        # Standardize input (identity if standardization disabled)
        x_std = self.standardize_input(x)

        # Evaluate log probability
        log_p = self.flow.log_prob(x_std)

        # Account for Jacobian of inverse transformation
        # For min-max scaling: log p(x) = log p(x_std) - sum(log(x_max - x_min))
        # If standardization was disabled (range=1), log_det_jacobian = 0
        log_det_jacobian = -jnp.sum(jnp.log(self.data_range))
        log_p = log_p + log_det_jacobian

        return log_p


def create_transformer(
    transformer_type: str = "affine",
    transformer_knots: int = 8,
    transformer_interval: float = 4.0,
) -> Any:
    """
    Create a transformer for masked_autoregressive_flow and coupling_flow.

    Args:
        transformer_type: Type of transformer ("affine", "rational_quadratic_spline")
        transformer_knots: Number of knots for RationalQuadraticSpline
        transformer_interval: Interval for RationalQuadraticSpline

    Returns:
        Transformer instance
    """
    if transformer_type == "affine":
        return Affine()
    elif transformer_type == "rational_quadratic_spline":
        return RationalQuadraticSpline(
            knots=transformer_knots, interval=transformer_interval
        )
    else:
        raise ValueError(
            f"Unknown transformer type: {transformer_type}. "
            "Must be one of: affine, rational_quadratic_spline"
        )


def create_flow(
    key: Array,
    flow_type: str = "triangular_spline_flow",
    nn_depth: int = 5,
    nn_block_dim: int = 8,
    nn_width: int = 50,
    flow_layers: int = 1,
    knots: int = 8,
    tanh_max_val: float = 3.0,
    invert: bool = True,
    cond_dim: int | None = None,
    transformer_type: str = "affine",
    transformer_knots: int = 8,
    transformer_interval: float = 4.0,
) -> Any:
    """
    Create a normalizing flow of the specified type.

    Args:
        key: JAX random key
        flow_type: Type of flow ("block_neural_autoregressive_flow",
            "masked_autoregressive_flow", "coupling_flow", "triangular_spline_flow")
        nn_depth: Depth of neural network (for block_neural_autoregressive_flow,
            masked_autoregressive_flow, coupling_flow)
        nn_block_dim: Block dimension (for block_neural_autoregressive_flow)
        nn_width: Width of hidden layers (for masked_autoregressive_flow, coupling_flow)
        flow_layers: Number of flow layers
        knots: Number of spline knots (for triangular_spline_flow)
        tanh_max_val: Maximum value for tanh tails (for triangular_spline_flow)
        invert: Whether to invert the flow
        cond_dim: Conditional dimension (None for unconditional flows)
        transformer_type: Type of transformer for masked_autoregressive_flow and coupling_flow
            ("affine", "rational_quadratic_spline")
        transformer_knots: Number of knots for RationalQuadraticSpline
        transformer_interval: Interval for RationalQuadraticSpline

    Returns:
        Untrained flowjax flow model
    """
    base_dist = Normal(jnp.zeros(4))  # 4D: m1, m2, lambda1, lambda2

    if flow_type == "block_neural_autoregressive_flow":
        flow = block_neural_autoregressive_flow(
            key=key,
            base_dist=base_dist,
            nn_depth=nn_depth,
            nn_block_dim=nn_block_dim,
            flow_layers=flow_layers,
            invert=invert,
            cond_dim=cond_dim,
        )
    elif flow_type == "masked_autoregressive_flow":
        transformer = create_transformer(
            transformer_type, transformer_knots, transformer_interval
        )
        flow = masked_autoregressive_flow(
            key=key,
            base_dist=base_dist,
            flow_layers=flow_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
            invert=invert,
            cond_dim=cond_dim,
            transformer=transformer,
        )
    elif flow_type == "coupling_flow":
        transformer = create_transformer(
            transformer_type, transformer_knots, transformer_interval
        )
        flow = coupling_flow(
            key=key,
            base_dist=base_dist,
            flow_layers=flow_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
            invert=invert,
            cond_dim=cond_dim,
            transformer=transformer,
        )
    elif flow_type == "triangular_spline_flow":
        flow = triangular_spline_flow(
            key=key,
            base_dist=base_dist,
            flow_layers=flow_layers,
            knots=knots,
            tanh_max_val=tanh_max_val,
            invert=invert,
            cond_dim=cond_dim,
        )
    else:
        raise ValueError(
            f"Unknown flow type: {flow_type}. Must be one of: "
            "block_neural_autoregressive_flow, masked_autoregressive_flow, "
            "coupling_flow, triangular_spline_flow"
        )

    return flow


def load_model(output_dir: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a trained flow model from saved files.

    Args:
        output_dir: Directory containing saved model files

    Returns:
        flow: Loaded flow model
        metadata: Training metadata (includes data_bounds if standardization was used)

    Example:
        >>> flow, metadata = load_model("./models/gw170817/")
    """
    # Load kwargs
    kwargs_path = os.path.join(output_dir, "flow_kwargs.json")
    with open(kwargs_path, "r") as f:
        flow_kwargs = json.load(f)

    # Recreate flow architecture
    key = jax.random.key(flow_kwargs["seed"])
    flow = create_flow(
        key=key,
        flow_type=flow_kwargs["flow_type"],
        nn_depth=flow_kwargs["nn_depth"],
        nn_block_dim=flow_kwargs["nn_block_dim"],
        nn_width=flow_kwargs["nn_width"],
        flow_layers=flow_kwargs["flow_layers"],
        knots=flow_kwargs["knots"],
        tanh_max_val=flow_kwargs["tanh_max_val"],
        invert=flow_kwargs["invert"],
        cond_dim=flow_kwargs["cond_dim"],
        transformer_type=flow_kwargs["transformer_type"],
        transformer_knots=flow_kwargs["transformer_knots"],
        transformer_interval=flow_kwargs["transformer_interval"],
    )

    # Load weights
    weights_path = os.path.join(output_dir, "flow_weights.eqx")
    flow = eqx.tree_deserialise_leaves(weights_path, flow)

    # Load metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return flow, metadata
