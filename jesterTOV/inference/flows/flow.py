r"""
Wrapper for trained normalizing flows with automatic data preprocessing.

This module provides a high-level interface for loading and using pre-trained
normalizing flow models for gravitational wave inference. The Flow class handles
the complexities of data standardization and model loading, allowing users to
sample from or evaluate trained flows with a simple API.

Normalizing flows trained on gravitational wave posterior samples can be used
for importance sampling in EOS inference, providing efficient proposals that
capture the correlations between binary component masses and tidal deformabilities.

The same :class:`Flow` class also supports *conditional* flows, i.e. models of
p(target | condition) such as p(λ1, λ2 | m1, m2). This is enabled simply by
training with ``cond_dim`` set (see :class:`~jesterTOV.inference.flows.config.FlowTrainingConfig`);
the resulting :class:`Flow` then requires a ``condition`` argument to
``sample``/``log_prob``, and standardizes it the same way it standardizes the
target variable, using statistics saved alongside the model.

Key Features
------------
- Automatic min-max/z-score standardization and inverse transformation
- Optional conditioning variable, standardized the same way as the target
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

For a conditional flow trained on p(λ1, λ2 | m1, m2), pass ``condition``:

>>> cflow = Flow.from_directory("./models/gw170817_conditional/")
>>> condition = jnp.array([1.4, 1.3])  # (m1, m2)
>>> samples = cflow.sample(jax.random.key(0), (1000,), condition=condition)
>>> print(samples.shape)  # (1000, 2) for (λ1, λ2)
>>> log_prob = cflow.log_prob(jnp.array([[100.0, 200.0]]), condition=condition)
"""

import json
import os
from typing import Any, Dict, Literal, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.flows import (
    block_neural_autoregressive_flow,
    coupling_flow,
    masked_autoregressive_flow,
)
from flowjax.bijections import (
    RationalQuadraticSpline,
    Affine,
)

try:
    # jax <= 0.9: context manager for locally overriding jax_enable_x64.
    from jax.experimental import disable_x64
except ImportError:
    # jax >= 0.11: disable_x64 was removed from jax.experimental; the
    # replacement is calling jax.enable_x64 itself as a context manager
    from contextlib import contextmanager

    @contextmanager
    def disable_x64():
        previous = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", False)
        try:
            yield
        finally:
            jax.config.update("jax_enable_x64", previous)


# Flow architectures for which the float32 evaluation recipe has actually been
# validated. Requesting dtype="float32" for any other combination raises a clear
# error rather than silently attempting an unverified architecture
_FLOAT32_SUPPORTED_FLOW_TYPES = {"masked_autoregressive_flow"}
_FLOAT32_SUPPORTED_TRANSFORMER_TYPES = {"rational_quadratic_spline"}


def _validate_float32_architecture(flow_kwargs: Dict[str, Any]) -> None:
    """Raise a clear error if dtype="float32" is requested for an
    architecture the recipe hasn't been validated against."""
    flow_type = flow_kwargs.get("flow_type")
    transformer_type = flow_kwargs.get("transformer_type")
    if flow_type not in _FLOAT32_SUPPORTED_FLOW_TYPES:
        raise ValueError(
            f"dtype='float32' was requested, but flow_type={flow_type!r} has not "
            f"been validated for float32 evaluation. Supported: "
            f"{sorted(_FLOAT32_SUPPORTED_FLOW_TYPES)}. See "
            "dev/float32_investigations/FINDINGS.md for what has and hasn't been tested."
        )
    if transformer_type not in _FLOAT32_SUPPORTED_TRANSFORMER_TYPES:
        raise ValueError(
            f"dtype='float32' was requested, but transformer_type={transformer_type!r} "
            f"has not been validated for float32 evaluation. Supported: "
            f"{sorted(_FLOAT32_SUPPORTED_TRANSFORMER_TYPES)}. See "
            "dev/float32_investigations/FINDINGS.md for what has and hasn't been tested."
        )


class Flow:
    """
    Wrapper class for flowjax normalizing flows with automatic standardization handling.

    This class encapsulates a trained normalizing flow and handles data standardization
    transparently. When sampling, it automatically converts samples back to the original
    scale if standardization was used during training.

    If the wrapped flow is conditional (i.e. it was built with ``cond_dim`` set, see
    :func:`create_flow`), a ``condition`` array must be passed to :meth:`sample` and
    :meth:`log_prob`. The condition is standardized the same way as the target
    variable, using statistics computed at training time and saved to
    ``metadata.json`` under the ``condition_*`` keys (mirroring the ``data_*`` keys
    used for the target). Standardizing the condition is purely an input
    transformation -- it does not require a Jacobian correction, unlike standardizing
    the target variable ``x`` (which is being modelled as a random variable).

    Attributes:
        flow: The underlying flowjax flow model
        metadata: Training metadata dictionary
        flow_kwargs: Flow architecture kwargs
        standardize: Whether standardization was used during training
        data_bounds: Min/max bounds for each feature (if standardization was used)
        cond_shape: Shape of the conditioning variable, or None for unconditional flows

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

        >>> # Conditional flow, e.g. p(lambda_1, lambda_2 | mass_1_source, mass_2_source)
        >>> cflow = Flow.from_directory("./models/gw170817_conditional/")
        >>> condition = jnp.array([1.4, 1.3])
        >>> samples = cflow.sample(jax.random.key(0), (1000,), condition=condition)
    """

    def __init__(
        self,
        flow: AbstractDistribution,
        metadata: Dict[str, Any],
        flow_kwargs: Dict[str, Any],
        dtype: Literal["float32", "float64"] = "float64",
    ):
        """
        Initialize Flow wrapper.

        Args:
            flow: Trained flowjax flow model
            metadata: Training metadata
            flow_kwargs: Flow architecture kwargs
            dtype: Precision the (de)standardization arrays are stored at.
                "float64" (default) preserves existing behaviour exactly. Only
                set "float32" for a `flow` that was itself already built
                float32-native (see `load_model`'s dtype argument) -- this
                does not, on its own, cast an existing float64 flow.
        """
        self.flow = flow
        self.metadata = metadata
        self.flow_kwargs = flow_kwargs
        self.dtype = dtype
        self.standardize = metadata.get("standardize", False)
        _dtype = jnp.float32 if dtype == "float32" else jnp.float64

        # Detect standardization method from metadata
        has_mean_std = "data_mean" in metadata and "data_std" in metadata
        has_bounds = "data_bounds_min" in metadata and "data_bounds_max" in metadata

        if self.standardize:
            if has_mean_std:
                # Z-score standardization (new default)
                self.standardization_method = "zscore"
                self.data_mean = jnp.array(metadata["data_mean"], dtype=_dtype)
                self.data_std = jnp.array(metadata["data_std"], dtype=_dtype)
                # Avoid division by zero
                self.data_std = jnp.where(self.data_std == 0, 1.0, self.data_std)
            elif has_bounds:
                # Min-max standardization (legacy)
                self.standardization_method = "minmax"
                self.data_min = jnp.array(metadata["data_bounds_min"], dtype=_dtype)
                self.data_max = jnp.array(metadata["data_bounds_max"], dtype=_dtype)
                self.data_range = self.data_max - self.data_min
                # Avoid division by zero
                self.data_range = jnp.where(self.data_range == 0, 1.0, self.data_range)
            else:
                raise ValueError(
                    "Standardization enabled but metadata missing both "
                    "(data_mean, data_std) and (data_bounds_min, data_bounds_max)"
                )
        else:
            # No standardization - create identity transform
            # Infer dimensionality from flow
            n_features = self.flow.shape[0]
            self.standardization_method = "none"
            # For identity: use minmax with min=0, range=1
            self.data_min = jnp.zeros(n_features, dtype=_dtype)
            self.data_max = jnp.ones(n_features, dtype=_dtype)
            self.data_range = jnp.ones(n_features, dtype=_dtype)

        # Conditional flow support: the conditioning variable is standardized the
        # same way as the target, using its own statistics (it generally lives on a
        # different scale, e.g. masses vs. tidal deformabilities). Unlike the target,
        # standardizing the condition is just an input transform -- no Jacobian
        # correction is needed since we are not modelling a density over it.
        self.cond_shape = self.flow.cond_shape
        self.condition_names = metadata.get("condition_names")

        if self.cond_shape is not None:
            n_cond = self.cond_shape[0]
            has_cond_mean_std = (
                "condition_mean" in metadata and "condition_std" in metadata
            )
            has_cond_bounds = (
                "condition_bounds_min" in metadata
                and "condition_bounds_max" in metadata
            )

            if self.standardize:
                if has_cond_mean_std:
                    self.condition_standardization_method = "zscore"
                    self.condition_mean = jnp.array(
                        metadata["condition_mean"], dtype=_dtype
                    )
                    self.condition_std = jnp.array(
                        metadata["condition_std"], dtype=_dtype
                    )
                    self.condition_std = jnp.where(
                        self.condition_std == 0, 1.0, self.condition_std
                    )
                elif has_cond_bounds:
                    self.condition_standardization_method = "minmax"
                    self.condition_min = jnp.array(
                        metadata["condition_bounds_min"], dtype=_dtype
                    )
                    self.condition_max = jnp.array(
                        metadata["condition_bounds_max"], dtype=_dtype
                    )
                    self.condition_range = self.condition_max - self.condition_min
                    self.condition_range = jnp.where(
                        self.condition_range == 0, 1.0, self.condition_range
                    )
                else:
                    raise ValueError(
                        "This flow is conditional (cond_shape="
                        f"{self.cond_shape}) and standardize=True, but metadata "
                        "is missing both (condition_mean, condition_std) and "
                        "(condition_bounds_min, condition_bounds_max)."
                    )
            else:
                self.condition_standardization_method = "none"
                self.condition_min = jnp.zeros(n_cond, dtype=_dtype)
                self.condition_max = jnp.ones(n_cond, dtype=_dtype)
                self.condition_range = jnp.ones(n_cond, dtype=_dtype)
        else:
            self.condition_standardization_method = None

    @classmethod
    def from_directory(
        cls, output_dir: str, dtype: Literal["float32", "float64"] = "float64"
    ) -> "Flow":
        """
        Load a trained flow from a directory.

        Args:
            output_dir: Directory containing flow_weights.eqx, flow_kwargs.json, metadata.json
            dtype: "float64" (default) loads the flow exactly as before. "float32"
                builds the flow architecture natively under
                `jax.experimental.disable_x64()` and deserializes the saved
                (float64-trained) weights into that template.
                Training is unaffected either way: the same saved weights file
                works for both dtypes.

        Returns:
            Flow instance with loaded model and metadata

        Example:
            >>> flow = Flow.from_directory("./models/gw170817/")
            >>> flow32 = Flow.from_directory("./models/gw170817/", dtype="float32")
        """
        # Load the flow model and metadata
        flow_model, metadata = load_model(output_dir, dtype=dtype)

        # Load kwargs
        kwargs_path = os.path.join(output_dir, "flow_kwargs.json")
        with open(kwargs_path, "r") as f:
            flow_kwargs = json.load(f)

        return cls(flow_model, metadata, flow_kwargs, dtype=dtype)

    def _prepare_condition(self, condition: Array | None) -> Array | None:
        """Validate and standardize a `condition` argument, or check it is absent.

        Raises a clear error if a conditional flow is called without a `condition`,
        or an unconditional flow is called with one, rather than letting flowjax
        raise a more cryptic shape error downstream.
        """
        if self.cond_shape is not None:
            if condition is None:
                raise ValueError(
                    "This flow is conditional (cond_shape="
                    f"{self.cond_shape}), but no `condition` was provided."
                )
            return self.standardize_condition(condition)
        if condition is not None:
            raise ValueError(
                "This flow is unconditional (cond_shape=None), but a `condition` "
                "was provided."
            )
        return None

    def standardize_condition(self, condition: Array) -> Array:
        """
        Standardize a conditioning variable using the method from training.

        Mirrors :meth:`standardize_input`, but for the conditioning variable of a
        conditional flow, using the ``condition_*`` statistics saved alongside the
        model. This is a plain input transformation -- no Jacobian correction is
        needed (unlike for the target variable), since the condition is not a random
        variable being modelled.

        Args:
            condition: Conditioning variable in original scale.

        Returns:
            Standardized conditioning variable.
        """
        if self.condition_standardization_method == "zscore":
            return (condition - self.condition_mean) / self.condition_std
        else:
            # Min-max or none: (x - min) / range
            return (condition - self.condition_min) / self.condition_range

    def sample(
        self, key: Array, shape: Tuple[int, ...], condition: Array | None = None
    ) -> Array:
        """
        Sample from the flow and return in original scale.

        If standardization was used during training, samples are automatically
        converted back to the original scale using the inverse transformation
        (z-score or min-max). If not, the transformation is identity (no-op).

        Args:
            key: JAX random key (jax.Array)
            shape: Shape of samples to generate (e.g., (1000,) for 1000 samples)
            condition: Conditioning variable, required if this is a conditional
                flow (``self.cond_shape is not None``), and disallowed otherwise.
                Standardized automatically, like the target variable. May include
                leading batch dimensions, in which case they broadcast against
                ``shape`` (see :meth:`flowjax.distributions.AbstractDistribution.sample`).

        Returns:
            Samples in original scale as JAX array of shape (``*shape``, n_features)

        Example:
            >>> samples = flow.sample(jax.random.key(0), (1000,))
            >>> print(samples.shape)  # (1000, 4) for 4D flow

            >>> # Conditional flow
            >>> samples = cflow.sample(jax.random.key(0), (1000,), condition=jnp.array([1.4, 1.3]))
        """
        condition_std = self._prepare_condition(condition)

        # Sample in standardized space
        samples = self.flow.sample(key, shape, condition=condition_std)

        # Inverse transformation to original scale (method-dependent)
        samples = self.destandardize_output(samples)

        return samples

    def standardize_input(self, data: Array) -> Array:
        """
        Standardize input data using the method from training.

        Applies the same standardization method used during training:
        - Z-score: (x - mean) / std → mean=0, std=1
        - Min-max: (x - min) / (max - min) → [0, 1]
        - None: identity (no-op)

        Args:
            data: Input data in original scale (JAX array)

        Returns:
            Standardized data (z-score, [0,1], or unchanged)

        Example:
            >>> original_data = jnp.array([[1.4, 1.3, 100, 200]])
            >>> standardized = flow.standardize_input(original_data)
        """
        if self.standardization_method == "zscore":
            # Z-score: (x - mean) / std
            return (data - self.data_mean) / self.data_std
        else:
            # Min-max or none: (x - min) / range
            # If standardization disabled, this is identity (min=0, range=1)
            return (data - self.data_min) / self.data_range

    def destandardize_output(self, data: Array) -> Array:
        """
        Convert standardized data back to original scale.

        Applies the inverse of the standardization method:
        - Z-score: x * std + mean
        - Min-max: x * (max - min) + min
        - None: identity (no-op)

        Args:
            data: Data in standardized space (z-score or [0, 1])

        Returns:
            Data in original scale (or unchanged if standardization not used)

        Example:
            >>> standardized_data = jnp.array([[0.5, 0.5, 0.5, 0.5]])
            >>> original = flow.destandardize_output(standardized_data)
        """
        if self.standardization_method == "zscore":
            # Inverse z-score: x * std + mean
            return data * self.data_std + self.data_mean
        else:
            # Inverse min-max or identity: x * range + min
            # If standardization disabled, this is identity (min=0, range=1)
            return data * self.data_range + self.data_min

    def log_prob(self, x: Array, condition: Array | None = None) -> Array:
        """
        Evaluate log probability of data under the flow.

        If standardization was used, input data is automatically standardized
        before evaluation and Jacobian correction is applied. If not, operations
        are identity (no-op).

        The Jacobian correction accounts for the change of variables:
        - Z-score: log p(x) = log p(x_std) - sum(log(std))
        - Min-max: log p(x) = log p(x_std) - sum(log(max - min))
        - None: log p(x) = log p(x_std) (no correction)

        No Jacobian correction is applied for standardizing `condition` -- it is an
        input transformation, not a change of variables of the modelled density.

        Args:
            x: Data in original scale, shape (n_samples, n_features).
               JAX array.
            condition: Conditioning variable, required if this is a conditional
                flow (``self.cond_shape is not None``), and disallowed otherwise.
                Standardized automatically, like ``x``.

        Returns:
            Log probabilities as JAX array, shape (n_samples,)

        Example:
            >>> data = jnp.array([[1.4, 1.3, 100, 200]])
            >>> log_prob = flow.log_prob(data)

            >>> # Conditional flow, e.g. p(lambda_1, lambda_2 | m1, m2)
            >>> log_prob = cflow.log_prob(jnp.array([[100.0, 200.0]]), condition=jnp.array([1.4, 1.3]))
        """
        # Standardize input (method-dependent or identity)
        x_std = self.standardize_input(x)
        condition_std = self._prepare_condition(condition)

        # Evaluate log probability in standardized space
        log_p = self.flow.log_prob(x_std, condition_std)

        # Account for Jacobian of inverse transformation
        if self.standardization_method == "zscore":
            # Z-score: log |det J| = sum(log(std))
            log_det_jacobian = -jnp.sum(jnp.log(self.data_std))
        else:
            # Min-max or none: log |det J| = sum(log(range))
            # If standardization disabled (range=1), log_det_jacobian = 0
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
    dim: int = 4,
    flow_type: str = "masked_autoregressive_flow",
    nn_depth: int = 5,
    nn_block_dim: int = 8,
    nn_width: int = 50,
    flow_layers: int = 1,
    invert: bool = True,
    cond_dim: int | None = None,
    transformer_type: str = "affine",
    transformer_knots: int = 8,
    transformer_interval: float = 4.0,
) -> Any:
    """
    Create a normalizing flow of the specified type with flexible dimensionality.

    Args:
        key: JAX random key
        dim: Dimensionality of the data (default: 4 for GW [m1, m2, λ1, λ2],
            can be 2 for NICER [M, R], etc.)
        flow_type: Type of flow ("block_neural_autoregressive_flow",
            "masked_autoregressive_flow", "coupling_flow")
        nn_depth: Depth of neural network (for block_neural_autoregressive_flow,
            masked_autoregressive_flow, coupling_flow)
        nn_block_dim: Block dimension (for block_neural_autoregressive_flow)
        nn_width: Width of hidden layers (for masked_autoregressive_flow, coupling_flow)
        flow_layers: Number of flow layers
        invert: Whether to invert the flow
        cond_dim: Conditional dimension (None for unconditional flows)
        transformer_type: Type of transformer for masked_autoregressive_flow and coupling_flow
            ("affine", "rational_quadratic_spline")
        transformer_knots: Number of knots for RationalQuadraticSpline
        transformer_interval: Interval for RationalQuadraticSpline

    Returns:
        Untrained flowjax flow model
    """
    base_dist = Normal(jnp.zeros(dim))

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
    else:
        raise ValueError(
            f"Unknown flow type: {flow_type}. Must be one of: "
            "block_neural_autoregressive_flow, masked_autoregressive_flow, "
            "coupling_flow"
        )

    return flow


def load_model(
    output_dir: str, dtype: Literal["float32", "float64"] = "float64"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a trained flow model from saved files.

    Args:
        output_dir: Directory containing saved model files
        dtype: "float64" (default) builds the architecture and deserializes
            weights at the ambient precision, exactly as before. "float32"
            builds the untrained architecture inside
            `jax.experimental.disable_x64()`.

    Returns:
        flow: Loaded flow model
        metadata: Training metadata (includes data statistics if standardization was used)

    Example:
        >>> flow, metadata = load_model("./models/gw170817/")
        >>> flow32, metadata = load_model("./models/gw170817/", dtype="float32")
    """
    # Load metadata first to infer dimensionality
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load kwargs
    kwargs_path = os.path.join(output_dir, "flow_kwargs.json")
    with open(kwargs_path, "r") as f:
        flow_kwargs = json.load(f)

    if dtype == "float32":
        _validate_float32_architecture(flow_kwargs)

    # Infer dimensionality from metadata. Prefer parameter_names -- it is always
    # present (written by load_posterior regardless of standardize) and its length
    # is the target dimensionality even when standardize=False, unlike data_mean /
    # data_bounds_min which are only written when standardization was used (and
    # previously left dim-inference falling back to a hardcoded 4 in that case,
    # silently wrong for any non-4D unconditional/conditional flow).
    if "parameter_names" in metadata:
        dim = len(metadata["parameter_names"])
    elif "data_mean" in metadata:
        dim = len(metadata["data_mean"])
    elif "data_bounds_min" in metadata:
        dim = len(metadata["data_bounds_min"])
    else:
        # Default to 4 for backward compatibility with old models without
        # standardization or parameter_names metadata.
        dim = 4

    def _build_and_deserialize():
        key = jax.random.key(flow_kwargs["seed"])
        flow = create_flow(
            key=key,
            dim=dim,
            flow_type=flow_kwargs["flow_type"],
            nn_depth=flow_kwargs["nn_depth"],
            nn_block_dim=flow_kwargs["nn_block_dim"],
            nn_width=flow_kwargs["nn_width"],
            flow_layers=flow_kwargs["flow_layers"],
            invert=flow_kwargs["invert"],
            cond_dim=flow_kwargs["cond_dim"],
            transformer_type=flow_kwargs.get("transformer_type", "affine"),
            transformer_knots=flow_kwargs.get("transformer_knots", 8),
            transformer_interval=flow_kwargs.get("transformer_interval", 4.0),
        )
        weights_path = os.path.join(output_dir, "flow_weights.eqx")
        return eqx.tree_deserialise_leaves(weights_path, flow)

    if dtype == "float32":
        with disable_x64():
            flow = _build_and_deserialize()
    else:
        flow = _build_and_deserialize()

    return flow, metadata
