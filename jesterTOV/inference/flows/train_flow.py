"""Training script for normalizing flows on gravitational wave posterior samples.

Trains normalizing flow models to approximate GW posteriors in (m1, m2, λ1, λ2) space.
The trained flows serve as efficient proposal distributions for EOS inference.

Training Pipeline
-----------------
1. Load configuration from YAML file
2. Load posterior samples from npz file
3. Apply optional physics constraints and standardization
4. Create flow architecture (triangular spline, autoregressive, coupling)
5. Fit flow using maximum likelihood with early stopping
6. Save trained weights, config, and metadata
7. Generate validation plots

Supported Architectures
-----------------------
- coupling_flow: Balanced speed and expressiveness
- triangular_spline_flow: Fast, flexible, exact sampling
- block_neural_autoregressive_flow: Good expressiveness
- masked_autoregressive_flow: Flexible but slower

Configuration-Driven Usage
---------------------------
Create a YAML config file (e.g., config.yaml):

    posterior_file: data/gw170817_posterior.npz
    output_dir: models/gw170817/
    flow_type: triangular_spline_flow
    num_epochs: 1000
    learning_rate: 1.0e-3
    standardize: true
    plot_corner: true
    plot_losses: true

Then run:

    uv run python -m jesterTOV.inference.flows.train_flow config.yaml

Or use the bash scripts for batch training:

    bash train_all_flows.sh

Programmatic Usage
------------------
For custom training workflows, use the provided functions:

>>> from jesterTOV.inference.flows.train_flow import train_flow_from_config
>>> from jesterTOV.inference.flows.config import FlowTrainingConfig
>>> config = FlowTrainingConfig.from_yaml("config.yaml")
>>> train_flow_from_config(config)

Or use the lower-level functions directly:

>>> from jesterTOV.inference.flows.train_flow import load_gw_posterior, create_flow, train_flow, save_model
>>> data, metadata = load_gw_posterior("gw170817.npz", max_samples=50000)
>>> flow = create_flow(jax.random.key(0), flow_type="triangular_spline_flow")
>>> trained_flow, losses = train_flow(flow, data, jax.random.key(1))
>>> save_model(trained_flow, "models/gw170817/", flow_kwargs, metadata)

Output Files
------------
The training script saves:
- flow_weights.eqx: Trained model parameters (Equinox serialization)
- flow_kwargs.json: Architecture configuration for reproducibility
- metadata.json: Training metadata (epochs, losses, data bounds, etc.)
- figures/losses.png: Training and validation loss curves
- figures/corner.png: Corner plot comparing data and flow samples
- figures/transformed_training_data.png: Visualization of transformed data
  (if physics constraints are enabled)

See Also
--------
jesterTOV.inference.flows.flow.Flow : High-level interface for loading trained flows
jesterTOV.inference.flows.config.FlowTrainingConfig : Configuration schema

Notes
-----
Training requires:
- JAX with GPU support recommended for large datasets
- flowjax for flow architectures
- equinox for model serialization
- PyYAML for configuration loading
- Optional: matplotlib and corner for plotting

Physics constraint modes (constrain_physics, use_chirp_mass) are experimental.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from flowjax.distributions import Normal, Transformed
from flowjax.flows import (
    block_neural_autoregressive_flow,
    coupling_flow,
    masked_autoregressive_flow,
    triangular_spline_flow,
)
from flowjax.bijections import (
    RationalQuadraticSpline,
    Affine,
    AbstractBijection,
    Stack,
    Chain,
    Exp,
    Sigmoid,
    Invert,
)
from flowjax.train import fit_to_data

from .config import FlowTrainingConfig

# # TODO: decide later on if this is necessary, but this might be much faster
# # # Enable 64-bit precision for numerical accuracy
# jax.config.update("jax_enable_x64", True)

# TODO: this is for now: we assume the NF is only for vanilla GW inference which has these 4 keys, later on, we could generalize this...
# Required keys in the posterior file
REQUIRED_KEYS = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]


class MassesAndLambdasToChirpMassRatio(AbstractBijection):
    r"""
    Bijection transforming GW parameters: (m1, m2, λ1, λ2) to (M_chirp, q, λ1, λ2).

    This bijection enforces the physical constraint m1 >= m2 by construction
    through the mass ratio parameterization, while leaving tidal parameters unchanged.

    Forward transformation:
        (m1, m2, λ1, λ2) -> (M_chirp, q, λ1, λ2)
        where:
            M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
            q = m2 / m1  (mass ratio, always <= 1 when m1 >= m2)
            λ1, λ2 unchanged

    Inverse transformation:
        (M_chirp, q, λ1, λ2) -> (m1, m2, λ1, λ2)
        where:
            M_total = M_chirp * (1 + q)^(1/5) / q^(3/5)
            m1 = M_total / (1 + q)
            m2 = M_total * q / (1 + q)
            λ1, λ2 unchanged

    The Jacobian determinant is computed using JAX automatic differentiation
    for numerical stability and correctness.

    Note: This bijection operates on 4D arrays (last dimension = 4).
    """

    shape: tuple = (4,)
    cond_shape: None = None

    def transform(self, x, condition=None):
        """Transform (m1, m2, λ1, λ2) to (M_chirp, q, λ1, λ2)."""
        m1, m2, lam1, lam2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        M_chirp = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
        q = m2 / m1
        return jnp.stack([M_chirp, q, lam1, lam2], axis=-1)

    def inverse(self, y, condition=None):
        """Transform (M_chirp, q, λ1, λ2) to (m1, m2, λ1, λ2)."""
        M_chirp, q, lam1, lam2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        prefac = (1 + q) ** (6 / 5) / q ** (3 / 5)
        M_total = M_chirp * prefac
        m1 = M_total / (1 + q)
        m2 = q * m1
        return jnp.stack([m1, m2, lam1, lam2], axis=-1)

    def transform_and_log_det(self, x, condition=None):
        """Transform (m1, m2, λ1, λ2) to (M_chirp, q, λ1, λ2) with log|det(Jacobian)|."""
        # Compute transformation
        y = self.transform(x, condition)

        # Compute Jacobian determinant using autodiff
        # Jacobian is 4x4 but block diagonal (masses don't affect lambdas)
        def transform_fn(x_single):
            return self.transform(x_single, condition)

        # Handle batched input
        if x.ndim == 1:
            jac = jax.jacobian(transform_fn)(x)
            log_det = jnp.log(jnp.abs(jnp.linalg.det(jac)))
        else:
            # Vectorize over batch dimension
            def single_log_det(x_single):
                jac = jax.jacobian(transform_fn)(x_single)
                return jnp.log(jnp.abs(jnp.linalg.det(jac)))

            log_det = jax.vmap(single_log_det)(x)

        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        """Transform (M_chirp, q, λ1, λ2) to (m1, m2, λ1, λ2) with log|det(Jacobian^-1)|."""
        # Compute inverse transformation
        x = self.inverse(y, condition)

        # For inverse Jacobian: log|det(J^-1)| = -log|det(J)|
        # Compute forward Jacobian at the resulting x
        _, log_det_forward = self.transform_and_log_det(x, condition)
        log_det_inverse = -log_det_forward

        return x, log_det_inverse



def load_gw_posterior(
    filepath: str, max_samples: int = 20_000
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load GW posterior samples from npz file.

    Args:
        filepath: Path to .npz file
        max_samples: Maximum number of samples to use (downsampling if needed)

    Returns:
        data: Array of shape (n_samples, 4) with columns [m1, m2, lambda1, lambda2]
        metadata: Dictionary with loading information

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required keys are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Posterior file not found: {filepath}")

    # Load data
    posterior = np.load(filepath)

    # Validate required keys
    missing_keys = [key for key in REQUIRED_KEYS if key not in posterior]
    if missing_keys:
        available_keys = list(posterior.keys())
        raise KeyError(
            f"Missing required keys: {missing_keys}\n"
            f"Available keys: {available_keys}\n"
            f"Required keys: {REQUIRED_KEYS}"
        )

    # Extract samples
    m1 = posterior["mass_1_source"].flatten()
    m2 = posterior["mass_2_source"].flatten()
    lambda1 = posterior["lambda_1"].flatten()
    lambda2 = posterior["lambda_2"].flatten()

    # Combine into array
    data = np.column_stack([m1, m2, lambda1, lambda2])
    n_samples_total = data.shape[0]

    # Downsample if needed
    if n_samples_total > max_samples:
        downsample_factor = int(np.ceil(n_samples_total / max_samples))
        data = data[::downsample_factor]
        print(
            f"Downsampled from {n_samples_total} to {data.shape[0]} samples "
            f"(factor: {downsample_factor})"
        )
    else:
        print(f"Using all {n_samples_total} samples")

    metadata = {
        "n_samples_total": n_samples_total,
        "n_samples_used": data.shape[0],
        "filepath": filepath,
    }

    return data, metadata


def standardize_data(
    data: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Standardize data to [0, 1] domain using min-max scaling.

    Args:
        data: Array of shape (n_samples, n_features)

    Returns:
        standardized_data: Data scaled to [0, 1]
        bounds: Dictionary with 'min' and 'max' arrays for each feature
    """
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)

    # Avoid division by zero (if a feature is constant)
    data_range = data_max - data_min
    data_range = np.where(data_range == 0, 1.0, data_range)

    standardized_data = (data - data_min) / data_range

    bounds = {"min": data_min, "max": data_max}

    return standardized_data, bounds


def inverse_standardize_data(
    standardized_data: np.ndarray, bounds: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Inverse transform standardized data back to original scale.

    Args:
        standardized_data: Data in [0, 1] domain
        bounds: Dictionary with 'min' and 'max' arrays for each feature

    Returns:
        data: Data in original scale
    """
    data_min = bounds["min"]
    data_max = bounds["max"]
    data_range = data_max - data_min
    data_range = np.where(data_range == 0, 1.0, data_range)

    data = standardized_data * data_range + data_min

    return data


def clip_data_for_bijection(data: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Clip data away from boundaries for numerical stability with bijections.

    Args:
        data: Array of shape (n_samples, 4) with columns [m1, m2, λ1, λ2]
        epsilon: Small value to avoid exact zeros and ones

    Returns:
        Clipped data ensuring:
            - All values > epsilon (for log transforms)
            - mass_ratio < 1 - epsilon (for logit transform)
    """
    data_clipped = data.copy()

    # Ensure all positive values are bounded away from zero
    data_clipped[:, 0] = np.maximum(data_clipped[:, 0], epsilon)  # m1
    data_clipped[:, 2] = np.maximum(data_clipped[:, 2], epsilon)  # λ1
    data_clipped[:, 3] = np.maximum(data_clipped[:, 3], epsilon)  # λ2

    # Ensure m1 >= m2 (should already be true, but enforce)
    # If m2 > m1, swap them
    mask = data_clipped[:, 1] > data_clipped[:, 0]
    if np.any(mask):
        print(
            f"Warning: {np.sum(mask)} samples had m2 > m1. Swapping to enforce m1 >= m2."
        )
        data_clipped[mask, [0, 1]] = data_clipped[mask, [1, 0]]
        data_clipped[mask, [2, 3]] = data_clipped[mask, [3, 2]]  # Swap lambdas too

    # Ensure m2 > epsilon and m2/m1 < 1 - epsilon
    data_clipped[:, 1] = np.maximum(data_clipped[:, 1], epsilon)  # m2
    # Ensure mass ratio q = m2/m1 < 1 - epsilon
    q = data_clipped[:, 1] / data_clipped[:, 0]
    q_clipped = np.minimum(q, 1 - epsilon)
    data_clipped[:, 1] = q_clipped * data_clipped[:, 0]

    return data_clipped


def create_physics_constraint_bijection(use_chirp_mass: bool = False):
    """
    Create bijection that enforces physical constraints on GW parameters.

    Args:
        use_chirp_mass: If True, reparameterize to chirp mass + mass ratio.
                       If False (default), only enforce positivity via log transforms.

    Constraints enforced:
        - m1, m2, lambda_1, lambda_2 > 0 (via log transform)
        - If use_chirp_mass=True: also enforce m1 >= m2 via mass ratio

    Returns:
        Bijection mapping physical space to unbounded R^4.
    """
    if use_chirp_mass:
        # Full reparameterization with chirp mass
        # Physical: (m1, m2, λ1, λ2) -> (M_chirp, q, λ1, λ2) -> unbounded
        mass_transform = MassesAndLambdasToChirpMassRatio()
        unbounded_transform = Stack(
            [
                Invert(Exp()),  # M_chirp in (0, ∞) -> R
                Invert(Sigmoid()),  # q in (0, 1) -> R
                Invert(Exp()),  # λ1 in (0, ∞) -> R
                Invert(Exp()),  # λ2 in (0, ∞) -> R
            ]
        )
        return Chain([mass_transform, unbounded_transform])
    else:
        # Simple mode: just enforce positivity via log transforms
        # Physical: (m1, m2, λ1, λ2) -> (log m1, log m2, log λ1, log λ2)
        # Note: This doesn't enforce m1 >= m2, but data already has this property
        return Stack(
            [
                Invert(Exp()),  # m1 in (0, ∞) -> R
                Invert(Exp()),  # m2 in (0, ∞) -> R
                Invert(Exp()),  # λ1 in (0, ∞) -> R
                Invert(Exp()),  # λ2 in (0, ∞) -> R
            ]
        )


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
        return RationalQuadraticSpline(knots=transformer_knots, interval=transformer_interval)
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


def train_flow(
    flow: Any,
    data: np.ndarray,
    key: Array,
    learning_rate: float = 1e-3,
    max_epochs: int = 600,
    max_patience: int = 50,
    val_prop: float = 0.2,
    batch_size: int = 128,
) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Train the normalizing flow on data.

    Args:
        flow: Untrained flowjax flow
        data: Training data of shape (n_samples, n_dims)
        key: JAX random key
        learning_rate: Learning rate for optimizer
        max_epochs: Maximum number of epochs
        max_patience: Early stopping patience
        val_prop: Proportion of data to use for validation
        batch_size: Batch size for training

    Returns:
        trained_flow: Trained flow model
        losses: Dictionary with 'train' and 'val' loss arrays
    """
    print(f"Training flow for up to {max_epochs} epochs...")
    print(f"Using {val_prop:.1%} of data for validation")
    print(f"Batch size: {batch_size}")
    trained_flow, losses = fit_to_data(
        key=key,
        dist=flow,
        data=data,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        val_prop=val_prop,
        batch_size=batch_size,
    )
    print(f"Training completed after {len(losses['train'])} epochs")
    return trained_flow, losses


def save_model(
    flow: Any,
    output_dir: str,
    flow_kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """
    Save trained flow model, architecture kwargs, and metadata.

    Args:
        flow: Trained flowjax flow
        output_dir: Directory to save files
        flow_kwargs: Dictionary of kwargs needed to recreate flow architecture
        metadata: Dictionary with training metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    weights_path = os.path.join(output_dir, "flow_weights.eqx")
    print(f"Saving model weights to {weights_path}")
    eqx.tree_serialise_leaves(weights_path, flow)

    # Save architecture kwargs
    kwargs_path = os.path.join(output_dir, "flow_kwargs.json")
    print(f"Saving flow kwargs to {kwargs_path}")
    with open(kwargs_path, "w") as f:
        json.dump(flow_kwargs, f, indent=2)

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def plot_losses(losses: Dict[str, np.ndarray], output_path: str) -> None:
    """Plot training and validation losses."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping loss plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(losses["train"], label="Train", color="red", alpha=0.7)
    plt.plot(losses["val"], label="Validation", color="blue", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to {output_path}")


def plot_transformed_data(
    transformed_data: np.ndarray, output_path: str, labels: list[str] | None = None
) -> None:
    """
    Plot the transformed (unbounded) data that the normalizing flow trains on.

    This visualization shows what the flow "sees" after applying bijections.
    For physics-constrained flows, this shows the data in unbounded R^4 space.

    Args:
        transformed_data: Array of shape (n_samples, 4) in unbounded space
        output_path: Path to save the corner plot
        labels: Optional custom labels for the 4 dimensions
    """
    try:
        import corner
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: corner package not available, skipping transformed data plot")
        return

    if labels is None:
        # Default labels (simple log transform mode)
        labels = [
            r"$\log(m_1)$",
            r"$\log(m_2)$",
            r"$\log(\Lambda_1)$",
            r"$\log(\Lambda_2)$",
        ]

    fig = corner.corner(
        transformed_data,
        labels=labels,
        color="purple",
        bins=40,
        smooth=1.0,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=False,
        levels=[0.68, 0.95],
        alpha=0.7,
        hist_kwargs={"color": "purple", "density": True},
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved transformed data plot to {output_path}")


def plot_corner(data: np.ndarray, flow_samples: np.ndarray, output_path: str) -> None:
    """Create corner plot comparing data and flow samples."""
    try:
        import corner
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: corner package not available, skipping corner plot")
        return

    labels = [
        r"$m_1$ [$M_\odot$]",
        r"$m_2$ [$M_\odot$]",
        r"$\Lambda_1$",
        r"$\Lambda_2$",
    ]

    hist_kwargs = {"color": "blue", "density": True}

    fig = corner.corner(
        data,
        labels=labels,
        color="blue",
        bins=40,
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        levels=[0.68, 0.95],
        alpha=0.6,
        hist_kwargs=hist_kwargs,
    )

    hist_kwargs["color"] = "red"

    corner.corner(
        flow_samples,
        fig=fig,
        color="red",
        bins=40,
        smooth=1.0,
        plot_datapoints=True, # DO plot them for the flow, to check if it violates bounds
        plot_density=False,
        fill_contours=False,
        levels=[0.68, 0.95],
        alpha=0.6,
        hist_kwargs=hist_kwargs,
    )

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="Data"),
        Line2D([0], [0], color="red", lw=2, label="Flow"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=12)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved corner plot to {output_path}")


def train_flow_from_config(config: FlowTrainingConfig) -> None:
    """
    Train a normalizing flow using a configuration object.

    Args:
        config: FlowTrainingConfig with all training parameters
    """
    # Print configuration
    print("=" * 60)
    print("GW Normalizing Flow Training")
    print("=" * 60)
    print(f"Posterior file: {config.posterior_file}")
    print(f"Output directory: {config.output_dir}")
    print(f"Max samples: {config.max_samples}")
    print(f"Flow type: {config.flow_type}")
    print(f"NN depth: {config.nn_depth}")
    print(f"NN block dim: {config.nn_block_dim}")
    print(f"NN width: {config.nn_width}")
    print(f"Flow layers: {config.flow_layers}")
    print(f"Knots: {config.knots}")
    print(f"Tanh max val: {config.tanh_max_val}")
    print(f"Invert: {config.invert}")
    print(f"Cond dim: {config.cond_dim}")
    print(f"Transformer: {config.transformer}")
    print(f"Transformer knots: {config.transformer_knots}")
    print(f"Transformer interval: {config.transformer_interval}")
    print(f"Standardize: {config.standardize}")
    print(f"Max epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Patience: {config.max_patience}")
    print(f"Val proportion: {config.val_prop}")
    print(f"Seed: {config.seed}")
    print(f"Constrain physics: {config.constrain_physics}")
    if config.constrain_physics:
        print(f"Use chirp mass: {config.use_chirp_mass}")
    print("=" * 60)

    # Check for GPU
    print(f"JAX devices: {jax.devices()}")

    # Load data
    print("\n[1/5] Loading posterior samples...")
    data, load_metadata = load_gw_posterior(config.posterior_file, config.max_samples)
    print(f"Data shape: {data.shape}")
    print("Original data ranges:")
    for i, name in enumerate(["m1", "m2", "lambda1", "lambda2"]):
        print(f"  {name}: [{data[:, i].min():.3f}, {data[:, i].max():.3f}]")

    # Apply physics constraints if requested
    physics_bijection = None
    original_data = data.copy()  # Keep original for plotting
    if config.constrain_physics:
        print("\nApplying physics constraints via bijections...")
        if config.use_chirp_mass:
            print("  - Enforcing m1 >= m2 via chirp mass + mass ratio")
            print("  - Enforcing λ1, λ2 > 0 via log transform")
            print("  - Transforming to unbounded R^4 space")
        else:
            print("  - Enforcing m1, m2, λ1, λ2 > 0 via log transform")
            print("  - Transforming to unbounded R^4 space (simple mode)")

        # Clip data near boundaries for numerical stability
        data = clip_data_for_bijection(data)

        # Create bijection
        physics_bijection = create_physics_constraint_bijection(config.use_chirp_mass)

        # Transform data to unbounded space
        data_jax = jnp.array(data)
        data_unbounded = jax.vmap(physics_bijection.transform)(data_jax)
        data = np.array(data_unbounded)

        print("Unbounded data ranges:")
        if config.use_chirp_mass:
            labels = ["log(M_chirp)", "logit(q)", "log(λ1)", "log(λ2)"]
        else:
            labels = ["log(m1)", "log(m2)", "log(λ1)", "log(λ2)"]
        for i, name in enumerate(labels):
            print(f"  {name}: [{data[:, i].min():.3f}, {data[:, i].max():.3f}]")

        # Plot transformed data to visualize what the flow will train on
        figures_dir = os.path.join(config.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        transformed_plot_path = os.path.join(figures_dir, "transformed_training_data.png")
        print(f"\nSaving transformed data visualization to {transformed_plot_path}")

        # Set appropriate labels based on transformation mode
        if config.use_chirp_mass:
            plot_labels = [
                r"$\log(\mathcal{M}_c)$",
                r"$\mathrm{logit}(q)$",
                r"$\log(\Lambda_1)$",
                r"$\log(\Lambda_2)$",
            ]
        else:
            plot_labels = None  # Use default labels

        plot_transformed_data(data, transformed_plot_path, plot_labels)

    # Standardize data if requested
    data_bounds = None
    if config.standardize:
        print("\nStandardizing data to [0, 1] domain...")
        data, data_bounds = standardize_data(data)
        print("Standardized data ranges:")
        for i, name in enumerate(["m1", "m2", "lambda1", "lambda2"]):
            print(f"  {name}: [{data[:, i].min():.3f}, {data[:, i].max():.3f}]")
        print("Data bounds saved for inverse transformation")

    # Create flow
    print("\n[2/5] Creating flow architecture...")
    flow_key, train_key, sample_key = jax.random.split(jax.random.key(config.seed), 3)
    flow = create_flow(
        key=flow_key,
        flow_type=config.flow_type,
        nn_depth=config.nn_depth,
        nn_block_dim=config.nn_block_dim,
        nn_width=config.nn_width,
        flow_layers=config.flow_layers,
        knots=config.knots,
        tanh_max_val=config.tanh_max_val,
        invert=config.invert,
        cond_dim=config.cond_dim,
        transformer_type=config.transformer,
        transformer_knots=config.transformer_knots,
        transformer_interval=config.transformer_interval,
    )

    # Train flow
    print("\n[3/5] Training flow...")
    print(f"Training dataset shape: {data.shape}")
    trained_flow, losses = train_flow(
        flow,
        data,
        train_key,
        learning_rate=config.learning_rate,
        max_epochs=config.num_epochs,
        max_patience=config.max_patience,
        val_prop=config.val_prop,
        batch_size=config.batch_size,
    )
    print(f"Final train loss: {losses['train'][-1]:.4f}")
    print(f"Final val loss: {losses['val'][-1]:.4f}")

    # Wrap flow with inverse bijection if physics constraints were applied
    if config.constrain_physics and physics_bijection is not None:
        print("\nWrapping flow with inverse physics bijection...")
        print("  Flow now samples in physical space: (m1, m2, λ1, λ2)")
        # Create inverse bijection: unbounded -> physical space
        # physics_bijection.transform(): physical -> unbounded
        # physics_bijection.inverse(): unbounded -> physical
        # We need Transformed to apply the inverse, so use Invert
        inverse_bijection = Invert(physics_bijection)
        print(f"  Bijection type: {type(inverse_bijection)}")
        trained_flow = Transformed(trained_flow, inverse_bijection)

    # Save model
    print("\n[4/5] Saving model...")
    flow_kwargs = {
        "flow_type": config.flow_type,
        "nn_depth": config.nn_depth,
        "nn_block_dim": config.nn_block_dim,
        "nn_width": config.nn_width,
        "flow_layers": config.flow_layers,
        "knots": config.knots,
        "tanh_max_val": config.tanh_max_val,
        "invert": config.invert,
        "cond_dim": config.cond_dim,
        "seed": config.seed,
        "standardize": config.standardize,
        "constrain_physics": config.constrain_physics,
        "use_chirp_mass": config.use_chirp_mass,
        "transformer_type": config.transformer,
        "transformer_knots": config.transformer_knots,
        "transformer_interval": config.transformer_interval,
    }

    # Add data bounds if standardization was used
    if config.standardize and data_bounds is not None:
        flow_kwargs["data_bounds_min"] = data_bounds["min"].tolist()
        flow_kwargs["data_bounds_max"] = data_bounds["max"].tolist()

    metadata = {
        **load_metadata,
        "flow_type": config.flow_type,
        "num_epochs": len(losses["train"]),
        "learning_rate": config.learning_rate,
        "max_patience": config.max_patience,
        "val_prop": config.val_prop,
        "standardize": config.standardize,
        "constrain_physics": config.constrain_physics,
        "use_chirp_mass": config.use_chirp_mass,
    }

    # Add data bounds to metadata if standardization was used
    if config.standardize and data_bounds is not None:
        metadata["data_bounds_min"] = data_bounds["min"].tolist()
        metadata["data_bounds_max"] = data_bounds["max"].tolist()

    save_model(trained_flow, config.output_dir, flow_kwargs, metadata)

    # Generate plots
    print("\n[5/5] Generating plots...")

    # Create figures subdirectory
    figures_dir = os.path.join(config.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    if config.plot_losses:
        loss_path = os.path.join(figures_dir, "losses.png")
        plot_losses(losses, loss_path)

    if config.plot_corner:
        try:
            # Sample from trained flow
            n_plot_samples = min(10_000, data.shape[0])
            flow_samples = trained_flow.sample(sample_key, (n_plot_samples,))
            flow_samples_np = np.array(flow_samples)

            # Inverse transform samples if data was standardized
            if config.standardize and data_bounds is not None:
                flow_samples_np = inverse_standardize_data(flow_samples_np, data_bounds)

            corner_path = os.path.join(figures_dir, "corner.png")
            # Use original_data for corner plot comparison
            plot_corner(original_data, flow_samples_np, corner_path)
        except Exception as e:
            print(f"Warning: Corner plot generation failed, skipping. Error: {type(e).__name__}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {config.output_dir}")
    print(f"Figures saved to: {os.path.join(config.output_dir, 'figures')}")
    print("=" * 60)
    print("\nTo use the trained flow:")
    print(">>> from jesterTOV.inference.flows.flow import Flow")
    print(f">>> flow = Flow.from_directory('{config.output_dir}')")
    print(">>> samples = flow.sample(jax.random.key(0), (1000,))")
    if config.standardize:
        print(">>> # Samples are automatically rescaled to original domain")
    print("=" * 60)


def main():
    """Main entry point for training script."""
    if len(sys.argv) < 2:
        print("Usage: python -m jesterTOV.inference.flows.train_flow <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    # Load config from YAML
    config = FlowTrainingConfig.from_yaml(config_path)

    # Train flow
    train_flow_from_config(config)


if __name__ == "__main__":
    main()
