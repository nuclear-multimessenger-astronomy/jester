"""
Train a normalizing flow on arbitrary posterior samples.

This module supports training flows on any set of parameters specified by the user,
rather than being hardcoded for GW inference parameters.

Dependencies:
- JAX and flowjax for normalizing flows
- equinox for model serialization
- PyYAML for configuration loading
- Optional: matplotlib and corner for plotting
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Mapping, List, Optional

import equinox as eqx
import jax
import numpy as np
from jax import Array
from flowjax.train import fit_to_data

from .config import FlowTrainingConfig
from .flow import create_flow

# # TODO: decide later on if this is necessary, but this might be much faster
# # # Enable 64-bit precision for numerical accuracy
# jax.config.update("jax_enable_x64", True)


def load_posterior(
    filepath: str,
    parameter_names: List[str] = None,
    max_samples: int = 20_000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load posterior samples from npz file.

    Args:
        filepath: Path to .npz file
        parameter_names: List of parameter names to extract from the file.
                        If None, defaults to GW parameters.
                        If empty list, raises ValueError.
        max_samples: Maximum number of samples to use (downsampling if needed)

    Returns:
        data: Array of shape (n_samples, n_params) with columns corresponding to parameter_names
        metadata: Dictionary with loading information

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required parameter names are missing
        ValueError: If parameter_names is an empty list
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Posterior file not found: {filepath}")

    # Set default parameter names if None
    if parameter_names is None:
        parameter_names = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]
        print("Warning: No parameter_names provided, using default GW parameters")

    # Check for empty list
    if len(parameter_names) == 0:
        raise ValueError(
            "parameter_names cannot be empty. Please provide at least one parameter name."
        )

    # Load data
    posterior = np.load(filepath)

    # Validate required keys
    missing_keys = [key for key in parameter_names if key not in posterior]
    if missing_keys:
        available_keys = list(posterior.keys())
        raise KeyError(
            f"Missing required parameters: {missing_keys}\n"
            f"Available parameters: {available_keys}\n"
            f"Requested parameters: {parameter_names}"
        )

    # Extract samples for each parameter
    param_arrays = []
    for param_name in parameter_names:
        param_arrays.append(posterior[param_name].flatten())

    # Combine into array
    data = np.column_stack(param_arrays)
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
        "parameter_names": parameter_names,
        "n_parameters": len(parameter_names),
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


def train_flow(
    flow: Any,
    data: np.ndarray,
    key: Array,
    learning_rate: float = 1e-3,
    max_epochs: int = 600,
    max_patience: int = 50,
    val_prop: float = 0.2,
    batch_size: int = 128,
) -> Tuple[Any, Dict[str, list]]:
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


def plot_losses(losses: Mapping[str, np.ndarray | list], output_path: str) -> None:
    """Plot training and validation losses (accepts dict or list values)."""
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


def plot_corner(
    data: np.ndarray,
    flow_samples: np.ndarray,
    output_path: str,
    parameter_names: Optional[List[str]] = None,
) -> None:
    """
    Create corner plot comparing data and flow samples.

    Args:
        data: Original data array
        flow_samples: Samples from trained flow
        output_path: Path to save the plot
        parameter_names: List of parameter names for axis labels (optional)
    """
    try:
        import corner
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: corner package not available, skipping corner plot")
        return

    # Use parameter names if provided, otherwise use generic labels
    if parameter_names is None:
        labels = [f"Param {i+1}" for i in range(data.shape[1])]
    else:
        labels = parameter_names

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
        plot_datapoints=True,  # DO plot them for the flow, to check if it violates bounds
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


def train_flow_from_config(
    config: FlowTrainingConfig, parameter_names: Optional[List[str]] = None
) -> None:
    """
    Train a normalizing flow using a configuration object.

    Args:
        config: FlowTrainingConfig with all training parameters
        parameter_names: List of parameter names to extract from posterior file.
                        If None, will try to get from config or use default GW parameters.
    """
    # Determine which parameters to use
    if parameter_names is None:
        # Try to get from config if available
        if hasattr(config, "parameter_names") and config.parameter_names is not None:
            parameter_names = config.parameter_names
        else:
            # Fall back to default GW parameters for backward compatibility
            print(
                "Warning: No parameter_names provided, using default GW parameters: "
                "mass_1_source, mass_2_source, lambda_1, lambda_2"
            )
            parameter_names = [
                "mass_1_source",
                "mass_2_source",
                "lambda_1",
                "lambda_2",
            ]

    # Print configuration
    print("=" * 60)
    print("Normalizing Flow Training")
    print("=" * 60)
    print(f"Posterior file: {config.posterior_file}")
    print(f"Output directory: {config.output_dir}")
    print(f"Parameters: {parameter_names}")
    print(f"Number of parameters: {len(parameter_names)}")
    print(f"Max samples: {config.max_samples}")
    print(f"Flow type: {config.flow_type}")
    print(f"NN depth: {config.nn_depth}")
    print(f"NN block dim: {config.nn_block_dim}")
    print(f"NN width: {config.nn_width}")
    print(f"Flow layers: {config.flow_layers}")
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
    print("=" * 60)

    # Check for GPU
    print(f"JAX devices: {jax.devices()}")

    # Load data
    print("\n[1/5] Loading posterior samples...")
    data, load_metadata = load_posterior(
        config.posterior_file, parameter_names, config.max_samples
    )
    print(f"Data shape: {data.shape}")
    print("Original data ranges:")
    for i, name in enumerate(parameter_names):
        print(f"  {name}: [{data[:, i].min():.3f}, {data[:, i].max():.3f}]")

    # Keep copy of original data for corner plot
    original_data = data.copy()

    # Standardize data if requested
    data_bounds = None
    if config.standardize:
        print("\nStandardizing data to [0, 1] domain...")
        data, data_bounds = standardize_data(data)
        print("Standardized data ranges:")
        for i, name in enumerate(parameter_names):
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

    # Save model
    print("\n[4/5] Saving model...")
    flow_kwargs = {
        "flow_type": config.flow_type,
        "nn_depth": config.nn_depth,
        "nn_block_dim": config.nn_block_dim,
        "nn_width": config.nn_width,
        "flow_layers": config.flow_layers,
        "invert": config.invert,
        "cond_dim": config.cond_dim,
        "seed": config.seed,
        "standardize": config.standardize,
        "transformer_type": config.transformer,
        "transformer_knots": config.transformer_knots,
        "transformer_interval": config.transformer_interval,
        "parameter_names": parameter_names,
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
        "parameter_names": parameter_names,
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
            plot_corner(original_data, flow_samples_np, corner_path, parameter_names)
        except Exception as e:
            print(
                f"Warning: Corner plot generation failed, skipping. Error: {type(e).__name__}"
            )

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
        print(
            "Usage: python -m jesterTOV.inference.flows.train_flow <config.yaml> [param1 param2 ...]"
        )
        print("\nExamples:")
        print("  # Use parameters from config file:")
        print("  python -m jesterTOV.inference.flows.train_flow config.yaml")
        print("\n  # Override with command-line parameters:")
        print(
            "  python -m jesterTOV.inference.flows.train_flow config.yaml mass_1_source mass_2_source lambda_1 lambda_2"
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])

    # Check if parameter names were provided as command-line arguments
    parameter_names = None
    if len(sys.argv) > 2:
        parameter_names = sys.argv[2:]
        print(f"Using parameter names from command line: {parameter_names}")

    # Load config from YAML
    config = FlowTrainingConfig.from_yaml(config_path)

    # Train flow
    train_flow_from_config(config, parameter_names=parameter_names)


if __name__ == "__main__":
    main()
