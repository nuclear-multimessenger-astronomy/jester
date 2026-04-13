"""Training script for normalizing flows on gravitational wave posterior samples.

Trains normalizing flow models to approximate GW posteriors in (m1, m2, λ1, λ2) space.
The trained flows serve as efficient proposal distributions for EOS inference.

Training Pipeline
-----------------
1. Load configuration from YAML file
2. Load posterior samples from npz file
3. Apply optional standardization
4. Create flow architecture (autoregressive, coupling)
5. Fit flow using maximum likelihood with early stopping
6. Save trained weights, config, and metadata
7. Generate validation plots

Supported Architectures
-----------------------
- coupling_flow: Balanced speed and expressiveness
- block_neural_autoregressive_flow: Good expressiveness
- masked_autoregressive_flow: Flexible but slower

Configuration-Driven Usage
---------------------------
Create a YAML config file (e.g., config.yaml):

    posterior_file: data/gw170817_posterior.npz
    output_dir: models/gw170817/
    flow_type: masked_autoregressive_flow
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

>>> from jesterTOV.inference.flows.flow import create_flow
>>> from jesterTOV.inference.flows.train_flow import load_gw_posterior, train_flow, save_model
>>> data, metadata = load_gw_posterior("gw170817.npz", max_samples=50000)
>>> flow = create_flow(jax.random.key(0), flow_type="masked_autoregressive_flow")
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
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Mapping, Iterable

import equinox as eqx
import jax
import numpy as np
from jax import Array
from flowjax.train import fit_to_data

from jesterTOV.logging_config import get_logger
from .config import FlowTrainingConfig
from .flow import create_flow

logger = get_logger("jester")


def load_posterior(
    filepath: str,
    parameter_names: list[str],
    max_samples: int = 20_000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load posterior samples from npz file with flexible parameter selection.

    Args:
        filepath: Path to .npz file
        parameter_names: List of parameter names to extract from file
        max_samples: Maximum number of samples to use (downsampling if needed)

    Returns:
        data: Array of shape (n_samples, n_params) with selected parameters
        metadata: Dictionary with loading information

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required parameter names are missing from file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Posterior file not found: {filepath}")

    # Load data
    posterior = np.load(filepath)

    # Validate required keys
    missing_keys = [key for key in parameter_names if key not in posterior]
    if missing_keys:
        available_keys = list(posterior.keys())
        raise KeyError(
            f"Missing required parameter names: {missing_keys}\n"
            f"Available keys in file: {available_keys}\n"
            f"Requested parameters: {parameter_names}"
        )

    # Extract samples for each parameter
    columns = [posterior[param].flatten() for param in parameter_names]

    # Combine into array
    data = np.column_stack(columns)
    n_samples_total = data.shape[0]

    # Downsample if needed
    if n_samples_total > max_samples:
        downsample_factor = int(np.ceil(n_samples_total / max_samples))
        data = data[::downsample_factor]
        logger.info(
            f"Downsampled from {n_samples_total} to {data.shape[0]} samples "
            f"(factor: {downsample_factor})"
        )
    else:
        logger.info(f"Using all {n_samples_total} samples")

    metadata = {
        "n_samples_total": n_samples_total,
        "n_samples_used": data.shape[0],
        "parameter_names": parameter_names,
        "filepath": filepath,
    }

    return data, metadata

def standardize_data(
        data: np.ndarray,
        standardization_method: str,
        parameter_names: list[str] = []
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Standardize data based on the selected standardiation method.
    
    Args:
        data: Array of shape (n_samples, n_features)
        standardization_method: str
            Which method to standardize with. 
            Can either be 'zscore' for zero mean and unit std,
            otherwise min-max scaling will be applied.
        parameter_names:
            Parameter names for which the rescaled data range should be printed.
            Defaults to [].
    """

    if standardization_method == "zscore":
        logger.info("Standardizing data using z-score (mean=0, std=1)...")
        data, data_statistics = standardize_data_zscore(data)
        logger.info("Standardized data statistics:")
        for i, name in enumerate(parameter_names):
            logger.info(
                f"  {name}: mean={data[:, i].mean():.3f}, std={data[:, i].std():.3f}"
            )
        logger.info("Data mean and std saved for inverse transformation")
    else:  # minmax
        logger.info("Standardizing data using min-max [0, 1] scaling...")
        data, data_statistics = standardize_data_minmax(data)
        logger.info("Standardized data ranges:")
        for i, name in enumerate(parameter_names):
            logger.info(
                f"  {name}: [{data[:, i].min():.3f}, {data[:, i].max():.3f}]"
            )
            logger.info("Data bounds saved for inverse transformation")

    return data, data_statistics

def standardize_data_zscore(
    data: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Standardize data to mean=0, std=1 using z-score normalization.

    Args:
        data: Array of shape (n_samples, n_features)

    Returns:
        standardized_data: Data with mean=0, std=1 per feature
        statistics: Dictionary with 'mean' and 'std' arrays for each feature
    """
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)

    # Avoid division by zero (if a feature is constant)
    data_std = np.where(data_std == 0, 1.0, data_std)

    standardized_data = (data - data_mean) / data_std

    statistics = {"mean": data_mean, "std": data_std}

    return standardized_data, statistics


def inverse_standardize_data_zscore(
    standardized_data: np.ndarray, statistics: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Inverse transform z-score standardized data back to original scale.

    Args:
        standardized_data: Data with mean=0, std=1
        statistics: Dictionary with 'mean' and 'std' arrays for each feature

    Returns:
        data: Data in original scale
    """
    data_mean = statistics["mean"]
    data_std = statistics["std"]
    data_std = np.where(data_std == 0, 1.0, data_std)

    data = standardized_data * data_std + data_mean

    return data


def standardize_data_minmax(
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


def inverse_standardize_data_minmax(
    standardized_data: np.ndarray, bounds: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Inverse transform min-max standardized data back to original scale.

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
    data: np.ndarray | Iterable[np.ndarray],
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
              or if conditional flow is trained iterable of two arrays
              where the last array are the conditional parameters.
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
    logger.info(f"Training flow for up to {max_epochs} epochs...")
    logger.info(f"Using {val_prop:.1%} of data for validation")
    logger.info(f"Batch size: {batch_size}")
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
    logger.info(f"Training completed after {len(losses['train'])} epochs")
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
    logger.info(f"Saving model weights to {weights_path}")
    eqx.tree_serialise_leaves(weights_path, flow)

    # Save architecture kwargs
    kwargs_path = os.path.join(output_dir, "flow_kwargs.json")
    logger.info(f"Saving flow kwargs to {kwargs_path}")
    with open(kwargs_path, "w") as f:
        json.dump(flow_kwargs, f, indent=2)

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def plot_losses(losses: Mapping[str, np.ndarray | list], output_path: str) -> None:
    """Plot training and validation losses (accepts dict or list values)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping loss plot")
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
    logger.info(f"Saved loss plot to {output_path}")


def plot_corner(
    data: np.ndarray,
    flow_samples: np.ndarray,
    output_path: str,
    labels: list[str],
) -> None:
    """Create corner plot comparing data and flow samples.

    Args:
        data: Original data samples
        flow_samples: Samples from trained flow
        output_path: Path to save plot
        labels: Parameter labels for plot
    """
    try:
        import corner
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("corner package not available, skipping corner plot")
        return

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
    logger.info(f"Saved corner plot to {output_path}")


def train_flow_from_config(config: FlowTrainingConfig) -> None:
    """
    Train a normalizing flow using a configuration object.

    Args:
        config: FlowTrainingConfig with all training parameters
    """
    # Log configuration
    logger.info("=" * 60)
    logger.info("Normalizing Flow Training")
    logger.info("=" * 60)
    logger.info(f"Posterior file: {config.posterior_file}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Parameter names: {config.parameter_names}")
    logger.info(f"Max samples: {config.max_samples}")
    logger.info(f"Flow type: {config.flow_type}")
    logger.info(f"NN depth: {config.nn_depth}")
    logger.info(f"NN block dim: {config.nn_block_dim}")
    logger.info(f"NN width: {config.nn_width}")
    logger.info(f"Flow layers: {config.flow_layers}")
    logger.info(f"Invert: {config.invert}")
    logger.info(f"Cond dim: {config.cond_dim}")
    logger.info(f"Cond. parameters: {config.cond_parameter_names}")
    logger.info(f"Transformer: {config.transformer}")
    logger.info(f"Transformer knots: {config.transformer_knots}")
    logger.info(f"Transformer interval: {config.transformer_interval}")
    logger.info(f"Standardize: {config.standardize}")
    logger.info(f"Standardization method: {config.standardization_method}")
    logger.info(f"Max epochs: {config.num_epochs}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Patience: {config.max_patience}")
    logger.info(f"Val proportion: {config.val_prop}")
    logger.info(f"Seed: {config.seed}")
    logger.info("=" * 60)

    # Check for GPU
    logger.info(f"JAX devices: {jax.devices()}")

    # Load data
    logger.info("[1/5] Loading posterior samples...")
    data, load_metadata = load_posterior(
        config.posterior_file,
        parameter_names=config.parameter_names,
        max_samples=config.max_samples,
    )
    parameter_names = load_metadata["parameter_names"]
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Parameters: {parameter_names}")
    logger.info("Original data ranges:")
    for i, name in enumerate(parameter_names):
        logger.info(f"  {name}: [{data[:, i].min():.3f}, {data[:, i].max():.3f}]")

    # Keep copy of original data for corner plot
    original_data = data.copy()

    # Standardize data if requested
    data_statistics = None
    if config.standardize:
        data, data_statistics = standardize_data(
            data,
            config.standardization_method,
            parameter_names,
        )
    dim = data.shape[1]  # Infer dimensionality from data

    if config.cond_dim:
        logger.info("[1.5/5] Loading conditional samples...")

        cond_parameter_names = config.cond_parameter_names or []
        if len(cond_parameter_names) != config.cond_dim:
            raise ValueError(
                  f"If conditional dimension is set, " 
                  f"you also need to provide {config.cond_dim} conditional "
                  f"parameter names. You provided {len(cond_parameter_names)}."
            )

        cond_samples, cond_samples_metadata = load_posterior(
            config.posterior_file, 
            parameter_names=cond_parameter_names,
            max_samples=config.max_samples
        )

        # Standardize data if requested
        original_cond_samples = cond_samples.copy()
        cond_data_statistics = None
        if config.standardize:
            cond_samples, cond_data_statistics = standardize_data(
                cond_samples,
                config.standardization_method,
            )

        original_data = np.hstack((original_data, original_cond_samples))
        data = (data, cond_samples)

    # Create flow
    logger.info("[2/5] Creating flow architecture...")
    flow_key, train_key, sample_key = jax.random.split(jax.random.key(config.seed), 3)
    logger.info(f"Flow dimensionality: {dim}D")

    flow = create_flow(
        key=flow_key,
        dim=dim,
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
    logger.info("[3/5] Training flow...")
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
    logger.info(f"Final train loss: {losses['train'][-1]:.4f}")
    logger.info(f"Final val loss: {losses['val'][-1]:.4f}")

    # Save model
    logger.info("[4/5] Saving model...")
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
        "standardization_method": config.standardization_method,
        "transformer_type": config.transformer,
        "transformer_knots": config.transformer_knots,
        "transformer_interval": config.transformer_interval,
    }

    # Add data statistics if standardization was used
    if config.standardize and data_statistics is not None:
        if config.standardization_method == "zscore":
            flow_kwargs["data_mean"] = data_statistics["mean"].tolist()
            flow_kwargs["data_std"] = data_statistics["std"].tolist()
        else:  # minmax
            flow_kwargs["data_bounds_min"] = data_statistics["min"].tolist()
            flow_kwargs["data_bounds_max"] = data_statistics["max"].tolist()

    metadata = {
        **load_metadata,
        "flow_type": config.flow_type,
        "num_epochs": len(losses["train"]),
        "learning_rate": config.learning_rate,
        "max_patience": config.max_patience,
        "val_prop": config.val_prop,
        "standardize": config.standardize,
        "standardization_method": config.standardization_method,
    }

    # Add data statistics to metadata if standardization was used
    if config.standardize and data_statistics is not None:
        if config.standardization_method == "zscore":
            metadata["data_mean"] = data_statistics["mean"].tolist()
            metadata["data_std"] = data_statistics["std"].tolist()
        else:  # minmax
            metadata["data_bounds_min"] = data_statistics["min"].tolist()
            metadata["data_bounds_max"] = data_statistics["max"].tolist()
    
    # Add conditional data statistics to metadata if standardization was used
    if config.cond_dim:
        if config.standardization_method == "zscore":
            metadata["cond_data_mean"] = cond_data_statistics["mean"].tolist()
            metadata["cond_data_std"] = cond_data_statistics["std"].tolist()
        else: 
            metadata["cond_data_min"] = cond_data_statistics["min"].tolist()
            metadata["cond_data_max"] = cond_data_statistics["max"].tolist()  

    save_model(trained_flow, config.output_dir, flow_kwargs, metadata)

    # Generate plots
    logger.info("[5/5] Generating plots...")

    # Create figures subdirectory
    figures_dir = os.path.join(config.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    if config.plot_losses:
        loss_path = os.path.join(figures_dir, "losses.png")
        plot_losses(losses, loss_path)

    if config.plot_corner:
        try:
            # Sample from trained flow
            n_plot_samples = min(10_000, original_data.shape[0])

            if config.cond_dim:
                # get flow samples and untransform them
                flow_samples = trained_flow.sample(sample_key, (1,), condition=cond_samples)
                flow_samples_np = np.array(flow_samples)
                if config.standardize and data_statistics is not None:
                    if config.standardization_method == "zscore":
                        flow_samples_np = inverse_standardize_data_zscore(
                            flow_samples_np, data_statistics
                        )
                    else:  # minmax
                        flow_samples_np = inverse_standardize_data_minmax(
                            flow_samples_np, data_statistics
                        )
                # stack together with conditional data for plot
                flow_samples_np = np.hstack(
                    (flow_samples_np.reshape(-1, len(config.parameter_names)), original_data[:, -config.cond_dim:])
                )
                labels = [*config.parameter_names, *config.cond_parameter_names]

            else:
                flow_samples = trained_flow.sample(sample_key, (n_plot_samples,))
                flow_samples_np = np.array(flow_samples)
                labels = parameter_names

                # Inverse transform samples if data was standardized
                if config.standardize and data_statistics is not None:
                    if config.standardization_method == "zscore":
                        flow_samples_np = inverse_standardize_data_zscore(
                            flow_samples_np, data_statistics
                        )
                    else:  # minmax
                        flow_samples_np = inverse_standardize_data_minmax(
                            flow_samples_np, data_statistics
                        )

            corner_path = os.path.join(figures_dir, "corner.png")
            # Use original_data for corner plot comparison
            # Update labels based on parameter names
            plot_corner(
                original_data, flow_samples_np, corner_path, labels=labels
            )
        except Exception as e:
            logger.warning(
                f"Corner plot generation failed, skipping. Error: {type(e).__name__}"
            )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {config.output_dir}")
    logger.info(f"Figures saved to: {os.path.join(config.output_dir, 'figures')}")
    logger.info("=" * 60)
    logger.info("To use the trained flow:")
    logger.info(">>> from jesterTOV.inference.flows.flow import Flow")
    logger.info(f">>> flow = Flow.from_directory('{config.output_dir}')")
    logger.info(">>> samples = flow.sample(jax.random.key(0), (1000,))")
    if config.standardize:
        logger.info(">>> # Samples are automatically rescaled to original domain")
    logger.info("=" * 60)


def main():
    """Main entry point for training script."""
    if len(sys.argv) < 2:
        logger.error(
            "Usage: python -m jesterTOV.inference.flows.train_flow <config.yaml>"
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])

    # Load config from YAML
    config = FlowTrainingConfig.from_yaml(config_path)

    # Train flow
    train_flow_from_config(config)


if __name__ == "__main__":
    main()
