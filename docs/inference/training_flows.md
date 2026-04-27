# Training normalizing flows

This documentation page helps users set up their own workflows for training normalizing flows. `jester` makes use of normalizing flows in evaluating the likelihoods for some of the multi-messenger observations. For example, analyses of the NICER and GW observations provide posterior densities that are used as input for the `jester` inference pipeline. Indeed, NICER provides a joint 2-dimensional posterior on the mass and radius of a pulsar, while GWs of BNSs provide a 4-dimensional posterior on the source-frame component masses and the tidal deformabilities of the binary components. These can be densities with complicated shapes, such that a density estimator is recommended to learn these shapes, rendering the densities that these samples make up tractable (i.e., we can sample from the density and evaluate the density at an arbitrary point in its parameter space).

## `Flow` class

`Jester` has its own {py:class}`Flow <jesterTOV.inference.flows.flow.Flow>` class, with the full API reference available in the API docs.
This provides a wrapper around the `flowjax` package, which is available on GitHub here: https://github.com/danielward27/flowjax. 
The flows also make use of `equinox` as the underlying framework for the neural networks that make up the flow, which is available on GitHub here: https://github.com/patrick-kidger/equinox

## Flow configuration

Just like the Bayesian inference in `jester`, training the flow can make use of its own config file. 

Below, we show an example for the default config file currently used by `jester`, taking a BNS GW posterior as an example to train the flow on. In this case, we will train a normalizing flow to approximate the source-frame masses (`mass_1_source`, `mass_2_source`) and the tidal deformabilities (`lambda_1`, `lambda_2`). 
The dataset is split into a training-validation set, with 20% of the data used for validation.
The defaults training hyperparameters chosen here train for a maximum of 1000 epochs, with a learning rate of 0.001 and an early stopping patience of 100 epochs (i.e., in case the validation loss values do not improve over 100 epochs, we quit the training). 
The flow architecture is a masked autoregressive flow with 1 layer, and the neural networks that make up the flow have a depth of 5, width of 50, and block dimension of 8. 
The data is standardized using z-score standardization, and the maximum number of samples used for training is 50,000. 
Finally, the configuration also specifies that corner plots and loss plots should be generated after training.

```bash
# Configuration for training normalizing flow
# Updated to use new default hyperparameters (PR #55)
posterior_file: posterior.npz
output_dir: .

# Parameter selection
parameter_names: ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]

# # Training parameters
num_epochs: 1000
learning_rate: 0.001
max_patience: 100
val_prop: 0.2
seed: 0

# Flow architecture
flow_type: masked_autoregressive_flow
flow_layers: 1
nn_depth: 5
nn_width: 50
nn_block_dim: 8
invert: true

# Data preprocessing
max_samples: 50000
standardize: true
standardization_method: zscore

# Plotting
plot_corner: true
plot_losses: true

# Conditional flow settings
cond_dim: null
```

```{note}
For the moment, `jester` does not yet support the use of conditional flows.
```

## Starting the training

Training of the flows is handled by the internals of `flowjax`, for which `jester` provides a small wrapper and utilities in `jester/jesterTOV/inference/flows/train_flow.py`. 

```bash
train_jester_flow flow_config.yaml
```

This will automatically perform the following tasks:
1. Load the posterior samples from the specified `posterior_file`, and (in case selected), standardize the data.
2. Set up the flow architecture according to the specified `flow_type` and its hyperparameters.
3. Train the flow using the specified training hyperparameters. 
4. Save the trained flow to the `output_dir`.
5. Generate plots for the loss values across the training epochs, and a corner plot of the flow samples compared to the original posterior samples.

## Training output

Once the training is complete, the following files will be saved to the `output_dir`:
* `flow_kwargs.json`: JSON representation of the flow architecture and hyperparameters, which can be used to load the flow later on.
* `flow_weights.eqx`: The weights of the trained flow, which can be used to load the flow later on. This uses the functionalities from `equinox` for saving and loading the weights of the neural networks that make up the flow.
* `metadata.json`: A few metadata entries about the training.
* `figures/corner.png` and `figures/losses.png`: The corner plot and loss plot generated after training, to sanity check the training results.

## Available flow types

Currently, the wrapper in `jester` supports the following `flowjax` flow types:
* `block_neural_autoregressive_flow`
* `masked_autoregressive_flow`
* `coupling_flow`

More flows might be available in `flowjax` in the future, and these can be added to the `jester` wrapper as needed.