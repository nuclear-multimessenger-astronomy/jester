# TODO: This is taken from fiesta for now (https://raw.githubusercontent.com/ThibeauWouters/fiesta/refs/heads/main/src/fiesta/train/neuralnets.py)
# We should merge them at some point?

from typing import Sequence, Callable
import time

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

import flax
from flax import linen as nn  # Linen API
from flax.training.train_state import TrainState
from ml_collections import ConfigDict
import optax
import pickle

###############
### CONFIGS ###
###############

class NeuralnetConfig(ConfigDict):
    """Configuration for a neural network model. For type hinting"""
    name: str
    output_size: Int
    layer_sizes: list[int]
    learning_rate: Float
    batch_size: Int
    nb_epochs: Int
    nb_report: Int
    
    def __init__(self,
                 name: str = "MLP",
                 output_size: Int = 10,
                 hidden_layer_sizes: list[int] = [64, 128, 64],
                 learning_rate: Float = 1e-3,
                 batch_size: int = 128,
                 nb_epochs: Int = 1_000,
                 nb_report: Int = None):
        
        super().__init__()
        self.name = name
        self.output_size = output_size
        hidden_layer_sizes.append(self.output_size)
        self.layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        if nb_report is None:
            nb_report = self.nb_epochs // 10
        self.nb_report = nb_report
    
#####################
### ARCHITECTURES ###
#####################

class BaseNeuralnet(nn.Module):
    """Abstract base class. Needs layer sizes and activation function used"""
    layer_sizes: Sequence[int]
    act_func: Callable = nn.relu
    
    def setup(self):
        raise NotImplementedError
    
    def __call__(self, x):
        raise NotImplementedError 
    
class MLP(BaseNeuralnet):
    """Basic multi-layer perceptron: a feedforward neural network with multiple Dense layers."""

    def setup(self):
        self.layers = [nn.Dense(n) for n in self.layer_sizes]

    @nn.compact
    def __call__(self, x: Array):
        """_summary_

        Args:
            x (Array): Input data of the neural network.
        """

        for i, layer in enumerate(self.layers):
            # Apply the linear part of the layer's operation
            x = layer(x)
            # If not the output layer, apply the given activation function
            if i != len(self.layer_sizes) - 1:
                x = self.act_func(x)

        return x
    
################
### TRAINING ###
################

def create_train_state(model: BaseNeuralnet, 
                       test_input: Array, 
                       rng: jax.random.PRNGKey, 
                       config: NeuralnetConfig):
    """
    Creates an initial `TrainState` from NN model and optimizer and initializes the parameters by passing dummy input.

    Args:
        model (BaseNeuralnet): Neural network model to be trained.
        test_input (Array): A test input used to initialize the parameters of the model.
        rng (jax.random.PRNGKey): Random number generator key used for initialization of the model.
        config (NeuralnetConfig): Configuration for the neural network training.

    Returns:
        TrainState: TrainState object for training
    """
    params = model.init(rng, test_input)['params']
    tx = optax.adam(config.learning_rate)
    state = TrainState.create(apply_fn = model.apply, params = params, tx = tx)
    return state

def apply_model(state: TrainState, 
                x_batched: Float[Array, "n_batch ndim_input"], 
                y_batched: Float[Array, "n_batch ndim_output"]):
    """
    Apply the model to a batch of data and compute the loss and gradients.

    Args:
        state (TrainState): TrainState object for training.
        x_batched (Float[Array, "n_batch ndim_input"]): Batch of input
        y_batched (Float[Array, "n_batch ndim_output"]): Batch of output
    """

    def loss_fn(params):
        def squared_error(x, y):
            # For a single datapoint
            pred = state.apply_fn({'params': params}, x)
            return jnp.inner(y - pred, y - pred) / 2.0
        # Vectorize the previous to compute the average of the loss on all samples.
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return loss, grads

@jax.jit
def train_step(state: TrainState, 
               train_X: Float[Array, "n_batch_train ndim_input"], 
               train_y: Float[Array, "n_batch_train ndim_output"], 
               val_X: Float[Array, "n_batch_val ndim_output"] = None, 
               val_y: Float[Array, "n_batch_val ndim_output"] = None) -> tuple[TrainState, Float[Array, "n_batch_train"], Float[Array, "n_batch_val"]]:
    """
    Train for a single step. Note that this function is functionally pure and hence suitable for jit.

    Args:
        state (TrainState): TrainState object
        train_X (Float[Array, "n_batch_train ndim_input"]): Training input data
        train_y (Float[Array, "n_batch_train ndim_output"]): Training output data
        val_X (Float[Array, "n_batch_val ndim_input"], optional): Validation input data. Defaults to None.
        val_y (Float[Array, "n_batch_val ndim_output"], optional): Valdiation output data. Defaults to None.

    Returns:
        tuple[TrainState, Float, Float]: TrainState with updated weights, and arrays of training and validation losses
    """

    # Compute losses
    train_loss, grads = apply_model(state, train_X, train_y)
    if val_X is not None:
        val_loss, _ = apply_model(state, val_X, val_y)
    else:
        val_loss = jnp.zeros_like(train_loss)

    # Update parameters
    state = state.apply_gradients(grads=grads)

    return state, train_loss, val_loss

def train_loop(state: TrainState, 
               config: NeuralnetConfig,
               train_X: Float[Array, "n_batch_train ndim_input"], 
               train_y: Float[Array, "n_batch_train ndim_output"], 
               val_X: Float[Array, "n_batch_val ndim_output"] = None, 
               val_y: Float[Array, "n_batch_val ndim_output"] = None,
               verbose: bool = True):

    train_losses, val_losses = [], []

    start = time.time()
    
    for i in range(config.nb_epochs):
        # Do a single step
        
        state, train_loss, val_loss = train_step(state, train_X, train_y, val_X, val_y)
        # Save the losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # Report once in a while
        if i % config.nb_report == 0 and verbose:
            print(f"Train loss at step {i+1}: {train_loss}")
            print(f"Valid loss at step {i+1}: {val_loss}")
            print(f"Learning rate: {config.learning_rate}")
            print("---")

    end = time.time()
    if verbose:
        print(f"Training for {config.nb_epochs} took {end-start} seconds.")

    return state, train_losses, val_losses

def serialize(state: TrainState, 
              config: NeuralnetConfig = None) -> dict:
    """
    Serialize function to save the model and its configuration.

    Args:
        state (TrainState): The TrainState object to be serialized.
        config (NeuralnetConfig, optional): The config to be serialized. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    # Get state dict, which has params
    params = flax.serialization.to_state_dict(state)["params"]
    
    serialized_dict = {"params": params,
                       "config": config,
                    }
    
    return serialized_dict

# TODO: add support for various activation functions and different model architectures to be loaded from serialized models
def save_model(state: TrainState, 
               config: ConfigDict = None, 
               out_name: str = "my_flax_model.pkl"):
    """
    Serialize and save the model to a file.
    
    Raises:
        ValueError: If the provided file extension is not .pkl or .pickle.

    Args:
        state (TrainState): The TrainState object to be saved.
        config (ConfigDict, optional): The config to be saved.. Defaults to None.
        out_name (str, optional): The pickle file to which we save the serialized model. Defaults to "my_flax_model.pkl".
    """
    
    if not out_name.endswith(".pkl") and not out_name.endswith(".pickle"):
        raise ValueError("For now, only .pkl or .pickle extensions are supported.")
    
    serialized_dict = serialize(state, config)
    with open(out_name, 'wb') as handle:
        pickle.dump(serialized_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_model(filename: str) -> tuple[TrainState, NeuralnetConfig]:
    """
    Load a model from a file.
    TODO: this is very cumbersome now and must be massively improved in the future

    Args:
        filename (str): Filename of the model to be loaded.

    Raises:
        ValueError: If there is something wrong with loading, since lots of things can go wrong here.

    Returns:
        tuple[TrainState, NeuralnetConfig]: The TrainState object loaded from the file and the NeuralnetConfig object.
    """
    
    with open(filename, 'rb') as handle:
        loaded_dict = pickle.load(handle)
        
    config: NeuralnetConfig = loaded_dict["config"]
    layer_sizes = config.layer_sizes
    act_func = nn.relu
    params = loaded_dict["params"]
        
    model = MLP(layer_sizes, act_func)
    
    # Create train state without optimizer
    state = TrainState.create(apply_fn = model.apply, params = params, tx = optax.adam(config.learning_rate))
    
    return state, config