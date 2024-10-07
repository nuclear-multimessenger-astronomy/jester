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
    output_size: Int
    layer_sizes: list[int]
    learning_rate: Float
    
    def __init__(self,
                 output_size: Int,
                 hidden_layer_sizes: list[int] = [64, 128, 64],
                 learning_rate: Float = 1e-3):
        
        super().__init__()
        self.output_size = output_size
        hidden_layer_sizes.append(self.output_size)
        self.layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
    
#####################
### ARCHITECTURES ###
#####################

class BaseNeuralnet(nn.Module):
    """Abstract base class. Needs layer sizes and activation function used"""
    layer_sizes: Sequence[int]
    act_func: Callable = nn.tanh
    
    def setup(self):
        raise NotImplementedError
    
    def __call__(self, x):
        raise NotImplementedError 
    
class CS2_MLP(BaseNeuralnet):
    """Basic multi-layer perceptron (a feedforward neural network with multiple Dense layers) that outputs the speed-of-sound (cs2) for a given input."""

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
                
        # The output layer has a sigmoid activation function, to guarantee cs2 is between 0 and 1
        x = nn.sigmoid(x)
        return x
    
def create_train_state(model: BaseNeuralnet, 
                       test_input: Array, 
                       rng: jax.random.PRNGKey, 
                       config: NeuralnetConfig) -> TrainState:
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
    act_func = nn.tanh
    params = loaded_dict["params"]
        
    model = CS2_MLP(layer_sizes, act_func)
    
    # Create train state without optimizer
    state = TrainState.create(apply_fn = model.apply, params = params, tx = optax.adam(config.learning_rate))
    
    return state, config