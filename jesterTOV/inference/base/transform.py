r"""Transform base classes for JESTER inference system.

These transforms encode the behavior to transform sets of parameters for the EOS, e.g. sampled from priors, to their TOV solutions.

This module contains transform classes that were originally from Jim (jimgw v0.2.0).
They are copied here to remove the dependency on jimgw.

Note: These classes follow the Jim/jimgw architecture and provide parameter
transformations with Jacobian corrections for Bayesian inference.
"""

from abc import ABC
from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

# Type aliases for better readability
ParamDict: TypeAlias = dict[str, Float] # dictionary containing parameter names and their values
NameMapping: TypeAlias = tuple[list[str], list[str]] # tuple of (input_names, output_names)


class Transform(ABC):
    """
    Base class for transforms.

    Note: This class follows the Jim/jimgw architecture. The purpose of this class
    is purely for keeping track of parameter name mappings.
    """

    name_mapping: NameMapping

    def __init__(
        self,
        name_mapping: NameMapping,
    ) -> None:
        """
        Parameters
        ----------
        name_mapping : tuple[list[str], list[str]]
            Tuple of (input_names, output_names) for the transform.
        """
        self.name_mapping = name_mapping

    def propagate_name(self, x: list[str]) -> list[str]:
        """
        Propagate parameter names through the transform.

        Parameters
        ----------
        x : list[str]
            Input parameter names.

        Returns
        -------
        list[str]
            Output parameter names after applying the transform.
        """
        input_set = set(x)
        from_set = set(self.name_mapping[0])
        to_set = set(self.name_mapping[1])
        return list(input_set - from_set | to_set)


class NtoMTransform(Transform):
    """
    N-to-M parameter transform (not necessarily invertible).

    Note: This class follows the Jim/jimgw architecture. Used for likelihood
    transforms where you map N parameters to M different parameters without
    requiring invertibility or Jacobian corrections.
    """

    transform_func: Callable[[ParamDict], ParamDict]  # Maps parameter dict to transformed parameter dict

    def forward(self, x: ParamDict) -> ParamDict:
        """
        Push forward the input x to transformed coordinate y.

        Parameters
        ----------
        x : dict[str, Float]
            The input dictionary.

        Returns
        -------
        y : dict[str, Float]
            The transformed dictionary.
        """
        x_copy = x.copy()
        output_params = self.transform_func(x_copy)
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy


class NtoNTransform(NtoMTransform):
    """
    N-to-N parameter transform with Jacobian calculation.

    Note: This class follows the Jim/jimgw architecture.
    """

    @property
    def n_dim(self) -> int:
        return len(self.name_mapping[0])

    def transform(self, x: ParamDict) -> tuple[ParamDict, Float]:
        """
        Transform the input x to transformed coordinate y and return the log Jacobian determinant.

        This only works if the transform is a N -> N transform.

        Parameters
        ----------
        x : ParamDict
            The input dictionary.

        Returns
        -------
        y : ParamDict
            The transformed dictionary.
        log_det : Float
            The log Jacobian determinant.
        """
        x_copy = x.copy()
        transform_params = dict((key, x_copy[key]) for key in self.name_mapping[0])
        output_params = self.transform_func(transform_params)
        jacobian = jax.jacfwd(self.transform_func)(transform_params)
        jacobian = jnp.array(jax.tree.leaves(jacobian))
        jacobian = jnp.log(jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim))))
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy, jacobian


class BijectiveTransform(NtoNTransform):
    """
    Bijective (invertible) N-to-N parameter transform with Jacobian corrections.

    Note: This class follows the Jim/jimgw architecture. Used for sample transforms
    where parameters are transformed during MCMC sampling and Jacobian corrections
    are applied to the prior.
    """

    inverse_transform_func: Callable[[ParamDict], ParamDict]  # Maps transformed dict back to original parameter dict

    def inverse(self, y: ParamDict) -> tuple[ParamDict, Float]:
        """
        Inverse transform the input y to original coordinate x.

        Parameters
        ----------
        y : ParamDict
            The transformed dictionary.

        Returns
        -------
        x : ParamDict
            The original dictionary.
        log_det : Float
            The log Jacobian determinant.
        """
        y_copy = y.copy()
        transform_params = dict((key, y_copy[key]) for key in self.name_mapping[1])
        output_params = self.inverse_transform_func(transform_params)
        jacobian = jax.jacfwd(self.inverse_transform_func)(transform_params)
        jacobian = jnp.array(jax.tree.leaves(jacobian))
        jacobian = jnp.log(jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim))))
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy, jacobian

    def backward(self, y: ParamDict) -> ParamDict:
        """
        Pull back the input y to original coordinate x (without Jacobian).

        Parameters
        ----------
        y : ParamDict
            The transformed dictionary.

        Returns
        -------
        x : ParamDict
            The original dictionary.
        """
        y_copy = y.copy()
        output_params = self.inverse_transform_func(y_copy)
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy


# ============================================================================
# Specific Transform Implementations
# ============================================================================
# These are used internally by UniformPrior and other prior distributions.


@jaxtyped(typechecker=typechecker)
class ScaleTransform(BijectiveTransform):
    """
    Scale transform: y = x * scale.

    Note: This class follows the Jim/jimgw architecture.
    """

    scale: Float

    def __init__(
        self,
        name_mapping: NameMapping,
        scale: Float,
    ) -> None:
        """
        Parameters
        ----------
        name_mapping : NameMapping
            Tuple of (input_names, output_names).
        scale : Float
            The scaling factor.
        """
        super().__init__(name_mapping)
        self.scale = scale
        self.transform_func = lambda x: {
            name_mapping[1][i]: x[name_mapping[0][i]] * self.scale
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: x[name_mapping[1][i]] / self.scale
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class OffsetTransform(BijectiveTransform):
    """
    Offset transform: y = x + offset.

    Note: This class follows the Jim/jimgw architecture.
    """

    offset: Float

    def __init__(
        self,
        name_mapping: NameMapping,
        offset: Float,
    ) -> None:
        """
        Parameters
        ----------
        name_mapping : NameMapping
            Tuple of (input_names, output_names).
        offset : Float
            The offset value.
        """
        super().__init__(name_mapping)
        self.offset = offset
        self.transform_func = lambda x: {
            name_mapping[1][i]: x[name_mapping[0][i]] + self.offset
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: x[name_mapping[1][i]] - self.offset
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class LogitTransform(BijectiveTransform):
    """
    Logit transform: y = 1 / (1 + exp(-x)).

    Note: This class follows the Jim/jimgw architecture.
    """

    def __init__(
        self,
        name_mapping: NameMapping,
    ) -> None:
        """
        Parameters
        ----------
        name_mapping : NameMapping
            Tuple of (input_names, output_names).
        """
        super().__init__(name_mapping)
        self.transform_func = lambda x: {
            name_mapping[1][i]: 1 / (1 + jnp.exp(-x[name_mapping[0][i]]))
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.log(
                x[name_mapping[1][i]] / (1 - x[name_mapping[1][i]])
            )
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class BoundToBound(BijectiveTransform):
    """
    Linear transform from [original_lower, original_upper] to [target_lower, target_upper].

    Used for nested sampling to map prior bounds to unit cube [0, 1].

    Note: This implementation handles per-parameter bounds, where each parameter can have
    different original and target bounds.
    """

    original_lower_bound: ParamDict  # Lower bounds for original parameters
    original_upper_bound: ParamDict  # Upper bounds for original parameters
    target_lower_bound: ParamDict    # Lower bounds for target parameters
    target_upper_bound: ParamDict    # Upper bounds for target parameters

    def __init__(
        self,
        name_mapping: NameMapping,
        original_lower_bound: ParamDict,  # Lower bounds for original parameters
        original_upper_bound: ParamDict,  # Upper bounds for original parameters
        target_lower_bound: ParamDict,    # Lower bounds for target parameters
        target_upper_bound: ParamDict,    # Upper bounds for target parameters
    ) -> None:
        """
        Parameters
        ----------
        name_mapping : NameMapping
            Tuple of (input_names, output_names).
        original_lower_bound : dict[str, Float]
            Lower bounds in original space (per parameter).
        original_upper_bound : dict[str, Float]
            Upper bounds in original space (per parameter).
        target_lower_bound : dict[str, Float]
            Lower bounds in target space (per parameter).
        target_upper_bound : dict[str, Float]
            Upper bounds in target space (per parameter).
        """
        super().__init__(name_mapping)
        self.original_lower_bound = original_lower_bound
        self.original_upper_bound = original_upper_bound
        self.target_lower_bound = target_lower_bound
        self.target_upper_bound = target_upper_bound

        # Forward: original → target
        # y = (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
        def _forward(x: ParamDict) -> ParamDict:
            result = {}
            for i, in_name in enumerate(name_mapping[0]):
                out_name = name_mapping[1][i]
                x_val = x[in_name]
                x_min = original_lower_bound[in_name]
                x_max = original_upper_bound[in_name]
                y_min = target_lower_bound[out_name]
                y_max = target_upper_bound[out_name]
                result[out_name] = (x_val - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
            return result

        # Inverse: target → original
        # x = (y - y_min) / (y_max - y_min) * (x_max - x_min) + x_min
        def _inverse(y: ParamDict) -> ParamDict:
            result = {}
            for i, out_name in enumerate(name_mapping[1]):
                in_name = name_mapping[0][i]
                y_val = y[out_name]
                x_min = original_lower_bound[in_name]
                x_max = original_upper_bound[in_name]
                y_min = target_lower_bound[out_name]
                y_max = target_upper_bound[out_name]
                result[in_name] = (y_val - y_min) / (y_max - y_min) * (x_max - x_min) + x_min
            return result

        self.transform_func = _forward
        self.inverse_transform_func = _inverse
