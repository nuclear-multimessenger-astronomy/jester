r"""Combined and utility transform classes"""
from typing import Any, Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.inference.base import NtoMTransform



class CombinedTransform(NtoMTransform):
    """
    Chain multiple transforms in a single object.

    Parameters
    ----------
    transform_list : list[NtoMTransform]
        List of transform objects to chain together.
        Note that the order can matter.

    Attributes
    ----------
    transform_list : list[NtoMTransform]
        Stored list of transforms
    """

    transform_list: list[NtoMTransform]
    counter: int

    def __init__(self, transform_list: list[NtoMTransform]) -> None:
        
        self.transform_list = transform_list

        self.fixed_params = {}
        for transform in self.transform_list:
            self.fixed_params.update(transform.fixed_params)

        name_mapping = (
            [
                x
                for transform in self.transform_list
                for x in transform.name_mapping[0]
            ],
            [
                x
                for transform in self.transform_list
                for x in transform.name_mapping[1]
            ],
        )

        super().__init__(name_mapping)

    def forward(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate combined transforms

        Parameters
        ----------
        params : dict[str, Float | Array]
            Parameter dictionary passed to all transforms

        Returns
        -------
        Dict
            All transformed parameters
        """

        result = dict()

        for transform in self.transform_list:
            result.update(transform.forward(params))

        return result
    
    def get_parameter_names(self,):

        parameter_names = []
        for transform in self.transform_list:
            parameter_names.extend(transform.get_parameter_names())

        return parameter_names


