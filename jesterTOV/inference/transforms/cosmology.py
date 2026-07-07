r"""Transform for cosmological parameters."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.random import PRNGKey

from jesterTOV.inference.base import NtoMTransform

from jesterTOV.utils import redshift_to_luminosity_distance

from jesterTOV.logging_config import get_logger
logger = get_logger("jester")

class CosmoJesterTransform(NtoMTransform):

    def __init__(self,
                 fixed_params: dict | None = None,
                 keep_names: list[str] | None = None,):

        if fixed_params is not None:
            self.fixed_params: dict[str, float] = fixed_params.copy()
        else:
            self.fixed_params = {}

        name_mapping = (["H0", "Omega0"], ["dL_fn_redshift_arr", "dL_fn_distance_arr"])

        # Set keep_names (default: all input names)
        if keep_names is None:
            keep_names = name_mapping[0]
        self.keep_names = keep_names

        # Initialize parent NtoMTransform
        super().__init__(
            name_mapping=name_mapping
        )
    
        self.transform_func = self.luminosity_distance_relation_from_cosmology
    
    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
        """Transform parameters and preserve keep_names.

        This overrides NtoMTransform.forward() to:

        1. Merge fixed parameters into ``x`` before the EOS/TOV pipeline runs.
        2. Preserve parameters specified in ``self.keep_names``.
        3. Add fixed parameters to the output so they appear in the result.

        Parameters
        ----------
        x : dict[str, Float]
            Input parameter dictionary (sampled parameters only)

        Returns
        -------
        dict[str, Float]
            Transformed parameters with keep_names and fixed_params included
        """
        # Inject fixed parameters so the EOS/TOV pipeline receives them.
        # Create a new dict to avoid mutating the caller's input.
        if self.fixed_params:
            x = {**x, **self.fixed_params}

        # Save parameters that should be kept
        kept_params = {name: x[name] for name in self.keep_names if name in x}

        # Call parent forward() for standard transformation
        result = super().forward(x)

        # Add back the kept parameters
        result.update(kept_params)

        # Add fixed parameters to the output for traceability
        if self.fixed_params:
            result.update(self.fixed_params)

        return result
    
    def luminosity_distance_relation_from_cosmology(   
        self,
        params: dict[str, Float],
    ) -> dict[str, Float | Float[Array, " n"]]:
        
        z_arr = jnp.linspace(0, 5, 1000)
        dL_arr = redshift_to_luminosity_distance(z_arr, params["H0"], params["Omega0"])

        return dict(dL_fn_redshift_arr=z_arr, dL_fn_distance_arr=dL_arr)
    
    def get_parameter_names(self):
        return ["H0", "Omega0"]