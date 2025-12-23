"""Combined and utility likelihood classes"""

import jax.numpy as jnp
from jaxtyping import Float
from jesterTOV.inference.base import LikelihoodBase


class CombinedLikelihood(LikelihoodBase):
    """Combine multiple likelihoods into a single log-likelihood sum"""

    def __init__(self, likelihoods_list: list[LikelihoodBase]):
        super().__init__()
        self.likelihoods_list = likelihoods_list
        self.counter = 0

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        all_log_likelihoods = jnp.array(
            [likelihood.evaluate(params, data) for likelihood in self.likelihoods_list]
        )
        return jnp.sum(all_log_likelihoods)


class ZeroLikelihood(LikelihoodBase):
    """Placeholder likelihood that always returns 0 (for testing/debugging)"""

    def __init__(self):
        super().__init__()
        self.counter = 0

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        return 0.0
