"""Sampler wrappers for jesterTOV inference

This module provides sampler implementations for Bayesian inference.
All samplers inherit from JesterSampler base class.
"""

from .jester_sampler import JesterSampler, SamplerOutput
from .flowmc import FlowMCSampler
from .blackjax.nested_sampling import BlackJAXNSAWSampler
from .blackjax.smc.random_walk import BlackJAXSMCRandomWalkSampler
from .blackjax.smc.nuts import BlackJAXSMCNUTSSampler

from ..base import LikelihoodBase, Prior, NtoMTransform
from ..config.schema import SamplerConfig

__all__ = [
    "JesterSampler",
    "SamplerOutput",
    "FlowMCSampler",
    "BlackJAXNSAWSampler",
    "BlackJAXSMCRandomWalkSampler",
    "BlackJAXSMCNUTSSampler",
    "create_sampler",
]

# Sampler registry: maps config.type to sampler class
SAMPLER_REGISTRY = {
    "flowmc": FlowMCSampler,
    "blackjax-ns-aw": BlackJAXNSAWSampler,
    "smc-rw": BlackJAXSMCRandomWalkSampler,
    "smc-nuts": BlackJAXSMCNUTSSampler,
}


def create_sampler(
    config: SamplerConfig,
    prior: Prior,
    likelihood: LikelihoodBase,
    likelihood_transforms: list[NtoMTransform] | None = None,
    seed: int = 0,
) -> JesterSampler:
    """Create sampler instance based on config.type.

    This factory function dispatches to the appropriate sampler implementation
    based on the sampler type specified in the configuration.

    Parameters
    ----------
    config : SamplerConfig
        Sampler configuration (discriminated union by type field)
    prior : Prior
        Prior distribution
    likelihood : LikelihoodBase
        Likelihood object
    likelihood_transforms : list[NtoMTransform], optional
        N-to-M transforms applied before likelihood evaluation.
        If None, defaults to empty list.
    seed : int, optional
        Random seed (default: 0)

    Returns
    -------
    JesterSampler
        Sampler instance (FlowMCSampler, BlackJAXNSAWSampler, or BlackJAXSMC*Sampler)

    Raises
    ------
    ValueError
        If sampler type is unknown

    Notes
    -----
    Sample transforms (unit cube transforms for NS-AW) are created automatically
    by each sampler's __init__ method, so we don't need to handle them here.

    Examples
    --------
    >>> from jesterTOV.inference.config.schema import FlowMCSamplerConfig
    >>> config = FlowMCSamplerConfig(type="flowmc", n_chains=20)
    >>> sampler = create_sampler(config, prior, likelihood, [transform])
    >>> sampler.sample(jax.random.PRNGKey(42))
    """
    if likelihood_transforms is None:
        likelihood_transforms = []

    # Look up sampler class in registry
    sampler_class = SAMPLER_REGISTRY.get(config.type)
    if sampler_class is None:
        raise ValueError(
            f"Unknown sampler type: {config.type}. "
            f"Available: {sorted(SAMPLER_REGISTRY.keys())}"
        )

    # Each sampler creates its own sample_transforms in __init__:
    # - NS-AW: creates unit cube [0,1] transforms if not provided
    # - FlowMC/SMC: use empty list (no transforms needed)
    return sampler_class(
        likelihood=likelihood,
        prior=prior,
        sample_transforms=[],  # Samplers handle their own transforms
        likelihood_transforms=likelihood_transforms,
        config=config,
        seed=seed,
    )
