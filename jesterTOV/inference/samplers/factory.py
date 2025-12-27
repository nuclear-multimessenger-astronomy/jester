"""Sampler factory for creating sampler instances based on configuration.

This module provides a central factory function that creates the appropriate
sampler instance based on the configuration type (flowmc, nested_sampling, or smc).
"""

from typing import Union

from ..base import LikelihoodBase, Prior, NtoMTransform
from ..config.schema import (
    FlowMCSamplerConfig,
    BlackJAXNSAWConfig,
    SMCSamplerConfig,
)
from .transform_factory import create_sample_transforms
from .jester_sampler import JesterSampler
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


def create_sampler(
    config: Union[FlowMCSamplerConfig, BlackJAXNSAWConfig, SMCSamplerConfig],
    prior: Prior,
    likelihood: LikelihoodBase,
    likelihood_transforms: list[NtoMTransform],
    seed: int = 0,
) -> JesterSampler:
    """Create sampler instance based on config.type.

    This factory function automatically creates appropriate sample_transforms
    for each sampler type and dispatches to the correct sampler implementation.

    Parameters
    ----------
    config : Union[FlowMCSamplerConfig, BlackJAXNSAWConfig, SMCSamplerConfig]
        Sampler configuration (discriminated by type field)
    prior : Prior
        Prior distribution
    likelihood : LikelihoodBase
        Likelihood object
    likelihood_transforms : list[NtoMTransform]
        N-to-M transforms applied before likelihood evaluation
    seed : int, optional
        Random seed (default: 0)

    Returns
    -------
    JesterSampler
        Sampler instance (FlowMCSampler, BlackJAXNSSampler, or BlackJAXSMCSampler)

    Raises
    ------
    ValueError
        If sampler type is unknown

    Examples
    --------
    >>> from jesterTOV.inference.config.schema import FlowMCSamplerConfig
    >>> config = FlowMCSamplerConfig(type="flowmc", n_chains=20)
    >>> sampler = create_sampler(config, prior, likelihood, [transform])
    >>> sampler.sample(jax.random.PRNGKey(42))
    """
    logger.info(f"Creating {config.type} sampler...")

    # Create sample transforms based on sampler type
    # - blackjax-ns-aw: BoundToBound [0,1] for all parameters
    # - smc: No transforms (empty list)
    # - flowmc: No transforms (current behavior)
    sample_transforms = create_sample_transforms(config, prior)

    logger.debug(f"Created {len(sample_transforms)} sample transforms for {config.type}")

    # Dispatch to appropriate sampler implementation
    if config.type == "flowmc":
        from .flowmc import FlowMCSampler

        return FlowMCSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=sample_transforms,
            likelihood_transforms=likelihood_transforms,
            config=config,
            seed=seed,
        )

    elif config.type == "blackjax-ns-aw":
        from .blackjax_ns_aw import BlackJAXNSAWSampler

        return BlackJAXNSAWSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=sample_transforms,
            likelihood_transforms=likelihood_transforms,
            config=config,
            seed=seed,
        )

    elif config.type == "smc":
        from .blackjax_smc import BlackJAXSMCSampler

        return BlackJAXSMCSampler(
            likelihood=likelihood,
            prior=prior,
            sample_transforms=sample_transforms,
            likelihood_transforms=likelihood_transforms,
            config=config,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown sampler type: {config.type}. "
            f"Expected one of: flowmc, blackjax-ns-aw, smc"
        )
