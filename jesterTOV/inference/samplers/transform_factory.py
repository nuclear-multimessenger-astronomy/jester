"""Transform factory for sampler-specific parameter space transforms.

This module creates appropriate transforms based on the sampler type:
- BlackJAX NS with Acceptance Walk: BoundToBound [0,1] unit cube transforms
- SMC: No transforms (works in prior space)
- FlowMC: No transforms (current behavior)
"""

from ..config.schema import SamplerConfig
from ..base.prior import Prior, CombinePrior, UniformPrior
from ..base.transform import BijectiveTransform, BoundToBound


def create_sample_transforms(
    sampler_config: SamplerConfig,
    prior: Prior,
) -> list[BijectiveTransform]:
    """Create sample transforms based on sampler type.

    Different samplers require different parameter space transformations:
    - BlackJAX NS-AW requires unit cube [0, 1] transforms for all parameters
    - SMC works best without transforms (in prior space)
    - FlowMC can optionally use unbounded transforms (currently none)

    Parameters
    ----------
    sampler_config : SamplerConfig
        Sampler configuration determining transform strategy (discriminated union).
    prior : Prior
        Prior distribution (must be CombinePrior of UniformPrior for NS).

    Returns
    -------
    list[BijectiveTransform]
        List of sample transforms to apply during sampling.

    Raises
    ------
    ValueError
        If nested sampling is used with non-uniform priors.
    """
    if sampler_config.type == "blackjax-ns-aw":
        # BlackJAX NS-AW requires unit cube transforms for acceptance walk algorithm
        return create_unit_cube_transforms(prior)
    elif sampler_config.type == "smc":
        # SMC works in prior space without transforms
        return []
    elif sampler_config.type == "flowmc":
        # FlowMC currently uses no sample transforms
        # (could optionally add BoundToUnbound here in future)
        return []
    else:
        raise ValueError(f"Unknown sampler type: {sampler_config.type}")


def create_unit_cube_transforms(prior: Prior) -> list[BijectiveTransform]:
    """Create BoundToBound [0,1] transforms for all prior parameters.

    This is required for BlackJAX nested sampling with acceptance walk,
    which samples in unit cube space and applies the inverse transform
    to evaluate in prior space.

    Parameters
    ----------
    prior : Prior
        Prior distribution (must be CombinePrior of UniformPrior).

    Returns
    -------
    list[BijectiveTransform]
        List containing single BoundToBound transform mapping all parameters to [0,1].

    Raises
    ------
    ValueError
        If prior is not a CombinePrior or contains non-uniform priors.

    Examples
    --------
    >>> from jesterTOV.inference.base.prior import CombinePrior, UniformPrior
    >>> prior = CombinePrior([
    ...     UniformPrior(150.0, 300.0, parameter_names=["K_sat"]),
    ...     UniformPrior(10.0, 200.0, parameter_names=["L_sym"]),
    ... ])
    >>> transforms = create_unit_cube_transforms(prior)
    >>> # transforms[0] maps K_sat: [150,300] → [0,1], L_sym: [10,200] → [0,1]
    """
    # Handle both single UniformPrior and CombinePrior
    if isinstance(prior, UniformPrior):
        # Wrap single prior in CombinePrior for uniform handling
        from ..base import CombinePrior as CombinePriorClass
        prior = CombinePriorClass([prior])
    elif not isinstance(prior, CombinePrior):
        raise ValueError(
            f"BlackJAX NS-AW requires UniformPrior or CombinePrior, got {type(prior).__name__}. "
            "Ensure your prior is a (combination of) UniformPrior distribution(s)."
        )

    # Extract bounds from all component priors
    original_lower = {}
    original_upper = {}
    param_names = []

    for component_prior in prior.base_prior:
        # Validate that each component is a UniformPrior
        if not isinstance(component_prior, UniformPrior):
            error_msg = (
                f"BlackJAX NS-AW requires UniformPrior components, got {type(component_prior).__name__}. "
                f"Parameter: {component_prior.parameter_names[0]}\n"
            )
            if isinstance(component_prior, CombinePrior):
                error_msg += (
                    "Hint: Nested CombinePrior detected. This likely means a CombinePrior was wrapped "
                    "in another CombinePrior. Instead of CombinePrior([prior1, prior2]), use "
                    "CombinePrior(prior1.base_prior + prior2.base_prior) to flatten the structure."
                )
            raise ValueError(error_msg)

        # Extract bounds (UniformPrior has xmin, xmax attributes)
        param_name = component_prior.parameter_names[0]
        param_names.append(param_name)
        original_lower[param_name] = component_prior.xmin
        original_upper[param_name] = component_prior.xmax

    # Create target bounds (unit cube [0, 1] for all parameters)
    target_lower = {name: 0.0 for name in param_names}
    target_upper = {name: 1.0 for name in param_names}

    # Create name mapping (same names in both spaces)
    name_mapping = (param_names, param_names)

    # Create single BoundToBound transform for all parameters
    transform = BoundToBound(
        name_mapping=name_mapping,
        original_lower_bound=original_lower,
        original_upper_bound=original_upper,
        target_lower_bound=target_lower,
        target_upper_bound=target_upper,
    )

    return [transform]
