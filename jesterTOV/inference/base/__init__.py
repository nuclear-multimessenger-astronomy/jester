"""Base classes for JESTER inference system.

These classes were originally from Jim (jimgw v0.2.0) and are copied here
to remove the dependency on jimgw.
"""

from .likelihood import LikelihoodBase
from .prior import Prior, CombinePrior, UniformPrior
from .transform import Transform, NtoMTransform, BijectiveTransform

__all__ = [
    "LikelihoodBase",
    "Prior",
    "CombinePrior",
    "UniformPrior",
    "Transform",
    "NtoMTransform",
    "BijectiveTransform",
]
