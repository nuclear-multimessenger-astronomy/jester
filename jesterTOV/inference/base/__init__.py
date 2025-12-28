"""Base classes for JESTER inference system.

These classes were originally from Jim (jimgw v0.2.0) and are copied here
to remove the dependency on jimgw. This allows JESTER to have full control
over these interfaces and maintain stability across jimgw version changes.

Original source: https://github.com/ThibeauWouters/jim
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
