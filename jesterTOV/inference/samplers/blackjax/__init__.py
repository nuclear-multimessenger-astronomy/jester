"""BlackJAX sampler implementations for JESTER inference.

This package provides BlackJAX-based sampling algorithms organized by type:
- smc: Sequential Monte Carlo with adaptive tempering
- nested_sampling: Nested sampling with acceptance walk

All BlackJAX samplers inherit from BlackjaxSampler base class.
"""

from .base import BlackjaxSampler

__all__ = ["BlackjaxSampler"]
