r"""Piecewise polytrope equation of state parametrization.

Implements the Read et al. (2009) 4-parameter piecewise polytrope EOS,
with an analytical SLy4 crust stitched to a 3-piece high-density core.

Import
------
    from jesterTOV.eos.piecewise_polytrope import PiecewisePolytrope_EOS_model
"""

from jesterTOV.eos.piecewise_polytrope.piecewise_polytrope import (
    PiecewisePolytrope_EOS_model,
)

__all__ = ["PiecewisePolytrope_EOS_model"]
