"""
TOV (Tolman-Oppenheimer-Volkoff) solver module.

This module contains TOV equation solvers for various theories of gravity:
- General Relativity (GR)
- Post-TOV with beyond-GR corrections
- Scalar-tensor theories

All solvers work with modular EOS representations via EOSData.
"""

from jesterTOV.tov.data_classes import EOSData, TOVSolution, FamilyData
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.gr import GRTOVSolver
from jesterTOV.tov.post import PostTOVSolver
from jesterTOV.tov.scalar_tensor import ScalarTensorTOVSolver

__all__ = [
    "EOSData",
    "TOVSolution",
    "FamilyData",
    "TOVSolverBase",
    "GRTOVSolver",
    "PostTOVSolver",
    "ScalarTensorTOVSolver",
]
