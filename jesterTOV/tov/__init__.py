"""TOV (Tolman-Oppenheimer-Volkoff) solver module.

This module contains TOV equation solvers for various theories of gravity:
- General Relativity (GR)
- Post-TOV with beyond-GR corrections
- Scalar-tensor theories

All solvers work with modular EOS representations via EOSData.

Import classes from their specific modules:

    from jesterTOV.tov.data_classes import EOSData, TOVSolution, FamilyData
    from jesterTOV.tov.base import TOVSolverBase
    from jesterTOV.tov.gr import GRTOVSolver
    from jesterTOV.tov.anisotropy import PostTOVSolver
    from jesterTOV.tov.scalar_tensor import ScalarTensorTOVSolver
    
    from jesterTOV.tov.data_classes import EOSData, TOVSolution, FamilyData
    from jesterTOV.tov.scalar_tensor_Creci import ScalarTensorTOVSolver_Creci
    from jesterTOV.tov.eibi import EiBITOVSolver
"""
