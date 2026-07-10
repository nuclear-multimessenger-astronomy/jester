"""Neural variational inference (NVI) for jester.

Fits a normalizing flow directly to ``prior x likelihood`` via reverse-KL minimization
(:func:`train_vi`), and supports sequential Bayesian updates as new data arrives
(:func:`sequential_vi_update`), following papers 2504.07197 and 2602.20277 (see
``./papers/`` at the project root) and the reference implementation in
``gwax/gwax/variational.py``.

This is a standalone library module, not yet wired into jester's
``SAMPLER_REGISTRY``/YAML config system — see the implementation plan referenced from
the project root ``CLAUDE.md`` for the staged rollout.
"""

from jesterTOV.inference.variational.diagnostics import (
    compare_posteriors,
    importance_diagnostics,
)
from jesterTOV.inference.variational.targets import (
    make_flow_prior_target,
    make_prior_likelihood_target,
)
from jesterTOV.inference.variational.trainer import (
    ELBOTrainingResult,
    sequential_vi_update,
    train_vi,
)

__all__ = [
    "ELBOTrainingResult",
    "compare_posteriors",
    "importance_diagnostics",
    "make_flow_prior_target",
    "make_prior_likelihood_target",
    "sequential_vi_update",
    "train_vi",
]
