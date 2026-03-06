r"""Unified neutron star crust equation of state.

Provides a thermodynamically consistent EOS where the crust is derived from the
same nuclear empirical parameters (NEPs) as the core, rather than a fixed
pre-tabulated table.

Two implementations are available:

- :class:`UnifiedCrustEOS_MetaModel`: JAX-native, suitable for Bayesian inference.
  Extends the existing metamodel to inner crust densities; outer crust uses a fixed
  BSk22/BSk24 Pearson fit.
- :class:`UnifiedCrustEOS_CUTER`: Subprocess wrapper around the CUTER Fortran/C
  backend. **Not for inference** — use for cross-checking and validation only.
"""

from jesterTOV.eos.unified_crust.unified_crust import UnifiedCrustEOS_MetaModel
from jesterTOV.eos.unified_crust.cuter_wrapper import UnifiedCrustEOS_CUTER

__all__ = [
    "UnifiedCrustEOS_MetaModel",
    "UnifiedCrustEOS_CUTER",
]
