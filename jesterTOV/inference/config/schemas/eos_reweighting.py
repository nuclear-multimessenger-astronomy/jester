r"""Pydantic model for EOS reweighting inference configuration.

This module defines :class:`EOSReweightingInferenceConfig`, a lightweight
configuration that replaces :class:`~jesterTOV.inference.config.schema.InferenceConfig`
when the sampler type is ``"eos-reweighting"``.  Unlike the standard config it
has **no** ``eos``, ``tov``, or ``prior`` sections, because the EOS is provided
as tabulated curves rather than generated from a parametric model.
"""

from pydantic import Field, field_validator, ConfigDict

from ._base import JesterBaseModel
from .likelihoods import LikelihoodConfig
from .samplers import EOSReweightingConfig


class EOSReweightingInferenceConfig(JesterBaseModel):
    r"""Top-level configuration for EOS reweighting inference.

    Used when ``sampler.type: "eos-reweighting"``.  The EOS is provided as a
    set of tabulated M--:math:`\Lambda`--R curves rather than sampled from a
    parametric model, so ``eos``, ``tov``, and ``prior`` fields are absent.

    Attributes
    ----------
    seed : int
        Random seed for reproducibility (default: 43)
    likelihoods : list[LikelihoodConfig]
        Likelihood configurations.  GW events must specify ``nf_model_dir``
        (pre-trained flow) or a built-in preset name.  Flow training from
        bilby results is not supported in this mode.
    sampler : EOSReweightingConfig
        EOS reweighting sampler configuration including EOS file paths.
    debug_nans : bool
        Enable JAX NaN debugging (default: False)

    Examples
    --------
    A minimal YAML config::

        seed: 42

        likelihoods:
          - type: gw
            events:
              - name: GW170817
          - type: nicer
            sources: [J0740]
          - type: radio
            database: FIDUCEO2

        sampler:
          type: eos-reweighting
          eos_file: path/to/eos.npz
          batch_size: 50
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = 43
    likelihoods: list[LikelihoodConfig]
    sampler: EOSReweightingConfig
    debug_nans: bool = Field(
        default=False,
        description="Enable JAX NaN debugging for catching numerical issues",
    )
    dry_run: bool = Field(
        default=False,
        description="Validate config and set up likelihoods without running evaluation",
    )

    @field_validator("likelihoods")
    @classmethod
    def _validate_likelihoods(cls, v: list[LikelihoodConfig]) -> list[LikelihoodConfig]:
        if not any(lk.enabled for lk in v):
            raise ValueError("At least one likelihood must be enabled")
        # TODO: validate that no incompatible likelihood types (chieft, rex,
        # eos_constraints, tov_constraints, gamma_constraints) are enabled —
        # these require EOS structure (n, p, nbreak, ...) that is not available
        # from tabulated M-Λ-R curves and will fail at sampling time with a
        # KeyError.  Add a clear validation error here once the full set of
        # incompatible types is confirmed.
        return v

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Seed must be non-negative, got: {v}")
        return v
