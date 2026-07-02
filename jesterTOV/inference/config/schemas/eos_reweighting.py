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

#: Likelihood types that only require the tabulated M-Λ-R family curves
#: (via "masses_EOS", "Lambdas_EOS", "radii_EOS") produced by the
#: reweighting sampler. Every other likelihood type reads EOS-level
#: structure (e.g. "n", "p", "nbreak", "_random_key") that only exists when
#: the EOS is built from a parametric model, and would raise a KeyError here.
_EOS_REWEIGHTING_ALLOWED_LIKELIHOOD_TYPES = {
    "gw",
    "nicer",
    "radio",
    "zero",
}


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
        # TODO: EOS-based likelihoods (gw_resampled, nicer_kde, chieft,
        # constraints_eos, constraints_tov, constraints_esym,
        # constraints_gamma, rex) require EOS-level structure
        # (n, p, nbreak, _random_key, ...) that is not available from
        # tabulated M-Λ-R curves. Check if we can include these likelihoods
        # in the future, e.g. by also tabulating the underlying EOS
        # quantities.
        invalid = [
            lk
            for lk in v
            if lk.enabled and lk.type not in _EOS_REWEIGHTING_ALLOWED_LIKELIHOOD_TYPES
        ]
        if invalid:
            bad_types = sorted({lk.type for lk in invalid})
            raise ValueError(
                f"Likelihood types {bad_types} are not supported by EOS reweighting: "
                "they require EOS-level structure (n, p, nbreak, _random_key, ...) "
                "that is not available from tabulated M-Λ-R curves. "
                f"Supported types are: {sorted(_EOS_REWEIGHTING_ALLOWED_LIKELIHOOD_TYPES)}."
            )
        return v

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Seed must be non-negative, got: {v}")
        return v
