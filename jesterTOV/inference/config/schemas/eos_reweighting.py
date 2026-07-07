r"""Pydantic model for EOS reweighting inference configuration.

This module defines :class:`EOSReweightingInferenceConfig`, a lightweight
configuration that replaces :class:`~jesterTOV.inference.config.schema.InferenceConfig`
when the sampler type is ``"eos-reweighting"``.  Unlike the standard config it
has **no** ``eos``, ``tov``, or ``prior`` sections, because the EOS is provided
as tabulated curves rather than generated from a parametric model.
"""

import os
from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator, ConfigDict

from ._base import JesterBaseModel
from .likelihoods import LikelihoodConfig
from .samplers import EOSReweightingConfig

#: Likelihood types (among :data:`_EOS_REWEIGHTING_ALLOWED_LIKELIHOOD_TYPES`)
#: that require the tabulated ``lambdas``/``radii`` curve respectively. Used
#: by :meth:`EOSReweightingInferenceConfig._validate_eos_file_has_required_curves`
#: to cross-check the EOS file's available keys against enabled likelihoods.
_LIKELIHOOD_TYPES_REQUIRING_LAMBDAS = {"gw"}
_LIKELIHOOD_TYPES_REQUIRING_RADII = {"nicer"}

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


class EOSReweightingPostprocessingConfig(JesterBaseModel):
    r"""Configuration for EOS reweighting postprocessing plots.

    EOS reweighting only produces resampled (M, :math:`\Lambda`, R) posterior
    curves — there is no NEP/CSE parameter posterior, density/pressure/cs2
    profile, or TOV central-density diagnostic to plot, so only the
    mass-radius and mass-Lambda plots are supported (unlike
    :class:`~jesterTOV.inference.config.schema.PostprocessingConfig`, which
    also covers cornerplots, pressure-density, cs2, histograms, and
    contours for the parametric samplers).

    Attributes
    ----------
    enabled : bool
        Whether to run postprocessing after inference (default: True)
    injection_eos_path : str | None
        Path to NPZ file containing injection EOS data for plotting
        (default: None). See
        :func:`~jesterTOV.inference.postprocessing.postprocessing.load_injection_eos`
        for the expected format.
    plot_format : {"pdf", "png"}
        Output file format for all plots (default: "pdf")
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    injection_eos_path: str | None = None
    plot_format: Literal["pdf", "png"] = "pdf"


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
    postprocessing : EOSReweightingPostprocessingConfig
        Postprocessing configuration (mass-radius/mass-Lambda plots only).
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
    postprocessing: EOSReweightingPostprocessingConfig = Field(
        default_factory=EOSReweightingPostprocessingConfig
    )
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

    @model_validator(mode="after")
    def _validate_eos_file_has_required_curves(self) -> "EOSReweightingInferenceConfig":
        """Check that the EOS file provides the curves the enabled likelihoods need.

        GW-type likelihoods read ``Lambdas_EOS`` and NICER-type likelihoods
        read ``radii_EOS`` (see
        :meth:`~jesterTOV.inference.samplers.eos_reweighting.EOSReweightingSampler.load_and_grid`);
        both are optional in the EOS NPZ file otherwise. If the file does
        not exist yet, this check is skipped and the (clearer)
        ``FileNotFoundError`` is left to surface when the file is actually
        loaded.
        """
        eos_file = self.sampler.eos_file
        if not os.path.exists(eos_file):
            return self

        needs_lambdas = any(
            lk.enabled and lk.type in _LIKELIHOOD_TYPES_REQUIRING_LAMBDAS
            for lk in self.likelihoods
        )
        needs_radii = any(
            lk.enabled and lk.type in _LIKELIHOOD_TYPES_REQUIRING_RADII
            for lk in self.likelihoods
        )
        if not needs_lambdas and not needs_radii:
            return self

        with np.load(eos_file) as data:
            available = set(data.files)

        missing = []
        if needs_lambdas and "lambdas" not in available:
            missing.append("'lambdas' (required by the enabled 'gw' likelihood)")
        if needs_radii and "radii" not in available:
            missing.append("'radii' (required by the enabled 'nicer' likelihood)")
        if missing:
            raise ValueError(
                f"EOS file '{eos_file}' is missing required curve(s): "
                f"{', '.join(missing)}."
            )
        return self
