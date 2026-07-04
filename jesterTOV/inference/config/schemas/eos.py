"""Pydantic models for EOS configuration."""

from typing import Literal, Union, Annotated
from pydantic import field_validator, ConfigDict, Discriminator

from ._base import JesterBaseModel


class BaseEOSConfig(JesterBaseModel):
    """Base configuration shared by all EOS types.

    Attributes
    ----------
    crust_name : Literal["DH", "BPS", "SLy", "Tolos"]
        Name of crust model to use (default: "DH"). "Tolos" (Tolos, Centelles & Ramos
        2017 outer-crust table) is the crust the RMF EOS was validated against in its
        reference implementation (CompactObject) -- see :class:`RMFEOSConfig`.
    """

    model_config = ConfigDict(extra="forbid")

    crust_name: Literal["DH", "BPS", "SLy", "Tolos"] = "DH"


class BaseMetamodelEOSConfig(BaseEOSConfig):
    """Base configuration shared by all MetaModel-based EOS types.

    Holds the grid parameters that control the metamodel density grid.
    This base class is inherited by :class:`MetamodelEOSConfig` and
    :class:`MetamodelCSEEOSConfig` but not by the spectral EOS, which
    has a different parameterization.

    Attributes
    ----------
    ndat_metamodel : int
        Number of data points for MetaModel EOS grid (default: 100)
    nmax_nsat : float
        Maximum density in units of saturation density (default: 25.0)
    nmin_MM_nsat : float
        Starting density for metamodel grid as fraction of nsat (default: 0.75)
    """

    ndat_metamodel: int = 100
    nmax_nsat: float = 25.0
    nmin_MM_nsat: float = 0.75


class MetamodelEOSConfig(BaseMetamodelEOSConfig):
    """Configuration for MetaModel EOS (without CSE).

    Attributes
    ----------
    type : Literal["metamodel"]
        EOS type identifier
    nb_CSE : int
        Must be 0 for standard metamodel (no CSE extension)
    """

    type: Literal["metamodel"] = "metamodel"
    nb_CSE: int = 0

    @field_validator("nb_CSE")
    @classmethod
    def _validate_nb_cse(cls, v: int) -> int:
        if v != 0:
            raise ValueError(
                "nb_CSE must be 0 for type='metamodel'. "
                "Use type='metamodel_cse' for CSE extension."
            )
        return v


class MetamodelCSEEOSConfig(BaseMetamodelEOSConfig):
    """Configuration for MetaModel with CSE extension.

    Attributes
    ----------
    type : Literal["metamodel_cse"]
        EOS type identifier
    nb_CSE : int
        Number of CSE parameters (must be > 0, typically 4-8)
    ndat_CSE : int
        Number of density grid points for the CSE region (default: 100)
    max_nbreak_nsat : float | None
        Maximum allowed breaking density in units of nsat (default: None,
        meaning no upper bound beyond the prior). If specified, this must
        be consistent with the upper bound of the ``nbreak`` prior; an
        error is raised if they disagree.
    """

    type: Literal["metamodel_cse"] = "metamodel_cse"
    nb_CSE: int = 8
    ndat_CSE: int = 100
    max_nbreak_nsat: float | None = None

    @field_validator("nb_CSE")
    @classmethod
    def _validate_nb_cse(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(
                "nb_CSE must be > 0 for type='metamodel_cse'. "
                "Use type='metamodel' for standard metamodel without CSE."
            )
        return v


class MetamodelPeakCSEEOSConfig(BaseMetamodelEOSConfig):
    """Configuration for MetaModel with peakCSE extension.

    Attributes
    ----------
    type : Literal["metamodel_peak_cse"]
        EOS type identifier
    ndat_CSE : int
        Number of density grid points for the peakCSE region (default: 100)
    max_nbreak_nsat : float | None
        Maximum allowed breaking density in units of nsat (default: None,
        meaning no upper bound beyond the prior). If specified, the metamodel
        grid is only computed up to this density, which can speed up inference.
    """

    type: Literal["metamodel_peak_cse"] = "metamodel_peak_cse"
    ndat_CSE: int = 100
    max_nbreak_nsat: float | None = None


class SpectralEOSConfig(BaseEOSConfig):
    r"""Configuration for Spectral Decomposition EOS.

    Attributes
    ----------
    type : Literal["spectral"]
        EOS type identifier
    n_points_high : int
        Number of high-density points for spectral EOS (default: 500)
    nb_CSE : int
        Must be 0 for spectral (no CSE support)
    reparametrized : bool
        If False (default), sample directly in :math:`(\gamma_0, \gamma_1, \gamma_2, \gamma_3)`.
        If True, sample in a whitened space :math:`(\tilde{\gamma}_0, \tilde{\gamma}_1, \tilde{\gamma}_2, \tilde{\gamma}_3)`
        centred on a Gaussian fit to a radio-timing inference result.  The bijection
        :math:`\boldsymbol{\gamma} = \boldsymbol{\mu} + L_\text{wide}\,\tilde{\boldsymbol{\gamma}}` maps the
        unit-normal tilde parameters back to physical spectral coefficients, where
        :math:`L_\text{wide} = \sigma_\text{scale}\,L` and :math:`\boldsymbol{\mu}` is
        the posterior mean.  Use a ``MultivariateGaussianPrior`` with default (unit)
        parameters in the prior file when this option is enabled.
    sigma_scale : float
        Multiplicative factor applied to the base Cholesky factor :math:`L` to form
        :math:`L_\text{wide} = \sigma_\text{scale}\,L`.  Only used when
        ``reparametrized=True``.  Default 1.0 (exact radio posterior covariance).
        Increase to widen the prior around the radio posterior.
    """

    type: Literal["spectral"] = "spectral"
    n_points_high: int = 500
    nb_CSE: int = 0
    reparametrized: bool = False
    sigma_scale: float = 1.0

    @field_validator("nb_CSE")
    @classmethod
    def _validate_nb_cse(cls, v: int) -> int:
        if v != 0:
            raise ValueError(
                "nb_CSE must be 0 for type='spectral'. "
                "CSE extension not supported for spectral EOS."
            )
        return v


class RMFEOSConfig(BaseEOSConfig):
    r"""Configuration for the relativistic mean-field (RMF) EOS.

    Attributes
    ----------
    type : Literal["rmf"]
        EOS type identifier
    crust_name : Literal["DH", "BPS", "SLy", "Tolos"]
        Crust model. Defaults to "Tolos" for this EOS type specifically (overriding
        :class:`BaseEOSConfig`'s "DH" default), matching the crust this RMF
        parametrization was fit and validated against in its reference implementation
        (CompactObject) -- joining it to "DH" instead produces a real pressure
        discontinuity at the crust-core junction.
    rho_0 : float
        RMF model's intrinsic nuclear saturation density [:math:`\mathrm{fm}^{-3}`]
        (default: 0.1505, matches the reference implementation).
    dt : float
        Density grid step size, in units of ``rho_0`` (default: 0.05).
    ndat_full : int
        Number of raw density grid points for the warm-started field continuation
        (default: 124, matches the reference implementation).
    min_core_nsat_rmf : float
        Minimum density the RMF core table may start at, as a fraction of ``rho_0``
        (default: 1.0). This nucleonic RMF model produces unphysical negative pressure
        well below saturation density.
    newton_max_steps : int
        Maximum Levenberg-Marquardt iterations per density point (default: 200).
    root_rtol : float
        Relative tolerance for the field-equation root solve (default: 1e-10).
    root_atol : float
        Absolute tolerance for the field-equation root solve (default: 1e-10).
    """

    type: Literal["rmf"] = "rmf"
    crust_name: Literal["DH", "BPS", "SLy", "Tolos"] = "Tolos"
    rho_0: float = 0.1505
    dt: float = 0.05
    ndat_full: int = 124
    min_core_nsat_rmf: float = 1.0
    newton_max_steps: int = 200
    root_rtol: float = 1e-10
    root_atol: float = 1e-10


class MITBagEOSConfig(BaseEOSConfig):
    r"""Configuration for the MIT bag model EOS (self-bound strange quark matter).

    Attributes
    ----------
    type : Literal["mit_bag"]
        EOS type identifier
    crust_name : Literal["DH", "BPS", "SLy", "Tolos"]
        Unused for this EOS type: strange quark matter is self-bound (no crust),
        so this field is accepted for interface consistency but ignored. Kept at
        the :class:`BaseEOSConfig` default.
    n_points : int
        Number of energy-density grid points (default: 1000, matches the reference
        implementation).
    e_max_over_b : float
        Upper end of the energy-density grid, in units of the bag constant
        (default: 10.0, matches the reference implementation).
    """

    type: Literal["mit_bag"] = "mit_bag"
    n_points: int = 1000
    e_max_over_b: float = 10.0


class StrangeonEOSConfig(BaseEOSConfig):
    r"""Configuration for the strangeon matter EOS (self-bound strangeon stars).

    Attributes
    ----------
    type : Literal["strangeon"]
        EOS type identifier
    crust_name : Literal["DH", "BPS", "SLy", "Tolos"]
        Unused for this EOS type: strangeon matter is self-bound (no crust), so
        this field is accepted for interface consistency but ignored. Kept at the
        :class:`BaseEOSConfig` default.
    Nq : int
        Number of quarks per strangeon (default: 18, matches the reference
        implementation's own inference examples). Fixed model-selection choice,
        not sampled.
    n_points : int
        Number of strangeon-density grid points (default: 2000, matches the
        reference implementation).
    """

    type: Literal["strangeon"] = "strangeon"
    Nq: int = 18
    n_points: int = 2000


# Discriminated union of all EOS types
EOSConfig = Annotated[
    Union[
        MetamodelEOSConfig,
        MetamodelCSEEOSConfig,
        MetamodelPeakCSEEOSConfig,
        SpectralEOSConfig,
        RMFEOSConfig,
        MITBagEOSConfig,
        StrangeonEOSConfig,
    ],
    Discriminator("type"),
]
