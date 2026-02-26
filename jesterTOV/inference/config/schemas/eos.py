"""Pydantic models for EOS configuration."""

from typing import Literal, Union, Annotated
from pydantic import BaseModel, field_validator, ConfigDict, Discriminator


class BaseEOSConfig(BaseModel):
    """Base configuration shared by all EOS types.

    Attributes
    ----------
    crust_name : Literal["DH", "BPS", "DH_fixed", "SLy"]
        Name of crust model to use (default: "DH")
    """

    model_config = ConfigDict(extra="forbid")

    crust_name: Literal["DH", "BPS", "DH_fixed", "SLy"] = "DH"


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
    def validate_nb_cse(cls, v: int) -> int:
        """Validate that nb_CSE is 0 for standard metamodel."""
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
    def validate_nb_cse(cls, v: int) -> int:
        """Validate that nb_CSE is positive for metamodel_cse."""
        if v <= 0:
            raise ValueError(
                "nb_CSE must be > 0 for type='metamodel_cse'. "
                "Use type='metamodel' for standard metamodel without CSE."
            )
        return v


class SpectralEOSConfig(BaseEOSConfig):
    """Configuration for Spectral Decomposition EOS.

    Attributes
    ----------
    type : Literal["spectral"]
        EOS type identifier
    n_points_high : int
        Number of high-density points for spectral EOS (default: 500)
    nb_CSE : int
        Must be 0 for spectral (no CSE support)
    """

    type: Literal["spectral"] = "spectral"
    n_points_high: int = 500
    nb_CSE: int = 0

    @field_validator("nb_CSE")
    @classmethod
    def validate_nb_cse(cls, v: int) -> int:
        """Validate that nb_CSE is 0 for spectral."""
        if v != 0:
            raise ValueError(
                "nb_CSE must be 0 for type='spectral'. "
                "CSE extension not supported for spectral EOS."
            )
        return v


class PiecewisePolytropeEOSConfig(BaseEOSConfig):
    """Configuration for the piecewise polytrope EOS (Read et al. 2009).

    The EOS is parametrised by four parameters: the log₁₀ of the pressure (in
    Pa) at the reference density 10^17.7 kg/m³, and three adiabatic indices
    for the high-density core.  The low-density crust uses the analytical SLy4
    4-piece fit from the LALSuite implementation.

    Attributes
    ----------
    type : Literal["piecewise_polytrope"]
        EOS type identifier.
    n_points : int
        Number of log-spaced pressure grid points (default: 500).
    nb_CSE : int
        Must be 0; CSE extension is not supported for this parametrisation.
    crust_name : str
        Ignored for this EOS (the SLy4 crust is built in analytically).
        Kept for interface compatibility; default ``"SLy"``.
    """

    type: Literal["piecewise_polytrope"] = "piecewise_polytrope"
    n_points: int = 500
    nb_CSE: int = 0
    crust_name: Literal["DH", "BPS", "DH_fixed", "SLy"] = "SLy"

    @field_validator("nb_CSE")
    @classmethod
    def validate_nb_cse(cls, v: int) -> int:
        """Validate that nb_CSE is 0 for piecewise polytrope."""
        if v != 0:
            raise ValueError(
                "nb_CSE must be 0 for type='piecewise_polytrope'. "
                "CSE extension not supported for piecewise polytrope EOS."
            )
        return v


# Discriminated union of all EOS types
EOSConfig = Annotated[
    Union[
        MetamodelEOSConfig,
        MetamodelCSEEOSConfig,
        SpectralEOSConfig,
        PiecewisePolytropeEOSConfig,
    ],
    Discriminator("type"),
]
