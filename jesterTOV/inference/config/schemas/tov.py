"""Pydantic models for TOV solver configuration."""

from typing import Annotated, Literal, Union
from pydantic import ConfigDict, Discriminator, Field

from ._base import JesterBaseModel


class BaseTOVConfig(JesterBaseModel):
    """Base configuration shared by all TOV solvers.

    Attributes
    ----------
    type : str
        TOV solver type identifier (discriminator field)
    min_nsat_TOV : float
        Minimum central density for TOV integration (units of nsat, default: 0.75)
    ndat_TOV : int
        Number of data points for TOV integration (default: 100)
    nb_masses : int
        Number of masses to sample (default: 100)
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    min_nsat_TOV: float = Field(
        default=0.75,
        gt=0.0,
        description="Minimum central density for TOV integration in units of nsat",
    )
    ndat_TOV: int = Field(
        default=100,
        gt=0,
        description="Number of data points for TOV integration",
    )
    nb_masses: int = Field(
        default=100,
        gt=0,
        description="Number of masses to sample when constructing the M-R-Λ family",
    )


class GRTOVConfig(BaseTOVConfig):
    """Configuration for the standard General Relativity TOV solver.

    This is the default solver. No additional parameters beyond those
    in BaseTOVConfig are required.

    Attributes
    ----------
    type : Literal["gr"]
        TOV solver type identifier
    """

    type: Literal["gr"] = "gr"  # type: ignore[override]  # Literal["gr"] ⊂ str


class ScalarTensorTOVConfig(BaseTOVConfig):
    """Configuration for the Scalar-Tensor TOV solver.

    Attributes
    ----------
    type : Literal["scalar_tensor"]
        TOV solver type identifier
    beta_ST : float
        Scalar-tensor coupling parameter beta_ST (default: 0.0)
    phi_inf_tgt : float
        Target asymptotic scalar field at infinity (default: 1e-3)
    phi_c : float
        Central scalar field value (default: 1.0)
    calculate_tidal : bool
        Whether to compute tidal deformability (k2 and related quantities).
        Set to False to save computational resources when tidal calculations
        are not needed (e.g., for M‑R constraints only). (default: True)
    """

    type: Literal["scalar_tensor"] = "scalar_tensor"  # type: ignore[override]

    beta_ST: float = Field(
        default=0.0, description="Scalar-tensor coupling parameter beta_ST"
    )
    phi_inf_tgt: float = Field(
        default=1e-3, description="Target asymptotic scalar field at infinity"
    )
    phi_c: float = Field(default=1.0, description="Central scalar field value")
    calculate_tidal: bool = Field(
        default=True,
        description="Whether to compute tidal deformability (k2 and related quantities). "
        "Set to False to save computational resources when tidal calculations "
        "are not needed (e.g., for M‑R constraints only)."
    )


class AnisotropyTOVConfig(BaseTOVConfig):
    """Configuration for the post-TOV solver with beyond-GR corrections.

    The six theory parameters (``lambda_BL``, ``lambda_DY``, ``lambda_HB``,
    ``gamma``, ``alpha``, ``beta``) are specified in the prior file, not here.
    Any subset of them may be sampled; the rest default to the GR limit.

    Attributes
    ----------
    type : Literal["anisotropy"]
        TOV solver type identifier
    """

    type: Literal["anisotropy"] = "anisotropy"  # type: ignore[override]  # Literal["anisotropy"] ⊂ str


# TOVConfig is a discriminated union of all available TOV solver configs.
TOVConfig = Annotated[
    Union[GRTOVConfig, AnisotropyTOVConfig, ScalarTensorTOVConfig],
    Discriminator("type"),
]
