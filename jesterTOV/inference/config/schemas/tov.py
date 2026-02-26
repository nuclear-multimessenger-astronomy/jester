"""Pydantic models for TOV solver configuration."""

from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Discriminator, Field


class BaseTOVConfig(BaseModel):
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


TOVConfig = Annotated[
    Union[GRTOVConfig, AnisotropyTOVConfig],
    Discriminator("type"),
]
