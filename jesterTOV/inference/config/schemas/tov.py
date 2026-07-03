"""Pydantic models for TOV solver configuration."""

from typing import Annotated, Literal, Union
from pydantic import ConfigDict, Discriminator, Field

from ._base import JesterBaseModel


class OdeConfig(JesterBaseModel):
    """Configuration for the ODE backend used to integrate the TOV equations.

    Shared/embedded (not a discriminated union) so every TOV solver config
    inherits it for free without duplicating fields. ``algorithm`` is
    validated against the selected backend's supported algorithm set at
    solver-construction time (in ``_create_tov_solver``), not via a Pydantic
    ``Literal``, so this config stays agnostic to which backends exist.

    Attributes
    ----------
    backend : Literal["diffrax", "modax"]
        ODE integration library to use (default: "diffrax")
    algorithm : str
        Backend-specific integrator name, e.g. "Dopri5", "Tsit5", "Rodas5P"
        (default: "Dopri5")
    rtol : float
        Relative tolerance for adaptive step-size control (default: 1e-5)
    atol : float
        Absolute tolerance for adaptive step-size control (default: 1e-6)
    max_steps : int
        Maximum number of integration steps (default: 4096, matching
        diffrax's own default)
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["diffrax", "modax"] = "diffrax"
    algorithm: str = Field(
        default="Dopri5",
        description="Backend-specific integrator name (e.g. Dopri5, Tsit5, Rodas5P)",
    )
    rtol: float = Field(default=1e-5, gt=0.0, description="Relative tolerance")
    atol: float = Field(default=1e-6, gt=0.0, description="Absolute tolerance")
    max_steps: int = Field(
        default=4096, gt=0, description="Maximum number of integration steps"
    )


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
    ode : OdeConfig
        ODE backend/algorithm/tolerance configuration for the TOV integration
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
    ode: OdeConfig = Field(
        default_factory=OdeConfig,
        description="ODE backend/algorithm/tolerance configuration",
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
