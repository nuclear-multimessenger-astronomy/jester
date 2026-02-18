"""Pydantic models for TOV solver configuration."""

from typing import Literal
from pydantic import BaseModel, ConfigDict

# TOVConfig is a type alias for the discriminated union of all TOV solver configs.
# Currently only GRTOVConfig exists; extend the union when adding AnisotropyTOVConfig,
# ScalarTensorTOVConfig, etc., and switch to:
#
#   TOVConfig = Annotated[
#       Union[GRTOVConfig, AnisotropyTOVConfig, ScalarTensorTOVConfig, ...],
#       Discriminator("tov_solver"),
#   ]


class BaseTOVConfig(BaseModel):
    """Base configuration shared by all TOV solvers.

    Attributes
    ----------
    tov_solver : str
        TOV solver type identifier (discriminator field)
    min_nsat_TOV : float
        Minimum central density for TOV integration (units of nsat, default: 0.75)
    ndat_TOV : int
        Number of data points for TOV integration (default: 100)
    nb_masses : int
        Number of masses to sample (default: 100)
    """

    model_config = ConfigDict(extra="forbid")

    tov_solver: str
    min_nsat_TOV: float = 0.75
    ndat_TOV: int = 100
    nb_masses: int = 100


class GRTOVConfig(BaseTOVConfig):
    """Configuration for the standard General Relativity TOV solver.

    This is the default solver. No additional parameters beyond those
    in BaseTOVConfig are required.

    Attributes
    ----------
    tov_solver : Literal["gr"]
        TOV solver type identifier
    """

    tov_solver: Literal["gr"] = "gr"  # type: ignore[override]  # Literal["gr"] ⊂ str


# TOVConfig is currently just GRTOVConfig since it is the only solver with a config.
# Replace this alias with the Annotated discriminated union (see comment at top of file)
# once AnisotropyTOVConfig / ScalarTensorTOVConfig are added.
TOVConfig = GRTOVConfig
