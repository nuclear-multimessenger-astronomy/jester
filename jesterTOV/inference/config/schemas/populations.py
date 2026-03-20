from pydantic import BaseModel, ConfigDict, Field
import jax

class PopulationConfig(BaseModel):
    """Configuration for population transform"""

    model_config = ConfigDict(extra="forbid")

    name: str
    N_masses_evaluation: int = Field(
        default = 8000,
        gt=10,
        description="Number of samples drawn from the population mass distribution."
    )
    pop_random_key: int = Field(
        default=170817,
        description="Fixed random key for sampling from the mass distribution."
    )