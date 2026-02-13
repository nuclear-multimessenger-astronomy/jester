# Adding a New EOS Model

This guide walks through adding a new equation of state (EOS) model to JESTER. EOS models define the thermodynamic relationship between pressure, energy density, and other state variables in neutron star matter.

## Architecture Overview

JESTER's modular design separates EOS models from TOV solvers and inference. All EOS models inherit from `Interpolate_EOS_model` and are automatically compatible with all TOV solvers through the unified `JesterTransform` system.

**Key files:**
- EOS implementation: `jesterTOV/eos/your_eos.py`
- Configuration schema: `jesterTOV/inference/config/schema.py`
- Transform factory: `jesterTOV/inference/transforms/transform.py`
- Tests: `tests/test_eos/test_your_eos.py`

## Step 1: Create EOS Class

Create your EOS model in `jesterTOV/eos/` inheriting from `Interpolate_EOS_model`:

```python
from jaxtyping import Array, Float
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.tov.data_classes import EOSData

class MyNewEOS(Interpolate_EOS_model):
    """One-line description of your EOS model.

    Detailed description including:
    - Physical motivation
    - Parameter definitions
    - Key references

    References:
        Author et al. (Year). Title. Journal Volume, Page.
    """

    def __init__(self, config_param: float = 1.0):
        """Initialize EOS model with configuration parameters.

        Args:
            config_param: Description of configuration parameter
        """
        self.config_param = config_param

    def construct_eos(self, params: dict[str, float]) -> EOSData:
        """Build EOS from parameters.

        Args:
            params: Dictionary with keys matching get_required_parameters()

        Returns:
            EOSData: JAX NamedTuple with fields:
                - ns: Number densities (fm^-3)
                - ps: Pressures (geometric units)
                - hs: Specific enthalpies (dimensionless)
                - es: Energy densities (geometric units)
                - dloge_dlogps: Derivative d(log e)/d(log p)
                - cs2: Sound speed squared (c_s^2/c^2)
                - mu: Chemical potential (optional)
                - extra_constraints: Additional constraints (optional)
        """
        # Extract parameters
        param1 = params["param1"]
        param2 = params["param2"]

        # Build EOS (your physics here)
        # ...

        return EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=cs2,
            mu=None,  # Optional
            extra_constraints=None  # Optional
        )

    def get_required_parameters(self) -> list[str]:
        """Return list of required parameter names.

        These parameter names must match those in the prior file.
        Missing parameters will cause validation errors before sampling.
        """
        return ["param1", "param2", "param3"]
```

**Critical considerations:**

1. **JAX compatibility**: Use `jax.numpy` operations, avoid Python `if` statements on traced values
2. **Geometric units**: Return pressures/energies in geometric units (c=G=1)
3. **Array shapes**: All returned arrays must have same length (number of density points)
4. **Sound speed**: Ensure $0 \leq c_s^2 \leq 1$ (causality constraint)
5. **Monotonicity**: Pressure and energy must be monotonically increasing with density

## Step 2: Update Configuration Schema

Add your EOS type to `jesterTOV/inference/config/schema.py`:

```python
class TransformConfig(BaseModel):
    """Transform configuration for EOS and TOV solver."""

    type: Literal["metamodel", "metamodel_cse", "spectral", "my_new_eos"]

    # Add any EOS-specific configuration fields
    my_eos_config: float = Field(
        default=1.0,
        description="Configuration parameter for my new EOS"
    )

    # ... rest of configuration ...
```

**Regenerate YAML documentation:**

```bash
uv run python -m jesterTOV.inference.config.generate_yaml_reference
```

This updates `docs/inference/yaml_reference.md` with your new EOS type.

## Step 3: Register in Transform Factory

Add your EOS to `jesterTOV/inference/transforms/transform.py`:

```python
def _create_eos(config: TransformConfig) -> Interpolate_EOS_model:
    """Create EOS model from configuration."""
    if config.type == "metamodel":
        # ... existing code ...
    elif config.type == "my_new_eos":
        from jesterTOV.eos.my_new import MyNewEOS
        return MyNewEOS(config_param=config.my_eos_config)
    else:
        raise ValueError(f"Unknown EOS type: {config.type}")
```

No need to create new transform classes—`JesterTransform` handles all EOS × TOV combinations automatically.

## Step 4: Create Prior File

Create a `.prior` file for your new EOS parameters:

```python
# examples/inference/my_new_eos/prior.prior

# Required parameters (must match get_required_parameters())
param1 = UniformPrior(0.0, 10.0, parameter_names=["param1"])
param2 = UniformPrior(-5.0, 5.0, parameter_names=["param2"])
param3 = UniformPrior(100.0, 500.0, parameter_names=["param3"])
```

**Parameter validation**: The inference system validates that all parameters in `get_required_parameters()` are present in the prior file. Missing parameters cause an error before sampling starts.

## Step 5: Write Tests

Create comprehensive tests in `tests/test_eos/test_my_new_eos.py`:

```python
import jax.numpy as jnp
import pytest
from jesterTOV.eos.my_new import MyNewEOS
from jesterTOV.tov import GRTOVSolver

class TestMyNewEOS:
    """Test suite for MyNewEOS model."""

    @pytest.fixture
    def test_params(self):
        """Valid parameter set for testing."""
        return {
            "param1": 5.0,
            "param2": 0.0,
            "param3": 250.0,
        }

    def test_construct_eos_shape(self, test_params):
        """Test that construct_eos returns correct array shapes."""
        model = MyNewEOS()
        eos_data = model.construct_eos(test_params)

        # All arrays must have same length
        n = len(eos_data.ns)
        assert len(eos_data.ps) == n
        assert len(eos_data.hs) == n
        assert len(eos_data.es) == n

    def test_causality(self, test_params):
        """Test that sound speed satisfies causality (cs^2 <= 1)."""
        model = MyNewEOS()
        eos_data = model.construct_eos(test_params)

        assert jnp.all(eos_data.cs2 >= 0.0)
        assert jnp.all(eos_data.cs2 <= 1.0)

    def test_monotonicity(self, test_params):
        """Test that pressure and energy increase with density."""
        model = MyNewEOS()
        eos_data = model.construct_eos(test_params)

        # Pressure monotonically increasing
        assert jnp.all(jnp.diff(eos_data.ps) > 0)

        # Energy monotonically increasing
        assert jnp.all(jnp.diff(eos_data.es) > 0)

    def test_integration_with_tov(self, test_params):
        """Test that EOS works with TOV solver."""
        model = MyNewEOS()
        eos_data = model.construct_eos(test_params)

        solver = GRTOVSolver()
        family_data = solver.construct_family(
            eos_data,
            ndat=50,
            min_nsat=0.75
        )

        # Should produce physical mass-radius curves
        assert jnp.all(family_data.masses > 0.0)
        assert jnp.all(family_data.radii > 0.0)

    def test_required_parameters(self):
        """Test that required parameters list is correct."""
        model = MyNewEOS()
        required = model.get_required_parameters()

        assert "param1" in required
        assert "param2" in required
        assert "param3" in required

    def test_missing_parameter_raises(self):
        """Test that missing parameters cause clear errors."""
        model = MyNewEOS()

        with pytest.raises(KeyError):
            model.construct_eos({"param1": 1.0})  # Missing param2, param3
```

**Test categories:**

1. **Shape validation**: Array dimensions match expected
2. **Physical constraints**: Causality, positivity, monotonicity
3. **Integration tests**: Works with TOV solvers and inference
4. **Error handling**: Missing/invalid parameters handled gracefully

## Step 6: Create Example Configuration

Create an example in `examples/inference/my_new_eos/`:

```yaml
# config.yaml
seed: 42

transform:
  type: "my_new_eos"
  my_eos_config: 1.0
  ndat: 100
  min_nsat: 0.75
  tov_solver: "gr"

prior: "prior.prior"

likelihoods:
  - type: "eos_constraints"
    enabled: true

sampler:
  type: "smc-rw"
  n_particles: 1000
  n_mcmc_steps: 10
  target_ess: 0.9

outdir: "outdir"
```

Test the example runs successfully:

```bash
cd examples/inference/my_new_eos
uv run run_jester_inference config.yaml
```

## Step 7: Documentation

Add documentation to `docs/overview/eos/`:

```markdown
# My New EOS

One-sentence description of the EOS model.

## Physical Motivation

Explain the physics behind the model...

## Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| param1    | Description | 0-10          |
| param2    | Description | -5 to 5       |
| param3    | Description | 100-500       |

## References

- Author et al. (Year). Title. *Journal* Volume, Page.

## Usage

\```python
from jesterTOV.eos.my_new import MyNewEOS

model = MyNewEOS()
params = {"param1": 5.0, "param2": 0.0, "param3": 250.0}
eos_data = model.construct_eos(params)
\```
```

Update `docs/overview/eos.rst` to include your new EOS.

## Checklist

Before submitting a PR:

- [ ] EOS class inherits from `Interpolate_EOS_model`
- [ ] `construct_eos()` returns valid `EOSData` NamedTuple
- [ ] `get_required_parameters()` lists all required parameters
- [ ] Added to `TransformConfig` in `schema.py`
- [ ] Regenerated YAML reference
- [ ] Registered in `_create_eos()` factory
- [ ] Comprehensive tests written and passing
- [ ] Example configuration runs successfully
- [ ] Documentation added
- [ ] All arrays are JAX-compatible (no Python `if` on traced values)
- [ ] Causality constraint satisfied ($c_s^2 \leq 1$)
- [ ] Monotonicity verified (pressure/energy increase with density)
- [ ] Type hints added for all functions
- [ ] Code passes `uv run pyright jesterTOV/eos/`
- [ ] Tests pass: `uv run pytest tests/test_eos/test_my_new_eos.py -v`

## Common Pitfalls

**Python control flow on JAX arrays**: Avoid `if` statements on traced values. Use `jnp.where()` instead.

```python
# BAD: Fails during JAX tracing
if param > 0:
    result = compute_positive(param)
else:
    result = compute_negative(param)

# GOOD: JAX-compatible
result = jnp.where(
    param > 0,
    compute_positive(param),
    compute_negative(param)
)
```

**Unit inconsistencies**: JESTER uses geometric units internally (c=G=1). Ensure your EOS returns pressures and energies in geometric units. Conversions to physical units (M☉, km) happen only in output.

**EOSData field requirements**: While `mu` and `extra_constraints` are optional, `ns`, `ps`, `hs`, `es`, `dloge_dlogps`, and `cs2` are required. All must have the same array length.

**Parameter naming**: Parameter names in `get_required_parameters()` must exactly match those used in prior files. Mismatches cause validation errors.

## Need Help?

- See existing implementations: `jesterTOV/eos/metamodel/base.py`, `jesterTOV/eos/spectral/`
- Review `jesterTOV/inference/CLAUDE.md` for inference system architecture
- Check JAX documentation for array operations: https://jax.readthedocs.io/
