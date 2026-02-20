# Adding a New TOV Solver

This guide walks through adding a new Tolman-Oppenheimer-Volkoff (TOV) equation solver to JESTER. TOV solvers integrate the structure equations for neutron stars, computing mass-radius-tidal deformability (M-R-Λ) relationships from an equation of state.

## Architecture Overview

TOV solvers are independent of specific EOS models and work with any EOS through the unified `EOSData` interface. All solvers inherit from `TOVSolverBase` and are automatically compatible with all EOS models through `JesterTransform`.

**Key files:**
- TOV solver implementation: `jesterTOV/tov/your_solver.py`
- Configuration schema: `jesterTOV/inference/config/schemas/tov.py`
- Transform factory: `jesterTOV/inference/transforms/transform.py`
- Tests: `tests/test_tov/test_your_solver.py`

## Step 1: Create TOV Solver Class

Create your solver in `jesterTOV/tov/` inheriting from `TOVSolverBase`:

```python
import jax.numpy as jnp
from jaxtyping import Array, Float
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, TOVSolution, FamilyData
import diffrax

class MyNewTOVSolver(TOVSolverBase):
    """One-line description of your TOV solver.

    Detailed description including:
    - Gravity theory or modification being implemented
    - Key equations
    - References

    References:
        Author et al. (Year). Title. Journal Volume, Page.
    """

    def __init__(self, coupling_constant: float = 0.0):
        """Initialize TOV solver with theory-specific parameters.

        Args:
            coupling_constant: Description of coupling parameter
        """
        self.coupling_constant = coupling_constant

    def solve(
        self,
        eos_data: EOSData,
        pc: float,
        **kwargs: Any
    ) -> TOVSolution:
        """Solve TOV equations for a single central pressure.

        Args:
            eos_data: EOS data from any EOS model
            pc: Central pressure (geometric units)
            **kwargs: Additional parameters from prior (if needed)

        Returns:
            TOVSolution: NamedTuple with:
                - M: Total mass (solar masses)
                - R: Radius (km)
                - k2: Tidal Love number (dimensionless)
        """
        # Extract additional parameters if needed
        theory_param = kwargs.get("theory_param", self.coupling_constant)

        # Define your modified TOV equations
        def ode_system(t, y, args):
            """ODE system: dy/dt = f(t, y).

            Args:
                t: Independent variable (typically radius)
                y: State vector [mass, pressure, ...]
                args: Additional arguments (eos_data, etc.)

            Returns:
                Derivatives dy/dt
            """
            # Your equations here
            # ...
            return derivatives

        # Solve using Diffrax
        solver = diffrax.Dopri5()
        term = diffrax.ODETerm(ode_system)

        # Set up initial conditions
        y0 = jnp.array([...])  # Initial state

        # Solve with adaptive stepping
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_final,
            dt0=0.01,
            y0=y0,
            args=(eos_data,),
            max_steps=10000,
            throw=False  # Graceful failure handling
        )

        # Extract solution
        M = sol.ys[0][-1]  # type: ignore[index]
        R = sol.ys[1][-1]  # type: ignore[index]

        # Compute Love number (theory-specific)
        k2 = self._compute_love_number(sol, eos_data)

        # Convert to physical units
        from jesterTOV.utils import Msun_km
        M_solar = M / Msun_km
        R_km = R

        return TOVSolution(M=M_solar, R=R_km, k2=k2)

    def construct_family(
        self,
        eos_data: EOSData,
        ndat: int,
        min_nsat: float,
        **kwargs: Any
    ) -> FamilyData:
        """Build M-R-Λ family curves.

        Args:
            eos_data: EOS data from any EOS model
            ndat: Number of central pressures to sample
            min_nsat: Minimum density as fraction of saturation density
            **kwargs: Additional theory parameters from prior

        Returns:
            FamilyData: NamedTuple with:
                - log10pcs: Log10 central pressures
                - masses: Gravitational masses (M☉)
                - radii: Radii (km)
                - lambdas: Tidal deformabilities (dimensionless)
        """
        # Generate central pressure grid
        log10_pc_min = ...  # Based on min_nsat
        log10_pc_max = ...
        log10pcs = jnp.linspace(log10_pc_min, log10_pc_max, ndat)

        # Vectorize solve() over central pressures
        solve_vec = jax.vmap(
            lambda pc: self.solve(eos_data, 10**pc, **kwargs)
        )

        solutions = solve_vec(log10pcs)

        # Extract arrays (vmap batches scalar fields → arrays)
        masses: Float[Array, "ndat"] = solutions.M  # type: ignore[assignment]
        radii: Float[Array, "ndat"] = solutions.R  # type: ignore[assignment]
        k2s: Float[Array, "ndat"] = solutions.k2  # type: ignore[assignment]

        # Compute dimensionless tidal deformability
        lambdas = self._compute_lambda(masses, radii, k2s)

        return FamilyData(
            log10pcs=log10pcs,
            masses=masses,
            radii=radii,
            lambdas=lambdas
        )

    def get_required_parameters(self) -> list[str]:
        """Return list of additional parameters beyond EOS.

        These are theory-specific coupling constants or other parameters
        that must be present in the prior file.

        Returns:
            List of parameter names required from prior
        """
        return ["theory_param"]  # Or empty list if no additional params
```

**Critical considerations:**

1. **JAX compatibility**: Use `jax.numpy`, `jax.vmap`, avoid Python control flow on traced values
2. **Diffrax for ODE solving**: Use `Dopri5` or `Tsit5` with adaptive stepping
3. **Graceful failure**: Set `throw=False` in `diffeqsolve` to handle divergent solutions
4. **Type ignores for vmap**: `vmap` batches scalar NamedTuple fields into arrays, requiring type ignores
5. **Unit conventions**: Return masses in M☉, radii in km, Λ dimensionless

## Step 2: Update Configuration Schema

Add a concrete config class for your solver to `jesterTOV/inference/config/schemas/tov.py` and include it in the `TOVConfig` discriminated union. `BaseTOVConfig` already provides `min_nsat_TOV`, `ndat_TOV`, and `nb_masses`; only add solver-specific fields to your subclass:

```python
# In jesterTOV/inference/config/schemas/tov.py

class MyNewTOVConfig(BaseTOVConfig):
    """Configuration for MyNewTOVSolver."""

    type: Literal["my_new_solver"] = "my_new_solver"  # type: ignore[override]

    # Solver-specific fields
    my_solver_coupling: float = Field(
        default=0.0,
        description="Coupling constant for my new solver"
    )


# Switch TOVConfig to a discriminated union
TOVConfig = Annotated[
    Union[GRTOVConfig, MyNewTOVConfig],
    Discriminator("type"),
]
```

Also export `MyNewTOVConfig` from `schema.py`.

**Regenerate YAML documentation:**

```bash
uv run python -m jesterTOV.inference.config.generate_yaml_reference
```

## Step 3: Register in Transform Factory

Add your solver to `jesterTOV/inference/transforms/transform.py` using an `isinstance` check:

```python
from jesterTOV.inference.config.schema import BaseTOVConfig, GRTOVConfig, MyNewTOVConfig

def _create_tov_solver(config: BaseTOVConfig) -> TOVSolverBase:
    """Create TOV solver from configuration."""
    if isinstance(config, GRTOVConfig):
        return GRTOVSolver()
    elif isinstance(config, MyNewTOVConfig):
        from jesterTOV.tov.my_new import MyNewTOVSolver
        return MyNewTOVSolver(coupling_constant=config.my_solver_coupling)
    else:
        raise ValueError(f"Unknown TOV config type: {type(config).__name__}")
```

## Step 4: Create Prior File

If your solver requires additional parameters beyond the EOS:

```python
# examples/inference/my_new_solver/prior.prior

# EOS parameters (e.g., for MetaModel)
E_sat = UniformPrior(-16.1, -15.9, parameter_names=["E_sat"])
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
# ... other EOS params ...

# Theory-specific parameters
theory_param = UniformPrior(-1.0, 1.0, parameter_names=["theory_param"])
```

**Parameter validation**: The inference system checks that all parameters in `get_required_parameters()` are present. Missing parameters cause errors before sampling.

## Step 5: Write Tests

Create comprehensive tests in `tests/test_tov/test_my_new_solver.py`:

```python
import jax.numpy as jnp
import pytest
from jesterTOV.tov.my_new import MyNewTOVSolver
from jesterTOV.eos.metamodel import MetaModel_EOS_model

class TestMyNewTOVSolver:
    """Test suite for MyNewTOVSolver."""

    @pytest.fixture
    def eos_data(self):
        """Valid EOS data for testing."""
        model = MetaModel_EOS_model()
        params = {
            "E_sat": -16.0,
            "K_sat": 240.0,
            "Q_sat": 400.0,
            "Z_sat": 0.0,
            "E_sym": 31.7,
            "L_sym": 58.7,
            "K_sym": -100.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }
        return model.construct_eos(params)

    def test_solve_single_star(self, eos_data):
        """Test single star solution."""
        solver = MyNewTOVSolver()
        pc = 1e34  # Example central pressure

        solution = solver.solve(eos_data, pc)

        # Physical mass and radius
        assert solution.M > 0.0
        assert solution.M < 3.0  # Reasonable mass range
        assert solution.R > 5.0
        assert solution.R < 20.0  # Reasonable radius range

        # Love number range
        assert solution.k2 >= 0.0
        assert solution.k2 <= 1.0

    def test_construct_family(self, eos_data):
        """Test M-R-Λ family construction."""
        solver = MyNewTOVSolver()

        family_data = solver.construct_family(
            eos_data,
            ndat=50,
            min_nsat=0.75
        )

        # Check shapes
        assert len(family_data.log10pcs) == 50
        assert len(family_data.masses) == 50
        assert len(family_data.radii) == 50
        assert len(family_data.lambdas) == 50

        # Physical ranges
        assert jnp.all(family_data.masses > 0.0)
        assert jnp.all(family_data.radii > 0.0)
        assert jnp.all(family_data.lambdas >= 0.0)

        # Find maximum mass
        max_mass_idx = jnp.argmax(family_data.masses)
        M_max = family_data.masses[max_mass_idx]
        assert M_max > 1.0  # Reasonable maximum mass

    def test_gr_limit(self, eos_data):
        """Test that solver reduces to GR when coupling = 0."""
        from jesterTOV.tov.gr import GRTOVSolver

        solver_new = MyNewTOVSolver(coupling_constant=0.0)
        solver_gr = GRTOVSolver()

        pc = 1e34

        sol_new = solver_new.solve(eos_data, pc)
        sol_gr = solver_gr.solve(eos_data, pc)

        # Should match GR to numerical precision
        assert jnp.allclose(sol_new.M, sol_gr.M, rtol=1e-6)
        assert jnp.allclose(sol_new.R, sol_gr.R, rtol=1e-6)

    def test_additional_parameters(self, eos_data):
        """Test that additional parameters from prior are used."""
        solver = MyNewTOVSolver()

        pc = 1e34
        sol1 = solver.solve(eos_data, pc, theory_param=0.0)
        sol2 = solver.solve(eos_data, pc, theory_param=1.0)

        # Different parameters should give different results
        assert not jnp.allclose(sol1.M, sol2.M)

    def test_required_parameters(self):
        """Test that required parameters list is correct."""
        solver = MyNewTOVSolver()
        required = solver.get_required_parameters()

        if required:  # If solver needs additional params
            assert "theory_param" in required

    @pytest.mark.slow
    def test_convergence_tolerance(self, eos_data):
        """Test numerical convergence with different tolerances."""
        solver = MyNewTOVSolver()
        pc = 1e34

        # Run with different max_steps (if configurable)
        # Verify solution converges
```

**Test categories:**

1. **Single star**: Test `solve()` produces physical M, R, k2
2. **Family curves**: Test `construct_family()` produces valid M-R-Λ
3. **Limit checks**: Verify reduction to GR when appropriate
4. **Parameter sensitivity**: Additional parameters affect results
5. **Convergence**: Numerical accuracy tests

## Step 6: Create Example Configuration

Create an example in `examples/inference/my_new_solver/`:

```yaml
# config.yaml
seed: 42

eos:
  type: "metamodel"
  ndat_metamodel: 100
  nmin_MM_nsat: 0.75

tov:
  type: "my_new_solver"
  my_solver_coupling: 0.1
  min_nsat_TOV: 0.75
  ndat_TOV: 100
  nb_masses: 100

prior: "prior.prior"

likelihoods:
  - type: "nicer"
    enabled: true
    sources: ["J0030"]

sampler:
  type: "smc-rw"
  n_particles: 1000
  n_mcmc_steps: 10
  target_ess: 0.9

outdir: "outdir"
```

Test the example runs successfully:

```bash
cd examples/inference/my_new_solver
uv run run_jester_inference config.yaml
```

## Step 7: Documentation

Add documentation to `docs/overview/tov/`:

```markdown
# My New TOV Solver

One-sentence description of the modified gravity theory.

## Theory Background

Explain the gravity theory modifications...

## Equations

Key equations (use LaTeX):

$$
\frac{dM}{dr} = 4\pi r^2 \epsilon(r) + f(\alpha, r)
$$

Where $f(\alpha, r)$ represents the modification.

## Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| theory_param | Coupling constant | -1 to 1 |

## GR Limit

Describe when the solver reduces to General Relativity...

## References

- Author et al. (Year). Title. *Journal* Volume, Page.

## Usage

\```python
from jesterTOV.tov.my_new import MyNewTOVSolver
from jesterTOV.eos.metamodel import MetaModel_EOS_model

# Build EOS
model = MetaModel_EOS_model()
params = {...}
eos_data = model.construct_eos(params)

# Solve TOV
solver = MyNewTOVSolver(coupling_constant=0.1)
family = solver.construct_family(eos_data, ndat=100, min_nsat=0.75)
\```
```

Update `docs/overview/tov_solvers.rst` to include your new solver.

## Checklist

Before submitting a PR:

- [ ] Solver inherits from `TOVSolverBase`
- [ ] `solve()` returns valid `TOVSolution` (M, R, k2)
- [ ] `construct_family()` returns valid `FamilyData`
- [ ] `get_required_parameters()` lists additional parameters
- [ ] Added `MyNewTOVConfig` to `schemas/tov.py` and included in `TOVConfig` union
- [ ] Regenerated YAML reference
- [ ] Registered in `_create_tov_solver()` factory
- [ ] Comprehensive tests written and passing
- [ ] Example configuration runs successfully
- [ ] Documentation added with equations and references
- [ ] Uses Diffrax for ODE integration
- [ ] JAX-compatible (no Python `if` on traced values)
- [ ] Type hints for all functions
- [ ] Graceful failure handling (`throw=False`)
- [ ] Reduces to GR when appropriate (if applicable)
- [ ] Type ignores documented for vmap patterns
- [ ] Code passes `uv run pyright jesterTOV/tov/`
- [ ] Tests pass: `uv run pytest tests/test_tov/test_my_new_solver.py -v`

## Common Pitfalls

**Type ignores for vmap**: When using `vmap` to batch `solve()`, scalar fields in `TOVSolution` become arrays:

```python
# This is correct and required
masses: Float[Array, "ndat"] = solutions.M  # type: ignore[assignment]
```

**Diffrax throw parameter**: Always use `throw=False` to handle divergent solutions gracefully:

```python
sol = diffrax.diffeqsolve(..., throw=False)

# Check if solution succeeded
if sol.result != 0:
    # Handle failure (return NaN or default values)
```

**Chemical potential access**: If your solver needs `mu` from `EOSData`, check if it's populated:

```python
if eos_data.mu is None:
    raise ValueError("This solver requires mu from EOS")

mu: Float[Array, "n"] = eos_data.mu  # type: ignore[assignment]
```

**Unit conversions**: Remember JESTER uses geometric units internally. Convert to physical units only at the end:

```python
from jesterTOV.utils import Msun_km, c_km, G_km

M_solar = M_geometric / Msun_km  # Convert to solar masses
R_km = R_geometric  # Already in km if using correct units
```

## Need Help?

- See existing implementations: `jesterTOV/tov/gr.py`, `jesterTOV/tov/scalar_tensor.py`
- Review BlackJAX source code and documentation for sampler best practices
- Check Diffrax documentation: https://docs.kidger.site/diffrax/
- Review `jesterTOV/inference/CLAUDE.md` for inference system architecture
