# Adding a New Likelihood

This guide walks through adding a new likelihood constraint to JESTER's inference system. Likelihoods encode observational constraints from multi-messenger astronomy (gravitational waves, X-ray timing, radio observations, nuclear experiments).

## Architecture Overview

Likelihoods evaluate the probability of observing data given EOS parameters. All likelihoods inherit from `LikelihoodBase` and are created through a factory pattern. The inference system automatically combines multiple likelihoods with equal weighting.

**Key files:**
- Likelihood implementation: `jesterTOV/inference/likelihoods/your_likelihood.py`
- Configuration schema: `jesterTOV/inference/config/schema.py`
- Likelihood factory: `jesterTOV/inference/likelihoods/factory.py`
- Data loading: `jesterTOV/inference/data/__init__.py`
- Tests: `tests/test_inference/test_likelihoods/test_your_likelihood.py`

## Step 1: Create Likelihood Class

Create your likelihood in `jesterTOV/inference/likelihoods/` inheriting from `LikelihoodBase`:

```python
from typing import Any
import jax.numpy as jnp
from jaxtyping import Array, Float
from jesterTOV.inference.base.likelihood import LikelihoodBase

class MyNewLikelihood(LikelihoodBase):
    """One-line description of observational constraint.

    Detailed description including:
    - What observation/experiment this represents
    - Data sources and references
    - Statistical model used

    References:
        Author et al. (Year). Title. Journal Volume, Page.
    """

    def __init__(self, data: Any):
        """Initialize likelihood with observational data.

        Args:
            data: Observational data structure (format depends on likelihood)
                  Common formats: dict, JAX arrays, KDE objects
        """
        self.data = data

    def evaluate(self, params: dict[str, float], data: Any) -> float:
        """Compute log probability for parameters.

        This method is called by the sampler during inference. It receives
        the transformed parameters (including M-R-Λ curves from JesterTransform)
        and returns the log likelihood.

        Args:
            params: Dictionary containing:
                - EOS parameters (e.g., K_sat, L_sym)
                - TOV parameters (if any)
                - Derived quantities from transform:
                  - masses: Stellar masses (M☉)
                  - radii: Stellar radii (km)
                  - lambdas: Tidal deformabilities
                  - eos_data: Full EOS if needed

            data: Observational data (passed through from __init__)

        Returns:
            Log probability: log P(data | params)
        """
        # Extract relevant quantities from params
        masses = params.get("masses")  # From JesterTransform
        radii = params.get("radii")    # From JesterTransform

        # Your likelihood calculation here
        # Example: Gaussian likelihood for mass-radius constraint
        log_likelihood = 0.0

        # Add contributions from each observation
        for obs in self.data["observations"]:
            # Interpolate model prediction at observation point
            predicted = self._interpolate_model(
                masses, radii, obs["mass"]
            )

            # Gaussian log likelihood
            residual = (predicted - obs["radius"]) / obs["error"]
            log_likelihood += -0.5 * residual**2

        return log_likelihood

    def _interpolate_model(
        self,
        masses: Float[Array, "n"],
        radii: Float[Array, "n"],
        target_mass: float
    ) -> float:
        """Helper to interpolate model prediction.

        Args:
            masses: Model mass grid
            radii: Model radius grid
            target_mass: Mass at which to evaluate

        Returns:
            Interpolated radius
        """
        return jnp.interp(target_mass, masses, radii)
```

**Critical considerations:**

1. **JAX compatibility**: Use `jax.numpy` for all computations
2. **Vectorization**: Design for efficient evaluation over posterior samples
3. **Numerical stability**: Handle edge cases (divide by zero, log of negative)
4. **Transform integration**: Know what quantities `JesterTransform` provides
5. **Return log probabilities**: Never return raw probabilities (numerical underflow)

## Step 2: Add Data Loading Function

If your likelihood requires external data, add loading to `jesterTOV/inference/data/__init__.py`:

```python
def load_my_observation_data() -> dict[str, Any]:
    """Load observational data for MyNewLikelihood.

    Returns:
        Dictionary with observational data:
            - observations: List of measurements
            - metadata: Additional information
    """
    # Option 1: Load from package data
    data_path = Path(__file__).parent / "data_files" / "my_observation.json"
    with open(data_path) as f:
        data = json.load(f)

    # Option 2: Download from Zenodo and cache
    from jesterTOV.inference.data.paths import get_data_path
    data_path = get_data_path(
        "my_observation.h5",
        zenodo_url="https://zenodo.org/record/XXXXX/files/my_observation.h5"
    )
    data = load_hdf5(data_path)

    return {
        "observations": data["measurements"],
        "metadata": data["metadata"]
    }
```

**Data management best practices:**

- Small data (<1 MB): Include in `jesterTOV/inference/data/data_files/`
- Large data (>1 MB): Host on Zenodo, cache locally with `get_data_path()`
- Validate data format in loading function
- Document data provenance and references

## Step 3: Update Configuration Schema

Add your likelihood config class to `jesterTOV/inference/config/schemas/likelihoods.py` and extend the `LikelihoodConfig` discriminated union there:

```python
# In jesterTOV/inference/config/schemas/likelihoods.py

class MyNewLikelihoodConfig(BaseLikelihoodConfig):
    """Configuration for MyNewLikelihood."""

    type: Literal["my_new_likelihood"] = "my_new_likelihood"

    # Add likelihood-specific parameters
    observation_set: str = Field(
        default="default",
        description="Which observation dataset to use"
    )
    systematic_error: float = Field(
        default=0.0,
        description="Additional systematic uncertainty (km)"
    )


# Extend the discriminated union at the bottom of the file
LikelihoodConfig = Annotated[
    Union[
        GWLikelihoodConfig,
        NICERLikelihoodConfig,
        # ... existing configs ...
        MyNewLikelihoodConfig,
    ],
    Discriminator("type"),
]
```

Then re-export `MyNewLikelihoodConfig` from both `config/schema.py` (in the import block and `__all__`) and `config/__init__.py` so it is accessible as `jesterTOV.inference.config.MyNewLikelihoodConfig`.

**Regenerate YAML documentation:**

```bash
uv run python -m jesterTOV.inference.config.generate_yaml_reference
```

## Step 4: Register in Likelihood Factory

Add creation logic to `jesterTOV/inference/likelihoods/factory.py`:

```python
def create_likelihood(
    config: LikelihoodConfig,
    transform: Optional[NtoMTransform] = None,
) -> LikelihoodBase:
    """Create likelihood from configuration."""

    if config.type == "my_new_likelihood":
        from jesterTOV.inference.data import load_my_observation_data
        from jesterTOV.inference.likelihoods.my_new import MyNewLikelihood

        data = load_my_observation_data()

        # Apply configuration options
        if config.systematic_error > 0:
            # Add systematic uncertainty
            for obs in data["observations"]:
                obs["error"] = jnp.sqrt(
                    obs["error"]**2 + config.systematic_error**2
                )

        return MyNewLikelihood(data)

    # ... other likelihood types ...
```

## Step 5: Write Tests

Create comprehensive tests in `tests/test_inference/test_likelihoods/test_my_new_likelihood.py`:

```python
import jax.numpy as jnp
import pytest
from jesterTOV.inference.likelihoods.my_new import MyNewLikelihood
from jesterTOV.eos.metamodel import MetaModel_EOS_model
from jesterTOV.tov.gr import GRTOVSolver

class TestMyNewLikelihood:
    """Test suite for MyNewLikelihood."""

    @pytest.fixture
    def mock_data(self):
        """Create mock observational data."""
        return {
            "observations": [
                {"mass": 1.4, "radius": 12.0, "error": 0.5},
                {"mass": 2.0, "radius": 11.5, "error": 0.7},
            ]
        }

    @pytest.fixture
    def mock_params(self):
        """Create mock parameter dict with M-R curves."""
        # Build realistic M-R curve
        model = MetaModel_EOS_model()
        eos_params = {
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
        eos_data = model.construct_eos(eos_params)

        solver = GRTOVSolver()
        family = solver.construct_family(eos_data, ndat=100, min_nsat=0.75)

        return {
            **eos_params,
            "masses": family.masses,
            "radii": family.radii,
            "lambdas": family.lambdas,
        }

    def test_likelihood_shape(self, mock_data, mock_params):
        """Test that likelihood returns scalar."""
        likelihood = MyNewLikelihood(mock_data)
        log_prob = likelihood.evaluate(mock_params, mock_data)

        assert isinstance(log_prob, (float, jnp.ndarray))
        if isinstance(log_prob, jnp.ndarray):
            assert log_prob.shape == ()

    def test_likelihood_is_finite(self, mock_data, mock_params):
        """Test that likelihood returns finite value."""
        likelihood = MyNewLikelihood(mock_data)
        log_prob = likelihood.evaluate(mock_params, mock_data)

        assert jnp.isfinite(log_prob)

    def test_likelihood_sensitivity(self, mock_data, mock_params):
        """Test that likelihood responds to parameter changes."""
        likelihood = MyNewLikelihood(mock_data)

        log_prob1 = likelihood.evaluate(mock_params, mock_data)

        # Modify radii (worse fit)
        mock_params_bad = mock_params.copy()
        mock_params_bad["radii"] = mock_params["radii"] + 2.0  # Shift by 2 km

        log_prob2 = likelihood.evaluate(mock_params_bad, mock_data)

        # Worse fit should have lower (more negative) log probability
        assert log_prob2 < log_prob1

    def test_configuration_options(self):
        """Test that configuration options are applied."""
        from jesterTOV.inference.config.schema import MyNewLikelihoodConfig
        from jesterTOV.inference.likelihoods.factory import create_likelihood

        config = MyNewLikelihoodConfig(
            type="my_new_likelihood",
            systematic_error=0.5
        )

        likelihood = create_likelihood(config)
        assert isinstance(likelihood, MyNewLikelihood)

    def test_integration_with_inference(self, tmp_path):
        """Test end-to-end inference with this likelihood."""
        config_yaml = """
seed: 42

eos:
  type: "metamodel"
  ndat_metamodel: 50

tov:
  tov_solver: "gr"

prior: "prior.prior"

likelihoods:
  - type: "my_new_likelihood"
    enabled: true

sampler:
  type: "smc-rw"
  n_particles: 100
  n_mcmc_steps: 5

outdir: "outdir"
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_yaml)

        prior_file = tmp_path / "prior.prior"
        prior_file.write_text("""
E_sat = UniformPrior(-16.1, -15.9, parameter_names=["E_sat"])
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
# ... other priors ...
""")

        # Run inference (lightweight parameters for testing)
        from jesterTOV.inference.run_inference import main
        main(config_path)

        # Check output exists
        assert (tmp_path / "outdir").exists()
```

**Test categories:**

1. **Shape/type validation**: Returns scalar log probability
2. **Numerical stability**: Finite values, no NaN/Inf
3. **Sensitivity**: Responds correctly to parameter changes
4. **Configuration**: Factory correctly applies options
5. **Integration**: Works in full inference pipeline

## Step 6: Create Example Configuration

Create an example in `examples/inference/my_new_likelihood/`:

```yaml
# config.yaml
seed: 42

eos:
  type: metamodel
  ndat_metamodel: 100

tov:
  tov_solver: gr

prior: prior.prior

likelihoods:
  - type: my_new_likelihood
    enabled: true
    observation_set: default
    systematic_error: 0.1

  # Can combine with other likelihoods
  - type: constraints_eos
    enabled: true

sampler:
  type: smc-rw
  n_particles: 2000
  n_mcmc_steps: 20
  target_ess: 0.9

outdir: outdir
```

Test the example runs successfully:

```bash
cd examples/inference/my_new_likelihood
uv run run_jester_inference config.yaml
```

## Step 7: Documentation

Add documentation to `docs/overview/likelihoods/`:

```markdown
# My New Likelihood

One-sentence description of the observational constraint.

## Observational Data

Describe the observation or experiment:
- What was measured
- Instruments/telescopes used
- Key results and uncertainties

## Statistical Model

Explain the likelihood function:

$$
\mathcal{L}(R | M, \theta) = \prod_i \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left[-\frac{(R_i - R_{\mathrm{model}}(M_i, \theta))^2}{2\sigma_i^2}\right]
$$

Where:
- $R_i$: Observed radius
- $M_i$: Measured mass
- $\theta$: EOS parameters
- $\sigma_i$: Measurement uncertainty

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| observation_set | Dataset to use | "default" |
| systematic_error | Additional systematic (km) | 0.0 |

## References

- Observer et al. (Year). Title. *Journal* Volume, Page.

## Usage

\```yaml
likelihoods:
  - type: "my_new_likelihood"
    enabled: true
    observation_set: "latest"
    systematic_error: 0.2
\```
```

Update `docs/overview/likelihoods.rst` to include your new likelihood.

## Checklist

Before submitting a PR:

- [ ] Likelihood inherits from `LikelihoodBase`
- [ ] `evaluate()` returns finite log probability (scalar)
- [ ] Data loading function added if needed
- [ ] Configuration model added to `schema.py`
- [ ] Regenerated YAML reference
- [ ] Registered in `factory.py`
- [ ] Comprehensive tests written and passing
- [ ] Integration test with full inference pipeline
- [ ] Example configuration runs successfully
- [ ] Documentation with equations and references
- [ ] JAX-compatible (uses `jax.numpy`)
- [ ] Handles edge cases (e.g., extrapolation beyond M-R range)
- [ ] Type hints for all functions
- [ ] Data provenance documented
- [ ] Code passes `uv run pyright jesterTOV/inference/likelihoods/`
- [ ] Tests pass: `uv run pytest tests/test_inference/test_likelihoods/test_my_new_likelihood.py -v`

## Common Pitfalls

**Not returning log probabilities**: Always return log P, never P. This avoids numerical underflow:

```python
# BAD: Probability underflows to zero
prob = 1e-300
return prob

# GOOD: Log probability stays in range
log_prob = -690.7755  # log(1e-300)
return log_prob
```

**Transform output not used**: The `params` dict contains outputs from `JesterTransform`. Know what's available:

```python
# Available from JesterTransform (note the _EOS suffix)
masses = params["masses_EOS"]      # Stellar masses (M☉)
radii = params["radii_EOS"]        # Stellar radii (km)
lambdas = params["Lambdas_EOS"]    # Tidal deformabilities
logpc = params["logpc_EOS"]        # Log10 central pressures

# EOS quantities (on the density grid)
n   = params["n"]    # Number density
p   = params["p"]    # Pressure
cs2 = params["cs2"]  # Sound speed squared

# Constraint violation counts (for physical validity checks)
n_tov_failures         = params["n_tov_failures"]
n_causality_violations = params["n_causality_violations"]

# Original EOS parameters also present
K_sat = params["K_sat"]
L_sym = params["L_sym"]
```

**Interpolation beyond M-R range**: Handle cases where observation lies outside model range:

```python
# Check if observation is in valid range
if target_mass < jnp.min(masses) or target_mass > jnp.max(masses):
    # Return very negative log likelihood (rejected)
    return -1e10

# Or: use fill_value in jnp.interp
radius = jnp.interp(target_mass, masses, radii, left=-jnp.inf, right=-jnp.inf)
```

**Combining likelihoods incorrectly**: The `CombinedLikelihood` wrapper handles summation. Just return your log probability:

```python
# In your likelihood - just return single log probability
return log_prob

# The framework handles combination:
# total_log_prob = sum(likelihood.evaluate(params, data) for likelihood in likelihoods)
```

**Not handling disabled likelihoods**: The factory respects the `enabled` flag:

```python
# In factory.py - check enabled flag
if not config.enabled:
    return None  # Or skip entirely
```

## Advanced: Resampling Likelihoods

Some likelihoods (like GW posteriors) require resampling during MCMC. See `GWLikelihoodResampled` for an example:

```python
class ResamplingLikelihood(LikelihoodBase):
    """Likelihood that resamples from posterior at each evaluation."""

    def __init__(self, posterior_samples: Array, n_resample: int = 1000):
        """Initialize with posterior samples from other analyses."""
        self.posterior_samples = posterior_samples
        self.n_resample = n_resample

    def evaluate(self, params: dict, data: Any, rng_key: Any = None) -> float:
        """Evaluate by resampling posterior."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # Resample from posterior
        indices = jax.random.choice(
            rng_key, len(self.posterior_samples), shape=(self.n_resample,)
        )
        samples = self.posterior_samples[indices]

        # Compute likelihood for each sample
        # ... (compare with model predictions)

        return log_likelihood
```

Resampling likelihoods need special handling in the sampler to pass PRNGKey.

## Need Help?

- See existing implementations: `jesterTOV/inference/likelihoods/nicer.py`, `gw.py`
- Review `jesterTOV/inference/CLAUDE.md` for inference architecture
- Check data loading examples in `jesterTOV/inference/data/__init__.py`
- For statistical model questions, consult relevant papers in `jesterTOV/references.bib`
