# JESTER Inference Quick Start Guide

**Get started with Bayesian EOS inference in 5 minutes**

## Installation

```bash
# Install JESTER with inference dependencies
cd jester
uv pip install -e ".[inference]"
```

## Run Your First Inference

### 1. Use an Example Configuration

JESTER provides ready-to-use example configurations:

```bash
# List available examples
ls examples/inference/

# Examples include:
# - prior/         : Prior-only sampling (no observational data)
# - GW170817/      : Gravitational wave constraint
# - NICER_J0030/   : NICER PSR J0030+0451 X-ray timing
# - NICER_J0740/   : NICER PSR J0740+6620 X-ray timing
# - radio/         : Radio pulsar timing constraints
# - chiEFT/        : Chiral effective field theory bounds
```

### 2. Run Prior-Only Sampling

**Fastest way to test the system:**

```bash
cd examples/inference/prior/
run_jester_inference config.yaml
```

**What this does:**
- Samples from the prior distribution (no observational data)
- Uses MetaModel transform (8 NEP parameters)
- Runs 2 training + 2 production loops with 10 chains
- Takes ~2-5 minutes on CPU

**Output:**
```bash
ls outdir/
# results_production.npz  - MCMC samples (K_sat, Q_sat, etc.)
# eos_samples.npz         - EOS curves (M, R, Λ)
# runtime.txt             - Timing info
```

### 3. Run With Real Data

**Gravitational wave constraint (GW170817):**

```bash
cd examples/inference/GW170817/
run_jester_inference config.yaml
```

**NICER X-ray timing (PSR J0030+0451):**

```bash
cd examples/inference/NICER_J0030/
run_jester_inference config.yaml
```

**NICER X-ray timing (PSR J0740+6620):**

```bash
cd examples/inference/NICER_J0740/
run_jester_inference config.yaml
```

**Radio pulsar timing:**

```bash
cd examples/inference/radio/
run_jester_inference config.yaml
```

## Configuration Files Explained

### Minimal Configuration

Create `config.yaml`:

```yaml
seed: 43

transform:
  type: "metamodel"  # Standard MetaModel EOS

prior:
  specification_file: "prior.prior"  # Path to .prior file

likelihoods:
  - type: "zero"      # Prior-only (no data)
    enabled: true

sampler:
  n_chains: 10
  n_loop_training: 2
  n_loop_production: 2
  output_dir: "./outdir/"
```

### Prior File

Create `prior.prior`:

```python
# Nuclear Empirical Parameters
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])
Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])
Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])
```

### Run It

```bash
run_jester_inference config.yaml
```

## Add Real Observational Data

### NICER X-ray Timing

```yaml
likelihoods:
  - type: "nicer"
    enabled: true
    parameters:
      pulsars:
        - name: "J0030"
          amsterdam_samples_file: "data/NICER/J0030_amsterdam.npz"
          maryland_samples_file: "data/NICER/J0030_maryland.npz"
        - name: "J0740"
          amsterdam_samples_file: "data/NICER/J0740_amsterdam.npz"
          maryland_samples_file: "data/NICER/J0740_maryland.npz"
      N_masses_evaluation: 100
```

### Gravitational Waves

```yaml
likelihoods:
  - type: "gw"
    enabled: true
    parameters:
      events:
        - name: "GW170817"
          model_dir: "data/GW/GW170817/model/"
      N_masses_evaluation: 20
```

### Radio Pulsar Timing

```yaml
likelihoods:
  - type: "radio"
    enabled: true
    parameters:
      pulsars:
        - name: "J0740+6620"
          mass_mean: 2.08  # Solar masses
          mass_std: 0.07
      nb_masses: 100
```

### Chiral Effective Field Theory

```yaml
likelihoods:
  - type: "chieft"
    enabled: true
    parameters:
      low_filename: "data/chiEFT/low.dat"  # Optional
      high_filename: "data/chiEFT/high.dat"  # Optional
      nb_n: 100
```

## Use MetaModel + CSE

For high-density physics (e.g., stiff EOS):

```yaml
transform:
  type: "metamodel_cse"  # MetaModel + Constant Speed Extension
  nb_CSE: 8              # Number of CSE grid points

# Add to prior.prior:
# nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])
# (CSE grid parameters auto-generated)
```

## Analyze Results

### Load Samples

```python
import numpy as np

# Load MCMC samples
results = np.load("outdir/results_production.npz")
log_prob = results["log_prob"]
K_sat = results["K_sat"]

print(f"Number of samples: {len(log_prob)}")
print(f"K_sat mean: {K_sat.mean():.2f} ± {K_sat.std():.2f} MeV")
```

### Load EOS Curves

```python
# Load EOS samples
eos = np.load("outdir/eos_samples.npz")
masses = eos["masses_EOS"]  # Shape: (n_samples, nb_masses)
radii = eos["radii_EOS"]

print(f"Mass range: {masses.min():.2f} - {masses.max():.2f} km (geometric)")
print(f"Radius range: {radii.min():.2f} - {radii.max():.2f} km (geometric)")
```

### Make Corner Plot

```python
import corner
import matplotlib.pyplot as plt

# Select parameters to plot
param_names = ["K_sat", "Q_sat", "E_sym", "L_sym"]
samples_array = np.column_stack([results[name] for name in param_names])

# Create corner plot
fig = corner.corner(
    samples_array,
    labels=param_names,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True
)
plt.savefig("corner_plot.png", dpi=150, bbox_inches="tight")
```

### Plot M-R Diagram

```python
import matplotlib.pyplot as plt

masses_SI = eos["masses_EOS"] * 4.926e-6  # Convert to solar masses
radii_SI = eos["radii_EOS"] * 1.477      # Convert to km

# Plot all EOS curves
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(masses_SI)):
    ax.plot(radii_SI[i], masses_SI[i], alpha=0.1, color="C0")

ax.set_xlabel("Radius [km]")
ax.set_ylabel("Mass [M_sun]")
ax.set_title("Mass-Radius Posteriors")
ax.grid(True, alpha=0.3)
plt.savefig("mass_radius.png", dpi=150, bbox_inches="tight")
```

## Command Line Options

```bash
# Basic usage
run_jester_inference config.yaml

# Override output directory
run_jester_inference config.yaml --output-dir ./my_results/

# Validate configuration without running
run_jester_inference config.yaml --validate-only

# Dry run (setup without sampling)
run_jester_inference config.yaml --dry-run

# Alternative syntax (module-based)
uv run python -m jesterTOV.inference.run_inference --config config.yaml
```

## Common Issues

### Issue: "Prior specification file not found"

**Solution**: Make sure `prior.prior` is in the same directory as `config.yaml`, or use an absolute path:

```yaml
prior:
  specification_file: "/absolute/path/to/prior.prior"
```

### Issue: "At least one likelihood must be enabled"

**Solution**: Enable at least one likelihood in config:

```yaml
likelihoods:
  - type: "zero"
    enabled: true  # Make sure this is true
```

### Issue: "nb_CSE must be 0 for type='metamodel'"

**Solution**: Either remove `nb_CSE` or set it to 0 for metamodel:

```yaml
transform:
  type: "metamodel"
  nb_CSE: 0  # Or just don't specify it
```

Or use metamodel_cse:

```yaml
transform:
  type: "metamodel_cse"
  nb_CSE: 8
```

### Issue: Low acceptance rates (<5%)

**Solution**: Decrease step size in `samplers/flowmc.py`:

```python
eps_mass_matrix = 1e-4  # Smaller step size
```

Or increase it if acceptance is too high (>90%).

## Performance Tips

### Fast Testing

```yaml
sampler:
  n_chains: 5              # Fewer chains
  n_loop_training: 1       # Minimal training
  n_loop_production: 1     # Minimal production
  n_local_steps: 50        # Fewer steps
  n_eos_samples: 1000      # Fewer EOS samples
```

### Production Runs

```yaml
sampler:
  n_chains: 20             # Good convergence
  n_loop_training: 3       # Adequate training
  n_loop_production: 5     # More production samples
  n_local_steps: 200       # Better mixing
  n_eos_samples: 10000     # Good statistics
```

### GPU Acceleration

JAX automatically uses GPU if available. Check with:

```python
import jax
print(jax.devices())  # Should show GPU if available
```

No configuration changes needed - JAX will use GPU automatically!

## Next Steps

1. **Read full documentation**: `docs/inference.md` for complete details
2. **Explore examples**: Check `examples/inference/` for more configurations
3. **Customize**: Add your own data, modify priors, adjust sampler settings
4. **Advanced features**: Custom likelihoods, transforms, analysis pipelines

## Getting Help

- **Examples**: `jester/examples/inference/` - Working configurations
- **Full docs**: `jester/docs/inference.md` - Complete reference
- **Issues**: Report bugs on GitHub
- **Source**: Read docstrings in module files

---

**Quick Start Version**: 1.0
**Last Updated**: December 2024
