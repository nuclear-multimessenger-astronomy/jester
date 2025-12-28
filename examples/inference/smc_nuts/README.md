# SMC with NUTS Kernel - Example Configurations

This directory contains example configurations for Sequential Monte Carlo (SMC) sampling with the NUTS (No-U-Turn Sampler) kernel from BlackJAX.

## Directory Structure

Each subdirectory contains a complete configuration for a specific constraint type:

- **`prior/`** - Prior-only sampling (no observational constraints, for testing)
- **`GW170817/`** - Gravitational wave event GW170817 constraint
- **`NICER_J0030/`** - NICER X-ray timing observations of PSR J0030+0451
- **`NICER_J0740/`** - NICER X-ray timing observations of PSR J0740+6620
- **`chiEFT/`** - Chiral Effective Field Theory constraints (low-density EOS)
- **`radio/`** - Radio pulsar mass measurements (J1614-2230, J0740+6620)

## Running Examples

### Quick Test (Prior-only)
```bash
cd prior
uv run run_jester_inference config.yaml
```

### With Real Constraints
```bash
# GW170817
cd GW170817
uv run run_jester_inference config.yaml

# NICER J0030
cd NICER_J0030
uv run run_jester_inference config.yaml
```

## Configuration Highlights

### NUTS Kernel Settings
- **`kernel_type: "nuts"`** - Use NUTS kernel with Hessian-based adaptation
- **`n_particles: 1000`** - Reduced for testing (use 10000 for production)
- **`n_mcmc_steps: 5`** - MCMC steps per tempering level
- **`init_step_size: 0.01`** - Initial step size for NUTS
- **`target_acceptance: 0.7`** - Target acceptance rate
- **`mass_matrix_base: 0.2`** - Base value for mass matrix

### When to Use NUTS
- Complex, high-dimensional posteriors
- When automatic tuning is preferred
- When computational resources allow (NUTS is more expensive per step)
- Production runs where efficiency matters

## Comparison with Random Walk

For the same problems using Gaussian Random Walk kernel, see `../smc_random_walk/`

**NUTS vs Random Walk Trade-offs**:
- NUTS: More efficient, needs fewer steps, automatic adaptation, higher per-step cost
- Random Walk: Simpler, lower per-step cost, needs more MCMC steps, easier to debug

## File Contents

Each subdirectory contains:
- `config.yaml` - Full inference configuration
- `prior.prior` - Prior specification (bilby-style format)
- `outdir/` - Output directory (created during sampling)

## Notes

- All examples use `metamodel_cse` transform with 8 CSE parameters
- Particle counts are reduced (1000) for testing; use 10000+ for production
- EOS samples are also reduced (1000) for testing
- See YAML reference documentation for complete parameter descriptions
