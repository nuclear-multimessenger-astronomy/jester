# SMC with Random Walk Kernel - Example Configurations

This directory contains example configurations for Sequential Monte Carlo (SMC) sampling with the Gaussian Random Walk Metropolis-Hastings kernel from BlackJAX.

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

### Random Walk Kernel Settings
- **`type: "smc-rw"`** - SMC with Gaussian random walk kernel
- **`n_particles: 1000`** - Reduced for testing (use 10000 for production)
- **`n_mcmc_steps: 10`** - More steps needed compared to NUTS
- **`random_walk_sigma: 0.1`** - Step size for Gaussian proposals
- **`target_acceptance: 0.5`** - Lower target than NUTS (typical for RW)
- **`adaptation_rate: 0.3`** - Adaptation rate for sigma tuning

### When to Use Random Walk
- Low-dimensional problems (< 20 parameters)
- When simplicity and robustness are priorities
- When computational budget is limited per step
- Debugging and initial exploration
- When NUTS has convergence issues

## Comparison with NUTS

For the same problems using NUTS kernel, use sampler type `"smc-nuts"` instead.

**Random Walk vs NUTS Trade-offs**:
- Random Walk (`smc-rw`): Simpler, lower per-step cost, needs more MCMC steps, easier to debug
- NUTS (`smc-nuts`): More efficient, needs fewer steps, automatic adaptation, higher per-step cost

## File Contents

Each subdirectory contains:
- `config.yaml` - Full inference configuration
- `prior.prior` - Prior specification (bilby-style format)
- `outdir/` - Output directory (created during sampling)

## Notes

- All examples use `metamodel_cse` transform with 8 CSE parameters
- Particle counts are reduced (1000) for testing; use 10000+ for production
- MCMC steps increased (10 vs 5 for NUTS) to compensate for simpler kernel
- EOS samples are also reduced (1000) for testing
- See YAML reference documentation for complete parameter descriptions

## Tuning Recommendations

If you find poor performance:
1. Increase `n_mcmc_steps` (try 20-50 for difficult posteriors)
2. Adjust `random_walk_sigma` (check acceptance rates in metadata)
3. Target acceptance rate ~0.3-0.5 is typical for random walk
4. Consider switching to NUTS if problem is high-dimensional
