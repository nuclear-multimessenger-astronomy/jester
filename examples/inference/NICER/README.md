# NICER-Only Inference Example

This example demonstrates how to constrain the neutron star equation of state (EOS) using NICER X-ray timing observations.

## Configuration

This configuration uses:
- **Transform**: MetaModel without CSE (lower computational cost)
- **Priors**: Standard NEP (Nuclear Empirical Parameters) priors
- **Likelihoods**: NICER X-ray timing observations for J0030+0451 and J0740+6620

This configuration is optimized for NICER data analysis:
- Uses MetaModel without CSE since NICER constrains lower densities (< 2 nsat)
- Faster computation compared to MetaModel+CSE
- Focuses on mass-radius measurements from X-ray pulse profile modeling

## Running the Example

From the jester repository root directory:

```bash
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/nicer_only/config.yaml
```

## Output

Results will be saved to `./outdir_nicer/` by default, containing:
- `results_production.npz` - MCMC samples and log probabilities
- `eos_samples.npz` - Generated EOS curves from sampled parameters
- `runtime.txt` - Sampling runtime and statistics
- `config_backup.yaml` - Copy of the configuration file used

## Customization

You can override the output directory:

```bash
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/nicer_only/config.yaml \
    --output-dir ./my_results/
```

Or validate the configuration without running:

```bash
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/nicer_only/config.yaml \
    --validate-only
```

## Requirements

Before running, ensure you have the required data files:
- NICER data: `./data/NICER/amsterdam_samples.npz`, `./data/NICER/maryland_samples.npz`

## Why MetaModel without CSE for NICER?

The MetaModel without CSE is appropriate for NICER data because:
- NICER constrains the EOS primarily through mass-radius measurements
- These measurements probe densities up to ~2 nsat (central density of typical neutron stars)
- MetaModel is valid up to ~2 nsat without requiring CSE extension
- Omitting CSE reduces computational cost while maintaining physical validity
