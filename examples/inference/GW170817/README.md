# GW170817-Only Inference Example

This example demonstrates how to constrain the neutron star equation of state (EOS) using only the GW170817 gravitational wave event.

## Configuration

This configuration uses:
- **Transform**: MetaModel + CSE (Constant Speed Extension) with 8 CSE grid points
- **Priors**: Standard NEP (Nuclear Empirical Parameters) priors
- **Likelihoods**: GW170817 gravitational wave event only

This is a simpler configuration that focuses solely on gravitational wave constraints, making it ideal for:
- Testing the inference pipeline
- Understanding GW170817 constraints on the EOS
- Faster runs compared to the full multi-messenger analysis

## Running the Example

From the jester repository root directory:

```bash
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/gw170817_only/config.yaml
```

## Output

Results will be saved to `./outdir_gw170817/` by default, containing:
- `results_production.npz` - MCMC samples and log probabilities
- `eos_samples.npz` - Generated EOS curves from sampled parameters
- `runtime.txt` - Sampling runtime and statistics
- `config_backup.yaml` - Copy of the configuration file used

## Customization

You can override the output directory:

```bash
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/gw170817_only/config.yaml \
    --output-dir ./my_results/
```

Or validate the configuration without running:

```bash
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/gw170817_only/config.yaml \
    --validate-only
```

## Requirements

Before running, ensure you have the required data files:
- GW170817 normalizing flow model: `./NFs/GW170817/model.eqx`

## Why MetaModel + CSE for GW170817?

The CSE (Constant Speed Extension) is particularly important for GW170817 analysis because:
- The binary neutron star merger probes high-density regions of the EOS
- CSE extends the MetaModel to densities up to ~6 nsat, ensuring coverage of the high-density regime
- This allows proper modeling of the tidal deformability during the inspiral phase
