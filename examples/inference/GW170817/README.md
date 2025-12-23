# GW170817-Only Inference Example

This example demonstrates how to constrain the neutron star equation of state (EOS) using only the GW170817 gravitational wave event.

## Configuration

This configuration uses:
- **Transform**: MetaModel + CSE (Constant Speed Extension) with 8 CSE grid points
- **Priors**: Standard NEP (Nuclear Empirical Parameters) priors (from `../prior/prior.prior`)
- **Likelihoods**: GW170817 gravitational wave event only
- **Random Key**: Automatically added `_random_key` parameter for stochastic mass sampling from the normalizing flow

This is a simpler configuration that focuses solely on gravitational wave constraints, making it ideal for:
- Testing the inference pipeline
- Understanding GW170817 constraints on the EOS
- Faster runs compared to the full multi-messenger analysis

## GW Likelihood Architecture

The GW likelihood uses a **normalizing flow** trained on GW170817 posterior samples:
1. During inference, the `_random_key` parameter is sampled (UniformPrior from 0 to 2³²-1)
2. For each EOS proposal, the likelihood samples binary neutron star masses (m1, m2) from the normalizing flow
3. Tidal deformabilities (Λ1, Λ2) are interpolated from the EOS using the sampled masses
4. The normalizing flow evaluates the log probability of the (m1, m2, Λ1, Λ2) combination
5. Batched sampling (batch size: 10) ensures memory efficiency

**Multiple GW events**: To add more events (e.g., GW190425), simply add them to the `events` list. Each event creates a separate likelihood, and the `CombinedLikelihood` sums them.

## Running the Example

### Testing Mode (Default)

The config is set to `dry_run: true` by default for testing. This will:
- Load the normalizing flow model
- Set up priors and transforms
- Initialize the likelihood
- **Stop before running MCMC sampling**

From the jester repository root directory:

```bash
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/GW170817/config.yaml
```

### Full Inference Run

To run the full MCMC sampling, edit `config.yaml` and set `dry_run: false`, then run:

```bash
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/GW170817/config.yaml
```

**Note**: Full runs can take several hours depending on your hardware.

## Output

Results will be saved to `./outdir_gw170817/` by default, containing:
- `results_production.npz` - MCMC samples (including `_random_key`) and log probabilities
- `eos_samples.npz` - Generated EOS curves from sampled parameters
- `runtime.txt` - Sampling runtime and statistics
- `config_backup.yaml` - Copy of the configuration file used

## Testing Modes

### Validate Only (Fastest)

To quickly check if the configuration is valid **without loading any data**:

```bash
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/GW170817/config.yaml \
    --validate-only
```

This only validates the YAML structure and parameter values.

### Dry Run (Current Default)

To test the full setup **including loading the normalizing flow** but without running MCMC:

```bash
# Already enabled in config.yaml: dry_run: true
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/GW170817/config.yaml
```

This is useful for verifying that:
- The normalizing flow model loads correctly
- The Flow wrapper works
- Priors and transforms initialize properly

## Customization

You can override the output directory:

```bash
uv run python -m jesterTOV.inference.run_inference \
    --config examples/inference/GW170817/config.yaml \
    --output-dir ./my_results/
```

## Requirements

The normalizing flow model must be available at:
```
jesterTOV/inference/flows/models/gw_maf/gw170817/gw170817_gwtc1_lowspin_posterior/
```

This directory should contain:
- `flow_weights.eqx` - Trained flow parameters
- `flow_kwargs.json` - Flow architecture configuration
- `metadata.json` - Training metadata and data bounds

## Why MetaModel + CSE for GW170817?

The CSE (Constant Speed Extension) is particularly important for GW170817 analysis because:
- The binary neutron star merger probes high-density regions of the EOS
- CSE extends the MetaModel to densities up to ~6 nsat, ensuring coverage of the high-density regime
- This allows proper modeling of the tidal deformability during the inspiral phase
