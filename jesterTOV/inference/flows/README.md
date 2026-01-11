# Normalizing Flow Training for GW Posteriors

TODO: this must be moved perhaps to a docs page.

This directory contains tools for training normalizing flows on gravitational wave posterior samples.

## Quick Start

### Local Training (Single Flow)

Train a flow on your local machine:

```bash
uv run train_jester_gw_flow ./configs/gw170817/low_spin.yaml
```

### SLURM Submission (Single Flow)

Submit a single flow training job to SLURM:

```bash
sbatch --job-name="gw170817_lowspin" \
       --export=CONFIG_FILE="./configs/gw170817/low_spin.yaml" \
       submit.sh
```

### SLURM Submission (All Flows)

Submit training jobs for all configured flows:

```bash
bash submit_all_flows.sh
```

This will submit 10 individual SLURM jobs:
- 4 GW170817 configurations
- 6 GW190425 configurations

## Directory Structure

```
flows/
├── configs/                  # YAML configuration files
│   ├── gw170817/            # GW170817 configs
│   │   ├── low_spin.yaml
│   │   ├── high_spin.yaml
│   │   ├── gwtc1_lowspin.yaml
│   │   └── gwtc1_highspin.yaml
│   └── gw190425/            # GW190425 configs
│       ├── phenomdnrt_hs.yaml
│       ├── phenomdnrt_ls.yaml
│       ├── phenompnrt_hs.yaml
│       ├── phenompnrt_ls.yaml
│       ├── taylorf2_hs.yaml
│       └── taylorf2_ls.yaml
├── submit.sh                 # Generic SLURM submit script
├── submit_all_flows.sh       # Submit all flow training jobs
├── train_flow.py             # Training script
├── config.py                 # Configuration schema
└── logs/                     # SLURM output logs

## Scripts

### `train_jester_gw_flow` (CLI command)

Command-line tool for training a single flow.

**Usage:**
```bash
train_jester_gw_flow <config.yaml>
```

**Example:**
```bash
train_jester_gw_flow ./configs/gw170817/low_spin.yaml
```

### `submit.sh` (SLURM script)

Generic SLURM submission script for a single flow training job.

**SLURM Parameters:**
- 1 H100 GPU
- 3 hours walltime
- 10GB memory
- Logs saved to `./logs/`

**Usage:**
```bash
sbatch --job-name="my_flow" \
       --export=CONFIG_FILE="./configs/path/to/config.yaml" \
       submit.sh
```

### `submit_all_flows.sh` (Batch submission)

Submits individual SLURM jobs for all flow configurations.

**Usage:**
```bash
bash submit_all_flows.sh
```

**What it does:**
1. Creates `./logs/` directory if it doesn't exist
2. Loops over all config files in `configs/`
3. Submits one SLURM job per config
4. Each job runs independently and in parallel

## Configuration Files

Each YAML config file specifies:
- Data file path (`posterior_file`)
- Output directory (`output_dir`)
- Flow architecture (type, layers, network size)
- Training hyperparameters (epochs, learning rate, patience)
- Data preprocessing options (standardization, max samples)
- Plotting options (corner plots, loss curves)

See `config.py` for the full configuration schema.

## Outputs

Each training run produces:

```
output_dir/
├── flow_weights.eqx          # Trained model parameters
├── flow_kwargs.json          # Architecture configuration
├── metadata.json             # Training metadata
└── figures/
    ├── losses.png            # Training/validation loss curves
    ├── corner.png            # Data vs flow samples comparison
    └── transformed_training_data.png  # (if physics constraints enabled)
```

## Monitoring Jobs

View running jobs:
```bash
squeue -u $USER
```

View logs (while running or after completion):
```bash
tail -f ./logs/gw170817_low_spin.out
```

Cancel a job:
```bash
scancel <job_id>
```

## Migration from Old Workflow

**Old workflow (deprecated):**
```bash
bash train_all_flows.sh  # Single job, sequential training
```

**New workflow (recommended):**
```bash
bash submit_all_flows.sh  # Multiple jobs, parallel training
```

The new workflow:
- ✅ Faster: All flows train in parallel
- ✅ Fault-tolerant: One failed flow doesn't stop others
- ✅ Better resource usage: Scheduler optimizes GPU allocation
- ✅ Individual logs: Each flow has its own log file

## Using Trained Flows

Load a trained flow in Python:

```python
from jesterTOV.inference.flows.flow import Flow
import jax

# Load flow
flow = Flow.from_directory("./models/gw_maf/gw170817/gw170817_gwtc1_lowspin_posterior")

# Sample from flow
samples = flow.sample(jax.random.key(0), (1000,))

# Evaluate log probability
log_prob = flow.log_prob(samples)
```

See `train_flow.py` docstring for detailed API documentation.
