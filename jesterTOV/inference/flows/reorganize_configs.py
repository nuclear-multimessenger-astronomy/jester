#!/usr/bin/env python3
"""Reorganize flow training configs into new directory structure."""

import os
import shutil
from pathlib import Path

# Base directories
FLOWS_DIR = Path(__file__).parent
OLD_CONFIGS = FLOWS_DIR / "models" / "configs"
NEW_BASE = FLOWS_DIR / "models" / "gw_maf"

# Mapping: old_config_file -> new_dataset_dir
CONFIG_MAPPING = {
    "gw170817/high_spin.yaml": "gw170817/gw170817_high_spin",
    "gw170817/low_spin.yaml": "gw170817/gw170817_low_spin",
    "gw170817/gwtc1_highspin.yaml": "gw170817/gw170817_gwtc1_highspin",
    "gw170817/gwtc1_lowspin.yaml": "gw170817/gw170817_gwtc1_lowspin",
    "gw190425/phenomdnrt_hs.yaml": "gw190425/gw190425_phenomdnrt_hs",
    "gw190425/phenomdnrt_ls.yaml": "gw190425/gw190425_phenomdnrt_ls",
    "gw190425/phenompnrt_hs.yaml": "gw190425/gw190425_phenompnrt_hs",
    "gw190425/phenompnrt_ls.yaml": "gw190425/gw190425_phenompnrt_ls",
    "gw190425/taylorf2_hs.yaml": "gw190425/gw190425_taylorf2_hs",
    "gw190425/taylorf2_ls.yaml": "gw190425/gw190425_taylorf2_ls",
}

def update_paths_in_config(config_text: str, event: str, dataset: str) -> str:
    """Update paths in config file for new directory structure.

    From: models/gw_maf/gw170817/gw170817_high_spin/config.yaml
    - Data path: ../../../../../data/gw170817/...
    - Output path: ../gw170817_high_spin_posterior
    """
    lines = []
    for line in config_text.split('\n'):
        if line.startswith('posterior_file:'):
            # Extract filename from old path
            old_path = line.split('"')[1]
            filename = old_path.split('/')[-1]
            # New path: 5 levels up to reach inference/, then data/
            new_path = f"../../../../../data/{event}/{filename}"
            lines.append(f'posterior_file: "{new_path}"')
        elif line.startswith('output_dir:'):
            # Output goes one level up, into a _posterior directory
            output_name = f"{dataset}_posterior"
            new_path = f"../{output_name}"
            lines.append(f'output_dir: "{new_path}"')
        else:
            lines.append(line)
    return '\n'.join(lines)

def create_submit_script(dataset_dir: Path, config_relpath: str):
    """Create individual submit.sh script for a dataset."""
    submit_content = f"""#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="../../../../logs/%x.out"
#SBATCH --job-name="{dataset_dir.name}"

# Training script for {dataset_dir.name}

now=$(date)
echo "$now"

echo "Training flow for: {dataset_dir.name}"
echo "Config: ./config.yaml"
echo ""

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
source activate /home/twouters2/projects/jester_review/jester/.venv/bin/activate

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=========================================="
echo "=== Training Normalizing Flow for GW ==="
echo "=========================================="
echo "Dataset: {dataset_dir.name}"
echo ""

# Train using config in current directory
train_jester_gw_flow "./config.yaml"

echo ""
echo "DONE"
echo "$now"
"""
    submit_path = dataset_dir / "submit.sh"
    submit_path.write_text(submit_content)
    submit_path.chmod(0o755)
    print(f"  Created {submit_path}")

def main():
    print("=" * 60)
    print("Reorganizing Flow Training Configs")
    print("=" * 60)

    # Create new directory structure
    for old_config_rel, new_dataset_rel in CONFIG_MAPPING.items():
        old_config = OLD_CONFIGS / old_config_rel
        new_dataset_dir = NEW_BASE / new_dataset_rel

        print(f"\nProcessing: {old_config_rel}")
        print(f"  → {new_dataset_dir.relative_to(FLOWS_DIR)}")

        # Create new directory
        new_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Read old config
        if not old_config.exists():
            print(f"  ✗ ERROR: {old_config} not found!")
            continue

        config_text = old_config.read_text()

        # Extract event name from dataset name
        event = "gw170817" if "gw170817" in new_dataset_rel else "gw190425"
        dataset_name = new_dataset_dir.name

        # Update paths
        updated_config = update_paths_in_config(config_text, event, dataset_name)

        # Write new config
        new_config = new_dataset_dir / "config.yaml"
        new_config.write_text(updated_config)
        print(f"  ✓ Created {new_config.relative_to(FLOWS_DIR)}")

        # Create submit script
        create_submit_script(new_dataset_dir, str(new_config.relative_to(FLOWS_DIR)))

    print("\n" + "=" * 60)
    print("Reorganization complete!")
    print("=" * 60)
    print(f"\nNew structure created in: {NEW_BASE.relative_to(FLOWS_DIR)}")
    print("\nNext steps:")
    print("1. Verify paths in config.yaml files")
    print("2. Update submit_all_flows.sh to use new structure")
    print("3. Test with a single config before running all")

if __name__ == "__main__":
    main()
