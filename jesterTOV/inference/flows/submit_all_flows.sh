#!/bin/bash

# This script submits individual SLURM jobs for training all normalizing flows
# Each dataset has its own directory with config.yaml and submit.sh

echo "==========================================="
echo "Submitting Flow Training Jobs"
echo "==========================================="

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Base directory for dataset configs
MODELS_DIR="./models/gw_maf"

# Find all config.yaml files in the models directory
CONFIG_DIRS=$(find "$MODELS_DIR" -type f -name "config.yaml" -exec dirname {} \; | sort)

# Count total configs
TOTAL_CONFIGS=$(echo "$CONFIG_DIRS" | wc -l | tr -d ' ')

echo ""
echo "Found $TOTAL_CONFIGS dataset configurations to train"
echo ""

# Submit jobs for all configs
for config_dir in $CONFIG_DIRS; do
    # Extract dataset name from directory path
    dataset_name=$(basename "$config_dir")
    event=$(basename "$(dirname "$config_dir")")

    echo "Submitting job for: $event/$dataset_name"
    echo "  Directory: $config_dir"

    # cd into the dataset directory and submit the job
    (
        cd "$config_dir" || exit 1

        # Check if submit.sh exists
        if [ ! -f "submit.sh" ]; then
            echo "  ✗ ERROR: submit.sh not found in $config_dir"
            exit 1
        fi

        # Submit the job
        sbatch submit.sh

        if [ $? -eq 0 ]; then
            echo "  ✓ Job submitted successfully"
        else
            echo "  ✗ Job submission failed"
        fi
    )

    echo ""
done

echo "==========================================="
echo "All jobs submitted!"
echo "==========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs in: ./logs/"
echo ""
