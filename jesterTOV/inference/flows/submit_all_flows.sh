#!/bin/bash

# This script submits individual SLURM jobs for training all normalizing flows

echo "==========================================="
echo "Submitting Flow Training Jobs"
echo "==========================================="

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Base directory for configs
CONFIG_DIR="./configs"

# GW170817 config files
GW170817_CONFIGS=(
    "gw170817/low_spin.yaml"
    "gw170817/high_spin.yaml"
    "gw170817/gwtc1_lowspin.yaml"
    "gw170817/gwtc1_highspin.yaml"
)

# GW190425 config files
GW190425_CONFIGS=(
    "gw190425/phenomdnrt_hs.yaml"
    "gw190425/phenomdnrt_ls.yaml"
    "gw190425/phenompnrt_hs.yaml"
    "gw190425/phenompnrt_ls.yaml"
    "gw190425/taylorf2_hs.yaml"
    "gw190425/taylorf2_ls.yaml"
)

# Combine all configs
ALL_CONFIGS=("${GW170817_CONFIGS[@]}" "${GW190425_CONFIGS[@]}")

echo ""
echo "Found ${#ALL_CONFIGS[@]} configurations to train"
echo ""

# Submit jobs for all configs
for config_file in "${ALL_CONFIGS[@]}"; do
    config_path="${CONFIG_DIR}/${config_file}"

    # Extract a job name from the config file path
    # e.g., gw170817/low_spin.yaml -> gw170817_low_spin
    job_name=$(echo "$config_file" | sed 's/\//_/g' | sed 's/.yaml//')

    echo "Submitting job for: $config_file"
    echo "  Job name: $job_name"
    echo "  Config path: $config_path"

    # Submit the job with the config file as an environment variable
    sbatch --job-name="$job_name" --export=CONFIG_FILE="$config_path" submit.sh

    echo "  ✓ Job submitted"
    echo ""
done

echo "==========================================="
echo "All jobs submitted successfully!"
echo "==========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs in: ./logs/"
echo ""
