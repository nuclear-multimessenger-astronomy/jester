#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="./logs/%x.out"
#SBATCH --job-name="train_flow"

# This script submits a single flow training job
# Usage: sbatch --export=CONFIG_FILE=<path_to_config.yaml> submit.sh

now=$(date)
echo "$now"

# Validate that CONFIG_FILE was provided
if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: CONFIG_FILE environment variable not set"
    echo "Usage: sbatch --export=CONFIG_FILE=<path_to_config.yaml> submit.sh"
    exit 1
fi

echo "Config file: $CONFIG_FILE"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
source activate /home/twouters2/projects/jester_review/jester/.venv/bin/activate

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=========================================="
echo "=== Training Normalizing Flow for GW ==="
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo ""

train_jester_gw_flow "$CONFIG_FILE"

echo ""
echo "DONE"
echo "$now"
