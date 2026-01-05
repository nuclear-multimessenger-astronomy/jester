#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="../../../../logs/%x.out"
#SBATCH --job-name="gw170817_high_spin"

# Training script for gw170817_high_spin

now=$(date)
echo "$now"

echo "Training flow for: gw170817_high_spin"
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
echo "Dataset: gw170817_high_spin"
echo ""

# Train using config in current directory
train_jester_gw_flow "./config.yaml"

echo ""
echo "DONE"
echo "$now"
