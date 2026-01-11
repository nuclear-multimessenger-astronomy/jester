#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="../../../../logs/%x.out"
#SBATCH --job-name="gw170817_gwtc1_highspin"

now=$(date)
echo "$now"
echo "Training flow for: gw170817_gwtc1_highspin"
source /home/twouters2/projects/jester_review/jester/.venv/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader
train_jester_gw_flow "./config.yaml"
echo "DONE"
echo "$now"
