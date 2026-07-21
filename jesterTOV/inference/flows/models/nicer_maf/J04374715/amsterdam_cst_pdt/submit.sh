#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="training_log.out"
#SBATCH --job-name="amsterdam_cst_pdt"

now=$(date)
echo "$now"
echo "Training flow for: amsterdam_cst_pdt"
source /home/twouters2/projects/43_eos_bayesian_updates/new_flowjax_jester/.venv/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader
train_jester_flow "./config.yaml"
echo "DONE"
echo "$now"
