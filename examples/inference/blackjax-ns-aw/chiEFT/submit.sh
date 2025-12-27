#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 10:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=40G
#SBATCH --output="./outdir/log_ns_aw.out"
#SBATCH --job-name="NS_chiEFT"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
source /home/twouters2/projects/jester_review/jester/.venv/bin/activate

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=========================================="
echo "=== Running jester inference (NS-AW) ==="
echo "=========================================="

run_jester_inference config.yaml
