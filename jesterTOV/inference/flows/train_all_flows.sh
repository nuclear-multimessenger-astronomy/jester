#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="./logs/train_all_flows.out"
#SBATCH --job-name="maf"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
source activate /home/twouters2/projects/jester_review/jester/.venv/bin/activate

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=========================================="
echo "Training Normalizing Flows for GW Events"
echo "=========================================="

# Base directory for configs
CONFIG_DIR="./configs"

# GW170817 config files
echo ""
echo "Training flows for GW170817..."
echo "------------------------------------------"

GW170817_CONFIGS=(
    "gw170817/low_spin.yaml"
    "gw170817/high_spin.yaml"
    "gw170817/gwtc1_lowspin.yaml"
    "gw170817/gwtc1_highspin.yaml"
)

for config_file in "${GW170817_CONFIGS[@]}"; do
    echo ""
    echo "Training from config: $config_file"
    echo "----------------------------------------"

    uv run python -m jesterTOV.inference.flows.train_flow "${CONFIG_DIR}/${config_file}"

    echo "✓ Completed: $config_file"
    echo ""
done

# GW190425 config files
echo ""
echo "Training flows for GW190425..."
echo "------------------------------------------"

GW190425_CONFIGS=(
    "gw190425/phenomdnrt_hs.yaml"
    "gw190425/phenomdnrt_ls.yaml"
    "gw190425/phenompnrt_hs.yaml"
    "gw190425/phenompnrt_ls.yaml"
    "gw190425/taylorf2_hs.yaml"
    "gw190425/taylorf2_ls.yaml"
)

for config_file in "${GW190425_CONFIGS[@]}"; do
    echo ""
    echo "Training from config: $config_file"
    echo "----------------------------------------"

    uv run python -m jesterTOV.inference.flows.train_flow "${CONFIG_DIR}/${config_file}"

    echo "✓ Completed: $config_file"
    echo ""
done

echo ""
echo "=========================================="
echo "All flows trained successfully!"
echo "=========================================="
echo "GW170817 models saved to: ${MODELS_DIR}/gw170817/"
echo "GW190425 models saved to: ${MODELS_DIR}/gw190425/"
echo ""
echo "DONE"
echo "$now"