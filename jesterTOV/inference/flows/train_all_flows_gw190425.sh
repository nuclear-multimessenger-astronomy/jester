#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 05:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="./logs/train_all_flows_gw190425_part2.out"
#SBATCH --job-name="GW190425_maf"

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

# Base directories
DATA_DIR="../data"
MODELS_DIR="./models/gw_maf"

# Training hyperparameters (matching test.sh configuration)
NUM_EPOCHS=3000
LEARNING_RATE=1e-4
MAX_PATIENCE=500
FLOW_TYPE="masked_autoregressive_flow"
TRANSFORMER="rational_quadratic_spline"
TRANSFORMER_KNOTS=8
TRANSFORMER_INTERVAL=4
NN_DEPTH=4

# gw190425 posterior files
echo ""
echo "Training flows for gw190425..."
echo "------------------------------------------"

gw190425_FILES=(
    "gw190425_phenompnrt-ls_posterior.npz"
    "gw190425_taylorf2-hs_posterior.npz"
    "gw190425_taylorf2-ls_posterior.npz"
)


for posterior_file in "${gw190425_FILES[@]}"; do
    # Extract model name from filename (remove .npz extension)
    model_name="${posterior_file%.npz}"

    echo ""
    echo "Training: $model_name"
    echo "----------------------------------------"

    uv run python -m jesterTOV.inference.flows.train_flow \
        --posterior-file "${DATA_DIR}/gw190425/${posterior_file}" \
        --output-dir "${MODELS_DIR}/gw190425/${model_name}" \
        --num-epochs ${NUM_EPOCHS} \
        --learning-rate ${LEARNING_RATE} \
        --max-patience ${MAX_PATIENCE} \
        --flow-type ${FLOW_TYPE} \
        --transformer ${TRANSFORMER} \
        --transformer-knots ${TRANSFORMER_KNOTS} \
        --transformer-interval ${TRANSFORMER_INTERVAL} \
        --nn-depth ${NN_DEPTH} \
        --standardize \
        --plot-corner \
        --plot-losses

    echo "✓ Completed: $model_name"
    echo ""
done
