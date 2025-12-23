#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="./logs/train_all_flows_block_neural.out"
#SBATCH --job-name="bnaf"

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
echo "Using block_neural_autoregressive_flow"
echo "=========================================="

# Base directories
DATA_DIR="../data"
MODELS_DIR="./models/gw_bnaf"

# Training hyperparameters
NUM_EPOCHS=3000
LEARNING_RATE=1e-4
MAX_PATIENCE=500
FLOW_TYPE="block_neural_autoregressive_flow"
NN_DEPTH=4

# GW170817 posterior files
echo ""
echo "Training flows for GW170817..."
echo "------------------------------------------"

GW170817_FILES=(
    "gw170817_low_spin_posterior.npz"
    "gw170817_high_spin_posterior.npz"
    "gw170817_gwtc1_lowspin_posterior.npz"
    "gw170817_gwtc1_highspin_posterior.npz"
)

for posterior_file in "${GW170817_FILES[@]}"; do
    # Extract model name from filename (remove .npz extension)
    model_name="${posterior_file%.npz}"

    echo ""
    echo "Training: $model_name"
    echo "----------------------------------------"

    uv run python -m jesterTOV.inference.flows.train_flow \
        --posterior-file "${DATA_DIR}/gw170817/${posterior_file}" \
        --output-dir "${MODELS_DIR}/gw170817/${model_name}" \
        --num-epochs ${NUM_EPOCHS} \
        --learning-rate ${LEARNING_RATE} \
        --max-patience ${MAX_PATIENCE} \
        --flow-type ${FLOW_TYPE} \
        --nn-depth ${NN_DEPTH} \
        --standardize \
        --plot-corner \
        --plot-losses

    echo "✓ Completed: $model_name"
    echo ""
done

# GW190425 posterior files
echo ""
echo "Training flows for GW190425..."
echo "------------------------------------------"

GW190425_FILES=(
    "gw190425_phenomdnrt-hs_posterior.npz"
    "gw190425_phenomdnrt-ls_posterior.npz"
    "gw190425_phenompnrt-hs_posterior.npz"
    "gw190425_phenompnrt-ls_posterior.npz"
    "gw190425_taylorf2-hs_posterior.npz"
    "gw190425_taylorf2-ls_posterior.npz"
)

for posterior_file in "${GW190425_FILES[@]}"; do
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
        --nn-depth ${NN_DEPTH} \
        --max-samples ${MAX_SAMPLES} \
        --standardize \
        --plot-corner \
        --plot-losses

    echo "✓ Completed: $model_name"
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
