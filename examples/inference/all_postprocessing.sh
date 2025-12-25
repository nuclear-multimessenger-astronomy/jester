#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p cpu
#SBATCH -t 00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output="./postprocessing_log.out"
#SBATCH --job-name="jester_postproc"

now=$(date)
echo "$now"

###
### This is to test updates to the jester postprocessing script easily on multiple inference runs.
###

# Loading modules
source /home/twouters2/projects/jester_review/jester/.venv/bin/activate

echo "=========================================="
echo "=== Running jester postprocessing ==="
echo "=========================================="

# Find all directories containing outdir/eos_samples.npz
INFERENCE_DIR="/home/twouters2/projects/jester_review/jester/examples/inference"

echo "Searching for directories with outdir/eos_samples.npz..."
DIRS=$(find "$INFERENCE_DIR" -type f -name "eos_samples.npz" -path "*/outdir/*" | sed 's|/outdir/eos_samples.npz||' | sort)

if [ -z "$DIRS" ]; then
    echo "No directories found with outdir/eos_samples.npz"
    exit 1
fi

echo "Found the following directories:"
echo "$DIRS"
echo ""

# Run postprocessing for each directory
for dir in $DIRS; do
    echo "=========================================="
    echo "Processing: $dir"
    echo "=========================================="

    # Change to the directory
    cd "$dir" || continue

    # Run postprocessing with all plots enabled
    run_jester_postprocessing \
        --outdir ./outdir \
        --make-all

    echo "Completed: $dir"
    echo ""
done

echo "=========================================="
echo "=== All postprocessing complete ==="
echo "=========================================="

now=$(date)
echo "$now"
