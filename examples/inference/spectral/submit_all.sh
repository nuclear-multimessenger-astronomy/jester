#!/bin/bash
# Submit all SMC-RandomWalk inference jobs

SKIP_DIRS=("prior" "radio")

echo "Submitting all SMC-RW inference jobs..."
echo "   (But skipping: ${SKIP_DIRS[*]})"

for dir in */; do
    dir=${dir%/}  # Remove trailing slash

    # Skip directories in SKIP_DIRS
    skip_dir=false
    for skip in "${SKIP_DIRS[@]}"; do
        if [[ "$skip" == "$dir" ]]; then
            echo "Skipping $dir"
            skip_dir=true
            break
        fi
    done
    if [ "$skip_dir" = true ]; then
        continue
    fi

    # Submit if submit.sh exists
    if [ -f "$dir/submit.sh" ]; then
        echo "Submitting $dir"
        pushd "$dir" >/dev/null || { echo "Failed to enter $dir" >&2; continue; }
        sbatch submit.sh
        popd >/dev/null || { echo "Failed to return from $dir" >&2; exit 1; }
    fi
done
