#!/bin/bash
# Submit all SMC-RandomWalk inference jobs

SKIP_DIRS=("prior" "radio")

echo "Submitting all SMC-RW inference jobs..."
echo "   (But skipping: ${SKIP_DIRS[*]})"

for dir in */; do
    dir=${dir%/}  # Remove trailing slash

    # Skip directories in SKIP_DIRS
    if [[ " ${SKIP_DIRS[@]} " =~ " ${dir} " ]]; then
        echo "Skipping $dir"
        continue
    fi

    # Submit if submit.sh exists
    if [ -f "$dir/submit.sh" ]; then
        echo "Submitting $dir"
        cd "$dir"
        sbatch submit.sh
        cd ..
    fi
done
