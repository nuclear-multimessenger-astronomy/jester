#!/bin/bash
# Submit all SMC-RandomWalk inference jobs

SKIP_DIRS=("prior")

for dir in */; do
    dir=${dir%/}  # Remove trailing slash

    # Skip if in skip list
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
