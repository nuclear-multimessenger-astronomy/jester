#!/bin/bash
# Clean all SMC-RandomWalk inference jobs, so we can check them from scratch

SKIP_DIRS=("prior")

for dir in */; do
    dir=${dir%/}  # Remove trailing slash

    # Skip if in skip list
    for skip in "${SKIP_DIRS[@]}"; do
        if [[ "$skip" == "$dir" ]]; then
            echo "Skipping $dir"
            continue 2
        fi
    done

    # Submit if submit.sh exists
    if [ -f "$dir/submit.sh" ]; then
        echo "Cleaning $dir"
        rm -rf "$dir/outdir"
    fi
done
