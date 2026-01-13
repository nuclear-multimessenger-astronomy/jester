#!/bin/bash
# Submit all SMC-RandomWalk inference jobs

for dir in */; do
    dir=${dir%/}  # Remove trailing slash

    # Submit if submit.sh exists
    if [ -f "$dir/submit.sh" ]; then
        echo "Submitting $dir"
        cd "$dir"
        sbatch submit.sh
        cd ..
    fi
done
