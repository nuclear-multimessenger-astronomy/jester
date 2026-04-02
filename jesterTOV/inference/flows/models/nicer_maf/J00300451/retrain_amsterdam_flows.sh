#!/usr/bin/env bash
# Submit all Amsterdam J0030 flow training jobs to the cluster.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for dir in "$SCRIPT_DIR"/amsterdam_*/; do
    echo "Submitting: $dir"
    cd "$dir"
    sbatch submit.sh
    cd "$SCRIPT_DIR"
done
