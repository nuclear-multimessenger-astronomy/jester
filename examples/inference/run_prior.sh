#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Manually specified subdirs (relative to this script). If non-empty, only
# these are run. Leave the array empty to auto-discover all "prior" dirs.
MANUAL_DIRS=(
    "smc_random_walk/prior"
    "mm_peakcse/prior"
    "spectral_reparam/prior"
    "anisotropy/prior"
)

run_in_dir() {
    local dir="$1"
    echo "==> Running inference in: $dir"
    cd "$dir"
    uv run run_jester_inference config.yaml
    cd "$SCRIPT_DIR"
}

if [[ ${#MANUAL_DIRS[@]} -gt 0 ]]; then
    for rel_dir in "${MANUAL_DIRS[@]}"; do
        run_in_dir "$SCRIPT_DIR/$rel_dir"
    done
else
    while IFS= read -r -d '' prior_dir; do
        run_in_dir "$prior_dir"
    done < <(find "$SCRIPT_DIR" -type d -name "prior" -print0 | sort -z)
fi
