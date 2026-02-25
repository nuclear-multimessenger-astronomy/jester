#!/bin/bash
# Submit all inference jobs found under subdirectories of this script's location.
# For each submit.sh found, cd into its directory and call sbatch submit.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Searching for submit.sh files under: $SCRIPT_DIR"
echo ""

submitted=0
skipped=0

while IFS= read -r submit_script; do
    job_dir="$(dirname "$submit_script")"
    rel_dir="${job_dir#"$SCRIPT_DIR/"}"

    echo "Submitting: $rel_dir"
    if (cd "$job_dir" && sbatch submit.sh); then
        submitted=$((submitted + 1))
    else
        echo "  WARNING: sbatch failed for $rel_dir"
        skipped=$((skipped + 1))
    fi
done < <(find "$SCRIPT_DIR" -mindepth 2 -name "submit.sh" | sort)

echo ""
echo "Done. Submitted: $submitted, Failed: $skipped"
