#!/bin/bash
# Wrapper script — alternative to the absolute-path approach in submit.sub.
# Use this as the executable if you need full venv environment activation
# (e.g. if run_jester_inference reads VIRTUAL_ENV or other venv variables).
#
# To use: set in submit.sub:
#   executable          = run_jester.sh
#   transfer_executable = true

set -e

VENV=/home/thibeau.wouters/jester_analyses/jester/.venv
source "${VENV}/bin/activate"

now=$(date)
echo "${now}"
echo "Python: $(which python) ($(python --version))"
echo "run_jester_inference: $(which run_jester_inference)"

nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No GPU detected"

echo "=========================================="
echo "=== Running jester inference (CIT GPU) ==="
echo "=========================================="

exec run_jester_inference "$@"
