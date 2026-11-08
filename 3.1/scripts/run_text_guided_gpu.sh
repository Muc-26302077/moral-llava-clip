#!/usr/bin/env bash
# Run only after a CUDA GPU has been enabled. Each run gets isolated artifacts.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/moral-llava/.venv/bin/python}"
RUN_ROOT="${1:-${PROJECT_ROOT}/results/runs/text-guided-gpu-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${RUN_ROOT}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/run_text_guided_vision.py" preflight --output "${RUN_ROOT}/preflight" | tee "${RUN_ROOT}/preflight.log"
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/run_text_guided_vision.py" smoke --output "${RUN_ROOT}/smoke" 2>&1 | tee "${RUN_ROOT}/smoke.log"
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/run_text_guided_vision.py" experiment --output "${RUN_ROOT}/formal" 2>&1 | tee "${RUN_ROOT}/formal.log"
echo "Completed. Results and Chinese outcome text: ${RUN_ROOT}/formal/"
