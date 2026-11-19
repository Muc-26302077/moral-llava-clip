#!/usr/bin/env bash
# Launch the recoverable validation-selected experiment and its 07:45 reporter.
set -euo pipefail
SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCENE_ROOT"
SCENE_PYTHON="${SCENE_PYTHON:-python}"
: "${BASE_SNAPSHOT:?Set BASE_SNAPSHOT to the pinned LLaVA snapshot directory}"
: "${MORALCLIP_MODEL:?Set MORALCLIP_MODEL to the MoralCLIP directory}"
SCENE_RUN="${1:?Pass a new output directory (or an existing one with --resume)}"
if [[ "$SCENE_RUN" != /* ]]; then SCENE_RUN="$SCENE_ROOT/$SCENE_RUN"; fi
SCENE_EXTRA=()
if [[ "${2:-}" == --resume ]]; then SCENE_EXTRA+=(--resume); fi
if [[ -e "$SCENE_RUN" && ${#SCENE_EXTRA[@]} -eq 0 ]]; then
  printf 'Output already exists: %s\n' "$SCENE_RUN" >&2
  exit 2
fi
command -v screen >/dev/null
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
"$SCENE_PYTHON" - <<'PYGPU'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit('A CUDA GPU supporting BF16 is required; no training started.')
PYGPU
mkdir -p "$SCENE_RUN"
SCENE_SESSION="scene-v2-$(date +%s)"
screen -dmS "$SCENE_SESSION" -L -Logfile "$SCENE_RUN/orchestrator.log" \
  "$SCENE_PYTHON" -u scripts/train_scene_gpu.py --run "$SCENE_RUN" "${SCENE_EXTRA[@]}"
screen -dmS "$SCENE_SESSION-report" "$SCENE_PYTHON" scripts/scene_training_report.py --run "$SCENE_RUN" --watch
printf 'Training session: %s\nRun directory: %s\n' "$SCENE_SESSION" "$SCENE_RUN"
