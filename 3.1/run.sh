#!/usr/bin/env bash
set -euo pipefail
section_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$section_root"
python_bin="${PYTHON:-python}"
stage="${1:-verify}"
if [[ $# -gt 0 ]]; then shift; fi
run_dir=""
if [[ $# -gt 0 ]]; then
  if [[ $# != 2 || "$1" != --run-dir ]]; then
    echo 'Usage: bash run.sh [verify|score|infer] [--run-dir PATH]' >&2
    exit 1
  fi
  run_dir="$2"
fi
case "$stage" in
  verify)
    [[ -z "$run_dir" ]] || { echo 'verify 不使用 --run-dir。' >&2; exit 1; }
    exec "$python_bin" scripts/verify_release.py
    ;;
  score)
    run_dir="${run_dir:-runs/local-score-$(date +%Y%m%d-%H%M%S)}"
    mkdir -- "$run_dir"
    for decoding in constrained free; do
      "$python_bin" -m moral_llava.crossmodal score \
        --protocol runs/crossmodal-v1/protocol.json \
        --base-predictions "runs/crossmodal-v1/full/base_${decoding}.jsonl" \
        --lora-predictions "runs/crossmodal-joint-v1-test/lora_${decoding}.jsonl" \
        --output "$run_dir/metrics_${decoding}.json"
    done
    printf '评分完成：%s\n' "$run_dir"
    ;;
  infer)
    : "${BASE_SNAPSHOT:?请设置 BASE_SNAPSHOT，指向固定 revision 命名的 LLaVA 本地快照目录。}"
    run_dir="${run_dir:-runs/local-inference-$(date +%Y%m%d-%H%M%S)}"
    mkdir -- "$run_dir"
    "$python_bin" -m moral_llava.crossmodal prepare --output "$run_dir/protocol.json"
    "$python_bin" -m moral_llava.infer_crossmodal \
      --protocol "$run_dir/protocol.json" --base-model "$BASE_SNAPSHOT" \
      --adapter runs/crossmodal-joint-v1/epoch-002 \
      --device cuda --precision bf16 --batch-size 8 --models both --decoding both \
      --output "$run_dir/predictions"
    ;;
  *)
    echo 'Usage: bash run.sh [verify|score|infer] [--run-dir PATH]' >&2
    exit 1
    ;;
esac
