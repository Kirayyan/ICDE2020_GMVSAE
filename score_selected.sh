#!/usr/bin/env bash
# Score selected stay/speed npy on Porto + Chengdu (no detour, no pattern scan)
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
export GPU_ID="${GPU_ID:-0}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

for CFG in porto_selected_anomalies cd_selected_anomalies; do
  echo "======== configs/${CFG}.json ========"
  "${PY}" "${ROOT}/scripts/score_selected_npy.py" \
    --config "${ROOT}/configs/${CFG}.json" \
    --gpu_id "${GPU_ID}" \
    --train_if_missing \
    --num_epochs "${NUM_EPOCHS:-10}" \
    "$@"
done
