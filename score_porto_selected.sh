#!/usr/bin/env bash
# Porto only. Prefer: sh score_selected.sh (Porto + CD together)
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
export GPU_ID="${GPU_ID:-0}"
python3 scripts/score_selected_npy.py \
  --config configs/porto_selected_anomalies.json \
  --gpu_id "${GPU_ID}" \
  --train_if_missing \
  --num_epochs "${NUM_EPOCHS:-10}" \
  "$@"
