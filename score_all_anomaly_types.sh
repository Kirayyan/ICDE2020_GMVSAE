#!/usr/bin/env bash
# Pattern-based scan (includes detour). For fixed file list without detour use: sh score_selected.sh
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
export GPU_ID="${GPU_ID:-0}"

python3 scripts/score_pattern_npy.py \
  --datasets porto cd \
  --patterns "_stay_" "speed_accelerate" "speed_decelerate" \
  --split "${SPLIT:-init}" \
  --eval_split val \
  --gpu_id "${GPU_ID}" \
  --train_if_missing \
  --num_epochs "${NUM_EPOCHS:-10}" \
  "$@"
