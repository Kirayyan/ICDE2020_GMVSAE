#!/usr/bin/env bash
# Score stay acceleration anomalies (stay_accelerate* npy pairs)
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
export GPU_ID="${GPU_ID:-0}"

# Pattern in filename, e.g. stay_accelerate3 / stay_accelerate4
PATTERN="${PATTERN:-stay_accelerate}"
# Empty = all splits (init, 1..10); or: SPLIT=init  SPLIT=10
SPLIT="${SPLIT:-}"

python3 scripts/score_pattern_npy.py \
  --datasets porto cd \
  --pattern "${PATTERN}" \
  ${SPLIT:+--split "$SPLIT"} \
  --eval_split val \
  --gpu_id "${GPU_ID}" \
  --train_if_missing \
  --num_epochs "${NUM_EPOCHS:-10}" \
  "$@"
