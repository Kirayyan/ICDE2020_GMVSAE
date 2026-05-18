#!/usr/bin/env bash
# Score only the Chengdu anomaly npy files listed in configs/cd_selected_anomalies.json
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
export GPU_ID="${GPU_ID:-0}"

# Preview matched files without running the model:
# python3 scripts/score_selected_npy.py --config configs/cd_selected_anomalies.json --dry_run

python3 scripts/score_selected_npy.py \
  --config configs/cd_selected_anomalies.json \
  --gpu_id "${GPU_ID}" \
  --train_if_missing \
  --num_epochs "${NUM_EPOCHS:-10}" \
  "$@"
