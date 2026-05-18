#!/usr/bin/env bash
# Score your pre-built stay/speed anomalies on Porto + Chengdu (no regeneration).
set -euo pipefail
export TF_USE_LEGACY_KERAS=1

# Put anomalies under:
#   data/porto/anomalies_stay_val.json  (or outliers_data*.npy + outliers_idx*.npy)
#   data/porto/anomalies_speed_val.json
#   data/cd/...

python3 scripts/score_prebuilt_anomalies.py \
  --datasets porto cd \
  --anomaly_types stay speed \
  --split val \
  --gpu_id "${GPU_ID:-0}" \
  "$@"
