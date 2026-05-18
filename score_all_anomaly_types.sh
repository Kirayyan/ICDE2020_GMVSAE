#!/usr/bin/env bash
# Score stay / speed / detour npy separately (only matches files you name-pattern)
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
export GPU_ID="${GPU_ID:-0}"

# Your naming examples:
#   outliers_data_init_stay_0p35_25_1.npy
#   outliers_data_init_speed_accelerate4_0p4_1.npy
#   outliers_data_init_detour_3_0p1_0p7.npy

python3 scripts/score_pattern_npy.py \
  --datasets porto cd \
  --patterns "_stay_" "speed_accelerate" "detour" \
  --split "${SPLIT:-init}" \
  --eval_split val \
  --gpu_id "${GPU_ID}" \
  --train_if_missing \
  --num_epochs "${NUM_EPOCHS:-10}" \
  "$@"
