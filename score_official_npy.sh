#!/usr/bin/env bash
# GMVSAE baseline with MST-OATD official .npy (Porto + Chengdu)
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
export GPU_ID="${GPU_ID:-0}"

# Expected under data/porto/ and data/cd/:
#   train_data_init.npy, test_data_init.npy
#   outliers_data_init_2_0.2_1.0.npy, outliers_idx_init_2_0.2_1.0.npy

python3 scripts/score_official_npy.py \
  --datasets porto cd \
  --distance "${DISTANCE:-2}" \
  --fraction "${FRACTION:-0.2}" \
  --observed_ratio "${OBSERVED_RATIO:-1.0}" \
  --split val \
  --gpu_id "${GPU_ID}" \
  --train_if_missing \
  --num_epochs "${NUM_EPOCHS:-10}" \
  "$@"
