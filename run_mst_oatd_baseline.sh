#!/usr/bin/env bash
# End-to-end: MST-OATD-style stay/speed anomalies on Porto + Chengdu
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
export GPU_ID="${GPU_ID:-0}"

# If you already have MST-OATD npy under data/porto and data/cd, skip this step.
if [ ! -f ./data/porto/test_data_init.npy ]; then
  echo "No MST npy found — creating synthetic smoke-test data."
  python3 scripts/create_synthetic_mst_npy.py --datasets porto cd
fi

for DS in porto cd; do
  python3 scripts/convert_mst_npy_to_gmvsae.py --dataset "${DS}"
  python3 scripts/generate_mst_anomalies.py --dataset "${DS}" --split val \
    --distance 2 --fraction 0.2 --observed_ratio 1.0 --ratio 0.05 --anomaly_type both
done

python3 scripts/score_mst_oatd_datasets.py --datasets porto cd --anomaly_types stay speed --split val --gpu_id "${GPU_ID}"

echo "Scores written to:"
echo "  data/porto/scores_stay_val.csv"
echo "  data/porto/scores_speed_val.csv"
echo "  data/cd/scores_stay_val.csv"
echo "  data/cd/scores_speed_val.csv"
