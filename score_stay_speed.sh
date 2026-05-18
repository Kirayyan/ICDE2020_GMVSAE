#!/usr/bin/env bash
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
# 1) generate stay/speed anomalies manifest
# 2) score with trained checkpoint and export CSV
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
python3 scripts/generate_anomalies.py --split val --ratio 0.05 \
  --output ./data/anomalies_stay_speed.json
python3 run_loop.py --mode=score --cluster_num=5 --gpu_id="${GPU_ID:-0}" \
  --model_dir=./ckpt --eval_data=val --anomaly_path=./data/anomalies_stay_speed.json \
  --output_scores=./data/anomaly_scores.csv
