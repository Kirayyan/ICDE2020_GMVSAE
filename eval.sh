#!/usr/bin/env bash
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
OTYPE="${OTYPE:-stay}"
python3 run_loop.py --mode=eval --cluster_num=5 --gpu_id="${GPU_ID:-0}" \
  --model_dir=./ckpt --partial_ratio=1.0 --eval_data=val --otype="${OTYPE}" --outlier_ratio=0.05
