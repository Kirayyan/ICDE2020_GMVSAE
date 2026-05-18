#!/usr/bin/env bash
set -euo pipefail
export TF_USE_LEGACY_KERAS=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
python3 run_loop.py --mode=pretrain --cluster_num=5 --num_epochs=2 --gpu_id="${GPU_ID:-0}" \
  --learning_rate=0.001 --model_dir=./pretrain --batch_size=64 --log_steps=5
