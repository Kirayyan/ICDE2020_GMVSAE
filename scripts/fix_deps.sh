#!/usr/bin/env bash
# Reinstall deps with compatible pins (fixes protobuf 7.x vs tensorboard conflict)
set -euo pipefail
pip install -U pip
pip uninstall -y tensorflow tf-keras keras tensorboard protobuf 2>/dev/null || true
pip install -r requirements.txt
python3 -c "import tensorflow as tf; import tf_keras; print('tensorflow', tf.__version__); import google.protobuf; print('protobuf', google.protobuf.__version__)"
