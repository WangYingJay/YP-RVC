#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PORT="${PORT:-7860}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export RVC_DATASET_DIR="${RVC_DATASET_DIR:-$(pwd)/datasets}"
export RVC_LOG_DIR="${RVC_LOG_DIR:-$(pwd)/logs}"
mkdir -p "$RVC_DATASET_DIR" "$RVC_LOG_DIR" "$RVC_LOG_DIR/weights" assets/weights
exec python webui.py
