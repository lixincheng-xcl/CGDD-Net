#!/usr/bin/env bash
set -euo pipefail
# Example: CUDA_VISIBLE_DEVICES=0 bash scripts/train_server.sh DRIVE full 2026
DATASET=${1:-DRIVE}
ABLATION=${2:-full}
SEED=${3:-2026}
CONFIG=${CONFIG:-configs/default.yaml}
python train.py --config "$CONFIG" --dataset "$DATASET" --ablation "$ABLATION" --seed "$SEED" \
  --output "runs/${DATASET}/${ABLATION}/seed_${SEED}"
