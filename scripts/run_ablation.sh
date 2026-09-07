#!/usr/bin/env bash
set -euo pipefail
DATASET=${1:-DRIVE}
CONFIG=${2:-configs/default.yaml}
SEED=${3:-2026}
for A in baseline csde samg dcdf detail_decoder selective_skip full; do
  python train.py --config "$CONFIG" --dataset "$DATASET" --ablation "$A" --seed "$SEED" \
    --output "runs/${DATASET}/${A}/seed_${SEED}"
done
