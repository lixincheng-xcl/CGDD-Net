#!/usr/bin/env bash
set -euo pipefail
DATASET=${1:-DRIVE}
CONFIG=${2:-configs/default.yaml}
for SEED in 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035; do
  python train.py --config "$CONFIG" --dataset "$DATASET" --ablation full --seed "$SEED" \
    --output "runs/${DATASET}/full/seed_${SEED}"
done
