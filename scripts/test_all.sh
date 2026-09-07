#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/default.yaml}
for DATASET in DRIVE CHASE_DB1 STARE HRF; do
  CKPT="runs/${DATASET}/full/seed_2026/best.pt"
  python test.py --config "$CONFIG" --dataset "$DATASET" --checkpoint "$CKPT" --save-predictions
done
