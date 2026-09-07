# Reproducibility Guide

This document records what is included in the public code release and what must be supplied externally.

## Included

- complete CGDD-Net architecture;
- seven cumulative ablation configurations;
- preprocessing and patch sampling;
- full-image sliding-window validation and testing;
- SE, SP, ACC, F1, and ROC-AUC evaluation;
- repeated-run scripts;
- cross-dataset evaluation entry point;
- parameter/FLOP proxy profiler;
- architecture smoke tests.

## Not redistributed

- public dataset image files;
- trained checkpoints;
- run-level probability maps;
- exact experiment logs from the manuscript.

These artifacts may be large or subject to third-party dataset terms.

## Default protocol encoded in `configs/default.yaml`

```yaml
train:
  patch_size: 48
  sampling_window: 64
  patches_per_epoch: 25600
  batch_size: 64
  epochs: 50
  lr: 0.0005
  early_stop_patience: 6

inference:
  patch_size: 96
  stride: 16
  threshold: 0.5
```

The optimizer is Adam and the learning-rate schedule is cosine annealing.

## Determinism

`seed_everything` seeds Python, NumPy, and PyTorch. DataLoader workers are also seeded. Set:

```yaml
train:
  deterministic: true
```

for deterministic backend settings where supported.

## Validation and checkpointing

The training script never selects a checkpoint from the test manifest. The best checkpoint is selected by validation AUC from the configured validation manifest.

## Metric aggregation

`test.py` computes metrics per image and then reports the arithmetic mean across images. Per-image values are written to `per_image.csv`, making the aggregation explicit.

## Repeated runs

Use:

```bash
bash scripts/run_repeats.sh DRIVE configs/default.yaml
python tools/summarize_repeats.py runs/DRIVE/full
```

to summarize repeated training seeds.

## Verification

Run:

```bash
pytest -q
python examples/quickstart.py
```

before starting full experiments.
