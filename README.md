# CGDD-Net

**Context-Guided Dynamic Detail Modeling for Retinal Vessel Segmentation**

Authors: **Xincheng Li, Xinyu Zhang, Xiaoqi Sheng**.

School of Computer Science, The University of Auckland, New Zealand; School of Future Technology, South China University of Technology, China.

A PyTorch implementation organized around the manuscript's completed architecture figures and the [VesselSeg-Pytorch](https://github.com/lee-zq/VesselSeg-Pytorch) data/training workflow. It includes CSDE, SAMG, DCDF, detail-guided decoding, selective skips and criss-cross attention, with seven cumulative ablations.

[中文说明](README_zh.md) · [Architecture](docs/architecture.md) · [Data protocol](docs/data_protocol.md) · [Release status](docs/release_status.md)

![CGDD-Net architecture](assets/architecture.png)

## Release status

This is a newly constructed, figure-aligned implementation. The authors confirm that the manuscript results were verified on their server, but this source release does not contain that server's original checkpoints, run configurations or split manifests. Its synthetic tests verify software behavior, not benchmark reproduction.

The default implementation has **1,956,802 trainable parameters (1.96 M)** with stage widths **8, 16, 32, 64, 128**. The manuscript now uses this measured parameter count. The reported **20.05 G FLOPs** remains an earlier server measurement whose input and counting convention have not been matched to this release. The supported-operator estimates in `docs/profile_measured.json` are partial counts, not a replacement for full FLOPs or measured latency.

## Installation

Python 3.10 or newer is required. Install a suitable PyTorch build using the [official installation selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke_test.py
```

Local CPU validation used Python 3.12.2 and PyTorch 2.3.1. The author-reported experiment environment is NVIDIA H200 141 GB, PyTorch 2.13.0 and CUDA 13.2. No CUDA extension or third-party deformable-convolution build is required.

## Data and explicit splits

Obtain DRIVE, STARE, CHASE_DB1 and HRF from their original providers, retain their supplied labels/FOV masks, and keep the image data outside this repository. The inherited directory layout and manifest schema are documented in [data_protocol.md](docs/data_protocol.md).

```bash
python scripts/prepare_data.py inspect --data-root ../datasets
```

Use the actual train/validation/test manifests from your experiment. If starting a **new experiment**, the following explicitly creates a new DRIVE protocol: it preserves the official 20 test images and selects 2 validation images from the 20-image development set.

```bash
python scripts/prepare_data.py generate --data-root ../datasets \
  --dataset DRIVE --strategy drive-official --seed 42 --output splits/new_drive
python scripts/prepare_data.py validate --data-root ../datasets \
  --train splits/new_drive/train.json --val splits/new_drive/val.json \
  --test splits/new_drive/test.json
```

CHASE eyes are grouped by subject. Other new holdouts and grouped folds are available through `--help`. Generated protocols are labelled as new experiments, not the historical manuscript partitions. No dataset images or fabricated result files are included. Training/validation separation occurs before patch sampling.

## Training and evaluation

```bash
python train.py --config configs/default.json \
  --train-manifest splits/new_drive/train.json --val-manifest splits/new_drive/val.json \
  --data-root ../datasets --output runs/drive_seed42 --device cuda

python evaluate.py --checkpoint runs/drive_seed42/best.pt \
  --manifest splits/new_drive/test.json --data-root ../datasets \
  --output runs/drive_seed42/test --device cuda
```

Choose `--device cpu` for a small functional check. The base learning rate in `configs/default.json` is `0.001`, confirmed by the author for the reported server experiments on 2026-09-10. Other configuration values remain release defaults unless separately confirmed; the file is not a recovered server configuration. Adam uses linear warmup followed by cosine restarts; the threshold is 0.5. The model emits logits and uses numerically stable FOV-weighted binary cross entropy. Inference averages overlapping patch probabilities before thresholding.

A run stores `config.json`, `environment.json`, copied manifests and their hashes, `history.csv`, `best.pt`, `latest.pt`, and `training_summary.json`. Evaluation exports per-image probabilities, binary predictions, `per_image.csv`, `summary.json`, and checkpoint/configuration provenance. Metrics are computed inside each image's FOV and then averaged without pixel-count weighting. Undefined AUCs remain explicit. Image-to-image standard deviation must not be reported as seed-to-seed training variation. Optional resizing changes the evaluation resolution; each CSV row records original and evaluated dimensions.

Resume using the **same configuration, manifests and original output directory**:

```bash
python train.py --config configs/default.json \
  --train-manifest splits/new_drive/train.json --val-manifest splits/new_drive/val.json \
  --data-root ../datasets --output runs/drive_seed42 --device cuda \
  --resume runs/drive_seed42/latest.pt
```

The matching `best.pt` must remain alongside the resume checkpoint. This preserves the historical best model even if resumed epochs do not improve it. Checkpoints are loaded with PyTorch's weights-only loader.

For cross-dataset evaluation, keep the source checkpoint/configuration fixed and pass a target dataset manifest to `evaluate.py`. Do not tune the checkpoint or threshold against the target test labels.

## Ablations, profiling and statistics

Set `model.ablation` to one of:

`baseline`, `csde`, `samg`, `dcdf`, `detail_decoder`, `selective_skip`, `full`.

The implementation choices for partial configurations and their actual parameter counts are documented in `docs/architecture.md`. Use matched seeds and partitions for new comparisons.

```bash
python scripts/profile_model.py --config configs/default.json --height 96 --width 96
# Optional operator-count estimate; unsupported operations are always disclosed:
python -m pip install -e '.[profile]'
python scripts/profile_model.py --config configs/default.json --height 96 --width 96 --fvcore

# Aggregate actual independent training runs; input schema is in --help:
python scripts/summarize_runs.py completed_runs.csv --reference CGDD-Net --output statistics.json
```

The statistics utility computes sample SD across seed-level rows and, if requested, two-sided paired Wilcoxon tests with Holm correction within each dataset. It does not infer a test from a table of means and SDs and does not retroactively validate the manuscript's existing p-values.

## Loss function

Training minimizes FOV-masked binary cross entropy from logits:

```python
loss = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * fov).sum() / fov.sum()
```

Only pixels inside the FOV contribute. Empty FOV masks are rejected. There are no additional Dice, boundary, topology or deep-supervision loss terms and no class reweighting. Sigmoid is applied for inference, not before this loss.

## Author-supplied paper values

| Dataset | SE | SP | ACC | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| DRIVE | 0.8421 | 0.9768 | 0.9704 | 0.8323 | 0.9824 |
| CHASE_DB1 | 0.8600 | 0.9815 | 0.9752 | 0.8102 | 0.9938 |
| STARE | 0.8661 | 0.9812 | 0.9775 | 0.8510 | 0.9895 |
| HRF | 0.8362 | 0.9823 | 0.9711 | 0.8157 | 0.9874 |

Machine-readable transcriptions and provenance are under `results/`. They are independent of locally generated test outputs. Hyperparameter Figure 8 is under author revision and is not used to choose this release's defaults.

## Publishing and attribution

This folder is the repository root: upload its contents to GitHub, or use `python scripts/build_release.py` to create a clean source ZIP. The source pack excludes datasets, checkpoints, local environments and runtime outputs. See [release notes](RELEASE_NOTES.md) for the migration from the earlier YAML-based implementation.

See [中文归档说明](docs/reproducibility_zh.md) for how to attach final configurations, split manifests and actual run outputs to a fixed release. Add the final paper citation and author metadata when approved; no journal acceptance or DOI is asserted here.

The inherited workflow is acknowledged in `NOTICE` and its Apache-2.0 license is retained in `LICENSE`. The earlier repository's MIT notice is preserved in `LICENSES/legacy-MIT.txt`. Dataset access and redistribution are governed by the original providers, separately from the software license.
