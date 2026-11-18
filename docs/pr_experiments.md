# CGDD-Net — Pattern Recognition controlled experiments

**Status:** implementation and software tests only. No retinal benchmark data or
H200 GPU are available in the authoring environment. This folder must not be
used as evidence that model training or metric tables have been completed.

## Research questions and architecture variants

| Variant | Change from released full CGDD-Net | Purpose |
|---|---|---|
| `full` | Exactly the registered full CGDDNet architecture | Reference |
| `independent_detail` | Three independently parameterized DCDF modules, one for each of D3/D2/D1, all using the same gated E2/E3/E4 input features | Test shared representation vs independently constructed details; report increased parameters |
| `fixed_offsets` | Keep four heads and sampling lattices; remove each learned offset predictor and fix its 18 displacement channels to zero | Test geometry adaptation |
| `uniform_samg` | Keep four convolution kernels; replace global-local softmax weights by arithmetic mean; delete unused gates | Test dynamic scale selection |
| `global_only_samg` | Keep only global adaptive weighting | Decompose gate |
| `local_only_samg` | Keep only spatial adaptive weighting | Decompose gate |
| `unet_same_width` | Five encoder widths `8,16,32,64,128`, two CBRs per stage; same input, loss and evaluation | Capacity-controlled architectural baseline |
| `unet_budget_matched` | Search integer U-Net base width minimizing the absolute parameter-count difference to full CGDDNet | Parameter-budget-controlled baseline, report residual mismatch |

The independent-detail variant is **not** parameter matched. It should be
compared jointly with complexity measurements; do not attribute an apparent gain
solely to sharing or remove the extra parameters from reported counts. A
parameter-matched independent-detail model may be added *after* the initial
measurements. The reference `full` retains a single CCA at D3.

## Original-image splits required before training

Actual source image IDs, label filenames, FOV masks, and subject IDs must be
provided. Never derive the historic split from example code or from a new seed.

| Dataset | Original images in train | Original images in test | Additional constraint |
|---|---:|---:|---|
| DRIVE | 20 | 20 | Preserve the official split |
| CHASE_DB1 | 20 | 8 | Keep both eyes of each child in a single split |
| STARE | 10 | 10 | Use the author-confirmed fixed 10/10 membership |
| HRF | 25 | 20 | Record H/DR/G composition and image IDs |

Store at `manifests/DRIVE/{train,test}.json`, etc., in the format documented in
`docs/data_protocol.md`. Original dataset imagery must stay outside GitHub.

## New controlled training protocol

The original paper scores remain immutable. **New experiments** train each
configuration on *all* designated training images for a fixed **100 epochs** and
select the **last checkpoint**, not the checkpoint preferred by test AUC. There
is no test-set validation or early stopping in this new experiment protocol.
Any historical manuscript protocol that used a different checkpoint-selection
rule must be distinguished; these are new experiments, not reconstructed runs.

- Grayscale in [0,1], 64x64 training patches, 64x64 overlapping inference patches,
  inference stride 16, reconstructed whole-image probabilities and FOV masks.
- Adam, base LR 0.001, training batch 64, dropout 0.1, 100 epochs, 150,000
  sampled training patches per epoch. Warmup and cosine restarts from config.
- Seed plan 42, 2026, 3407 (change only before launching the complete matrix).
- Test metrics: per-image AUC, SE, SP, ACC, F1. Also save probabilities and
  thresholded masks for all original test images.
- When comparing models, use exactly the same train/test image IDs, loss,
  augmentation, training budget, reconstruction and threshold.

### Dry-run planning

```bash
python experiments/launch_grid.py \
  --manifest-root manifests --data-root /path/to/original_datasets \
  --output-root runs/pr_controls \
  --plan-output pr_experiment_plan.json
```

The default matrix is 5 variants x 4 datasets x 3 seeds = **60 runs**. The
dry-run creates JSON commands only; it does not produce AUC values. Add
`--execute --device cuda` **only on a machine with an appropriate GPU and the
real dataset directories mounted**. A `.pt` checkpoint and original full-image
predictions are produced for each completed run; existing run directories are
not overwritten. Use `--resume` to resume an interrupted run with compatible
config/manifests. Only use `--smoke --device cpu --execute` to validate software;
smoke results are flagged and omitted from aggregate benchmark tables.

The fixed training budget of 100 x 150,000 patches is substantial. Estimate
hours and GPU resource use from one representative real-data run before
launching the full 60-job matrix.

### Aggregate only completed runs

```bash
python experiments/collect_results.py \
  --runs-root runs/pr_controls \
  --output runs/pr_summary
```

The aggregator requires all requested seeds for each included method/dataset
and excludes smoke tests. It exports `run_metrics.csv`, `per_image_metrics.csv`
and `aggregate_metrics.csv` with **between-seed SD explicitly distinct from
between-image SD**. Do not mix patches, images and random seeds as independent
statistical replicates. Preselect paired comparison, multiple-testing family,
bootstrap unit and subject clustering, especially for CHASE_DB1.

### Additional topology and thin-vessel metrics

```bash
python experiments/structure_metrics.py \
  --evaluation runs/pr_controls/DRIVE/full/seed_42/test \
  --test-manifest manifests/DRIVE/test.json \
  --data-root /path/to/original_datasets \
  --config runs/pr_controls/DRIVE/full/seed_42/config.json \
  --output runs/pr_controls/DRIVE/full/seed_42/structural
```

Requires `scikit-image` in addition to scipy. Define thin centerlines by the
lowest quartile of ground-truth skeleton radii *within each image*, with a
2-pixel matching tolerance. Report clDice, thin-skeleton recall and FOV false
positive rate. Parameters must be fixed before looking at comparative scores.
Absolute vessel thickness in pixels is resolution dependent, so compare within
dataset and predeclare any alternative physical/normalized width rule.

### Complexity and external baselines

Use the repository `scripts/profile_model.py` with explicit input, counting
library and unsupported operations. A complete comparison must use the *same*
operator rules across every model. The existing 29.96 GFLOPs paper claim is
**not** independently reproduced by the partial operator diagnostics in the
public repository. The new variants' parameter counts are measured from their
actual registered trainable parameters. For true latency measure the actual
H200 forward/reconstruction workload with warmup and device synchronization.

PA-Net and FSG-Net must be trained/evaluated only from verified source releases
with their input preprocessing and split assignments documented. Their
published scores cannot be relabelled as same-protocol measured baselines.

## Integrity and audit rules

1. Save source commit SHA, Python, torch/CUDA versions, GPU name, full config,
   manifest hashes, logs and per-image prediction files.
2. Name every software-only test as a **smoke test**, not an experiment result.
3. Require every reported quantitative row to resolve to saved originals,
   checkpoint, prediction maps and a known evaluation script.
4. Report outcomes even when a control ablation underperforms or a gain lacks
   statistical significance. No invented values, no tuned results using test.
5. Keep new runs separate from `results/paper_*.csv` and paper Table 7/10.

## Dependencies and test suite

```bash
python -m pip install -e '.[dev]'
python -m pip install scikit-image
python -m pytest -q tests/test_pr_experiments.py
```

These tests verify shape, gradient connectivity, parameter and gating variants,
planning, and one entirely synthetic I/O smoke run. They are *not* retinal
training or performance evidence.