# Data protocol and provenance

## Reporting protocol and acquisition field of view

The author requested the following reporting settings on 12 October 2026. Counts are train/test image counts, not recovered experiment manifests.

| Dataset | Original images | Native width x height | Acquisition FOV | Train / test |
|---|---:|---|---:|---:|
| DRIVE | 40 | 565 x 584 | 45° | 20 / 20 |
| CHASE_DB1 | 28 | 999 x 960 | 30° | 20 / 8 |
| STARE | 20 | 700 x 605 | 35° | 10 / 10 |
| HRF | 45 | 3504 x 2336 | 60° | 25 / 20 |

The DRIVE, CHASE_DB1, and STARE angular values follow the author-specified Table 1 update. HRF is verified against Section 3 of Odstrcilik et al., *Retinal vessel segmentation by improved matched filtering: evaluation on a new high-resolution fundus image database*, IET Image Processing 7(4), 373-383 (2013), [doi:10.1049/iet-ipr.2012.0455](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2012.0455). That original segmentation dataset contains 45 images acquired with a CANON CF-60 UVi / EOS-20D system at 60°. The [provider's page](https://www5.cs.fau.de/research/data/fundus-images/) separately describes an image-quality-assessment dataset acquired at 45°; it must not be substituted for the segmentation dataset.

An angular FOV is an acquisition property. The manifest field `fov` instead identifies a binary pixel mask defining the evaluated image region. An angle does not create or replace such a mask.

The supplied manuscript previously described STARE as five-fold evaluation. The requested change to 10/10 does not establish which images produced the existing results. Original split IDs, checkpoint-selection records, and the association between the STARE scores and the requested protocol remain pending. Existing result CSVs are preserved as manuscript transcriptions, not labelled as newly rerun 10/10 results. See `results/provenance.json` and `splits/protocol_status.json`.

## Evidence and original data

Obtain data from their original providers; do not commit images or derivative patches. The earlier inspected attachment contained 40 DRIVE, 20 STARE, 28 CHASE_DB1, and 45 HRF originals. DRIVE's supplied training/test directories each contained 20 images. The legacy example scripts selected 15/5 STARE and 21/7 CHASE splits; the latter divided one child's eyes between partitions. These were legacy examples, not recovered CGDD-Net experiment manifests. Original image assignments for the reported CHASE, STARE, and HRF experiments were not supplied.

The earlier `datasets/augmented` directory contained derivatives, not additional independent originals, and is excluded from discovery. Legacy patch-level train/validation splitting is not used to establish image independence. New experiments validate original-image and available subject identities before extracting patches. Both CHASE eyes of a child must remain in one partition. HRF has 15 healthy, 15 diabetic-retinopathy, and 15 glaucoma images; its current reporting count is 25/20, superseding the old incomplete 15/20 metadata.

## Preprocessing and evaluation

The reference configuration uses one grayscale channel scaled to `[0,1]`, with no CLAHE, gamma correction, ImageNet normalization, or target-statistics re-estimation. The preprocessing settings are:

```json
{"color_mode": "gray", "max_side": null, "pad_multiple": 1}
```

`max_side` optionally shrinks images while preserving aspect ratio; it changes the evaluated geometry and must be recorded for any new experiment. `pad_multiple` adds bottom/right padding only. Image resizing uses bilinear interpolation; labels and FOV masks use nearest-neighbor interpolation. Padded mask values are zero. Masks encoded as 0/1 or 0/255 are accepted. A missing FOV is explicit (`null`) and means evaluation on the full original image; empty masks and incompatible shapes are rejected. Record mask origin rather than inferring it from the angular FOV.

Training patches are sampled from original images or explicitly recorded resized images. Flips and right-angle rotations are applied jointly to image, label, and FOV; derivatives must retain original identity. Call `PatchDataset.set_epoch(epoch)` and use fresh or nonpersistent workers so the epoch-dependent sample stream changes as intended. The cache is configurable; HRF images can require substantial host memory.

Inference averages sigmoid probabilities from overlapping 96x96 patches with stride 16 before applying threshold 0.5. Metrics are computed per image inside its FOV and then averaged without pixel-count weighting. AUC uses reconstructed probabilities. Resized/padded outputs must not be compared as if they had identical native geometry without documenting restoration and evaluation settings.

## Portable manifest and validation

```json
{
  "schema_version": 1,
  "dataset": "CHASEDB1",
  "split": "train",
  "protocol_status": "newly_generated_not_original_server_split",
  "strategy": "holdout",
  "seed": 42,
  "data_root": "../../datasets",
  "records": [{
    "id": "CHASEDB1/Image_01L",
    "original_id": "CHASEDB1/Image_01L",
    "dataset": "CHASEDB1",
    "subject": "CHASEDB1:01",
    "image": "CHASEDB1/images/Image_01L.jpg",
    "label": "CHASEDB1/1st_label/Image_01L_1stHO.png",
    "fov": "CHASEDB1/mask/Image_01L.png"
  }]
}
```

This is a schema example, not an actual manuscript assignment. Paths must remain within the selected data root. Root priority is an explicit `data_root`, then `CGDD_DATA_ROOT`, then the manifest-relative root. `load_case` returns float32 image `(C,H,W)`, label `(1,H,W)`, and FOV `(1,H,W)` arrays.

`validate_splits` rejects empty sets, missing files, duplicate record/original identities, overlapping resolved paths and subjects, and exact-byte duplicate image content. CHASE left/right identities are also derived from filenames. These checks cannot identify every re-encoding or renamed augmentation; retain original IDs. Discovery excludes augmented directories. `splits/protocol_status.json` records evidence status and is not a runnable split manifest.

## Explicitly generate a new protocol

The following examples create new experiments, not reconstructed manuscript runs. Existing manifests are never overwritten.

```bash
python scripts/prepare_data.py inspect --data-root ../datasets

# Preserve the official DRIVE test set; newly split the development set into 18/2.
python scripts/prepare_data.py generate --dataset DRIVE --strategy drive-official \
  --data-root ../datasets --output my_protocols/DRIVE --seed 42

# NEW STARE holdout: 10 development images (8 train + 2 validation), 10 test.
# This is NOT evidence that the original manuscript trained on these image IDs.
python scripts/prepare_data.py generate --dataset STARE --strategy holdout \
  --ratios 0.4 0.1 0.5 --data-root ../datasets --output my_protocols/STARE --seed 42

# NEW grouped CHASE holdout; both eyes of each child remain together.
python scripts/prepare_data.py generate --dataset CHASEDB1 --strategy holdout \
  --ratios 0.6 0.2 0.2 --data-root ../datasets --output my_protocols/CHASEDB1 --seed 42

# NEW stratified HRF holdout: 27 train / 9 val / 9 test, not the reported 25/20.
python scripts/prepare_data.py generate --dataset HRF --strategy holdout \
  --ratios 0.6 0.2 0.2 --data-root ../datasets --output my_protocols/HRF --seed 42

python scripts/prepare_data.py validate --train my_protocols/STARE/train.json \
  --val my_protocols/STARE/val.json --test my_protocols/STARE/test.json --data-root ../datasets
```

For an exact historical 10/10 or 25/20 reproduction, use the corresponding original image IDs and actual validation/checkpoint-selection policy; do not select new IDs and call them historical. The generic k-fold generator remains available for explicitly different new experiments. DRIVE, STARE, and HRF use image-level grouping unless validated subject information is supplied; this is not a claim that image IDs prove subject independence.

## Statistical sample-size audit

`python scripts/exact_p_bounds.py` reports the minimum two-sided exact rank-sum p-value, `2 / comb(n1+n2, n1)`, under independent exchangeable groups with no ties and a doubled smaller tail. Assuming one AUC per test image gives group sizes 20/20 (DRIVE), 8/8 (CHASE_DB1), and 10/10 (STARE). These are mathematical bounds, not empirical Table 7 values, and have no significance stars. All methods share the same bound within a dataset. Self-comparison is not applicable.

The statistic, actual sample unit, ties, sidedness, and exact/asymptotic method must be identified before any empirical recalculation. `summarize_runs.py` is a separate paired repeated-training-run utility, not an implementation of the manuscript's rank-sum analysis. Mean and standard deviation alone do not recover the required observations or ranks.
