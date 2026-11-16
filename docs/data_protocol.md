# Data protocol and provenance

## Reporting protocol and acquisition field of view

Dissertation Table 4.2 gives the training and test counts in original fundus images. All STARE experiments used the same 10/10 split.

| Dataset | Original images | Native width x height | Acquisition FOV | Train / test |
|---|---:|---|---:|---:|
| [DRIVE](https://doi.org/10.1109/TMI.2004.825627) | 40 | 565 x 584 | 45° | 20 / 20 |
| [CHASE_DB1](https://doi.org/10.1109/TBME.2012.2205687) | 28 | 999 x 960 | 30° | 20 / 8 |
| [STARE](https://doi.org/10.1109/42.845178) | 20 | 700 x 605 | 35° | 10 / 10 |
| [HRF](https://doi.org/10.1155/2013/154860) | 45 | 3504 x 2336 | 60° | 25 / 20 |

For HRF, Section 3 of Odstrcilik et al., *Retinal vessel segmentation by improved matched filtering: evaluation on a new high-resolution fundus image database*, describes the 60° acquisition used for the 45 segmentation images. The provider also lists a separate 45° image-quality dataset. [Acquisition paper](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2012.0455) · [Dataset page](https://www5.cs.fau.de/research/data/fundus-images/)

The manifest field `fov` points to the binary pixel mask used for evaluation. Appendix A identifies DRIVE images 21–40 as the training set and 01–20 as the test set. For CHASE_DB1, STARE and HRF, the records give split sizes but omit the individual filenames. The historical subject assignments for the paired CHASE_DB1 eyes therefore remain unknown.

## Evidence and original data

Obtain images from the original providers; do not commit images or derivative patches. The previously inspected attachment contained 40 DRIVE, 20 STARE, 28 CHASE_DB1, and 45 HRF originals. DRIVE's supplied training and test directories each contained 20 images. Legacy example scripts selected 15/5 STARE and 21/7 CHASE; these were not records of the actual CGDD-Net experiments. The author-confirmed protocols above take precedence for the manuscript.

New experiments validate original-image and available subject identities before extracting patches. Both CHASE eyes of a child must remain in one partition. HRF contains 15 healthy, 15 diabetic-retinopathy, and 15 glaucoma images. Augmented images are derivatives, not additional independent original observations. The generic generator does not recover historical image membership.

## Preprocessing and evaluation

The reference configuration uses one grayscale channel scaled to `[0,1]`, with no CLAHE, gamma correction, ImageNet normalization, or target-statistics re-estimation:

```json
{"color_mode": "gray", "max_side": null, "pad_multiple": 1}
```

`max_side` optionally shrinks images while preserving aspect ratio; `pad_multiple` adds bottom/right padding. Such changes must be recorded. Image resizing uses bilinear interpolation; labels and FOV masks use nearest-neighbor interpolation. Padded mask values are zero. A missing FOV is explicit (`null`) and means full-image evaluation; empty masks and incompatible shapes are rejected. Record mask provenance separately from the acquisition angle.

Training uses 64x64 patches, joint horizontal/vertical flips and rotations of image, label and FOV, batch size 64, Adam at 0.001, dropout 0.1, and 100 epochs. Default early stopping is disabled to preserve the fixed training budget. The current entry point changes the patch-sampling seed each epoch. New repository runs use image-disjoint training/validation manifests; their checkpoint-selection records do not redefine historical runs.

At inference, a 64×64 window moves 16 pixels at a time along each axis, with eight windows per mini-batch. For the native image dimensions below, the evaluator uses reflection padding at the right and bottom edges where needed. Dissertation Table A.1 gives the resulting number of windows per image.

| Dataset | Padded width × height (pixels) | Horizontal × vertical positions | Windows per image |
|---|---:|---:|---:|
| DRIVE | 576 × 592 | 33 × 34 | 1,122 |
| CHASE_DB1 | 1,008 × 960 | 60 × 57 | 3,420 |
| STARE | 704 × 608 | 41 × 35 | 1,435 |
| HRF | 3,504 × 2,336 | 216 × 143 | 30,888 |

Each fully covered interior pixel receives 16 sigmoid probabilities. Pixels near the edges receive fewer. The evaluator divides each pixel's accumulated probability by its own window count and crops the map to the original image size. AUC uses the cropped probabilities. SE, SP, ACC and F1 use the vessel mask obtained by thresholding at 0.5. Each metric is evaluated inside the supplied FOV or over the full image if the FOV is absent. Each image with a defined value contributes equally to that metric's dataset mean.

The dissertation also reports mIoU and HD95 for the complete model and the seven ablation configurations. HD95 is given as a mean and standard deviation. Section 4.1.4 gives the metric definitions and records which details are unspecified: the classes averaged for mIoU, and the distance unit and directional convention for HD95.

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

This illustrates the schema, not a manuscript assignment. Paths stay within the selected data root. Root priority is explicit `data_root`, then `CGDD_DATA_ROOT`, then the manifest-relative root. `load_case` returns float32 image `(C,H,W)`, label `(1,H,W)`, and FOV `(1,H,W)` arrays.

`validate_splits` rejects empty sets, missing files, duplicate original identities, overlapping paths/subjects, and exact-byte duplicate image content. CHASE eye identities are also derived from filenames. Retain original IDs for derivatives because no file-name or byte-hash check detects every re-encoding. `splits/protocol_status.json` records protocol provenance and is not a runnable image manifest.

## Explicitly generate a new protocol

These commands create new experiments, not recovered manuscript runs. Existing manifests are never overwritten.

```bash
python scripts/prepare_data.py inspect --data-root ../datasets

# Preserve DRIVE's official test set; newly split development images into 18/2.
python scripts/prepare_data.py generate --dataset DRIVE --strategy drive-official \
  --data-root ../datasets --output my_protocols/DRIVE --seed 42

# NEW STARE experiment: 8 train + 2 validation + 10 test.
# This is a new validation-based protocol, not the historical 10-training-image run.
python scripts/prepare_data.py generate --dataset STARE --strategy holdout \
  --ratios 0.4 0.1 0.5 --data-root ../datasets --output my_protocols/STARE --seed 42

# NEW grouped CHASE holdout; keep both eyes of each child together.
python scripts/prepare_data.py generate --dataset CHASEDB1 --strategy holdout \
  --ratios 0.6 0.2 0.2 --data-root ../datasets --output my_protocols/CHASEDB1 --seed 42

# NEW stratified HRF holdout: 27 train / 9 val / 9 test, not historical 25/20.
python scripts/prepare_data.py generate --dataset HRF --strategy holdout \
  --ratios 0.6 0.2 0.2 --data-root ../datasets --output my_protocols/HRF --seed 42

python scripts/prepare_data.py validate --train my_protocols/STARE/train.json \
  --val my_protocols/STARE/val.json --test my_protocols/STARE/test.json --data-root ../datasets
```

Historical reproduction requires the original image IDs and actual checkpoint-selection policy, not merely equal split counts. Generic k-fold generation remains an option for explicitly different new experiments. DRIVE, STARE and HRF use image-level grouping unless validated subject information is supplied.

## Statistical sample-size audit

Table 7 of the earlier manuscript compared per-image AUC using the Wilcoxon rank-sum test, with 20 images for DRIVE, eight for CHASE_DB1 and ten for STARE. The dissertation comparison tables report descriptive scores. The original per-image AUC vectors for those rank-sum comparisons are unavailable in this release.

`python scripts/exact_p_bounds.py` reports `2 / comb(n1+n2, n1)` under an untied, independent-exchangeable, doubled-tail exact rank-sum model. It is a theoretical diagnostic, not an observed p-value. Equal sample sizes give equal bounds, regardless of the methods' AUCs. The actual raw vectors, ties and computational method are required for recalculation; no exact-method claim is added to the manuscript. `summarize_runs.py` is a separate repeated-training-run analysis utility.
