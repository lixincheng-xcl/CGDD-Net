# Data protocol and provenance

## What the supplied attachment establishes

The baseline repository is `VesselSeg-Pytorch-master`. Its original dataset directories were inspected on 2026-09-10 without altering the supplied files.

| Dataset | Original images | Native width × height | Labels / FOV masks | Existing image split evidence |
|---|---:|---|---|---|
| DRIVE | 40 | 565 × 584 | `1st_manual` / `mask` | The supplied `training` and `test` folders contain 20 images each. |
| STARE | 20 | 700 × 605 | `1st_labels_ah` / `mask` | `prepare_dataset/stare.py` selects sorted indices `[0:5]` for testing and the other 15 for training. This is a single example holdout, not a five-fold definition. |
| CHASE_DB1 | 28 | 999 × 960 | `1st_label` / `mask` | `prepare_dataset/chasedb1.py` selects the first 7 sorted images for testing and the other 21 for training. This divides subject 04's left and right eyes between splits. |
| HRF | 45 | 3504 × 2336 | `manual1` / `mask` | The attachment contains 15 h, 15 dr, and 15 g images, but no HRF split script or train/validation/test lists. |

The `prepare_dataset/data_path_list` directory contains no actual `.txt` image lists. There is no evidence in this attachment for an existing HRF 15/20 split with ten remaining images, a server STARE five-fold image assignment, or subject-disjoint CHASE server results. This release does not represent newly generated partitions as recovered experimental records.

The separate `datasets/augmented` folder contains 80 training images and 20 test images. These derivatives are not independent originals and are excluded from discovery. `mask` denotes field of view (FOV); vessel ground truth is in the label directories listed above.

The baseline's `function.py` divides a pool of extracted patches into training and validation subsets. That procedure can place patches from the same original image in both subsets. The new release requires separate image manifests before any patches are extracted, and CHASE grouping additionally keeps both eyes of each child together.

## Preprocessing used by this release

The author specified normalization and size handling. This implementation uses 8-bit RGB values divided by 255; `color_mode: gray` explicitly selects Pillow's grayscale conversion followed by division by 255. It does not assume the baseline's CLAHE, gamma correction, or dataset-wide standardization were used in the verified server runs.

The configuration schema is:

```json
{"color_mode": "rgb", "max_side": null, "pad_multiple": 1}
```

- `color_mode`: `rgb` (three channels) or `gray` (one channel). The model input channels must match this value.
- `max_side`: optional maximum image side. Larger images are shrunk uniformly to preserve aspect ratio; images are never stretched to a square. This option changes the input resolution and must be recorded in the experiment configuration.
- `pad_multiple`: optional padding on the bottom and right. Padded label and FOV values are zero; padding never creates evaluable background pixels.

Images use bilinear interpolation when shrinking; ground truth and FOV masks use nearest-neighbor interpolation. Binary masks stored as either 0/1 or 0/255 are supported. A missing FOV is explicit (`null`) and means the entire original image is evaluated. Every supplied original dataset has FOV masks, so generated manifests use them. Empty FOV masks and mismatched image/label/FOV shapes are rejected.

`load_case` returns the processed image dimensions. With the defaults, these are the native dimensions. If `max_side` or `pad_multiple` is changed, downstream predictions are at the resized/padded resolution unless the evaluator explicitly restores their geometry. Record the original and evaluated shapes, and do not compare metrics at different resolutions as if they used an identical protocol. The FOV excludes padded pixels but does not reverse a resize.

Patch sampling uses original images or explicitly resized images, chooses a point inside the FOV, and crops a patch around that point. Images smaller than the patch are zero-padded. Horizontal/vertical flips and right-angle rotations are applied jointly to image, label and FOV; rectangular patches only use rotations that preserve their shape. No derivative image can enter validation or testing as an independent sample.

`PatchDataset.set_epoch(epoch)` changes the deterministic sample stream. The training entry point should call it once per epoch and recreate workers or use nonpersistent DataLoader workers so worker copies see the changed epoch. Cached cases are stored as uint8 to reduce memory; `cache_size` can bound the number of retained cases. High-resolution HRF data can still require substantial host RAM when all images are cached or replicated across DataLoader workers.

## Portable manifest format

```json
{
  "schema_version": 1,
  "dataset": "CHASEDB1",
  "split": "train",
  "protocol_status": "newly_generated_not_original_server_split",
  "strategy": "holdout",
  "seed": 42,
  "data_root": "../../datasets",
  "records": [
    {
      "id": "CHASEDB1/Image_01L",
      "original_id": "CHASEDB1/Image_01L",
      "dataset": "CHASEDB1",
      "subject": "CHASEDB1:01",
      "image": "CHASEDB1/images/Image_01L.jpg",
      "label": "CHASEDB1/1st_label/Image_01L_1stHO.png",
      "fov": "CHASEDB1/mask/Image_01L.png"
    }
  ]
}
```

This example illustrates the schema; it is not a provided training assignment. Image/label/FOV paths are relative to a dataset root and cannot escape that root. The root is selected by the explicit `data_root` argument, then `CGDD_DATA_ROOT`, then the manifest's `data_root` relative to its own directory. Move or download the datasets separately and override the root on another machine. Dataset images must not be committed or bundled with the repository.

`load_manifest(path, data_root=None)` returns validated records with resolved `Path` values. `load_case(record, preprocessing=None)` returns three NumPy float32 arrays: image `(C,H,W)`, vessel label `(1,H,W)`, and FOV `(1,H,W)`. `PatchDataset` returns the same shapes as three float32 PyTorch tensors.

Always call the group validator on all partitions before training:

```python
from cgddnet.data import load_manifest, validate_splits
splits = {name: load_manifest(f"my_protocol/{name}.json", data_root)
          for name in ("train", "val", "test")}
validate_splits(splits)
```

Validation rejects empty sets, missing files, duplicate record/original IDs, duplicate resolved image/label paths, and overlapping subjects. It also hashes image files to detect copied originals with changed filenames and IDs, and derives source identities from known DRIVE, STARE, CHASEDB1, and HRF filename prefixes. CHASE subject IDs are independently derived from `Image_XXL` and `Image_XXR`; conflicting supplied identities are rejected. Exact-byte hashing and filename checks cannot detect arbitrary re-encodings or geometric augmentations whose identities have also been completely renamed, so derivatives must retain `original_id` and must never be independently split. Automatic discovery accepts only original dataset folders, and the manifest loader rejects paths inside an `augmented` directory.

## Inspect without creating a protocol

```bash
python scripts/prepare_data.py inspect --data-root ../datasets
```

No train/validation/test assignments are created by inspection. `splits/protocol_status.json` records the evidence status only and is intentionally not a loadable experiment manifest.

## Explicitly generate a new protocol

These commands create new experiments. They are not reconstructions of the server runs reported in the manuscript. Outputs are not prepopulated in this release, and existing manifests are never overwritten.

```bash
# Keep DRIVE's original 20-image test set; newly select validation images
# from the original 20-image development set (default: 18 train, 2 val).
python scripts/prepare_data.py generate --dataset DRIVE --strategy drive-official \
  --data-root ../datasets --output my_protocols/DRIVE --seed 42

# A new five-fold STARE protocol: each fold uses 12 train / 4 val / 4 test.
# Outer fold k is test; the next outer fold is validation; remaining folds train.
python scripts/prepare_data.py generate --dataset STARE --strategy kfold --folds 5 \
  --data-root ../datasets --output my_protocols/STARE --seed 42

# A new CHASE holdout groups both eyes of each subject before splitting.
python scripts/prepare_data.py generate --dataset CHASEDB1 --strategy holdout \
  --ratios 0.6 0.2 0.2 --data-root ../datasets --output my_protocols/CHASEDB1 --seed 42

# An explicitly NEW HRF holdout, stratified across h/dr/g categories.
# Default ratios give 27 train / 9 val / 9 test, not the old manuscript's 15/20.
python scripts/prepare_data.py generate --dataset HRF --strategy holdout \
  --ratios 0.6 0.2 0.2 --data-root ../datasets --output my_protocols/HRF --seed 42

python scripts/prepare_data.py validate --train my_protocols/HRF/train.json \
  --val my_protocols/HRF/val.json --test my_protocols/HRF/test.json --data-root ../datasets
```

For HRF, the h/dr/g labels describe image categories. The attachment does not establish person-level identity across categories; the new default grouping is therefore image-level. DRIVE and STARE also use image-level grouping unless separate validated subject information is supplied. Only CHASE has explicit left/right eye subject grouping established by the supplied filenames.

The generator stores the seed, strategy and settings in every manifest and sorts output records by ID. For multi-fold validation, each image appears exactly once across outer test folds; the validation fold never contributes training patches within the same outer fold. Use a separate results directory per fold and retain all generated manifests with the corresponding experiment outputs.
