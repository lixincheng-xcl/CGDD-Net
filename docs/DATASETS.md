# Dataset Preparation

This repository does not redistribute DRIVE, CHASE_DB1, STARE, or HRF.

## Manifest format

Every manifest is a JSON list:

```json
[
  {
    "id": "sample_001",
    "image": "relative/path/to/image.tif",
    "mask": "relative/path/to/manual_mask.gif",
    "fov": "relative/path/to/fov_mask.gif"
  }
]
```

Paths are interpreted relative to the dataset root configured in `configs/default.yaml`.

`fov` is optional. When omitted, `cgddnet.data.preprocess.infer_fov` estimates a conservative field-of-view mask.

## Required manifests

```text
splits/DRIVE/train.json
splits/DRIVE/val.json
splits/DRIVE/test.json
splits/CHASE_DB1/train.json
splits/CHASE_DB1/val.json
splits/CHASE_DB1/test.json
splits/STARE/train.json
splits/STARE/val.json
splits/STARE/test.json
splits/HRF/train.json
splits/HRF/val.json
splits/HRF/test.json
```

The checked-in files are intentionally empty templates. Populate them with the exact split used for your experiment.

## Building a manifest

`tools/manifest_from_dirs.py` pairs files by normalized filename stems.

Example for DRIVE:

```bash
python tools/manifest_from_dirs.py \
  --root data/DRIVE \
  --images training/images \
  --masks training/1st_manual \
  --fovs training/mask \
  --output splits/DRIVE/train_all.json
```

After generating a complete list, create fixed train/validation/test manifests before training. Use the same manifests for all methods and seeds in a controlled comparison.

For STARE cross-validation, keep fold-specific manifests and point a copied YAML config to each fold.
