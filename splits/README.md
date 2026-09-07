# Split manifests

Each JSON manifest is a list of sample objects relative to the dataset root:

```json
[
  {"id": "sample_001", "image": "images/sample_001.png", "mask": "masks/sample_001.png", "fov": "fov/sample_001.png"}
]
```

`fov` is optional. When it is omitted, the code estimates the field of view from the fundus image.

The final paper protocol uses fixed development/test partitions. Keep the same manifests for every model and every seed. For STARE five-fold experiments, create fold-specific manifests (e.g. `fold1_train.json`, `fold1_val.json`, `fold1_test.json`) and point a copied YAML config to the desired fold.
