"""Image-level manifests and FOV-aware patches; dataset files stay outside the repo."""
from __future__ import annotations

from collections import OrderedDict
import json
import hashlib
import os
from pathlib import Path
import re
from typing import Mapping

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _subject(record):
    dataset = str(record.get("dataset", "")).upper().replace("_", "")
    candidate = str(record.get("id", "")) + " " + str(record.get("image", ""))
    if dataset == "CHASEDB1" or "chasedb1" in candidate.lower().replace("_", ""):
        # A changed record ID must not override the subject in the original file.
        pattern = r"Image[_-]?(\d+)[LR]"
        match = re.search(pattern, Path(record["image"]).stem, flags=re.IGNORECASE)
        if match is None:
            match = re.search(pattern, str(record.get("id", "")), flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot establish CHASE subject from case {record.get('id')!r}")
        inferred = f"CHASEDB1:{int(match.group(1)):02d}"
        supplied = record.get("subject")
        if supplied is not None and str(supplied) != inferred:
            raise ValueError(f"CHASE subject must be {inferred}, got {supplied!r}")
        return inferred
    return str(record.get("subject") or record.get("id", ""))


def _source_identity(record):
    """Recognize original case prefixes even if a derivative has a new record ID."""
    dataset = str(record.get("dataset", "")).upper().replace("_", "")
    stem = Path(record["image"]).stem
    patterns = {"DRIVE": r"^(\d+)(?:_|$)", "STARE": r"^(im\d+)",
                "CHASEDB1": r"^(Image_\d+[LR])", "HRF": r"^(\d+_(?:dr|h|g))(?:_|$)"}
    match = re.match(patterns[dataset], stem, re.IGNORECASE) if dataset in patterns else None
    return f"{dataset}/{match.group(1).lower()}" if match else str(record["id"])


def validate_records(records, *, check_files=True):
    """Reject empty, ambiguous, duplicated, or missing cases. Return the input list."""
    if not isinstance(records, list) or not records:
        raise ValueError("A manifest must contain a non-empty list of image records")
    seen = {key: set() for key in ("id", "original_id", "source_identity", "image", "label")}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every manifest record must be an object")
        for key in ("id", "image", "label"):
            if not record.get(key):
                raise ValueError(f"Missing required {key!r} in manifest record")
        record["subject"] = _subject(record)
        record["source_identity"] = _source_identity(record)
        record.setdefault("original_id", record["source_identity"])
        for key in seen:
            value = str(Path(record[key]).resolve()) if key in ("image", "label") else str(record[key])
            if value in seen[key]:
                raise ValueError(f"Duplicate {key}: {value}")
            seen[key].add(value)
        if Path(record["image"]).resolve() == Path(record["label"]).resolve():
            raise ValueError("Image and ground-truth label must be different files")
        if record.get("fov") and Path(record["fov"]).resolve() == Path(record["label"]).resolve():
            raise ValueError("FOV mask and vessel ground truth must be different files")
        if check_files:
            for key in ("image", "label", "fov"):
                if record.get(key) and not Path(record[key]).is_file():
                    raise FileNotFoundError(f"Missing {key} for {record['id']}: {record[key]}")
    return records


def load_manifest(path, data_root=None):
    """Resolve portable paths and validate a manifest.

    Root priority: explicit argument, CGDD_DATA_ROOT, document data_root (relative
    to the manifest), then the manifest directory. Record paths must be relative
    and cannot escape their dataset root. JSON lists are accepted for compatibility.
    """
    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict):
        records = document.get("records")
        dataset = document.get("dataset")
        recorded_root = document.get("data_root", ".")
    else:
        records, dataset, recorded_root = document, None, "."
    chosen_root = data_root or os.environ.get("CGDD_DATA_ROOT")
    root = Path(chosen_root).expanduser().resolve() if chosen_root else (path.parent / recorded_root).resolve()
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest {path} has no image records")
    resolved = []
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("Each manifest record must be an object")
        record = dict(raw)
        if dataset:
            record.setdefault("dataset", dataset)
        for key in ("image", "label", "fov"):
            value = record.get(key)
            if not value:
                if key == "fov":
                    record[key] = None
                    continue
                raise ValueError(f"Missing required path {key!r}")
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{key} must be a relative path within data_root: {value}")
            if "augmented" in {part.lower() for part in relative.parts}:
                raise ValueError("Use original images and on-the-fly augmentation, not datasets/augmented")
            full = (root / relative).resolve()
            if not full.is_relative_to(root):
                raise ValueError(f"{key} resolves outside data_root: {value}")
            record[key] = full
        resolved.append(record)
    return validate_records(resolved)


def validate_splits(splits: Mapping[str, list], *, check_files=True, check_content=True):
    """Require disjoint image IDs, originals, files, and subject identities."""
    if not splits:
        raise ValueError("No data splits supplied")
    owners = {key: {} for key in ("id", "original_id", "source_identity", "image", "label", "subject")}
    image_digests = {}
    for split, records in splits.items():
        validate_records(records, check_files=check_files)
        for record in records:
            for key, found in owners.items():
                value = str(Path(record[key]).resolve()) if key in ("image", "label") else str(record[key])
                if value in found and found[value] != split:
                    raise ValueError(f"Cross-split {key} overlap: {value} in {found[value]} and {split}")
                found[value] = split
            if check_files and check_content:
                digest = hashlib.sha256()
                with Path(record["image"]).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                fingerprint = digest.hexdigest()
                if fingerprint in image_digests:
                    previous_split, previous_id = image_digests[fingerprint]
                    raise ValueError(f"Duplicate image content: {record['id']} in {split} "
                                     f"matches {previous_id} in {previous_split}")
                image_digests[fingerprint] = (split, record["id"])
    return {name: len(records) for name, records in splits.items()}


def _binary(image):
    array = np.asarray(image.convert("L"))
    # The original datasets use 0/255. Also accept explicitly binary 0/1 PNGs.
    threshold = 0.5 if int(array.max(initial=0)) <= 1 else 127.5
    return (array.astype(np.float32) >= threshold).astype(np.float32)


def load_case(record, preprocessing=None):
    """Return float32 numpy arrays (C,H,W), (1,H,W), (1,H,W).

    preprocessing accepts color_mode='rgb' (default) or 'gray', max_side=None
    (only shrink, maintaining aspect ratio), and pad_multiple=1. No CLAHE,
    gamma adjustment, per-dataset fitting, or anisotropic resizing is applied.
    Missing FOV means the full image is evaluable and is explicit in the manifest.
    """
    options = dict(preprocessing or {})
    unknown = set(options) - {"color_mode", "max_side", "pad_multiple"}
    if unknown:
        raise ValueError(f"Unsupported preprocessing options: {sorted(unknown)}")
    mode = options.get("color_mode", "rgb").lower()
    if mode not in {"rgb", "gray", "grayscale"}:
        raise ValueError("color_mode must be 'rgb' or 'gray'")
    with Image.open(record["image"]) as source:
        image = source.convert("RGB" if mode == "rgb" else "L")
    with Image.open(record["label"]) as source:
        label = source.convert("L")
    if record.get("fov"):
        with Image.open(record["fov"]) as source:
            fov = source.convert("L")
    else:
        fov = Image.new("L", image.size, 255)
    if label.size != image.size or fov.size != image.size:
        raise ValueError(f"Image/label/FOV shape mismatch for {record.get('id')}")
    maximum = options.get("max_side")
    if maximum is not None:
        if int(maximum) != maximum or maximum <= 0:
            raise ValueError("max_side must be a positive integer")
        ratio = min(1.0, int(maximum) / max(image.size))
        if ratio < 1:
            size = tuple(max(1, int(round(value * ratio))) for value in image.size)
            image = image.resize(size, Image.Resampling.BILINEAR)
            label = label.resize(size, Image.Resampling.NEAREST)
            fov = fov.resize(size, Image.Resampling.NEAREST)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    if image_array.ndim == 2:
        image_array = image_array[None]
    else:
        image_array = image_array.transpose(2, 0, 1)
    label_array, fov_array = _binary(label)[None], _binary(fov)[None]
    if not np.any(fov_array):
        raise ValueError(f"Empty FOV for {record.get('id')}")
    multiple = options.get("pad_multiple", 1)
    if int(multiple) != multiple or multiple < 1:
        raise ValueError("pad_multiple must be a positive integer")
    multiple = int(multiple)
    height, width = image_array.shape[-2:]
    padding = ((0, 0), (0, (-height) % multiple), (0, (-width) % multiple))
    arrays = tuple(np.ascontiguousarray(np.pad(a, padding), dtype=np.float32)
                   for a in (image_array, label_array, fov_array))
    return arrays


class PatchDataset(Dataset):
    """Deterministic FOV-centered patches with paired augmentation.

    set_epoch() changes the sequence. Case arrays are cached as uint8, keeping
    full-size HRF storage substantially smaller than float32 arrays. cache_size
    limits the number of retained cases; None caches all cases encountered.
    """
    def __init__(self, records, patch_size=64, patches_per_epoch=150000,
                 seed=42, augment=True, preprocessing=None, cache_size=None):
        self.records = validate_records([dict(record) for record in records])
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        if len(self.patch_size) != 2 or any(int(x) != x or x < 1 for x in self.patch_size):
            raise ValueError("patch_size must contain positive integers")
        self.patch_size = tuple(int(x) for x in self.patch_size)
        if int(patches_per_epoch) != patches_per_epoch or patches_per_epoch < 1:
            raise ValueError("patches_per_epoch must be positive")
        if int(seed) != seed or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if cache_size is not None and (int(cache_size) != cache_size or cache_size < 1):
            raise ValueError("cache_size must be positive or None")
        self.patches_per_epoch = int(patches_per_epoch)
        self.seed, self.epoch, self.augment = int(seed), 0, bool(augment)
        self.preprocessing, self.cache_size = preprocessing, cache_size
        self._cache = OrderedDict()

    def set_epoch(self, epoch):
        if int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = int(epoch)

    def __len__(self):
        return self.patches_per_epoch

    def _case(self, index):
        if index not in self._cache:
            arrays = load_case(self.records[index], self.preprocessing)
            self._cache[index] = tuple(np.rint(a * 255).astype(np.uint8) for a in arrays)
            if self.cache_size is not None and len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(index)
        return self._cache[index]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        arrays = self._case(int(rng.integers(len(self.records))))
        height, width = arrays[0].shape[-2:]
        patch_h, patch_w = self.patch_size
        for _ in range(128):
            y, x = int(rng.integers(height)), int(rng.integers(width))
            if arrays[2][0, y, x]:
                break
        else:
            positions = np.flatnonzero(arrays[2][0])
            if not len(positions):
                raise ValueError("Empty FOV cannot produce a patch")
            y, x = np.unravel_index(int(rng.choice(positions)), (height, width))
        top = min(max(int(y) - patch_h // 2, 0), max(height - patch_h, 0))
        left = min(max(int(x) - patch_w // 2, 0), max(width - patch_w, 0))
        patches = tuple(a[:, top:top + patch_h, left:left + patch_w] for a in arrays)
        patches = tuple(np.pad(a, ((0, 0), (0, patch_h - a.shape[1]),
                                  (0, patch_w - a.shape[2]))) for a in patches)
        if self.augment:
            if rng.random() < 0.5:
                patches = tuple(a[:, :, ::-1] for a in patches)
            if rng.random() < 0.5:
                patches = tuple(a[:, ::-1, :] for a in patches)
            rotations = int(rng.integers(4)) if patch_h == patch_w else 2 * int(rng.integers(2))
            patches = tuple(np.rot90(a, rotations, axes=(1, 2)) for a in patches)
        return tuple(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32) / 255.0)
                     for a in patches)
