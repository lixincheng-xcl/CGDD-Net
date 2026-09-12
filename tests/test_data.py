"""Behavioral checks for image-level separation, FOV masks, and paired patches."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from cgddnet.data import load_case, load_manifest, PatchDataset, validate_records, validate_splits
from scripts.prepare_data import discover, make_drive_official, make_holdout, make_kfold, write_protocol


def case(tmp_path, index=0, shape=(11, 17), *, dataset="SYNTHETIC", name=None, subject=None):
    name = name or f"case_{index}"
    rng = np.random.default_rng(index)
    image = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    label = np.zeros(shape, dtype=np.uint8)
    label[:, shape[1] // 2] = 255
    fov = np.zeros(shape, dtype=np.uint8)
    fov[1:-1, 1:-1] = 255
    paths = {}
    for key, data in (("image", image), ("label", label), ("fov", fov)):
        path = tmp_path / f"{name}_{key}.png"
        Image.fromarray(data).save(path)
        paths[key] = path
    return {"id": f"{dataset}/{name}", "dataset": dataset,
            "subject": subject or f"{dataset}:{name}", **paths}


def write_manifest(tmp_path, records, name="manifest.json"):
    portable = [{**record, **{key: str(record[key].relative_to(tmp_path))
                             if record.get(key) else None for key in ("image", "label", "fov")}}
                for record in records]
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "data_root": ".", "records": portable}))
    return path


def test_manifest_resolves_portable_paths_and_rejects_empty_missing(tmp_path):
    record = case(tmp_path)
    loaded = load_manifest(write_manifest(tmp_path, [record]))
    assert loaded[0]["image"] == record["image"]
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    with pytest.raises(ValueError, match="no image records"):
        load_manifest(empty)
    record["label"].unlink()
    with pytest.raises(FileNotFoundError, match="label"):
        load_manifest(tmp_path / "manifest.json")


def test_manifest_refuses_absolute_and_escaping_paths(tmp_path):
    record = case(tmp_path)
    for path_value in (str(record["image"]), "../outside.png"):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{**record, "image": path_value,
                                    "label": "label.png", "fov": None}]))
        with pytest.raises(ValueError, match="relative path"):
            load_manifest(path)


def test_duplicates_and_cross_split_same_image_with_changed_id_rejected(tmp_path):
    original = case(tmp_path)
    renamed = {**original, "id": "different", "subject": "different"}
    with pytest.raises(ValueError, match="Duplicate image"):
        validate_records([original, renamed])
    with pytest.raises(ValueError, match="overlap"):
        validate_splits({"train": [original], "val": [renamed]})


def test_duplicate_content_detected_after_file_and_id_change(tmp_path):
    original, duplicate = case(tmp_path, 1), case(tmp_path, 2)
    shutil.copyfile(original["image"], duplicate["image"])
    with pytest.raises(ValueError, match="Duplicate image content"):
        validate_splits({"train": [original], "val": [duplicate]})


def test_explicit_original_id_keeps_augmented_variants_together(tmp_path):
    records = [case(tmp_path, i) for i in range(2)]
    for record in records:
        record["original_id"] = "same_original"
    with pytest.raises(ValueError, match="original_id overlap"):
        validate_splits({"train": records[:1], "val": records[1:]})


def test_dataset_source_prefix_rejects_transformed_original_with_changed_id(tmp_path):
    original = case(tmp_path, 1, dataset="DRIVE", name="21_training")
    transformed = case(tmp_path, 2, dataset="DRIVE", name="21_training_rot90")
    # Different bytes, labels, record IDs, and declared provenance cannot disguise
    # the known original filename identity.
    original["original_id"], transformed["original_id"] = "declared_a", "declared_b"
    with pytest.raises(ValueError, match="source_identity overlap"):
        validate_splits({"train": [original], "val": [transformed]})


def test_manifest_rejects_preaugmented_data_directory(tmp_path):
    directory = tmp_path / "augmented"
    directory.mkdir()
    record = case(directory)
    path = write_manifest(tmp_path, [record])
    with pytest.raises(ValueError, match="Use original images"):
        load_manifest(path)


def test_chase_subject_inferred_and_cannot_be_overridden(tmp_path):
    left = case(tmp_path, 1, dataset="CHASEDB1", name="Image_01L", subject="CHASEDB1:01")
    right = case(tmp_path, 2, dataset="CHASEDB1", name="Image_01R", subject="CHASEDB1:01")
    with pytest.raises(ValueError, match="subject overlap"):
        validate_splits({"train": [left], "val": [right]})
    with pytest.raises(ValueError, match="CHASE subject must"):
        validate_records([{**right, "subject": "fake_subject"}])
    right_with_new_id = {**right, "id": "CHASEDB1/Image_99R"}
    with pytest.raises(ValueError, match="subject overlap"):
        validate_splits({"train": [left], "val": [right_with_new_id]})


def test_load_case_normalization_label_fov_distinction_and_aspect(tmp_path):
    record = case(tmp_path, shape=(10, 20))
    image, label, fov = load_case(record)
    assert image.shape == (3, 10, 20)
    assert label.shape == fov.shape == (1, 10, 20)
    assert image.dtype == label.dtype == fov.dtype == np.float32
    assert 0 <= image.min() <= image.max() <= 1
    assert set(np.unique(label)) == {0, 1}
    assert not np.array_equal(label, fov)
    small = load_case(record, {"color_mode": "gray", "max_side": 10, "pad_multiple": 4})
    assert small[0].shape == (1, 8, 12)  # 5x10 aspect-preserved resize, then padding
    assert not small[2][:, 5:, :].any()
    assert not small[2][:, :, 10:].any()
    with pytest.raises(ValueError, match="Unsupported preprocessing"):
        load_case(record, {"gamma": 1.2})


def test_empty_fov_and_mismatched_label_fail_before_sampling(tmp_path):
    record = case(tmp_path)
    Image.fromarray(np.zeros((11, 17), dtype=np.uint8)).save(record["fov"])
    with pytest.raises(ValueError, match="Empty FOV"):
        load_case(record)
    with pytest.raises(ValueError, match="Empty FOV"):
        PatchDataset([record], patches_per_epoch=1)[0]
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(record["label"])
    with pytest.raises(ValueError, match="shape mismatch"):
        load_case(record)


def test_patches_deterministic_padded_and_jointly_augmented(tmp_path):
    record = case(tmp_path, shape=(7, 9))
    # Make the image a direct copy of the label to detect image/mask misalignment.
    with Image.open(record["label"]) as label:
        label.convert("RGB").save(record["image"])
    first = PatchDataset([record], patch_size=16, patches_per_epoch=8, seed=19)
    second = PatchDataset([record], patch_size=16, patches_per_epoch=8, seed=19)
    for index in range(8):
        a, b = first[index], second[index]
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert a[0].shape == (3, 16, 16)
        assert torch.equal(a[0][0:1], a[1])
        assert a[2].sum() == (7 - 2) * (9 - 2)
    first.set_epoch(1)
    assert any(not torch.equal(first[index][0], second[index][0]) for index in range(8))
    with pytest.raises(IndexError):
        first[8]


def test_new_grouped_holdout_and_folds_are_disjoint_reproducible(tmp_path):
    records = []
    for subject in range(1, 8):
        for side in "LR":
            index = 2 * subject + (side == "R")
            records.append(case(tmp_path, index, dataset="CHASEDB1", name=f"Image_{subject:02d}{side}",
                                subject=f"CHASEDB1:{subject:02d}"))
    first, second = make_holdout(records, seed=9), make_holdout(records, seed=9)
    assert first == second
    assert sum(validate_splits(first).values()) == 14
    folds = make_kfold(records, folds=5, seed=9)
    for fold in folds:
        assert sum(validate_splits(fold).values()) == 14
    assert sorted(r["id"] for fold in folds for r in fold["test"]) == sorted(r["id"] for r in records)
    output = tmp_path / "splits"
    counts = write_protocol(first, output, tmp_path, "CHASEDB1", "holdout", 9)
    assert sum(counts.values()) == 14
    for name in first:
        document = json.loads((output / f"{name}.json").read_text())
        assert document["protocol_status"] == "newly_generated_not_original_server_split"
        assert all(not Path(r["image"]).is_absolute() for r in document["records"])
        assert len(load_manifest(output / f"{name}.json")) == counts[name]
    with pytest.raises(FileExistsError):
        write_protocol(first, output, tmp_path, "CHASEDB1", "holdout", 9)


def test_drive_official_test_partition_is_untouched(tmp_path):
    records = [case(tmp_path, index) for index in range(10)]
    for index, record in enumerate(records):
        record["official_partition"] = "training" if index < 6 else "test"
    split = make_drive_official(records, val_ratio=1 / 3, seed=4)
    assert validate_splits(split) == {"train": 4, "val": 2, "test": 4}
    assert {r["id"] for r in split["test"]} == {r["id"] for r in records[6:]}


def test_discovery_pairs_stare_label_and_fov_by_identity_ignoring_augmented(tmp_path):
    for directory in ("STARE/images", "STARE/1st_labels_ah", "STARE/mask", "augmented/images"):
        (tmp_path / directory).mkdir(parents=True)
    for index in range(1, 4):
        Image.new("RGB", (6, 8), color=(index, 0, 0)).save(tmp_path / f"STARE/images/im{index:04d}.ppm")
        Image.new("L", (6, 8), color=0).save(tmp_path / f"STARE/1st_labels_ah/im{index:04d}.ah.ppm")
        Image.new("L", (6, 8), color=255).save(tmp_path / f"STARE/mask/mask_{index:04d}.png")
        Image.new("RGB", (6, 8)).save(tmp_path / f"augmented/images/im{index:04d}.png")
    records = discover(tmp_path, "STARE")
    assert len(records) == 3
    assert all("1st_labels_ah" in str(r["label"]) and "mask_" in r["fov"].name for r in records)
    assert all("augmented" not in str(r["image"]) for r in records)
