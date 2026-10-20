"""Publisher-archive parsing and new-split software tests; NO retinal performance."""
from __future__ import annotations

import gzip
import io
from pathlib import Path
import tarfile
import zipfile
import json
import numpy as np
from PIL import Image
import pytest
from experiments.kaggle_data import (
    STARE_EXPECTED, extract_chase, extract_stare, make_new_split,
    safe_archive_name, write_manifests,
)


def fake_png(kind: str) -> bytes:
    stream = io.BytesIO()
    if kind == "rgb":
        pixels = np.zeros((24, 24, 3), dtype=np.uint8)
        pixels[..., 0] = 45
        pixels[..., 1] = np.arange(24)[None] * 8
        image = Image.fromarray(pixels, "RGB")
        image.save(stream, format="PNG")
    elif kind == "label":
        pixels = np.zeros((24, 24), dtype=np.uint8)
        pixels[:, 12] = 255
        Image.fromarray(pixels, "L").save(stream, format="PNG")
    else:
        raise ValueError(kind)
    return stream.getvalue()


def fake_ppm(kind: str) -> bytes:
    stream = io.BytesIO()
    pix = np.zeros((20, 20, 3), dtype=np.uint8)
    if kind == "rgb":
        pix[:, :, 1] = 80
    else:
        pix[:, 9] = 255
    Image.fromarray(pix).save(stream, format="PPM")
    return gzip.compress(stream.getvalue())


def build_chase_archive(path: Path):
    with zipfile.ZipFile(path, "w") as z:
        for i in range(1, 15):
            for side in "LR":
                name = f"Image_{i:02d}{side}"
                z.writestr(f"CHASE/{name}.jpg", fake_png("rgb"))
                z.writestr(f"CHASE/{name}_1stHO.png", fake_png("label"))
                z.writestr(f"CHASE/{name}_2ndHO.png", fake_png("label"))


def build_stare_archive(path: Path, *, is_label: bool):
    with tarfile.open(path, "w") as t:
        for name in sorted(STARE_EXPECTED):
            ext = ".ah.ppm.gz" if is_label else ".ppm.gz"
            data = fake_ppm("label" if is_label else "rgb")
            item = tarfile.TarInfo(name=f"stare/{name}{ext}")
            item.size = len(data)
            t.addfile(item, io.BytesIO(data))


def test_chase_zip_parse_and_subject_disjointness(tmp_path):
    archive = tmp_path / "CHASEDB1.zip"
    build_chase_archive(archive)
    dataset_root = tmp_path / "data"
    rec = extract_chase(archive, dataset_root)
    assert len(rec) == 28
    assert len(set(x["subject"] for x in rec)) == 14
    train, test = make_new_split(rec, dataset="CHASEDB1", seed=20261020)
    assert (len(train), len(test)) == (20, 8)
    assert not set(r["subject"] for r in train) & set(r["subject"] for r in test)
    sizes = write_manifests(rec, dataset_root, dataset_root / "manifests" / "CHASEDB1",
                            dataset="CHASEDB1", seed=20261020,
                            source_hashes={"CHASEDB1.zip": "synthetic-not-real"}, mask_policy="full")
    assert sizes["subjects_test"] == 4
    data = json.loads((dataset_root / "manifests/CHASEDB1/test.json").read_text())
    assert len(data["records"]) == 8
    assert data["protocol_status"] == "new_kaggle_controlled_not_historical"
    with pytest.raises(FileExistsError):
        write_manifests(rec, dataset_root, dataset_root / "manifests" / "CHASEDB1",
                        dataset="CHASEDB1", seed=43,
                        source_hashes={"CHASEDB1.zip": "synthetic-not-real"}, mask_policy="full")


def test_stare_tars_and_fixed_10_10(tmp_path):
    img, mask = tmp_path / "stare-images.tar", tmp_path / "labels-ah.tar"
    build_stare_archive(img, is_label=False)
    build_stare_archive(mask, is_label=True)
    root = tmp_path / "data"
    rec = extract_stare(img, mask, root)
    assert len(rec) == 20
    train, test = make_new_split(rec, dataset="STARE", seed=42)
    assert (len(train), len(test)) == (10, 10)
    assert not set(x["id"] for x in train) & set(x["id"] for x in test)
    assert [(x["id"]) for x in make_new_split(rec, dataset="STARE", seed=42)[1]] == [x["id"] for x in test]


def test_disallowed_archive_path_and_missing_image(tmp_path):
    with pytest.raises(ValueError):
        safe_archive_name("../../secret.ppm")
    with pytest.raises(ValueError):
        safe_archive_name("/abs/secret.ppm")
    bad = tmp_path / "CHASEDB1.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("Image_01L.jpg", fake_png("rgb"))
    with pytest.raises(ValueError, match="28 matched"):
        extract_chase(bad, tmp_path / "bad")