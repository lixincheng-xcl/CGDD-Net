"""Prepare NEW, provenance-tracked CHASE_DB1 and STARE Kaggle experiments.

Source: official provider archives, or the SAME archives supplied by the user
under /kaggle/input. Does not silently download unauthorized mirrors or claim
that a deterministic random split reconstructs historical manuscript images.

Usage:
  python -m experiments.kaggle_data --dataset CHASEDB1 --output /kaggle/working/cgdd_data
  python -m experiments.kaggle_data --dataset STARE --output /kaggle/working/cgdd_data
  python -m experiments.kaggle_data --dataset CHASEDB1 --chase-zip /kaggle/input/.../CHASEDB1.zip
  python -m experiments.kaggle_data --dataset STARE --stare-images /kaggle/input/.../stare-images.tar --stare-labels /kaggle/input/.../labels-ah.tar
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tarfile
import urllib.error
import urllib.request
import zipfile
from PIL import Image

# Only provider URLs; direct links can require manual browser download.
CHASE_URL = "https://researchinnovation.kingston.ac.uk/files/40659508/CHASEDB1.zip"
STARE_IMAGES_URL = "https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar"
STARE_LABELS_URL = "https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar"

STARE_EXPECTED = {
    "im0001", "im0002", "im0003", "im0004", "im0005", "im0044", "im0077", "im0081", "im0082", "im0139",
    "im0162", "im0163", "im0235", "im0236", "im0239", "im0240", "im0255", "im0291", "im0319", "im0324",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provider_download(url: str, dest: Path, *, max_bytes: int = 140 * 1024 * 1024) -> Path:
    """Download only the named university archive; fail clearly on denial/HTML."""
    dest = Path(dest)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + ".part")
    if temp.exists():
        temp.unlink()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CGDD-Net-academic-research/1.0"})
        with urllib.request.urlopen(req, timeout=90) as stream, temp.open("wb") as out:
            total = 0
            while True:
                block = stream.read(1 << 20)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise ValueError(f"Provider archive exceeds safety limit of {max_bytes} bytes")
                out.write(block)
        if dest.suffix.lower() == ".zip" and not zipfile.is_zipfile(temp):
            raise ValueError("Downloaded content is not a valid ZIP (possibly an HTML access page)")
        if dest.name.endswith(".tar") and not tarfile.is_tarfile(temp):
            raise ValueError("Downloaded content is not a valid TAR")
        temp.replace(dest)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Official download did not succeed: {url}\n"
            "Open the official provider website in your browser, obtain permission/agree to terms, "
            "and attach its ORIGINAL ZIP/TAR as a private Kaggle Input. "
            "Then set --chase-zip or --stare-images/--stare-labels."
        ) from error
    return dest


def safe_archive_name(name: str) -> str:
    norm = name.replace("\\", "/")
    if norm.startswith("/") or ".." in norm.split("/"):
        raise ValueError(f"Unsafe archive entry: {name!r}")
    return Path(norm).name


def image_as_png(raw: bytes, path: Path, *, mask: bool) -> None:
    # Do not invent or augment original images; convert losslessly to PNG.
    with Image.open(io.BytesIO(raw)) as opened:
        if mask:
            image = opened.convert("L")
            # Official observer masks use high-valued vessels, zero-valued background.
            import numpy as np
            a = np.asarray(image)
            threshold = 0 if a.max(initial=0) <= 1 else 127
            binary = ((a > threshold) * 255).astype("uint8")
            image = Image.fromarray(binary, "L")
        else:
            image = opened.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")


def extract_chase(archive: Path, dataset_root: Path) -> list[dict]:
    images: dict[str, bytes] = {}
    masks: dict[str, bytes] = {}
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if entry.is_dir():
                continue
            name = safe_archive_name(entry.filename)
            im = re.fullmatch(r"(Image_\d{2}[LR])\.(?:jpg|jpeg|png|tif|tiff)", name, re.I)
            gt = re.fullmatch(r"(Image_\d{2}[LR])_1stHO\.(?:png|gif|tif|tiff|jpg)", name, re.I)
            if not (im or gt):
                continue
            raw = z.read(entry)
            if len(raw) > 20 * 1024 * 1024:
                raise ValueError("Unexpected oversized image inside CHASE archive")
            key = (im or gt).group(1)
            group = images if im else masks
            if key in group:
                raise ValueError(f"Duplicate CHASE entry for {key}")
            group[key] = raw
    if len(images) != 28 or len(masks) != 28 or set(images) != set(masks):
        raise ValueError(f"CHASE archive expected 28 matched images/masks, got {len(images)}/{len(masks)}")
    records = []
    for name in sorted(images):
        num = re.search(r"_(\d{2})[LR]", name, flags=re.I).group(1)
        image = dataset_root / "CHASEDB1" / "images" / f"{name}.png"
        label = dataset_root / "CHASEDB1" / "1st_label" / f"{name}_1stHO.png"
        image_as_png(images[name], image, mask=False)
        image_as_png(masks[name], label, mask=True)
        records.append({"id": f"CHASEDB1/{name}", "original_id": f"CHASEDB1/{name}",
                        "dataset": "CHASEDB1", "subject": f"CHASEDB1:{num}",
                        "image": image, "label": label, "fov": None})
    counts = {}
    for r in records:
        counts[r["subject"]] = counts.get(r["subject"], 0) + 1
    if len(counts) != 14 or set(counts.values()) != {2}:
        raise ValueError("CHASE archive lacks 14 complete left/right subject pairs")
    return records


def tar_image_bytes(archive: Path, *, label: bool) -> dict[str, bytes]:
    output = {}
    with tarfile.open(archive, "r:*") as t:
        for member in t.getmembers():
            if not member.isfile():
                continue
            name = safe_archive_name(member.name)
            pattern = r"(im\d{4})\.ah\.ppm(?:\.gz)?" if label else r"(im\d{4})\.ppm(?:\.gz)?"
            match = re.fullmatch(pattern, name, re.I)
            if not match:
                continue
            handle = t.extractfile(member)
            if handle is None:
                continue
            raw = handle.read(20 * 1024 * 1024 + 1)
            if len(raw) > 20 * 1024 * 1024:
                raise ValueError("Unexpected oversized STARE image")
            if name.lower().endswith(".gz"):
                raw = gzip.decompress(raw)
            key = match.group(1).lower()
            if key in output:
                raise ValueError(f"Duplicate STARE image {key}")
            output[key] = raw
    return output


def extract_stare(image_tar: Path, label_tar: Path, dataset_root: Path) -> list[dict]:
    images = tar_image_bytes(image_tar, label=False)
    masks = tar_image_bytes(label_tar, label=True)
    if set(images) != STARE_EXPECTED or set(masks) != STARE_EXPECTED:
        raise ValueError(f"STARE images/labels do not match official 20 IDs: images={sorted(images)} masks={sorted(masks)}")
    records = []
    for name in sorted(STARE_EXPECTED):
        image = dataset_root / "STARE" / "images" / f"{name}.png"
        label = dataset_root / "STARE" / "1st_labels_ah" / f"{name}.ah.png"
        image_as_png(images[name], image, mask=False)
        image_as_png(masks[name], label, mask=True)
        records.append({"id": f"STARE/{name}", "original_id": f"STARE/{name}",
                        "dataset": "STARE", "subject": f"STARE:{name}",
                        "image": image, "label": label, "fov": None})
    return records


def make_new_split(records: list[dict], *, dataset: str, seed: int) -> tuple[list[dict], list[dict]]:
    if dataset == "CHASEDB1":
        # 10 subjects (20 eyes) train; 4 subjects (8 eyes) held out together.
        subjects = sorted({r["subject"] for r in records})
        if len(subjects) != 14:
            raise ValueError("Expected 14 subjects in CHASE")
        random.Random(seed).shuffle(subjects)
        test_subjects = set(subjects[:4])
        train = [r for r in records if r["subject"] not in test_subjects]
        test = [r for r in records if r["subject"] in test_subjects]
        expected = (20, 8)
    elif dataset == "STARE":
        ids = sorted(r["id"] for r in records)
        if len(ids) != 20:
            raise ValueError("Expected 20 STARE images")
        random.Random(seed).shuffle(ids)
        test_ids = set(ids[:10])
        train = [r for r in records if r["id"] not in test_ids]
        test = [r for r in records if r["id"] in test_ids]
        expected = (10, 10)
    else:
        raise ValueError(f"Unsupported first-batch dataset {dataset}")
    train, test = sorted(train, key=lambda x: x["id"]), sorted(test, key=lambda x: x["id"])
    if (len(train), len(test)) != expected:
        raise AssertionError("Unexpected split sizes")
    if set(x["subject"] for x in train) & set(x["subject"] for x in test):
        raise AssertionError("Subject leakage detected")
    return train, test


def write_manifests(records: list[dict], dataset_root: Path, split_root: Path, *, dataset: str,
                    seed: int, source_hashes: dict, mask_policy: str) -> dict:
    from cgddnet.data import validate_splits
    train, test = make_new_split(records, dataset=dataset, seed=seed)
    validate_splits({"train": [dict(x) for x in train], "test": [dict(x) for x in test]})
    split_root.mkdir(parents=True, exist_ok=True)
    for split, cases in (("train", train), ("test", test)):
        portable = []
        for r in cases:
            rr = dict(r)
            for key in ("image", "label", "fov"):
                rr[key] = rr[key].relative_to(dataset_root).as_posix() if rr[key] else None
            portable.append(rr)
        body = {"schema_version": 1, "dataset": dataset, "split": split,
                "protocol_status": "new_kaggle_controlled_not_historical",
                "strategy": "subject_holdout_10_4" if dataset == "CHASEDB1" else "fixed_image_holdout_10_10",
                "seed": seed, "fov_mask_policy": mask_policy,
                "source_archive_sha256": source_hashes,
                "data_root": os.path.relpath(dataset_root, split_root),
                "records": portable}
        dest = split_root / f"{split}.json"
        if dest.exists():
            if json.loads(dest.read_text()) != body:
                raise FileExistsError(f"Existing manifest has different contents: {dest}")
        else:
            dest.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    return {"train": len(train), "test": len(test), "subjects_train": len({x["subject"] for x in train}),
            "subjects_test": len({x["subject"] for x in test})}


def execute(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["CHASEDB1", "STARE"])
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/cgdd_data"))
    p.add_argument("--archive-cache", type=Path, default=None)
    p.add_argument("--chase-zip", type=Path)
    p.add_argument("--stare-images", type=Path)
    p.add_argument("--stare-labels", type=Path)
    p.add_argument("--split-seed", type=int, default=20261020)
    p.add_argument("--no-download", action="store_true", help="Only use provider archives already supplied locally")
    args = p.parse_args(argv)
    if args.split_seed < 0:
        p.error("Split seed must be nonnegative")
    root = args.output.resolve()
    cache = (args.archive_cache or root / "official_archives").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    def source(passed: Path | None, url: str, basename: str) -> Path:
        if passed:
            if not passed.is_file():
                raise FileNotFoundError(f"Missing attached provider archive: {passed}")
            return passed.resolve()
        path = cache / basename
        if args.no_download and not path.exists():
            raise FileNotFoundError(f"Attach the original provider archive or download it with the provider permission: {url}")
        return provider_download(url, path)
    if args.dataset == "CHASEDB1":
        archive = source(args.chase_zip, CHASE_URL, "CHASEDB1.zip")
        records = extract_chase(archive, root)
        hashes = {"CHASEDB1.zip": sha256(archive)}
    else:
        archive_images = source(args.stare_images, STARE_IMAGES_URL, "stare-images.tar")
        archive_labels = source(args.stare_labels, STARE_LABELS_URL, "labels-ah.tar")
        records = extract_stare(archive_images, archive_labels, root)
        hashes = {"stare-images.tar": sha256(archive_images), "labels-ah.tar": sha256(archive_labels)}
    split_root = root / "manifests" / args.dataset
    sizes = write_manifests(records, root, split_root, dataset=args.dataset,
                            seed=args.split_seed, source_hashes=hashes,
                            mask_policy="full_rectangular_image: no official pixelwise FOV masks included in archives")
    report = {"dataset": args.dataset, "provenance": "new, NOT author original data assignment",
              "provider_source": CHASE_URL if args.dataset == "CHASEDB1" else [STARE_IMAGES_URL, STARE_LABELS_URL],
              "source_hashes": hashes, "training_count": sizes["train"], "test_count": sizes["test"],
              "training_subjects": sizes["subjects_train"], "test_subjects": sizes["subjects_test"],
              "split_seed": args.split_seed, "label": "CHASE first observer" if args.dataset == "CHASEDB1" else "STARE Adam Hoover",
              "fov_mask_policy": "full_image: no publisher-provided FOV mask in the attached source archive",
              "restriction": "Do not present as historical manuscript split. Test set not used for checkpoint selection.",
              "manifests": {"train": str(split_root / "train.json"), "test": str(split_root / "test.json")}}
    (split_root / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))