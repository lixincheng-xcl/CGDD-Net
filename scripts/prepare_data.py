#!/usr/bin/env python3
"""Inspect original datasets or explicitly generate a NEW reproducible protocol."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgddnet.data import load_manifest, validate_records, validate_splits

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".ppm", ".bmp"}


def images(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing original-data directory: {directory}")
    return sorted((p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS),
                  key=lambda p: p.name.lower())


def pair_file(directory, stem, candidates):
    available = {p.name.lower(): p for p in images(directory)}
    matches = [available[name.lower()] for name in candidates if name.lower() in available]
    if len(matches) != 1:
        raise ValueError(f"Expected one paired file for {stem} in {directory}, found {matches}")
    return matches[0]


def discover(data_root, dataset):
    """Read only known original directories; never traverse datasets/augmented."""
    root = Path(data_root).expanduser().resolve()
    dataset = dataset.upper().replace("_", "")
    if dataset not in {"DRIVE", "STARE", "CHASEDB1", "HRF"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    directory = root / dataset
    if dataset == "CHASEDB1" and not directory.is_dir():
        directory = root / "CHASE_DB1"
    records = []
    image_dirs = [(directory / phase / "images", phase) for phase in ("training", "test")] \
        if dataset == "DRIVE" else [(directory / "images", None)]
    for image_dir, phase in image_dirs:
        for image in images(image_dir):
            stem = image.stem
            if dataset == "DRIVE":
                number = stem.split("_")[0]
                label = pair_file(image_dir.parent / "1st_manual", stem,
                                  [f"{number}_manual1{ext}" for ext in EXTENSIONS])
                fov = pair_file(image_dir.parent / "mask", stem,
                                [f"{stem}_mask{ext}" for ext in EXTENSIONS])
                subject, category = f"DRIVE:{number}", "all"
            elif dataset == "STARE":
                label = pair_file(directory / "1st_labels_ah", stem,
                                  [f"{stem}.ah{ext}" for ext in EXTENSIONS] +
                                  [f"{stem}{ext}" for ext in EXTENSIONS])
                fov = pair_file(directory / "mask", stem,
                                [f"{stem}{ext}" for ext in EXTENSIONS] +
                                [f"{stem}_mask{ext}" for ext in EXTENSIONS] +
                                [f"mask_{stem.removeprefix('im')}{ext}" for ext in EXTENSIONS])
                subject, category = f"STARE:{stem}", "all"
            elif dataset == "CHASEDB1":
                label = pair_file(directory / "1st_label", stem,
                                  [f"{stem}_1stHO{ext}" for ext in EXTENSIONS])
                fov = pair_file(directory / "mask", stem,
                                [f"{stem}{ext}" for ext in EXTENSIONS])
                subject, category = f"CHASEDB1:{int(stem.split('_')[-1][:-1]):02d}", "all"
            else:
                label = pair_file(directory / "manual1", stem,
                                  [f"{stem}{ext}" for ext in EXTENSIONS])
                fov = pair_file(directory / "mask", stem,
                                [f"{stem}_mask{ext}" for ext in EXTENSIONS] +
                                [f"{stem}{ext}" for ext in EXTENSIONS])
                subject, category = f"HRF:{stem}", stem.split("_")[-1].lower()
            case_id = f"{dataset}/{stem}"
            records.append({"id": case_id, "original_id": case_id, "dataset": dataset,
                            "subject": subject, "category": category,
                            "official_partition": phase,
                            "image": image, "label": label, "fov": fov})
    validate_records(records)
    if dataset == "CHASEDB1":
        counts = defaultdict(int)
        for record in records:
            counts[record["subject"]] += 1
        if any(count != 2 for count in counts.values()):
            raise ValueError("CHASE discovery requires both eyes for every subject")
    return records


def _groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["subject"]].append(record)
    return groups


def _counts(n, ratios):
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Three positive split ratios must sum to one")
    if n < 3:
        raise ValueError("At least three independent groups are required")
    raw = [n * ratio for ratio in ratios]
    values = [int(math.floor(value)) for value in raw]
    order = sorted(range(3), key=lambda i: (-(raw[i] - values[i]), i))
    for index in order[:n - sum(values)]:
        values[index] += 1
    if min(values) == 0:
        raise ValueError("Ratios produce an empty split; use more groups or different ratios")
    return values


def make_holdout(records, ratios=(0.6, 0.2, 0.2), seed=42):
    """NEW holdout; keep subjects together and stratify HRF h/dr/g when available."""
    grouped = _groups(records)
    strata = defaultdict(list)
    for subject, cases in sorted(grouped.items()):
        categories = {case.get("category", "all") for case in cases}
        if len(categories) != 1:
            raise ValueError(f"Subject {subject} has conflicting strata")
        strata[next(iter(categories))].append(subject)
    rng, result = random.Random(seed), {name: [] for name in ("train", "val", "test")}
    for _, subjects in sorted(strata.items()):
        rng.shuffle(subjects)
        counts = _counts(len(subjects), ratios)
        start = 0
        for name, count in zip(result, counts):
            for subject in subjects[start:start + count]:
                result[name].extend(grouped[subject])
            start += count
    return {name: sorted(cases, key=lambda case: case["id"]) for name, cases in result.items()}


def make_drive_official(records, val_ratio=0.1, seed=42):
    """Preserve DRIVE's directory-defined test images; NEW train/val selection."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must lie strictly between zero and one")
    development = sorted((r for r in records if r.get("official_partition") == "training"),
                         key=lambda r: r["id"])
    test = [r for r in records if r.get("official_partition") == "test"]
    if len(development) < 2 or not test:
        raise ValueError("DRIVE official strategy requires nonempty training and test directories")
    random.Random(seed).shuffle(development)
    count = max(1, min(len(development) - 1, int(round(len(development) * val_ratio))))
    return {"train": development[count:], "val": development[:count], "test": test}


def make_kfold(records, folds=5, seed=42):
    """NEW grouped outer folds; next fold is validation, remaining folds train."""
    grouped = _groups(records)
    if folds < 3 or folds > len(grouped):
        raise ValueError("folds must be at least 3 and no greater than independent groups")
    strata = defaultdict(list)
    for subject, cases in sorted(grouped.items()):
        strata[cases[0].get("category", "all")].append(subject)
    buckets, rng, offset = [[] for _ in range(folds)], random.Random(seed), 0
    for _, subjects in sorted(strata.items()):
        rng.shuffle(subjects)
        for index, subject in enumerate(subjects):
            buckets[(offset + index) % folds].extend(grouped[subject])
        offset = (offset + len(subjects)) % folds
    result = []
    for test_fold in range(folds):
        val_fold = (test_fold + 1) % folds
        result.append({"test": buckets[test_fold], "val": buckets[val_fold],
                       "train": [record for index, bucket in enumerate(buckets)
                                 if index not in (test_fold, val_fold) for record in bucket]})
    return result


def write_protocol(splits, output, data_root, dataset, strategy, seed, *, settings=None):
    """Write only paths/identifiers and provenance, never image arrays."""
    validate_splits(splits)
    output, root = Path(output).resolve(), Path(data_root).resolve()
    targets = [output / f"{name}.json" for name in splits]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing to replace existing manifests in {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        portable = []
        for record in sorted(records, key=lambda case: case["id"]):
            item = dict(record)
            for key in ("image", "label", "fov"):
                item[key] = Path(item[key]).resolve().relative_to(root).as_posix() if item.get(key) else None
            portable.append(item)
        document = {"schema_version": 1, "dataset": dataset, "split": name,
                    "protocol_status": "newly_generated_not_original_server_split",
                    "strategy": strategy, "seed": seed, "settings": settings or {},
                    "data_root": os.path.relpath(root, output), "records": portable}
        (output / f"{name}.json").write_text(json.dumps(document, indent=2) + "\n")
    return {name: len(records) for name, records in splits.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Count original paired data without assigning splits")
    inspect.add_argument("--data-root", required=True, type=Path)
    inspect.add_argument("--dataset", choices=["DRIVE", "STARE", "CHASEDB1", "CHASE_DB1", "HRF"])
    generate = commands.add_parser("generate", help="Explicitly generate NEW manifests, not recover a server split")
    generate.add_argument("--data-root", required=True, type=Path)
    generate.add_argument("--dataset", required=True, choices=["DRIVE", "STARE", "CHASEDB1", "CHASE_DB1", "HRF"])
    generate.add_argument("--strategy", required=True, choices=["drive-official", "holdout", "kfold"])
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    generate.add_argument("--folds", type=int, default=5)
    generate.add_argument("--val-ratio", type=float, default=0.1)
    validate = commands.add_parser("validate", help="Check all supplied train/val/test manifests together")
    validate.add_argument("--train", required=True, type=Path)
    validate.add_argument("--val", required=True, type=Path)
    validate.add_argument("--test", required=True, type=Path)
    validate.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        report = {}
        for dataset in [args.dataset] if args.dataset else ["DRIVE", "STARE", "CHASEDB1", "HRF"]:
            records = discover(args.data_root, dataset)
            report[dataset] = {"images": len(records), "groups": len(_groups(records)),
                               "source": "original_data_directories_only", "protocol": "unassigned"}
        print(json.dumps(report, indent=2))
    elif args.command == "validate":
        splits = {name: load_manifest(getattr(args, name), args.data_root) for name in ("train", "val", "test")}
        print(json.dumps(validate_splits(splits), indent=2))
    else:
        records = discover(args.data_root, args.dataset)
        if args.seed < 0:
            parser.error("seed must be nonnegative")
        settings = {"ratios": args.ratios, "folds": args.folds, "val_ratio": args.val_ratio}
        if args.strategy == "drive-official":
            if args.dataset != "DRIVE":
                parser.error("drive-official is only valid for DRIVE")
            partitions = [(args.output, make_drive_official(records, args.val_ratio, args.seed))]
        elif args.strategy == "holdout":
            partitions = [(args.output, make_holdout(records, args.ratios, args.seed))]
        else:
            partitions = [(args.output / f"fold_{index}", splits)
                          for index, splits in enumerate(make_kfold(records, args.folds, args.seed))]
        # Check every fold before writing the first one; avoid partially accepted protocols.
        for output, splits in partitions:
            validate_splits(splits)
            if any((output / f"{name}.json").exists() for name in splits):
                raise FileExistsError(f"Output already contains manifests: {output}")
        counts = {str(output): write_protocol(splits, output, args.data_root, args.dataset,
                                              args.strategy, args.seed, settings=settings)
                  for output, splits in partitions}
        print(json.dumps({"status": "newly_generated_not_original_server_split", "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
