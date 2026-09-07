#!/usr/bin/env python3
"""Create a CGDD-Net manifest from image/mask directories.

Files are paired by normalized stem. Optional substrings can be stripped from
stems, which covers common conventions such as ``_manual1`` and ``_mask``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".ppm", ".bmp"}


def norm(stem: str, remove: list[str]):
    out = stem.lower()
    for x in remove:
        out = out.replace(x.lower(), "")
    out = re.sub(r"[^a-z0-9]+", "", out)
    return out


def files(d: Path):
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--images", required=True, help="image directory relative to root")
    p.add_argument("--masks", required=True, help="mask directory relative to root")
    p.add_argument("--fovs", default=None, help="optional FOV directory relative to root")
    p.add_argument("--remove-image", nargs="*", default=["_training", "_test", "image_"])
    p.add_argument("--remove-mask", nargs="*", default=["_manual1", "_1stho", "_mask", "image_"])
    p.add_argument("--remove-fov", nargs="*", default=["_training", "_test", "_mask", "image_"])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    imgs = files(root / args.images)
    masks = {norm(x.stem, args.remove_mask): x for x in files(root / args.masks)}
    fovs = {norm(x.stem, args.remove_fov): x for x in files(root / args.fovs)} if args.fovs else {}
    out = []
    missing = []
    for im in sorted(imgs):
        key = norm(im.stem, args.remove_image)
        ma = masks.get(key)
        if ma is None:
            missing.append(im.name); continue
        item = {"id": im.stem, "image": str(im.relative_to(root)), "mask": str(ma.relative_to(root))}
        if key in fovs:
            item["fov"] = str(fovs[key].relative_to(root))
        out.append(item)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} samples to {args.output}")
    if missing:
        print(f"Warning: {len(missing)} images had no mask match: {missing[:10]}")


if __name__ == "__main__":
    main()
