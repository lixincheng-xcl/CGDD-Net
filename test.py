#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cgddnet.data import FundusImageDataset, load_manifest
from cgddnet.models import build_model
from cgddnet.utils import aggregate_metrics, binary_metrics, load_yaml, sliding_window_predict


def main():
    p = argparse.ArgumentParser(description="Evaluate CGDD-Net")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--output", default=None)
    p.add_argument("--save-predictions", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    ds_cfg = cfg["datasets"][args.dataset]
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ablation = ckpt.get("ablation", cfg["model"].get("ablation", "full"))
    model = build_model(ablation=ablation,
                        in_channels=cfg["model"].get("in_channels", 3),
                        base_channels=cfg["model"].get("base_channels", 12),
                        detail_channels=cfg["model"].get("detail_channels", 12),
                        dropout=cfg["model"].get("dropout", 0.1))
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    manifest = ds_cfg[f"{args.split}_manifest"]
    ds = FundusImageDataset(load_manifest(manifest), root=ds_cfg["root"])
    out_dir = Path(args.output or f"results/{args.dataset}/{Path(args.checkpoint).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        (out_dir / "predictions").mkdir(exist_ok=True)

    rows, metrics = [], []
    for sample in tqdm(ds, desc=f"{args.dataset} {args.split}"):
        prob = sliding_window_predict(model, sample["image"], cfg["inference"]["patch_size"],
                                     cfg["inference"]["stride"], device)
        m = binary_metrics(prob.numpy(), sample["mask"].numpy(), sample["fov"].numpy(),
                           cfg["inference"]["threshold"])
        metrics.append(m)
        rows.append({"id": sample["id"], **{k: m[k] for k in ["SE","SP","ACC","F1","AUC"]}})
        if args.save_predictions:
            arr = (prob.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / "predictions" / f"{sample['id']}.png")

    summary = aggregate_metrics(metrics)
    print(json.dumps(summary, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id","SE","SP","ACC","F1","AUC"])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
