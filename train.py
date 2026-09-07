#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from cgddnet.data import FundusImageDataset, RandomPatchDataset, load_manifest
from cgddnet.models import build_model
from cgddnet.utils import aggregate_metrics, binary_metrics, load_yaml, seed_everything, sliding_window_predict, worker_init_fn


def parse_args():
    p = argparse.ArgumentParser(description="Train CGDD-Net")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", default=None, help="Override dataset section (DRIVE/CHASE_DB1/STARE/HRF)")
    p.add_argument("--ablation", default=None, choices=["baseline","csde","samg","dcdf","detail_decoder","selective_skip","full"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def resolve(cfg, args):
    name = args.dataset or cfg["dataset"]
    ds = cfg["datasets"][name]
    train_cfg = cfg["train"]
    return name, ds, train_cfg


@torch.no_grad()
def validate(model, ds_cfg, root, device, infer_cfg):
    val_samples = load_manifest(ds_cfg["val_manifest"])
    ds = FundusImageDataset(val_samples, root=root)
    vals = []
    for sample in tqdm(ds, desc="validation", leave=False):
        prob = sliding_window_predict(model, sample["image"], infer_cfg["patch_size"], infer_cfg["stride"], device)
        vals.append(binary_metrics(prob.numpy(), sample["mask"].numpy(), sample["fov"].numpy(), infer_cfg["threshold"]))
    return aggregate_metrics(vals)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset_name, ds_cfg, tc = resolve(cfg, args)
    seed = args.seed if args.seed is not None else tc.get("seed", 2026)
    seed_everything(seed, tc.get("deterministic", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ablation = args.ablation or cfg["model"].get("ablation", "full")
    out_dir = Path(args.output or f"runs/{dataset_name}/{ablation}/seed_{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2))

    root = ds_cfg["root"]
    train_samples = load_manifest(ds_cfg["train_manifest"])
    train_ds = RandomPatchDataset(
        train_samples, root=root,
        patch_size=tc["patch_size"], sampling_window=tc.get("sampling_window", 64),
        patches_per_epoch=tc.get("patches_per_epoch", 25600), seed=seed,
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=tc.get("workers", 8), pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn, generator=gen, persistent_workers=tc.get("workers", 8) > 0,
    )

    model = build_model(
        ablation=ablation,
        in_channels=cfg["model"].get("in_channels", 3),
        base_channels=cfg["model"].get("base_channels", 12),
        detail_channels=cfg["model"].get("detail_channels", 12),
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Dataset={dataset_name}  Ablation={ablation}  Device={device}  Trainable params={nparams/1e6:.3f} M")

    optim = Adam(model.parameters(), lr=tc["lr"], betas=tuple(tc.get("betas", [0.9, 0.999])),
                 eps=tc.get("eps", 1e-8), weight_decay=tc.get("weight_decay", 0.0))
    sched = CosineAnnealingLR(optim, T_max=tc["epochs"], eta_min=tc.get("min_lr", 1e-7))
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    start_epoch, best_auc, bad_epochs = 0, -math.inf, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt: sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_auc = ckpt.get("best_auc", best_auc)

    log_path = out_dir / "history.csv"
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if log_path.stat().st_size == 0:
            writer.writerow(["epoch", "train_loss", "val_SE", "val_SP", "val_ACC", "val_F1", "val_AUC", "lr"])

        for epoch in range(start_epoch, tc["epochs"]):
            model.train()
            running = 0.0
            pbar = tqdm(loader, desc=f"epoch {epoch+1}/{tc['epochs']}")
            for batch in pbar:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)
                fov = batch["fov"].to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                logits = model(x)
                loss_map = criterion(logits, y)
                loss = (loss_map * fov).sum() / fov.sum().clamp_min(1.0)
                loss.backward()
                if tc.get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                optim.step()
                running += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")
            sched.step()
            train_loss = running / max(len(loader), 1)
            metrics = validate(model, ds_cfg, root, device, cfg["inference"])
            lr = optim.param_groups[0]["lr"]
            writer.writerow([epoch+1, train_loss, metrics["SE"], metrics["SP"], metrics["ACC"], metrics["F1"], metrics["AUC"], lr])
            f.flush()
            print(f"val: {metrics}")

            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optim.state_dict(),
                     "scheduler": sched.state_dict(), "best_auc": max(best_auc, metrics["AUC"]),
                     "dataset": dataset_name, "ablation": ablation, "seed": seed, "params": nparams}
            torch.save(state, out_dir / "last.pt")
            if metrics["AUC"] > best_auc:
                best_auc = metrics["AUC"]
                bad_epochs = 0
                torch.save(state, out_dir / "best.pt")
                (out_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2))
            else:
                bad_epochs += 1
            if bad_epochs >= tc.get("early_stop_patience", 6):
                print(f"Early stopping after {bad_epochs} epochs without validation AUC improvement.")
                break


if __name__ == "__main__":
    main()
