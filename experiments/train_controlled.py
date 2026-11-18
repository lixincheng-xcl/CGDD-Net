"""Run one controlled PR ablation on original retinal images.

This is a NEW experimental protocol: all declared training images are used for
100 fixed epochs, and the final checkpoint is tested exactly once. There is NO
validation-based checkpoint selection and no test-driven tuning. Never present
this protocol as a reconstruction of the historical manuscript runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgddnet.data import PatchDataset, load_manifest, validate_splits
from cgddnet.inference import evaluate_records
from cgddnet.runtime import (
    capture_rng, device_from_name, environment, load_config, restore_rng,
    seed_everything, sha256, write_json,
)
from cgddnet.schedule import WarmupCosine
from experiments.pr_models import VARIANTS, build_model, count_trainable


def masked_bce(logits, label, fov):
    if not (logits.shape == label.shape == fov.shape):
        raise ValueError("Logits, labels and FOV masks must have identical shapes")
    total = fov.sum()
    if total <= 0:
        raise ValueError("Training batch has no valid FOV pixels")
    return (F.binary_cross_entropy_with_logits(logits, label, reduction="none") * fov).sum() / total


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs-override", type=int,
                        help="FOR SOFTWARE SMOKE TESTS ONLY; not for benchmark tables")
    parser.add_argument("--patches-override", type=int,
                        help="FOR SOFTWARE SMOKE TESTS ONLY; not for benchmark tables")
    parser.add_argument("--max-epochs-this-session", type=int, default=None,
                        help="Stop cleanly after at most this many NEW epochs. Save latest.pt, "
                             "skip test evaluation and exit successfully unless final epoch is reached. "
                             "Resume next session with the same config and --resume.")
    return parser.parse_args(argv)


def run(args):
    if args.seed < 0:
        raise ValueError("Seed must be nonnegative")
    if args.max_epochs_this_session is not None and args.max_epochs_this_session < 1:
        raise ValueError("--max-epochs-this-session must be positive")
    cfg = load_config(args.config)
    cfg = json.loads(json.dumps(cfg))  # copy without mutating the original
    cfg["training"]["seed"] = args.seed
    for override, key in ((args.epochs_override, "epochs"),
                          (args.patches_override, "patches_per_epoch")):
        if override is not None:
            if override < 1:
                raise ValueError(f"{key} override must be positive")
            cfg["training"][key] = override
    smoke = args.epochs_override is not None or args.patches_override is not None
    cfg["controlled_experiment"] = {
        "variant": args.variant,
        "protocol": "new_fixed_epochs_final_checkpoint_no_test_selection",
        "smoke_test_only": smoke,
        "not_historical_manuscript_reproduction": True,
    }
    tr = cfg["training"]
    if min(tr["batch_size"], tr["epochs"], tr["patches_per_epoch"]) < 1:
        raise ValueError("Invalid training budget")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    device = device_from_name(args.device)
    seed_everything(args.seed)
    train_records = load_manifest(args.train_manifest, args.data_root)
    test_records = load_manifest(args.test_manifest, args.data_root)
    validate_splits({"train": train_records, "test": test_records})
    identity = {
        "train_manifest_sha256": sha256(args.train_manifest),
        "test_manifest_sha256": sha256(args.test_manifest),
    }
    model = build_model(args.variant, **cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"],
                                 weight_decay=tr.get("weight_decay", 0))
    steps_per_epoch = math.ceil(tr["patches_per_epoch"] / tr["batch_size"])
    schedule = WarmupCosine(
        tr["warmup_epochs"] * steps_per_epoch,
        tr["restart_epochs"] * steps_per_epoch,
        tr["restart_mult"], tr["warmup_start_factor"], tr["min_lr_factor"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start = 0
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(f"Cannot resume without {latest}")
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["config"] != cfg or state["manifests"] != identity:
            raise ValueError("Run config and input manifests must match checkpoint on resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = int(state["epoch"]) + 1
        restore_rng(state["rng"])
    else:
        write_json(output / "config.json", cfg)
        write_json(output / "manifests.json", identity)
        write_json(output / "environment.json", environment(device))
        shutil.copy2(args.train_manifest, output / "train_manifest.json")
        shutil.copy2(args.test_manifest, output / "test_manifest.json")
        with (output / "history.csv").open("w", newline="") as f:
            csv.DictWriter(f, ["epoch", "train_loss", "learning_rate"]).writeheader()

    dataset = PatchDataset(train_records, patch_size=tr["patch_size"],
                           patches_per_epoch=tr["patches_per_epoch"], seed=args.seed,
                           augment=True, preprocessing=cfg["preprocessing"])
    stop = min(tr["epochs"], start + args.max_epochs_this_session) \
        if args.max_epochs_this_session is not None else tr["epochs"]
    for epoch in range(start, stop):
        dataset.set_epoch(epoch)
        # Nonpersistent workers see the current epoch. Worker count is stored.
        gen = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=tr["batch_size"], shuffle=True,
                            num_workers=tr["num_workers"], generator=gen,
                            persistent_workers=False)
        model.train()
        loss_total, evaluated_pixels = 0.0, 0
        for image, label, fov in loader:
            image, label, fov = (v.to(device) for v in (image, label, fov))
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = masked_bce(logits, label, fov)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training encountered nonfinite loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            pixel_count = int(fov.sum().item())
            evaluated_pixels += pixel_count
            loss_total += loss.item() * pixel_count
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": capture_rng(),
            "epoch": epoch, "config": cfg, "manifests": identity,
        }
        torch.save(state, latest)
        row = {"epoch": epoch + 1, "train_loss": loss_total / evaluated_pixels,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        with (output / "history.csv").open("a", newline="") as f:
            writer = csv.DictWriter(f, row.keys())
            writer.writerow(row)
        print(json.dumps({"variant": args.variant, "seed": args.seed, **row}), flush=True)

    if stop < tr["epochs"]:
        # Kaggle's Save & Run All creates a separate clean execution session.
        # Successful exit preserves latest.pt as a versioned Notebook output.
        # Crucially NO TEST evaluation happens on intermediate checkpoints.
        progress = {"status": "checkpointed_incomplete_no_test_evaluation",
                    "variant": args.variant, "seed": args.seed,
                    "completed_epochs": stop, "target_epochs": tr["epochs"],
                    "resume_required": True, "smoke_test_only": smoke}
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        return progress
    # Only the final epoch checkpoint is evaluated; never tune on test outcomes.
    (output / "progress.json").unlink(missing_ok=True)
    model.eval()
    summary_dir = output / "test"
    if summary_dir.exists():
        raise FileExistsError("A test result already exists; do not retest to select an outcome")
    summary = evaluate_records(model, test_records, cfg["preprocessing"],
                               cfg["inference"], device, summary_dir)
    write_json(output / "experiment_summary.json", {
        "variant": args.variant, "seed": args.seed,
        "trained_epochs": tr["epochs"], "final_epoch_evaluated": True,
        "parameter_count": count_trainable(model),
        "smoke_test_only": smoke,
        "run_protocol": cfg["controlled_experiment"]["protocol"],
        "test_scores": summary,
    })
    torch.save({"model": model.state_dict(), "config": cfg,
                "manifests": identity, "epoch": tr["epochs"] - 1},
               output / "final.pt")
    return summary


def main(argv=None):
    args = options(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()