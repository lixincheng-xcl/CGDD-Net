"""Plan a reproducible experiment matrix; execution requires real manifests/GPU.

Use --execute to run, otherwise only report commands. No scores are invented.
Each dataset directory must contain validated train.json and test.json for the
ACTUAL chosen split. Creating a fresh random split changes the comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.pr_models import VARIANTS

CORE_VARIANTS = ("full", "independent_detail", "fixed_offsets", "uniform_samg", "unet_same_width")
DATASETS = ("DRIVE", "CHASE_DB1", "STARE", "HRF")


def plan(args):
    jobs = []
    for dataset in args.datasets:
        manifest_base = Path(args.manifest_root) / dataset
        train, test = manifest_base / "train.json", manifest_base / "test.json"
        if args.execute and (not train.is_file() or not test.is_file()):
            raise FileNotFoundError(f"Missing real split manifests: {train} / {test}")
        for variant in args.variants:
            for seed in args.seeds:
                output = Path(args.output_root) / dataset / variant / f"seed_{seed}"
                command = [sys.executable, str(ROOT / "experiments/train_controlled.py"),
                           "--config", str(args.config), "--variant", variant,
                           "--train-manifest", str(train), "--test-manifest", str(test),
                           "--data-root", str(args.data_root),
                           "--output", str(output), "--seed", str(seed),
                           "--device", args.device]
                if args.smoke:
                    command += ["--epochs-override", "1", "--patches-override", "2"]
                if args.resume and (output / "latest.pt").is_file():
                    command.append("--resume")
                jobs.append({"dataset": dataset, "variant": variant, "seed": seed,
                             "output": str(output), "command": command})
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="runs/pr_controls")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=list(CORE_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true", help="Actually launch training")
    parser.add_argument("--resume", action="store_true", help="Resume existing latest.pt")
    parser.add_argument("--smoke", action="store_true",
                        help="One epoch with 2 patches, never a publishable benchmark")
    parser.add_argument("--plan-output", default="pr_experiment_plan.json")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be unique nonnegative integers")
    if args.execute and args.device == "cpu" and not args.smoke:
        parser.error("Full benchmark is GPU-intensive. For CPU run only --smoke; H200 training requires CUDA")
    jobs = plan(args)
    document = {"protocol": "new_fixed_100_epochs_final_checkpoint",
                "publishable": not args.smoke and args.execute,
                "manifest_origin": "user-supplied actual image split, never generated here",
                "num_jobs": len(jobs), "jobs": jobs}
    destination = Path(args.plan_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Prepared {len(jobs)} runs; execution={'enabled' if args.execute else 'disabled'}")
    if args.execute:
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {job['dataset']}/{job['variant']}/seed_{job['seed']}", flush=True)
            subprocess.run(job["command"], check=True)
    return document


if __name__ == "__main__":
    main()