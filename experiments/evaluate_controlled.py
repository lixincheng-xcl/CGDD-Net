"""Evaluate a completed controlled checkpoint on a declared test/transfer set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgddnet.data import load_manifest
from cgddnet.inference import evaluate_records
from cgddnet.runtime import device_from_name, environment, sha256, write_json
from experiments.pr_models import build_model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Never overwrite an existing evaluation")
    device = device_from_name(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = state["config"]
    experiment = cfg["controlled_experiment"]
    model = build_model(experiment["variant"], **cfg["model"]).to(device)
    model.load_state_dict(state["model"])
    records = load_manifest(args.manifest, args.data_root)
    summary = evaluate_records(model.eval(), records, cfg["preprocessing"],
                               cfg["inference"], device, output)
    write_json(output / "evaluation_context.json", {
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": sha256(args.manifest),
        "variant": experiment["variant"], "environment": environment(device),
        "training_seed": cfg["training"]["seed"],
        "test_used_for_checkpoint_selection": False,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()