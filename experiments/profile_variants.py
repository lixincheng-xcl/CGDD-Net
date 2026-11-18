"""Measure actual parameter counts and fvcore-supported operations for PR variants.

Operation estimates exclude unsupported operators and patch reconstruction.
The output is a NEW measurement and never validates the manuscript's original
29.96 GFLOPs without a matching operator/input convention.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgddnet.runtime import load_config
from experiments.pr_models import VARIANTS, build_model, count_trainable


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fvcore", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if min(args.batch_size, args.height, args.width) < 1:
        p.error("Batch and image dimensions must be positive")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but not available")
    cfg = load_config(args.config)
    report = {"profile_type": "model_forward_only; supported_ops_only_when_fvcore",
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda_runtime": torch.version.cuda, "host_os": platform.platform(),
                              "device": str(device)},
              "model_input": [args.batch_size, cfg["model"]["in_channels"], args.height, args.width],
              "fvcore_convention": "one multiply-accumulate is one counted operation",
              "no_training_or_retinal_evaluation": True, "results": []}
    if args.fvcore:
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError as error:
            raise RuntimeError("Install with pip install -e '.[profile]'") from error
        report["environment"]["fvcore"] = importlib.metadata.version("fvcore")
    for variant in args.variants:
        model = build_model(variant, **cfg["model"]).to(device).eval()
        inputs = torch.zeros(args.batch_size, cfg["model"]["in_channels"], args.height, args.width,
                             dtype=torch.float32, device=device)
        with torch.no_grad():
            output = model(inputs)
        row = {"variant": variant, "parameters": count_trainable(model),
               "all_parameters": sum(x.numel() for x in model.parameters()),
               "output_shape": list(output.shape),
               "unet_base_width": getattr(model, "base_channels", None),
               "supported_ops": None, "supported_ops_g": None,
               "unsupported_ops": {}, "full_coverage": None}
        if args.fvcore:
            # Do not drop warnings: unsupported_ops must be documented.
            analysis = FlopCountAnalysis(model, inputs)
            row["supported_ops"] = int(analysis.total())
            row["supported_ops_g"] = row["supported_ops"] / 1e9
            row["unsupported_ops"] = dict(analysis.unsupported_ops())
            row["full_coverage"] = not bool(row["unsupported_ops"])
        report["results"].append(row)
        print(f"{variant}: params={row['parameters']}, partial_Gops={row['supported_ops_g']}", flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Will not overwrite prior profile: {destination}")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    csv_path = destination.with_suffix(".csv")
    if csv_path.exists():
        raise FileExistsError(f"Will not overwrite prior CSV: {csv_path}")
    keys = ("variant", "parameters", "all_parameters", "supported_ops", "supported_ops_g",
            "full_coverage", "unet_base_width")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in keys} for row in report["results"]])


if __name__ == "__main__":
    main()