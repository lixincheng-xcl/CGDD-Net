"""Profile an explicitly shaped forward pass; never substitute manuscript values.

An optional fvcore report covers supported operators only. A batch of patches
and a single image with the same total pixel count are not interchangeable
profiling inputs. Patch extraction, reconstruction, metrics and latency are
outside the operation-count scope of this script.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cgddnet.model import CGDDNet
from cgddnet.runtime import load_config, write_json


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Expected a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--height", type=positive_int, default=64)
    parser.add_argument("--width", type=positive_int, default=64)
    parser.add_argument("--device", default="cpu", help="Execution device, e.g. cpu or cuda:0")
    parser.add_argument("--fvcore", action="store_true",
                        help="Optional supported-operator estimate; install .[profile]")
    parser.add_argument("--output", default="profile.json")
    return parser.parse_args(argv)


def collect_profile(args) -> dict:
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this runtime")
    cfg = load_config(args.config)
    model = CGDDNet(**cfg["model"]).to(device).eval()
    x = torch.zeros(args.batch_size, cfg["model"]["in_channels"],
                    args.height, args.width, device=device, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    result = {
        "model_config": cfg["model"],
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "dtype": str(x.dtype),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "scope": "model_forward_only; excludes patch extraction, reconstruction and metrics",
        "environment": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "execution_os": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "status": "measured from the public implementation at the explicit input shape",
        "is_manuscript_reproduction": False,
    }
    if args.fvcore:
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError as error:
            raise RuntimeError("Install profiling dependencies with: pip install '.[profile]'") from error
        analysis = FlopCountAnalysis(model, x)
        value = float(analysis.total())
        unsupported = dict(analysis.unsupported_ops())
        result.update(
            profiling_library="fvcore",
            profiling_library_version=importlib.metadata.version("fvcore"),
            estimated_supported_ops=value,
            estimated_supported_ops_per_patch=value / args.batch_size,
            estimated_supported_gops=value / 1e9,
            unit="fvcore operations; one fused multiply-add counts as one operation",
            by_operator=dict(analysis.by_operator()),
            unsupported_ops=unsupported,
            full_coverage=not unsupported,
            note="The estimate is not a complete FLOP total when operators are unsupported. It is not a latency measurement or an independent verification of the manuscript's 29.96 GFLOPs.",
        )
    return result


def main(argv=None) -> None:
    args = parse_args(argv)
    result = collect_profile(args)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
