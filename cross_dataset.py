#!/usr/bin/env python3
"""Source-only cross-dataset evaluation without target fine-tuning."""
import argparse
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/default.yaml")
p.add_argument("--source", required=True)
p.add_argument("--target", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--output", default=None)
a = p.parse_args()
cmd = [sys.executable, "test.py", "--config", a.config, "--dataset", a.target,
       "--checkpoint", a.checkpoint, "--split", "test"]
if a.output:
    cmd += ["--output", a.output]
raise SystemExit(subprocess.call(cmd))
