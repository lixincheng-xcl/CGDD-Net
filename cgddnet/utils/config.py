from __future__ import annotations
from pathlib import Path
import yaml


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())
