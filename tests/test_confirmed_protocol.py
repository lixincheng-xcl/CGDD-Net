"""Regression checks for the author-confirmed protocol and explicit profiling."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cgddnet.inference import sliding_window
from scripts.profile_model import collect_profile, parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_author_confirmed_defaults():
    cfg = json.loads((ROOT / "configs/default.json").read_text())
    assert cfg["training"]["patch_size"] == cfg["inference"]["patch_size"] == 64
    assert cfg["training"]["batch_size"] == 64
    assert cfg["training"]["epochs"] == 100
    assert cfg["training"]["early_stopping"] == 0
    assert cfg["training"]["lr"] == 0.001
    assert cfg["model"]["dropout"] == 0.1
    assert inspect.signature(sliding_window).parameters["patch_size"].default == 64


def test_default_patch_reconstruction_keeps_full_image_shape():
    class ConstantLogits(torch.nn.Module):
        def forward(self, x):
            assert tuple(x.shape[-2:]) == (64, 64)
            return x.new_zeros((len(x), 1, 64, 64))
    image = np.ones((1, 71, 83), dtype=np.float32)
    result = sliding_window(ConstantLogits(), image)
    assert result.shape == (71, 83)
    np.testing.assert_allclose(result, 0.5)


def test_explicit_profiling_batch_is_recorded():
    args = parse_args(["--config", str(ROOT / "configs/default.json"),
                       "--batch-size", "2", "--height", "16", "--width", "16"])
    result = collect_profile(args)
    assert result["input_shape"] == result["output_shape"] == [2, 1, 16, 16]
    assert result["trainable_parameters"] == 1956802
    assert result["is_manuscript_reproduction"] is False
    assert "estimated_supported_ops" not in result


@pytest.mark.parametrize("option,value", [("--batch-size", "0"), ("--batch-size", "-1"),
                                          ("--height", "0"), ("--width", "-3")])
def test_profiler_rejects_invalid_dimensions(option, value):
    with pytest.raises(SystemExit):
        parse_args([option, value])
