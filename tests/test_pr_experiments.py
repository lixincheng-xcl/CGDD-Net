"""Architecture and end-to-end software tests; NOT retinal benchmark results."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from experiments.pr_models import (
    ControlledSAMG, IndependentDetailNet, SameProtocolUNet,
    ZeroOffsets, build_model, count_trainable,
)
from experiments.launch_grid import plan
from experiments.train_controlled import main as controlled_main


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("variant", [
    "full", "independent_detail", "fixed_offsets", "uniform_samg",
    "global_only_samg", "local_only_samg", "unet_same_width",
    "unet_budget_matched",
])
def test_variant_forward_backward(variant):
    model = build_model(variant, base_channels=4, dropout=0)
    image = torch.randn(2, 1, 17, 19)
    out = model(image)
    F.binary_cross_entropy_with_logits(out, torch.ones_like(out)).backward()
    assert out.shape == (2, 1, 17, 19)
    assert torch.isfinite(out).all()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient to {variant}: {name}"
        assert torch.isfinite(p.grad).all(), f"Bad gradient to {variant}: {name}"


def test_fixed_offset_removes_all_predictors():
    model = build_model("fixed_offsets", base_channels=4)
    heads = [head for encoder in model.encoders for head in encoder.attention_heads]
    assert len(heads) == 20
    assert all(isinstance(head.offset, ZeroOffsets) for head in heads)
    assert not any("offset.weight" in key or "offset.bias" in key
                   for key in model.state_dict())


def test_uniform_gate_removes_unused_gate_parameters():
    model = build_model("uniform_samg", base_channels=4)
    assert all(isinstance(layer, ControlledSAMG) and layer.mode == "uniform"
               for layer in model.samg.values())
    assert not any("global_gate" in name or "local_gate" in name
                   for name, _ in model.named_parameters())


def test_independent_branch_parameters_are_not_shared():
    model = build_model("independent_detail", base_channels=4)
    assert isinstance(model, IndependentDetailNet)
    assert model.independent_dcdf["1"] is not model.independent_dcdf["2"]
    assert model.independent_dcdf["2"] is not model.independent_dcdf["3"]
    assert count_trainable(model) > count_trainable(build_model("full", base_channels=4))


def test_unet_width_control_and_parameter_matching():
    same = build_model("unet_same_width", base_channels=8)
    matched = build_model("unet_budget_matched", base_channels=8)
    reference = build_model("full", base_channels=8)
    assert isinstance(same, SameProtocolUNet)
    assert same.base_channels == 8
    assert isinstance(matched, SameProtocolUNet)
    assert matched.base_channels >= 8
    assert abs(count_trainable(matched) - count_trainable(reference)) / count_trainable(reference) < 0.15


def test_planning_is_nonexecuting_and_counted(tmp_path):
    args = SimpleNamespace(manifest_root=tmp_path / "manifests", data_root=tmp_path,
                           output_root=tmp_path / "runs", config=tmp_path / "cfg.json",
                           device="cuda", datasets=["DRIVE", "STARE"],
                           variants=["full", "independent_detail"], seeds=[42, 2026, 3407],
                           execute=False, smoke=False, resume=False)
    jobs = plan(args)
    assert len(jobs) == 12
    assert all("--epochs-override" not in j["command"] for j in jobs)
    assert all(j["seed"] in (42, 2026, 3407) for j in jobs)


def _make_synthetic_files(root, prefix, offset):
    # Deliberately distinct original-image content, vessel labels and FOV masks.
    y, x = np.mgrid[:32, :32]
    image = np.uint8((x * 5 + y * 3 + offset) % 255)
    label = np.uint8(((x + offset) % 9 < 3) * 255)
    fov = np.uint8(np.ones((32, 32)) * 255)
    for name, pixels in (("image", image), ("label", label), ("fov", fov)):
        Image.fromarray(pixels).save(root / f"{prefix}_{name}.png")
    return {"id": f"SYNTH/{prefix}", "dataset": "SYNTH", "subject": prefix,
            "image": f"{prefix}_image.png", "label": f"{prefix}_label.png",
            "fov": f"{prefix}_fov.png"}


def test_synthetic_fixed_epoch_pipeline(tmp_path):
    root = tmp_path / "sample_images"
    root.mkdir()
    train = _make_synthetic_files(root, "train", 3)
    test = _make_synthetic_files(root, "test", 7)
    path_train, path_test = tmp_path / "train.json", tmp_path / "test.json"
    path_train.write_text(json.dumps({"dataset": "SYNTH", "records": [train]}))
    path_test.write_text(json.dumps({"dataset": "SYNTH", "records": [test]}))
    cfg = {
        "model": {"in_channels": 1, "base_channels": 4, "dropout": 0.0, "ablation": "full"},
        "preprocessing": {"color_mode": "gray", "max_side": None, "pad_multiple": 1},
        "training": {"seed": 42, "epochs": 100, "batch_size": 2, "patch_size": 32,
                     "patches_per_epoch": 100, "num_workers": 0, "lr": 0.001,
                     "weight_decay": 0.0, "warmup_epochs": 0, "restart_epochs": 10,
                     "restart_mult": 2, "warmup_start_factor": 0.1, "min_lr_factor": 0.0},
        "inference": {"patch_size": 64, "stride": 16, "batch_size": 2, "threshold": 0.5},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    output = tmp_path / "smoke_run"
    controlled_main(["--config", str(config_path), "--variant", "unet_same_width",
                     "--train-manifest", str(path_train), "--test-manifest", str(path_test),
                     "--data-root", str(root), "--output", str(output), "--seed", "19",
                     "--device", "cpu", "--epochs-override", "1", "--patches-override", "2"])
    summary = json.loads((output / "experiment_summary.json").read_text())
    assert summary["smoke_test_only"] is True
    assert summary["trained_epochs"] == 1
    assert (output / "final.pt").is_file()
    assert (output / "test" / "per_image.csv").is_file()
    assert summary["test_scores"]["images"] == 1
    assert 0 <= summary["test_scores"]["metrics"]["AUC"]["mean"] <= 1