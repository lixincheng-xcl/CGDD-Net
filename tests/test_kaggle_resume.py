"""Check short, synthetic session checkpoints; never report toy performance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from experiments.train_controlled import main as controlled_main


def _record(root: Path, name: str, salt: int):
    yy, xx = np.mgrid[:32,:32]
    a = ((xx * 3 + yy * 7 + salt) % 255).astype("uint8")
    y = (((xx + salt) % 9 < 4) * 255).astype("uint8")
    f = np.full((32,32),255, dtype="uint8")
    for suffix,v in (("image",a),("label",y),("fov",f)):
        Image.fromarray(v).save(root / f"{name}_{suffix}.png")
    return {"dataset":"SYNTH","id":"SYNTH/"+name,"subject":name,
            "image":name+"_image.png", "label":name+"_label.png", "fov":name+"_fov.png"}


def test_kaggle_checkpoint_segments_recover_without_intermediate_test(tmp_path):
    data = tmp_path / "cases"; data.mkdir()
    a,b=_record(data,"train",1),_record(data,"test",13)
    for split, row in (("train",a),("test",b)):
        (tmp_path/f"{split}.json").write_text(json.dumps({"dataset":"SYNTH","records":[row]}))
    cfg = {"model":{"in_channels":1,"base_channels":4,"dropout":0,"ablation":"full"},
           "preprocessing":{"color_mode":"gray","max_side":None,"pad_multiple":1},
           "training":{"seed":1,"epochs":3,"batch_size":2,"patch_size":32,
                       "patches_per_epoch":2,"num_workers":0,"lr":0.001,
                       "weight_decay":0.0,"warmup_epochs":0,"restart_epochs":10,
                       "restart_mult":2,"warmup_start_factor":0.1,"min_lr_factor":0.0},
           "inference":{"patch_size":64,"stride":16,"batch_size":2,"threshold":0.5}}
    c=tmp_path/"cfg.json";c.write_text(json.dumps(cfg))
    target=tmp_path/"run"
    args=["--config",str(c),"--variant","unet_same_width","--train-manifest",str(tmp_path/"train.json"),
          "--test-manifest",str(tmp_path/"test.json"),"--data-root",str(data),
          "--output",str(target),"--seed","11","--device","cpu", "--max-epochs-this-session","1"]
    controlled_main(args)
    assert (target/"latest.pt").is_file()
    assert (target/"progress.json").is_file()
    assert not (target/"test").exists()
    assert not (target/"experiment_summary.json").exists()
    assert json.loads((target/"progress.json").read_text())["completed_epochs"] == 1
    controlled_main(args+["--resume"])
    assert json.loads((target/"progress.json").read_text())["completed_epochs"] == 2
    assert not (target/"test").exists()
    controlled_main(args+["--resume"])
    assert (target/"final.pt").is_file()
    assert (target/"experiment_summary.json").is_file()
    assert (target/"test/per_image.csv").is_file()
    assert not (target/"progress.json").exists()