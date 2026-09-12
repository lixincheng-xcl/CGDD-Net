"""Exercise real CGDD-Net training/resume/evaluation on synthetic images only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir",help="Fresh directory; defaults to a temporary directory")
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    work=Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="cgdd-smoke-"))
    if work.exists() and any(work.iterdir()): raise FileExistsError("Use a fresh smoke directory")
    work.mkdir(parents=True,exist_ok=True)
    data=work/"synthetic"; data.mkdir()
    rng=np.random.default_rng(17)
    records=[]
    for i in range(4):
        image=rng.integers(0,256,(32,32),dtype=np.uint8)
        label=np.zeros((32,32),np.uint8); label[:,5+i*3:8+i*3]=255
        fov=np.ones((32,32),np.uint8)*255
        for key,array in (("image",image),("label",label),("fov",fov)):
            Image.fromarray(array).save(data/f"{i}_{key}.png")
        records.append(dict(id=f"synthetic-{i}",subject=f"subject-{i}",
                            **{key:f"{i}_{key}.png" for key in ("image","label","fov")}))
    for name,items in (("train",records[:2]),("val",records[2:3]),("test",records[3:])):
        (work/f"{name}.json").write_text(json.dumps({"records":items,"data_root":"synthetic"}))
    config=json.loads((repo/"configs/default.json").read_text())
    config["description"]="Synthetic software smoke test, not a benchmark experiment"
    config["training"].update(epochs=2,batch_size=2,patch_size=32,patches_per_epoch=2,
                              early_stopping=0,warmup_epochs=1,restart_epochs=1)
    config["inference"].update(patch_size=32,stride=16,batch_size=2)
    cfg=work/"smoke.json"; cfg.write_text(json.dumps(config,indent=2))
    env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    def run(*command): subprocess.run([sys.executable,*map(str,command)],cwd=repo,env=env,check=True)
    common=["train.py","--config",cfg,"--train-manifest",work/"train.json","--val-manifest",work/"val.json",
            "--data-root",data,"--output",work/"run","--device","cpu"]
    run(*common)
    run(*common,"--resume",work/"run/latest.pt")
    run("evaluate.py","--checkpoint",work/"run/best.pt","--manifest",work/"test.json", "--data-root",data,
        "--output",work/"evaluation","--device","cpu")
    summary=json.loads((work/"evaluation/summary.json").read_text())
    assert summary["images"]==1 and summary["metrics"]["AUC"]["defined_images"]==1
    assert (work/"evaluation/predictions/0001.npz").is_file()
    checkpoint=work/"run/best.pt"
    assert checkpoint.stat().st_size>0
    report={"status":"passed","data":"four synthetic 32x32 images","epochs":2,
            "checks":["real model forward/backward", "checkpoint serialization", "same-run resume",
                      "saved-checkpoint inference", "per-image metrics and probability export"],
            "benchmark_reproduction":False,"working_directory":str(work)}
    (work/"smoke_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
