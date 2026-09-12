"""Train CGDD-Net with explicit, image-disjoint training and validation manifests.

Training orchestration reimplements the supplied Apache-2.0 VesselSeg-Pytorch
workflow. Modified in 2026: CGDD-Net, masked BCE, warmup/restarts, manifest
validation, portable configuration, and complete resume/provenance handling.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cgddnet.data import load_manifest, PatchDataset, validate_splits
from cgddnet.inference import evaluate_records
from cgddnet.model import CGDDNet
from cgddnet.runtime import (load_config, seed_everything, device_from_name,
                             environment, write_json, sha256, capture_rng, restore_rng)
from cgddnet.schedule import WarmupCosine


def masked_bce(logits, target, fov):
    if logits.shape != target.shape or target.shape != fov.shape:
        raise ValueError("Logits, target and FOV must have identical shapes")
    if torch.any(fov.flatten(1).sum(1) <= 0):
        raise ValueError("Training batch contains an empty FOV")
    return (F.binary_cross_entropy_with_logits(logits,target,reduction="none")*fov).sum()/fov.sum()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/default.json")
    parser.add_argument("--train-manifest",required=True)
    parser.add_argument("--val-manifest",required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--resume",help="Checkpoint created by this release")
    args=parser.parse_args(argv)
    cfg=load_config(args.config); opt=cfg["training"]
    if min(opt["epochs"],opt["batch_size"],opt["patches_per_epoch"])<1 or opt["num_workers"]<0:
        raise ValueError("Training lengths must be positive and worker count nonnegative")
    device=device_from_name(args.device); seed_everything(opt["seed"])
    train_records=load_manifest(args.train_manifest,args.data_root)
    val_records=load_manifest(args.val_manifest,args.data_root)
    validate_splits({"train":train_records,"val":val_records})
    output=Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Output is nonempty; select a fresh directory or use --resume")
    if args.resume:
        if Path(args.resume).resolve().parent != output.resolve():
            raise ValueError("Resume in the checkpoint's original run directory; --output must match it")
        if not (output/"best.pt").is_file():
            raise FileNotFoundError("Resume requires the original best.pt alongside the checkpoint")
    output.mkdir(parents=True,exist_ok=True)
    model=CGDDNet(**cfg["model"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=opt["lr"],weight_decay=opt["weight_decay"])
    steps=math.ceil(opt["patches_per_epoch"]/opt["batch_size"])
    schedule=WarmupCosine(opt["warmup_epochs"]*steps,opt["restart_epochs"]*steps,
                          opt["restart_mult"],opt["warmup_start_factor"],opt["min_lr_factor"])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    identity={"train_sha256":sha256(args.train_manifest),"val_sha256":sha256(args.val_manifest)}
    start,best,best_epoch,wait=0,float("-inf"),None,0
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=True)
        if state["config"] != cfg or state["manifests"] != identity:
            raise ValueError("Resume requires exactly the same config and manifest files")
        best_state=torch.load(output/"best.pt",map_location="cpu",weights_only=True)
        if (best_state["best_epoch"],best_state["best_auc"]) != (state["best_epoch"],state["best_auc"]):
            raise ValueError("Saved best.pt and resume checkpoint disagree; use the matching run snapshot")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start,best,best_epoch,wait=state["epoch"]+1,state["best_auc"],state["best_epoch"],state["patience"]
        restore_rng(state["rng"])
    write_json(output/"config.json",cfg)
    write_json(output/"environment.json",environment(device))
    write_json(output/"manifest_hashes.json",identity)
    shutil.copy2(args.train_manifest,output/"train_manifest.json")
    shutil.copy2(args.val_manifest,output/"val_manifest.json")
    log_path=output/"history.csv"
    columns=["epoch","train_loss","val_auc","val_f1","lr"]
    if args.resume and log_path.exists():
        with log_path.open(newline="") as f: existing=list(csv.DictReader(f))
        existing=[r for r in existing if int(r["epoch"])<=start]
        with log_path.open("w",newline="") as f:
            writer=csv.DictWriter(f,columns); writer.writeheader(); writer.writerows(existing)
    if not log_path.exists():
        with log_path.open("w",newline="") as f: csv.DictWriter(f,columns).writeheader()
    for epoch in range(start,opt["epochs"]):
        dataset=PatchDataset(train_records,patch_size=opt["patch_size"],
                             patches_per_epoch=opt["patches_per_epoch"],seed=opt["seed"]+epoch,
                             augment=True,preprocessing=cfg["preprocessing"])
        generator=torch.Generator().manual_seed(opt["seed"]+epoch)
        loader=DataLoader(dataset,batch_size=opt["batch_size"],shuffle=True,
                          num_workers=opt["num_workers"],generator=generator)
        model.train(); loss_sum=0; pixels=0
        for image,label,fov in loader:
            image,label,fov=(t.to(device) for t in (image,label,fov))
            optimizer.zero_grad(set_to_none=True)
            loss=masked_bce(model(image),label,fov)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")
            loss.backward(); optimizer.step(); scheduler.step()
            n=int(fov.sum().item()); loss_sum+=loss.item()*n; pixels+=n
        summary=evaluate_records(model,val_records,cfg["preprocessing"],cfg["inference"],device)
        auc=summary["metrics"]["AUC"]["mean"]
        if auc is None: raise ValueError("No defined validation AUC; inspect validation labels/FOVs")
        improved=auc>best
        if improved: best,best_epoch,wait=auc,epoch+1,0
        else: wait+=1
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
               "config":cfg,"manifests":identity,"epoch":epoch,"best_auc":best,"best_epoch":best_epoch,
               "patience":wait,"rng":capture_rng()}
        torch.save(state,output/"latest.pt")
        if improved: torch.save(state,output/"best.pt")
        row=dict(epoch=epoch+1,train_loss=loss_sum/pixels,val_auc=auc,
                 val_f1=summary["metrics"]["F1"]["mean"],lr=optimizer.param_groups[0]["lr"])
        with log_path.open("a",newline="") as f: csv.DictWriter(f,columns).writerow(row)
        print(json.dumps(row),flush=True)
        if opt["early_stopping"] and wait>=opt["early_stopping"]: break
    write_json(output/"training_summary.json",{"best_epoch":best_epoch,"best_validation_auc":best,
                "parameters":sum(p.numel() for p in model.parameters()),"test_set_used_for_selection":False})


if __name__=="__main__": main()
