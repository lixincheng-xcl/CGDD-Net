"""Evaluate one saved checkpoint on a within-dataset or transfer manifest."""
import argparse
from pathlib import Path
import shutil
import torch
from cgddnet.data import load_manifest
from cgddnet.model import CGDDNet
from cgddnet.inference import evaluate_records
from cgddnet.runtime import device_from_name, environment, write_json, sha256


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--data-root")
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="auto")
    args=p.parse_args(argv)
    output=Path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError("Select an empty evaluation directory")
    device=device_from_name(args.device)
    state=torch.load(args.checkpoint,map_location=device,weights_only=True)
    config=state["config"]
    model=CGDDNet(**config["model"]).to(device)
    model.load_state_dict(state["model"])
    records=load_manifest(args.manifest,args.data_root)
    summary=evaluate_records(model,records,config["preprocessing"],config["inference"],device,output)
    write_json(output/"evaluation_context.json",{"environment":environment(device),"config":config,
               "checkpoint_sha256":sha256(args.checkpoint),"manifest_sha256":sha256(args.manifest),
               "training_seed":config["training"]["seed"],"threshold_tuned_on_evaluation_set":False})
    shutil.copy2(args.manifest,output/"evaluation_manifest.json")
    print(summary)


if __name__=="__main__": main()
