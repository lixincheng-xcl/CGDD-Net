"""Measure this implementation, without substituting manuscript table values."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from cgddnet.model import CGDDNet
from cgddnet.runtime import load_config, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/default.json")
    p.add_argument("--height",type=int,default=64)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--fvcore",action="store_true",help="Optional operator-count estimate; install .[profile]")
    p.add_argument("--output",default="profile.json")
    args=p.parse_args(); torch.set_num_threads(1)
    cfg=load_config(args.config); model=CGDDNet(**cfg["model"]).eval()
    x=torch.zeros(1,cfg["model"]["in_channels"],args.height,args.width)
    with torch.no_grad(): y=model(x)
    result={"model_config":cfg["model"],"input_shape":list(x.shape),"output_shape":list(y.shape),
            "parameters":sum(p.numel() for p in model.parameters()),
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "status":"measured on the newly constructed release, not the historical server model"}
    if args.fvcore:
        try: from fvcore.nn import FlopCountAnalysis
        except ImportError: raise SystemExit("Install profiling dependencies with: pip install '.[profile]'")
        analysis=FlopCountAnalysis(model,x)
        value=analysis.total()
        unsupported=dict(analysis.unsupported_ops())
        result.update(estimated_supported_ops=value,unit="fvcore operations; one fused multiply-add counts as one",
                      by_operator=dict(analysis.by_operator()),unsupported_ops=unsupported,
                      full_coverage=not unsupported,
                      note="A partial operator count is not a complete FLOP total and is not comparable with the manuscript until input and conventions match.")
    write_json(args.output,result); print(result)


if __name__=="__main__": main()
