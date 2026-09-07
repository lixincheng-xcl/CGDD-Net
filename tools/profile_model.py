#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cgddnet.models import build_model

p=argparse.ArgumentParser()
p.add_argument('--ablation',default='full')
p.add_argument('--size',type=int,default=256)
a=p.parse_args()
m=build_model(ablation=a.ablation).eval()
params=sum(x.numel() for x in m.parameters() if x.requires_grad)
print(f'Trainable parameters: {params:,} ({params/1e6:.2f} M)')
try:
    from fvcore.nn import FlopCountAnalysis
    x=torch.randn(1,3,a.size,a.size)
    f=FlopCountAnalysis(m,x)
    print(f'fvcore counted FLOPs: {f.total()/1e9:.2f} G')
    unsupported=f.unsupported_ops()
    if unsupported:
        print('Unsupported/un-counted operators:', dict(unsupported))
except ImportError:
    print('Install fvcore to report counted FLOPs: pip install fvcore')
