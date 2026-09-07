#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('run_dir', help='e.g. runs/DRIVE/full')
a=p.parse_args()
vals=[]
for f in sorted(Path(a.run_dir).glob('seed_*/best_metrics.json')):
    vals.append(json.loads(f.read_text()))
if not vals:
    raise SystemExit('No seed_*/best_metrics.json files found')
for k in ['SE','SP','ACC','F1','AUC']:
    x=np.array([v[k] for v in vals],dtype=float)
    print(f'{k}: {x.mean():.6f} ± {x.std(ddof=1):.6f}  (n={len(x)})')
