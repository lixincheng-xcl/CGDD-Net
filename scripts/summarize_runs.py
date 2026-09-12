"""Aggregate independent runs and optionally paired Wilcoxon/Holm comparisons.

Input CSV: method,seed,dataset,AUC,F1,ACC,SE,SP. Each row is one completed
run's dataset-level metrics; images must not be passed as independent seeds.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--reference",help="Exact method name for paired tests")
    p.add_argument("--output",default="run_statistics.json")
    args=p.parse_args()
    with open(args.csv,newline="") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError("Empty CSV")
    if not {"method","seed","dataset","AUC"}.issubset(rows[0]): raise ValueError("Missing required columns")
    grouped={}
    for row in rows:
        key=(row["dataset"],row["method"])
        if row["seed"] in grouped.setdefault(key,{}): raise ValueError("Duplicate seed within a dataset/method")
        value=float(row["AUC"])
        if not np.isfinite(value) or not 0<=value<=1: raise ValueError("AUC must be a fraction in [0,1]")
        grouped[key][row["seed"]]=value
    output={"unit":"independent training seed","dispersion":"sample standard deviation (ddof=1)","summaries":[],"tests":[]}
    for (dataset,method),runs in sorted(grouped.items()):
        values=list(runs.values())
        output["summaries"].append(dict(dataset=dataset,method=method,seeds=sorted(runs),n=len(runs),
                                       mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(runs)>1 else None))
    if args.reference:
        for dataset in sorted({key[0] for key in grouped}):
            ref=grouped.get((dataset,args.reference))
            if ref is None: raise ValueError(f"Reference missing for {dataset}")
            comparisons=[]
            for (ds,method),runs in sorted(grouped.items()):
                if ds!=dataset or method==args.reference: continue
                if set(runs)!=set(ref): raise ValueError(f"Seed pairing mismatch: {dataset}/{method}")
                if len(runs)<2: raise ValueError("Paired comparison requires at least two matched runs")
                delta=np.array([runs[s]-ref[s] for s in sorted(ref)])
                pval=1.0 if np.all(delta==0) else float(wilcoxon(delta,alternative="two-sided",zero_method="wilcox",method="auto").pvalue)
                comparisons.append(dict(dataset=dataset,method=method,reference=args.reference,n=len(runs),p_raw=pval))
            running=0.0
            for rank,item in enumerate(sorted(comparisons,key=lambda x:x["p_raw"])):
                running=max(running,min(1.0,item["p_raw"]*(len(comparisons)-rank)))
                item["p_holm"]=running
            output["tests"].extend(comparisons)
        output["test_protocol"]="two-sided paired Wilcoxon, zero_method=wilcox, method=auto; Holm family = all methods vs reference within each dataset"
    Path(args.output).write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__=="__main__": main()
