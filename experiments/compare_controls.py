"""Compare completed controlled runs without treating images/seeds as duplicates.

Pair by (dataset, seed, image_id). First reduce AUC differences to one paired
mean per independent training seed, then report the exact two-sided sign-flip
randomization test across those seed-level differences. An assumption of
exchangeable sign under the null is necessary. Three seeds have low power;
report effects even when significance is absent. This NEW analysis is not the
historical paper's Table 7.
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
import math
from pathlib import Path
from statistics import mean, stdev


def exact_signflip_p(differences):
    values = [float(x) for x in differences]
    if not values or len(values) > 20:
        raise ValueError("Require 1..20 independent seed-level paired differences")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite difference")
    observed = abs(sum(values))
    extreme = sum(1 for flips in product((-1, 1), repeat=len(values))
                  if abs(sum(a * b for a, b in zip(values, flips))) >= observed - 1e-12)
    return extreme / (2 ** len(values))


def holm_adjust(items):
    """Return Holm step-down adjusted p-values in original item order."""
    indexed = sorted(enumerate(items), key=lambda x: x[1])
    output = [None] * len(items)
    current = 0.0
    for j, (i, p) in enumerate(indexed):
        current = max(current, min(1.0, p * (len(items) - j)))
        output[i] = current
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-image", required=True, help="Output of collect_results.py")
    p.add_argument("--reference", default="full")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    by_dataset = {}
    with open(args.per_image, newline="") as f:
        for row in csv.DictReader(f):
            if not row["AUC"]:
                raise ValueError("Undefined AUC in input")
            ds, model, seed, image = row["dataset"], row["variant"], int(row["seed"]), row["image_id"]
            key = (seed, image)
            d = by_dataset.setdefault(ds, {}).setdefault(model, {})
            if key in d:
                raise ValueError(f"Duplicate pair: {ds}/{model}/{key}")
            d[key] = float(row["AUC"])
    results = []
    for dataset, methods in sorted(by_dataset.items()):
        reference = methods.get(args.reference)
        if not reference:
            raise ValueError(f"Missing reference {args.reference} on {dataset}")
        family = []
        for variant, scores in sorted(methods.items()):
            if variant == args.reference:
                continue
            if set(scores) != set(reference):
                raise ValueError(f"Unpaired images or seeds in {dataset}/{variant}")
            by_seed = {}
            for (seed, image_id), score in scores.items():
                by_seed.setdefault(seed, []).append(reference[seed, image_id] - score)
            delta = [mean(values) for seed, values in sorted(by_seed.items())]
            family.append({"dataset": dataset, "method": variant, "reference": args.reference,
                           "n_independent_seeds": len(delta),
                           "mean_paired_auc_gain": mean(delta),
                           "sd_paired_gain_between_seeds": stdev(delta) if len(delta)>1 else None,
                           "p_signflip_raw": exact_signflip_p(delta)})
        adjusted = holm_adjust([r["p_signflip_raw"] for r in family])
        for entry, adj in zip(family, adjusted):
            entry["p_holm_within_dataset"] = adj
            results.append(entry)
    if not results:
        raise ValueError("No comparable methods")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Compared {len(results)} matched method/dataset pairs; no old Table 7 values changed")


if __name__ == "__main__":
    main()