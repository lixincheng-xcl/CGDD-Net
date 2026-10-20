"""Aggregate actual controlled runs; never infer missing runs or image scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--require-seeds", nargs="+", type=int, default=[42, 2026, 3407])
    args = p.parse_args(argv)
    root = Path(args.runs_root)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to replace a previous results report")
    output.mkdir(parents=True, exist_ok=True)
    run_rows = []
    per_image_rows = []
    for report in sorted(root.glob("*/*/seed_*/experiment_summary.json")):
        rel = report.relative_to(root).parts
        dataset, variant = rel[0], rel[1]
        seed = int(rel[2].removeprefix("seed_"))
        data = json.loads(report.read_text())
        if data["smoke_test_only"]:
            continue
        image_csv = report.parent / "test" / "per_image.csv"
        if not image_csv.is_file():
            raise FileNotFoundError(f"No per-image metrics: {image_csv}")
        row = {"dataset": dataset, "variant": variant, "seed": seed,
               "parameters": data["parameter_count"],
               **{name: data["test_scores"]["metrics"][name]["mean"]
                  for name in ("AUC", "F1", "SE", "SP", "ACC")}}
        if (dataset, variant, seed) in {(r["dataset"], r["variant"], r["seed"]) for r in run_rows}:
            raise ValueError("Duplicate run dataset/variant/seed")
        run_rows.append(row)
        with image_csv.open(newline="") as f:
            for item in csv.DictReader(f):
                per_image_rows.append({"dataset": dataset, "variant": variant, "seed": seed,
                                       "image_id": item["id"], "AUC": item["AUC"],
                                       "F1": item["F1"]})
    if not run_rows:
        raise ValueError("No completed, non-smoke experimental runs found")
    with (output / "run_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, run_rows[0].keys())
        writer.writeheader()
        writer.writerows(run_rows)
    with (output / "per_image_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, per_image_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_image_rows)
    grouped = {}
    for row in run_rows:
        grouped.setdefault((row["dataset"], row["variant"]), []).append(row)
    summary = []
    for (dataset, variant), rows in sorted(grouped.items()):
        seeds = sorted(r["seed"] for r in rows)
        expected = sorted(args.require_seeds)
        if seeds != expected:
            raise ValueError(f"Incomplete experiment {dataset}/{variant}: seeds {seeds} != {expected}")
        entry = {"dataset": dataset, "variant": variant, "num_seeds": len(seeds),
                 "seeds": ",".join(map(str, seeds))}
        for metric in ("AUC", "F1", "SE", "SP", "ACC"):
            values = [r[metric] for r in rows]
            entry[f"{metric}_mean"] = statistics.mean(values)
            entry[f"{metric}_sd_between_seeds"] = statistics.stdev(values) if len(values)>1 else None
        summary.append(entry)
    with (output / "aggregate_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (output / "readme.txt").write_text(
        "Each run metric is the unweighted mean across TEST IMAGES.\n"
        "Aggregate SD is across TRAINING SEEDS, not test images.\n"
        "Per-image metrics must be paired by image_id within each seed.\n"
        "Do not treat patches or repeated-seed scores as independent image observations.\n")
    print(f"Aggregated {len(run_rows)} finished real runs into {output}")


if __name__ == "__main__":
    main()