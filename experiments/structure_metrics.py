"""Evaluate thin-vessel centerlines and topology on saved full-image predictions.

No training or new inference is required. Only predictions from actual
retinal datasets may be reported as benchmark evidence. Thin width is defined
from the ground-truth skeleton radius percentile within each image, not tuned
from predictions or test performance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
try:
    from skimage.morphology import skeletonize
except ImportError as error:
    raise ImportError("Install scikit-image to compute vessel skeleton metrics") from error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgddnet.data import load_case, load_manifest
from cgddnet.runtime import sha256


def metrics_from_maps(probability, vessel, fov, threshold=0.5, thin_quantile=0.25,
                      matching_radius=2):
    if not 0 < thin_quantile < 1 or matching_radius < 0 or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold, quantile or matching radius")
    prob, gt, mask = map(np.asarray, (probability, vessel, fov))
    if not (prob.shape == gt.shape == mask.shape) or prob.ndim != 2:
        raise ValueError("Probability, vessel and FOV must be equally shaped 2-D arrays")
    if not np.isfinite(prob).all():
        raise ValueError("Nonfinite predicted probability")
    mask = mask > 0
    gt = (gt > 0.5) & mask
    pred = (prob >= threshold) & mask
    gt_skel = skeletonize(gt)
    pred_skel = skeletonize(pred)
    tp_skel = (gt_skel & pred).sum()
    predicted_skel_total = int(pred_skel.sum())
    reference_skel_total = int(gt_skel.sum())
    tprec = (pred_skel & gt).sum() / predicted_skel_total if predicted_skel_total else None
    tsens = tp_skel / reference_skel_total if reference_skel_total else None
    cldice = (2*tprec*tsens / (tprec+tsens)
              if tprec is not None and tsens is not None and (tprec+tsens) > 0 else None)
    radius = distance_transform_edt(gt)
    true_radius = radius[gt_skel]
    if true_radius.size:
        cutoff = float(np.quantile(true_radius, thin_quantile))
        thin_center = gt_skel & (radius <= cutoff)
        distance_from_pred = distance_transform_edt(~pred_skel)
        correct_thin = int((thin_center & (distance_from_pred <= matching_radius)).sum())
        thin_total = int(thin_center.sum())
        thin_recall = correct_thin / thin_total if thin_total else None
    else:
        cutoff, thin_total, thin_recall = None, 0, None
    false_positives = int((pred & ~gt & mask).sum())
    background = int((~gt & mask).sum())
    return {
        "thin_skeleton_recall": thin_recall,
        "cldice": cldice,
        "skeleton_reference_pixels": reference_skel_total,
        "skeleton_predicted_pixels": predicted_skel_total,
        "thin_skeleton_reference_pixels": thin_total,
        "thin_radius_cutoff_pixels": cutoff,
        "fov_false_positive_rate": false_positives / background if background else None,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", required=True, help="Saved output/test with per_image.csv")
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--data-root")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--thin-quantile", type=float, default=0.25)
    p.add_argument("--matching-radius", type=int, default=2)
    args = p.parse_args(argv)
    root = Path(args.evaluation)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(args.config).read_text())
    records = {r["id"]: r for r in load_manifest(args.test_manifest, args.data_root)}
    with (root / "per_image.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(records):
        raise ValueError("Evaluation row count does not match manifest")
    output_rows = []
    for row in rows:
        if row["id"] not in records:
            raise ValueError(f"Unknown image: {row['id']}")
        if not row.get("prediction"):
            raise ValueError("Saved prediction .npz is required")
        pred_path = (root / row["prediction"]).resolve()
        if not pred_path.is_relative_to(root.resolve()):
            raise ValueError("Prediction path escapes evaluation directory")
        with np.load(pred_path) as item:
            probability = np.asarray(item["probability"])
        _, label, fov = load_case(records[row["id"]], cfg["preprocessing"])
        result = metrics_from_maps(probability, label[0], fov[0],
                                   cfg["inference"]["threshold"], args.thin_quantile,
                                   args.matching_radius)
        output_rows.append({"image_id": row["id"], **result})
    with (output / "per_image_structure.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {"n_images": len(output_rows), "thin_quantile": args.thin_quantile,
               "matching_radius_pixels": args.matching_radius,
               "test_manifest_sha256": sha256(args.test_manifest),
               "metrics": {key: float(np.mean(values)) if values else None for key in
                           ("thin_skeleton_recall", "cldice", "fov_false_positive_rate")
                           for values in [[r[key] for r in output_rows if r[key] is not None]]}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()