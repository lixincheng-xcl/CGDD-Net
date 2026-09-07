from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_metrics(prob: np.ndarray, target: np.ndarray, fov: np.ndarray | None = None,
                   threshold: float = 0.5) -> dict[str, float]:
    prob = np.asarray(prob).reshape(-1)
    target = np.asarray(target).reshape(-1).astype(np.uint8)
    if fov is not None:
        keep = np.asarray(fov).reshape(-1) > 0
        prob, target = prob[keep], target[keep]
    pred = prob >= threshold
    tp = int(np.logical_and(pred, target == 1).sum())
    tn = int(np.logical_and(~pred, target == 0).sum())
    fp = int(np.logical_and(pred, target == 0).sum())
    fn = int(np.logical_and(~pred, target == 1).sum())
    eps = 1e-12
    se = tp / (tp + fn + eps)
    sp = tn / (tn + fp + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    try:
        auc = roc_auc_score(target, prob)
    except ValueError:
        auc = float("nan")
    return {"SE": se, "SP": sp, "ACC": acc, "F1": f1, "AUC": auc,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def aggregate_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    keys = ["SE", "SP", "ACC", "F1", "AUC"]
    return {k: float(np.nanmean([x[k] for x in items])) for k in keys}
