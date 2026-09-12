"""Explicit field-of-view metrics; undefined quantities remain undefined."""
import numpy as np
from sklearn.metrics import roc_auc_score


def binary_metrics(probability, target, fov, threshold=0.5):
    probability, target, fov = map(np.asarray, (probability, target, fov))
    if probability.shape != target.shape or target.shape != fov.shape:
        raise ValueError("Probability, label and FOV shapes must match")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    inside = fov > 0
    if not inside.any():
        raise ValueError("Empty field of view")
    y, pred = target[inside] > 0.5, probability[inside] >= threshold
    tp, tn = int((y & pred).sum()), int((~y & ~pred).sum())
    fp, fn = int((~y & pred).sum()), int((y & ~pred).sum())
    divide = lambda a, b: a / b if b else None
    return dict(SE=divide(tp, tp+fn), SP=divide(tn, tn+fp),
                ACC=(tp+tn)/len(y), F1=divide(2*tp, 2*tp+fp+fn),
                AUC=float(roc_auc_score(y, probability[inside])) if np.unique(y).size == 2 else None,
                TP=tp, TN=tn, FP=fp, FN=fn, pixels=len(y))


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    output = {"images": len(rows), "aggregation": "unweighted mean of image-level metrics", "metrics": {}}
    for metric in ("SE", "SP", "ACC", "F1", "AUC"):
        values = [r[metric] for r in rows if r[metric] is not None]
        output["metrics"][metric] = {
            "mean": float(np.mean(values)) if values else None,
            "std_between_images": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "defined_images": len(values), "undefined_images": len(rows)-len(values)}
    output["confusion_totals"] = {k: sum(r[k] for r in rows) for k in ("TP", "TN", "FP", "FN")}
    return output
