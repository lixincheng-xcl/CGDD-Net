from __future__ import annotations

import cv2
import numpy as np


def preprocess_fundus(rgb: np.ndarray, fov: np.ndarray | None = None,
                       clahe_clip: float = 2.0, gamma: float = 1.2) -> np.ndarray:
    """Paper preprocessing. Returns float32 CHW, replicated to three channels."""
    if rgb.ndim == 2:
        gray = rgb.astype(np.float32)
    else:
        # Input is expected RGB.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if fov is None:
        valid = np.ones_like(gray, dtype=bool)
    else:
        valid = fov > 0
    vals = gray[valid]
    mean = float(vals.mean()) if vals.size else float(gray.mean())
    std = float(vals.std()) if vals.size else float(gray.std())
    gray = (gray - mean) / max(std, 1e-6)
    # Map standardized intensities to 8-bit before CLAHE.
    lo, hi = np.percentile(gray[valid], [1, 99]) if valid.any() else np.percentile(gray, [1, 99])
    gray = np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1)
    gray8 = (gray * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray8).astype(np.float32) / 255.0
    gray = np.power(np.clip(gray, 0, 1), gamma).astype(np.float32)
    gray = np.stack([gray, gray, gray], axis=0)
    return gray


def infer_fov(rgb: np.ndarray) -> np.ndarray:
    """Conservative FOV fallback for datasets without a supplied FOV mask."""
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb
    mask = (gray > 8).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == idx).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
