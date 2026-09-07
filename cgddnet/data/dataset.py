from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocess import infer_fov, preprocess_fundus


@dataclass(frozen=True)
class Sample:
    image: str
    mask: str
    fov: str | None = None
    id: str | None = None


def load_manifest(path: str | Path) -> list[Sample]:
    path = Path(path)
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    return [Sample(**x) for x in data]


def _read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _read_mask(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


class FundusImageDataset(Dataset):
    """Full-image dataset used for validation and inference."""
    def __init__(self, samples: Sequence[Sample], root: str | Path = ".", preprocess: bool = True):
        self.samples = list(samples)
        self.root = Path(root)
        self.do_preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = _read_rgb(self.root / s.image)
        mask = _read_mask(self.root / s.mask)
        fov = _read_mask(self.root / s.fov) if s.fov else infer_fov(image)
        x = preprocess_fundus(image, fov) if self.do_preprocess else image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return {
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "fov": torch.from_numpy(fov[None].astype(np.float32)),
            "id": s.id or Path(s.image).stem,
        }


class RandomPatchDataset(Dataset):
    """On-the-fly in-FOV patch sampler used for training.

    The paper samples a 64x64 in-FOV region, then takes a random 48x48 crop,
    followed by paired flips and right-angle rotations.
    """
    def __init__(self, samples: Sequence[Sample], root: str | Path = ".",
                 patch_size: int = 48, sampling_window: int = 64,
                 patches_per_epoch: int = 25_600, seed: int = 2026):
        self.samples = list(samples)
        self.root = Path(root)
        self.patch_size = patch_size
        self.sampling_window = sampling_window
        self.patches_per_epoch = patches_per_epoch
        self.seed = seed
        self.cache = []
        for s in self.samples:
            image = _read_rgb(self.root / s.image)
            mask = _read_mask(self.root / s.mask)
            fov = _read_mask(self.root / s.fov) if s.fov else infer_fov(image)
            image = preprocess_fundus(image, fov)
            coords = np.argwhere(fov > 0)
            self.cache.append((image, mask, fov, coords))

    def __len__(self):
        return self.patches_per_epoch

    def __getitem__(self, idx: int):
        # Worker-specific Python RNG is seeded by worker_init_fn in train.py.
        im, mask, fov, coords = random.choice(self.cache)
        ps, ws = self.patch_size, self.sampling_window
        _, h, w = im.shape
        for _ in range(64):
            cy, cx = coords[random.randrange(len(coords))]
            y0 = int(np.clip(cy - ws // 2, 0, max(h - ws, 0)))
            x0 = int(np.clip(cx - ws // 2, 0, max(w - ws, 0)))
            if h < ws or w < ws:
                continue
            fw = fov[y0:y0+ws, x0:x0+ws]
            if fw.mean() >= 0.75:
                break
        else:
            y0 = random.randint(0, max(h - ws, 0))
            x0 = random.randint(0, max(w - ws, 0))

        maxoff = max(ws - ps, 0)
        dy = random.randint(0, maxoff) if maxoff else 0
        dx = random.randint(0, maxoff) if maxoff else 0
        y, x = y0 + dy, x0 + dx
        p_im = im[:, y:y+ps, x:x+ps].copy()
        p_mask = mask[y:y+ps, x:x+ps].copy()[None]
        p_fov = fov[y:y+ps, x:x+ps].copy()[None]

        if random.random() < 0.5:
            p_im = p_im[:, :, ::-1].copy(); p_mask = p_mask[:, :, ::-1].copy(); p_fov = p_fov[:, :, ::-1].copy()
        if random.random() < 0.5:
            p_im = p_im[:, ::-1, :].copy(); p_mask = p_mask[:, ::-1, :].copy(); p_fov = p_fov[:, ::-1, :].copy()
        k = random.randrange(4)
        if k:
            p_im = np.rot90(p_im, k, axes=(1, 2)).copy()
            p_mask = np.rot90(p_mask, k, axes=(1, 2)).copy()
            p_fov = np.rot90(p_fov, k, axes=(1, 2)).copy()

        return {
            "image": torch.from_numpy(p_im.astype(np.float32)),
            "mask": torch.from_numpy(p_mask.astype(np.float32)),
            "fov": torch.from_numpy(p_fov.astype(np.float32)),
        }
