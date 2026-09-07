from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sliding_window_predict(model, image: torch.Tensor, patch_size: int = 96, stride: int = 16,
                           device: torch.device | str = "cuda") -> torch.Tensor:
    """Overlapping patch inference with reflection padding and probability averaging.

    Args:
        image: CHW or 1CHW preprocessed image tensor.
    Returns:
        1xHxW probability tensor on CPU.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    _, _, h, w = image.shape
    pad_h = max(patch_size - h, 0)
    pad_w = max(patch_size - w, 0)
    if h + pad_h > patch_size:
        rem = (h + pad_h - patch_size) % stride
        if rem:
            pad_h += stride - rem
    if w + pad_w > patch_size:
        rem = (w + pad_w - patch_size) % stride
        if rem:
            pad_w += stride - rem
    x = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    hp, wp = x.shape[-2:]
    acc = torch.zeros((1, 1, hp, wp), dtype=torch.float32, device=device)
    cnt = torch.zeros_like(acc)
    model.eval()
    for y in range(0, hp - patch_size + 1, stride):
        for x0 in range(0, wp - patch_size + 1, stride):
            patch = x[:, :, y:y+patch_size, x0:x0+patch_size].to(device, non_blocking=True)
            prob = torch.sigmoid(model(patch))
            acc[:, :, y:y+patch_size, x0:x0+patch_size] += prob
            cnt[:, :, y:y+patch_size, x0:x0+patch_size] += 1
    out = acc / cnt.clamp_min(1)
    return out[0, :, :h, :w].cpu()
