from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                 padding: int | None = None, groups: int = 1, bias: bool = False):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DownsamplingProjection(nn.Module):
    """2x2 max-pooling followed by a 1x1 channel-expansion CBR."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.proj = ConvBNReLU(in_ch, out_ch, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(x))


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW tensors."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def _make_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # H,W,2


class ScaleAdaptiveDeformableAttentionHead(nn.Module):
    """Local deformable attention over a 3x3 lattice at one base sampling scale.

    This implementation uses ``grid_sample`` and therefore does not depend on
    custom CUDA extensions. Offsets are predicted in pixel units and added to a
    scale-specific 3x3 reference lattice. Modulation gates are learned jointly.
    """
    def __init__(self, channels: int, base_scale: int):
        super().__init__()
        self.channels = channels
        self.base_scale = float(base_scale)
        self.offset = nn.Conv2d(channels, 18, 3, padding=1, bias=True)
        self.modulation = nn.Conv2d(channels, 9, 3, padding=1, bias=True)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

        lattice = []
        for dy in (-1.0, 0.0, 1.0):
            for dx in (-1.0, 0.0, 1.0):
                lattice.append((dx, dy))
        self.register_buffer("lattice", torch.tensor(lattice, dtype=torch.float32), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, c, h, w = q.shape
        offsets = self.offset(q).view(b, 9, 2, h, w).permute(0, 1, 3, 4, 2)  # B,9,H,W,2
        gates = torch.sigmoid(self.modulation(q)).view(b, 9, h, w)

        base = _make_base_grid(h, w, q.device, q.dtype)[None, None].expand(b, 9, -1, -1, -1)
        lattice = self.lattice.to(device=q.device, dtype=q.dtype)[None, :, None, None, :]
        disp = lattice * self.base_scale + offsets
        if w > 1:
            disp_x = disp[..., 0] * (2.0 / (w - 1))
        else:
            disp_x = torch.zeros_like(disp[..., 0])
        if h > 1:
            disp_y = disp[..., 1] * (2.0 / (h - 1))
        else:
            disp_y = torch.zeros_like(disp[..., 1])
        grid = base + torch.stack([disp_x, disp_y], dim=-1)

        # Sample all nine locations in one grid_sample call per K/V tensor.
        grid_flat = grid.reshape(b, 9 * h, w, 2)
        sampled_k = F.grid_sample(k, grid_flat, mode="bilinear", padding_mode="border",
                                  align_corners=True).view(b, c, 9, h, w).permute(0, 2, 1, 3, 4)
        sampled_v = F.grid_sample(v, grid_flat, mode="bilinear", padding_mode="border",
                                  align_corners=True).view(b, c, 9, h, w).permute(0, 2, 1, 3, 4)

        logits = (sampled_k * q[:, None]).sum(dim=2) / math.sqrt(max(c, 1))
        attn = torch.softmax(logits, dim=1) * gates
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-6)
        return (sampled_v * attn[:, :, None]).sum(dim=1)


class LargeKernelEmbedding(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.dw = ConvBNReLU(channels, channels, kernel_size, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.dw(x), inplace=True)


class CSDEBlock(nn.Module):
    """Context-Guided Scale-Adaptive Deformable Encoding (CSDE)."""
    def __init__(self, channels: int, scales: Sequence[int] = (1, 3, 5, 7)):
        super().__init__()
        if channels % len(scales) != 0:
            raise ValueError(f"channels={channels} must be divisible by number of heads={len(scales)}")
        self.channels = channels
        self.num_heads = len(scales)
        self.head_dim = channels // self.num_heads

        self.local_branch = nn.Sequential(
            ConvBNReLU(channels, channels, 3),
            ConvBNReLU(channels, channels, 3),
        )

        self.norm = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, groups=self.num_heads, bias=False)
        self.refine = nn.Conv2d(channels * 3, channels * 3, 3, padding=1,
                                groups=channels * 3, bias=False)
        self.refine_bn = nn.BatchNorm2d(channels * 3)
        self.heads = nn.ModuleList([
            ScaleAdaptiveDeformableAttentionHead(self.head_dim, s) for s in scales
        ])
        self.out_proj = nn.Conv2d(channels, channels, 1, groups=self.num_heads, bias=False)
        self.lk = LargeKernelEmbedding(channels, 7)
        self.compress = ConvBNReLU(channels * 2, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fixed = self.local_branch(x)
        qkv = self.refine_bn(self.refine(self.qkv(self.norm(x))))
        q, k, v = qkv.chunk(3, dim=1)
        qh, kh, vh = q.chunk(self.num_heads, dim=1), k.chunk(self.num_heads, dim=1), v.chunk(self.num_heads, dim=1)
        ah = [head(qi, ki, vi) for head, qi, ki, vi in zip(self.heads, qh, kh, vh)]
        adaptive = self.out_proj(torch.cat(ah, dim=1))
        adaptive = self.lk(x + adaptive)
        fused = self.compress(torch.cat([fixed, adaptive], dim=1))
        return F.relu(x + fused, inplace=True)


class CBRBlock(nn.Module):
    """Two 3x3 CBR units used by the baseline encoder and decoder."""
    def __init__(self, in_ch: int, out_ch: int | None = None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3),
            ConvBNReLU(out_ch, out_ch, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SAMG(nn.Module):
    """Spatially Adaptive Multi-Kernel Gating.

    The scale-specific responses use depthwise-separable CBR branches to keep
    the module compact while retaining the 1/3/5/7 receptive-field family.
    """
    def __init__(self, channels: int, kernels: Iterable[int] = (1, 3, 5, 7)):
        super().__init__()
        self.kernels = tuple(kernels)
        branches = []
        for k in self.kernels:
            if k == 1:
                branches.append(ConvBNReLU(channels, channels, 1, padding=0))
            else:
                branches.append(nn.Sequential(
                    ConvBNReLU(channels, channels, k, groups=channels),
                    ConvBNReLU(channels, channels, 1, padding=0, groups=2),
                ))
        self.branches = nn.ModuleList(branches)
        hidden = 8
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, len(self.kernels), 1),
        )
        self.local_gate = nn.Sequential(
            ConvBNReLU(channels, hidden, 3),
            nn.Conv2d(hidden, len(self.kernels), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        responses = torch.stack([branch(x) for branch in self.branches], dim=1)  # B,K,C,H,W
        logits = self.global_gate(x) + self.local_gate(x)
        weights = torch.softmax(logits, dim=1)
        return (responses * weights[:, :, None]).sum(dim=1)


class DCDF(nn.Module):
    """Dynamic Cross-Scale Detail Fusion at H/4 x W/4."""
    def __init__(self, c2: int, c3: int, c4: int, detail_ch: int):
        super().__init__()
        self.p2 = ConvBNReLU(c2, detail_ch, 1, padding=0)
        self.p3 = ConvBNReLU(c3, detail_ch, 1, padding=0)
        self.p4 = ConvBNReLU(c4, detail_ch, 1, padding=0)
        self.fuse = ConvBNReLU(detail_ch * 3, detail_ch, 1, padding=0)

    def forward(self, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        f2 = F.max_pool2d(f2, 2, 2)
        f4 = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.p2(f2), self.p3(f3), self.p4(f4)], dim=1))


class FusionUnit(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fuse = ConvBNReLU(in_ch, out_ch, 1, padding=0)

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.fuse(torch.cat(xs, dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = CBRBlock(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


class CrissCrossAttention(nn.Module):
    """Single recurrent-free criss-cross attention pass.

    Computes row and column attention independently and sums both contexts.
    This keeps memory use bounded for the H/4 decoder stage.
    """
    def __init__(self, channels: int):
        super().__init__()
        qk = max(channels // 8, 1)
        self.query = nn.Conv2d(channels, qk, 1)
        self.key = nn.Conv2d(channels, qk, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Row attention: B*H, W, Cq
        qr = q.permute(0, 2, 3, 1).reshape(b * h, w, -1)
        kr = k.permute(0, 2, 1, 3).reshape(b * h, -1, w)
        ar = torch.softmax(torch.bmm(qr, kr) / math.sqrt(q.shape[1]), dim=-1)
        vr = v.permute(0, 2, 3, 1).reshape(b * h, w, c)
        orow = torch.bmm(ar, vr).reshape(b, h, w, c).permute(0, 3, 1, 2)

        # Column attention: B*W, H, Cq
        qc = q.permute(0, 3, 2, 1).reshape(b * w, h, -1)
        kc = k.permute(0, 3, 1, 2).reshape(b * w, -1, h)
        ac = torch.softmax(torch.bmm(qc, kc) / math.sqrt(q.shape[1]), dim=-1)
        vc = v.permute(0, 3, 2, 1).reshape(b * w, h, c)
        ocol = torch.bmm(ac, vc).reshape(b, w, h, c).permute(0, 3, 2, 1)
        return x + self.gamma * (orow + ocol)
