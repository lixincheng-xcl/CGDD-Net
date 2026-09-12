"""CGDD-Net implementation of the current manuscript Figures 2--5.

This is a new, configurable implementation, not a reconstruction of an archived
training checkpoint. See ``docs/architecture.md`` for the explicit choices made
where the figures leave numerical settings unspecified.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ABLATIONS = (
    "baseline", "csde", "samg", "dcdf", "detail_decoder", "selective_skip", "full"
)


class BatchNorm2d(nn.BatchNorm2d):
    """BatchNorm with a running-statistics fallback for a singleton feature map.

    Ordinary training batches use exactly nn.BatchNorm2d. A single value per
    channel (B=H=W=1) cannot estimate batch variance; using the stored statistics
    makes small-image smoke checks well-defined without altering normal batches.
    """

    def forward(self, x: Tensor) -> Tensor:
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, momentum=0.0, eps=self.eps,
            )
        return super().forward(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


class ChannelLayerNorm(nn.Module):
    """Layer normalization along channels independently at each pixel."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class DeformableAttentionHead(nn.Module):
    """Nine-point local attention with an offset for each key/value location.

    The query is sampled using the center-point offset; keys and values use all
    nine offsets. Offsets are in feature-pixel units. A dense 3x3 predictor
    produces nine (dx, dy) pairs. There is no additional
    modulation gate: the manuscript's local-attention equation uses softmax.
    """

    def __init__(self, channels: int, base_scale: float):
        super().__init__()
        self.base_scale = float(base_scale)
        self.offset = nn.Conv2d(channels, 18, 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        lattice = [(dx, dy) for dy in (-1.0, 0.0, 1.0)
                   for dx in (-1.0, 0.0, 1.0)]
        self.register_buffer("lattice", torch.tensor(lattice), persistent=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch, channels, height, width = query.shape
        offsets = self.offset(query).reshape(batch, 9, 2, height, width)
        offsets = offsets.permute(0, 1, 3, 4, 2)
        ys = torch.arange(height, device=query.device, dtype=query.dtype)
        xs = torch.arange(width, device=query.device, dtype=query.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack((xx, yy), dim=-1)[None, None]
        lattice = self.lattice.to(dtype=query.dtype)[None, :, None, None, :]
        positions = centers + lattice * self.base_scale + offsets
        # Pixel centers with align_corners=False also support H=1 or W=1.
        normalizer = query.new_tensor((width, height))
        grid = (positions + 0.5) * (2.0 / normalizer) - 1.0
        # Figure 3 sends Q through the sampling stage too. The center lattice
        # point supplies one sampled query per output pixel, while all nine
        # points supply keys/values. Each head predicts its own center offset.
        sampled_query = F.grid_sample(
            query, grid[:, 4], mode="bilinear", padding_mode="border", align_corners=False
        )
        grid = grid.reshape(batch, 9 * height, width, 2)
        sampled_key = F.grid_sample(
            key, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).reshape(batch, channels, 9, height, width)
        sampled_value = F.grid_sample(
            value, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).reshape(batch, channels, 9, height, width)
        logits = (sampled_query[:, :, None] * sampled_key).sum(dim=1) / math.sqrt(channels)
        weights = logits.softmax(dim=1)
        return (sampled_value * weights[:, None]).sum(dim=2)


class CSDE(nn.Module):
    """Two-branch CSDE including the GELU/BN pointwise block in Figure 3."""

    def __init__(self, channels: int, attention_scales: Sequence[float],
                 large_kernel: int = 7, ffn_expansion: float = 2.0):
        super().__init__()
        heads = len(attention_scales)
        if not heads or channels % heads:
            raise ValueError("Each CSDE width must be divisible by the number of attention scales")
        self.local = nn.Sequential(
            ConvBNReLU(channels, channels, 3), ConvBNReLU(channels, channels, 3)
        )
        self.norm = ChannelLayerNorm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.qkv_spatial = nn.Conv2d(
            3 * channels, 3 * channels, 3, padding=1, groups=3 * channels, bias=False
        )
        self.attention_heads = nn.ModuleList(
            DeformableAttentionHead(channels // heads, scale) for scale in attention_scales
        )
        self.attention_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.large_kernel = nn.Sequential(
            nn.Conv2d(channels, channels, large_kernel, padding=large_kernel // 2,
                      groups=channels, bias=False),
            nn.GELU(), BatchNorm2d(channels),
        )
        hidden = max(1, int(round(channels * ffn_expansion)))
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(), BatchNorm2d(hidden),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.GELU(), BatchNorm2d(channels),
        )
        self.fusion = ConvBNReLU(2 * channels, channels, 3)

    def forward(self, x: Tensor) -> Tensor:
        local = self.local(x)
        query, key, value = self.qkv_spatial(self.qkv(self.norm(x))).chunk(3, dim=1)
        heads = len(self.attention_heads)
        outputs = [head(q, k, v) for head, q, k, v in zip(
            self.attention_heads, query.chunk(heads, dim=1),
            key.chunk(heads, dim=1), value.chunk(heads, dim=1)
        )]
        attention = x + self.attention_projection(torch.cat(outputs, dim=1))
        context = attention + self.large_kernel(attention)
        adaptive = context + self.pointwise(context)
        return x + self.fusion(torch.cat((local, adaptive), dim=1))


class SAMG(nn.Module):
    """Figure 4: dense multi-kernel CBR branches and global/local scale logits."""

    def __init__(self, channels: int, global_hidden: int = 8):
        super().__init__()
        self.kernels = (1, 3, 5, 7)
        self.branches = nn.ModuleList(
            ConvBNReLU(channels, channels, kernel) for kernel in self.kernels
        )
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, global_hidden, 1),
            nn.ReLU(inplace=False), nn.Conv2d(global_hidden, 4, 1),
        )
        # Figure 4 labels the local intermediate feature with C_s channels.
        self.local_gate = nn.Sequential(
            ConvBNReLU(channels, channels, 3), nn.Conv2d(channels, 4, 1)
        )

    def scale_weights(self, x: Tensor) -> Tensor:
        return (self.global_gate(x) + self.local_gate(x)).softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        responses = torch.stack([branch(x) for branch in self.branches], dim=1)
        weights = self.scale_weights(x)
        return (responses * weights[:, :, None]).sum(dim=1)


class DCDF(nn.Module):
    """Align gated E2/E3/E4 features, then project and fuse at the E3 size."""

    def __init__(self, channels: Sequence[int], detail_channels: int):
        super().__init__()
        self.projections = nn.ModuleList(
            ConvBNReLU(channels_i, detail_channels, 1) for channels_i in channels
        )
        self.fusion = ConvBNReLU(3 * detail_channels, detail_channels, 1)

    def forward(self, f2: Tensor, f3: Tensor, f4: Tensor) -> Tensor:
        lower = F.max_pool2d(f2, 2, 2, ceil_mode=True)
        if lower.shape[-2:] != f3.shape[-2:]:
            raise ValueError("DCDF expects adjacent encoder scales E2, E3, E4")
        upper = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        aligned = [proj(feature) for proj, feature in
                   zip(self.projections, (lower, f3, upper))]
        return self.fusion(torch.cat(aligned, dim=1))


class CrissCrossAttention(nn.Module):
    """One separately normalized row/column attention pass, with residual scale."""

    def __init__(self, channels: int):
        super().__init__()
        self.query_channels = max(channels // 8, 1)
        self.query = nn.Conv2d(channels, self.query_channels, 1)
        self.key = nn.Conv2d(channels, self.query_channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        query, key, value = self.query(x), self.key(x), self.value(x)
        qr = query.permute(0, 2, 3, 1).reshape(batch * height, width, -1)
        kr = key.permute(0, 2, 1, 3).reshape(batch * height, -1, width)
        vr = value.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        row_weights = (torch.bmm(qr, kr) / math.sqrt(self.query_channels)).softmax(-1)
        row = torch.bmm(row_weights, vr).reshape(batch, height, width, channels)
        row = row.permute(0, 3, 1, 2)

        qc = query.permute(0, 3, 2, 1).reshape(batch * width, height, -1)
        kc = key.permute(0, 3, 1, 2).reshape(batch * width, -1, height)
        vc = value.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        column_weights = (torch.bmm(qc, kc) / math.sqrt(self.query_channels)).softmax(-1)
        column = torch.bmm(column_weights, vc).reshape(batch, width, height, channels)
        column = column.permute(0, 3, 2, 1)
        return x + self.gamma * (row + column)


class DecoderBlock(nn.Sequential):
    def __init__(self, channels: int, dropout: float):
        super().__init__(
            ConvBNReLU(channels, channels, 3), ConvBNReLU(channels, channels, 3),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
        )


class CGDDNet(nn.Module):
    """Five-stage CGDD-Net returning unnormalized B x 1 x H x W logits.

    Args:
        in_channels: Input channels; 1 for grayscale, 3 for RGB.
        base_channels: Encoder widths are [C, 2C, 4C, 8C, 16C].
        detail_channels: Shared DCDF width; defaults to base_channels.
        dropout: Spatial dropout after each decoder block.
        attention_scales: Fixed reference-lattice spacings, one per attention head.
        samg_hidden: Width of the global SAMG gating pathway. The local pathway
            retains the corresponding encoder width, as drawn in Figure 4.
        ffn_expansion: Hidden/output ratio of the two-PWConv block in Figure 3.
        large_kernel: Odd spatial kernel for the depthwise context operation.
        ablation: One of the seven cumulative configurations in ABLATIONS.

    Odd and small spatial sizes are preserved. Pooling uses ceil_mode=True and
    each upsampling step targets the actual encoder size. For dimensions divisible
    by 16 this is the exact H, H/2, ..., H/16 schedule in the architecture figure.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 8,
                 detail_channels: int | None = None, dropout: float = 0.1,
                 attention_scales: Sequence[float] = (1, 3, 5, 7),
                 samg_hidden: int = 8, ffn_expansion: float = 2.0,
                 large_kernel: int = 7, ablation: str = "full"):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation {ablation!r}; choose from {ABLATIONS}")
        if in_channels < 1 or base_channels < 1:
            raise ValueError("in_channels and base_channels must be positive")
        if detail_channels is None:
            detail_channels = base_channels
        if detail_channels < 1 or samg_hidden < 1:
            raise ValueError("detail_channels and samg_hidden must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if ffn_expansion <= 0 or not math.isfinite(ffn_expansion):
            raise ValueError("ffn_expansion must be finite and positive")
        if large_kernel < 1 or large_kernel % 2 != 1:
            raise ValueError("large_kernel must be a positive odd integer")
        attention_scales = tuple(float(s) for s in attention_scales)
        if not attention_scales or any(s <= 0 or not math.isfinite(s) for s in attention_scales):
            raise ValueError("attention_scales must contain finite positive spacings")
        rank = ABLATIONS.index(ablation)
        if rank and base_channels % len(attention_scales):
            raise ValueError("base_channels must be divisible by the attention head count")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.detail_channels = detail_channels
        self.ablation = ablation
        self.channels = tuple(base_channels * (2 ** i) for i in range(5))
        self.skip_levels = (4, 1) if rank >= 5 else (4, 3, 2, 1)
        self.detail_levels = (3, 2, 1) if rank >= 4 else ((3,) if rank >= 2 else ())
        self.stem = ConvBNReLU(in_channels, base_channels, 3)
        self.encoders = nn.ModuleList(
            CSDE(channels, attention_scales, large_kernel, ffn_expansion) if rank else
            nn.Sequential(ConvBNReLU(channels, channels, 3),
                          ConvBNReLU(channels, channels, 3))
            for channels in self.channels
        )
        self.down_projections = nn.ModuleList(
            ConvBNReLU(self.channels[i], self.channels[i + 1], 1) for i in range(4)
        )

        samg_levels = (2, 3, 4) if rank >= 3 else ((3,) if rank >= 2 else ())
        self.samg = nn.ModuleDict({
            str(level): SAMG(self.channels[level - 1], samg_hidden) for level in samg_levels
        })
        self.dcdf = DCDF(self.channels[1:4], detail_channels) if rank >= 3 else None
        self.single_detail = ConvBNReLU(self.channels[2], detail_channels, 1) if rank == 2 else None
        self.detail_projections = nn.ModuleDict({
            str(level): ConvBNReLU(detail_channels, self.channels[level - 1], 1)
            for level in self.detail_levels
        })
        self.up_projections = nn.ModuleDict({
            str(level): ConvBNReLU(self.channels[level], self.channels[level - 1], 1)
            for level in (4, 3, 2, 1)
        })
        self.fusions = nn.ModuleDict({
            str(level): ConvBNReLU(
                self.channels[level - 1] * (1 + (level in self.skip_levels) + (level in self.detail_levels)),
                self.channels[level - 1], 1,
            ) for level in (4, 3, 2, 1)
        })
        self.decoders = nn.ModuleDict({
            str(level): DecoderBlock(self.channels[level - 1], dropout) for level in (5, 4, 3, 2, 1)
        })
        self.cca = CrissCrossAttention(self.channels[2]) if rank == 6 else nn.Identity()
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected B x {self.in_channels} x H x W input")
        if min(x.shape[0], x.shape[2], x.shape[3]) < 1:
            raise ValueError("Batch and spatial dimensions must be nonempty")
        encoded = [self.encoders[0](self.stem(x))]
        for projection, encoder in zip(self.down_projections, self.encoders[1:]):
            encoded.append(encoder(projection(F.max_pool2d(encoded[-1], 2, 2, ceil_mode=True))))

        detail = None
        if self.dcdf is not None:
            detail = self.dcdf(*(self.samg[str(level)](encoded[level - 1]) for level in (2, 3, 4)))
        elif self.single_detail is not None:
            detail = self.single_detail(self.samg["3"](encoded[2]))

        decoded = self.decoders["5"](encoded[4])
        for level in (4, 3, 2, 1):
            key = str(level)
            size = encoded[level - 1].shape[-2:]
            main = self.up_projections[key](F.interpolate(
                decoded, size=size, mode="bilinear", align_corners=False
            ))
            parts = [main]
            if level in self.skip_levels:
                parts.append(encoded[level - 1])
            if level in self.detail_levels:
                aligned = F.interpolate(detail, size=size, mode="bilinear", align_corners=False)
                parts.append(self.detail_projections[key](aligned))
            fused = self.fusions[key](torch.cat(parts, dim=1))
            if level == 3:
                fused = self.cca(fused)
            decoded = self.decoders[key](fused)
        return self.head(decoded)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the actual registered parameters of a chosen configuration."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad or not trainable_only)
