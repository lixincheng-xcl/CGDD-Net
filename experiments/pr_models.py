"""Controlled architecture variants for PR evidence, separate from paper ablations.

All configurations must be trained from scratch on the same image split and
training budget. Constructing these variants does not reproduce benchmark scores.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cgddnet.model import CGDDNet, ConvBNReLU, DeformableAttentionHead, SAMG

VARIANTS = (
    "full", "independent_detail", "fixed_offsets", "uniform_samg",
    "global_only_samg", "local_only_samg", "unet_same_width",
    "unet_budget_matched",
)


class ZeroOffsets(nn.Module):
    """Zero displacement at all nine reference samples, without a predictor."""

    def forward(self, x: Tensor) -> Tensor:
        return x.new_zeros(x.shape[0], 18, x.shape[-2], x.shape[-1])


class ControlledSAMG(nn.Module):
    """Keep original convolution branches while changing only gate structure."""

    def __init__(self, source: SAMG, mode: str):
        super().__init__()
        if mode not in {"uniform", "global_only", "local_only"}:
            raise ValueError(f"Unknown gate mode: {mode}")
        self.mode = mode
        self.branches = source.branches
        if mode == "global_only":
            self.global_gate = source.global_gate
        elif mode == "local_only":
            self.local_gate = source.local_gate

    def forward(self, x: Tensor) -> Tensor:
        responses = torch.stack([branch(x) for branch in self.branches], dim=1)
        if self.mode == "uniform":
            return responses.mean(dim=1)
        if self.mode == "global_only":
            weights = self.global_gate(x).softmax(dim=1)
        else:
            weights = self.local_gate(x).softmax(dim=1)
        return (responses * weights[:, :, None]).sum(dim=1)


class IndependentDetailNet(CGDDNet):
    """Three separate DCDF modules, fed by the same gated E2/E3/E4 features.

    Decoder projections, CCA, skips, encoder and SAMG are unchanged. Parameter
    count is allowed to increase and must be reported rather than hidden.
    """

    def __init__(self, **kwargs):
        kwargs["ablation"] = "full"
        super().__init__(**kwargs)
        assert self.dcdf is not None
        self.independent_dcdf = nn.ModuleDict({
            "3": self.dcdf,
            "2": deepcopy(self.dcdf),
            "1": deepcopy(self.dcdf),
        })
        self.dcdf = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected B x {self.in_channels} x H x W input")
        if min(x.shape[0], x.shape[2], x.shape[3]) < 1:
            raise ValueError("Batch and spatial dimensions must be nonempty")

        encoded = [self.encoders[0](self.stem(x))]
        for projection, encoder in zip(self.down_projections, self.encoders[1:]):
            encoded.append(encoder(projection(F.max_pool2d(encoded[-1], 2, 2, ceil_mode=True))))
        gated = tuple(self.samg[str(level)](encoded[level - 1]) for level in (2, 3, 4))
        detail = {level: module(*gated) for level, module in self.independent_dcdf.items()}

        decoded = self.decoders["5"](encoded[4])
        for level in (4, 3, 2, 1):
            key = str(level)
            size = encoded[level - 1].shape[-2:]
            main = self.up_projections[key](F.interpolate(
                decoded, size=size, mode="bilinear", align_corners=False))
            parts = [main]
            if level in self.skip_levels:
                parts.append(encoded[level - 1])
            if key in detail:
                aligned = F.interpolate(detail[key], size=size, mode="bilinear", align_corners=False)
                parts.append(self.detail_projections[key](aligned))
            fused = self.fusions[key](torch.cat(parts, dim=1))
            if level == 3:
                fused = self.cca(fused)
            decoded = self.decoders[key](fused)
        return self.head(decoded)


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(
            ConvBNReLU(input_channels, output_channels, 3),
            ConvBNReLU(output_channels, output_channels, 3),
        )


class SameProtocolUNet(nn.Module):
    """Five-scale U-Net; bilinear resizing, two CBRs/stage, sigmoid-free logits."""

    def __init__(self, base_channels=8, in_channels=1, dropout=0.1):
        super().__init__()
        if base_channels < 1 or in_channels < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid U-Net width, channels or dropout")
        widths = [base_channels * (2 ** i) for i in range(5)]
        self.encoders = nn.ModuleList([
            DoubleConv(in_channels if i == 0 else widths[i - 1], widths[i])
            for i in range(5)
        ])
        self.up_projections = nn.ModuleList([
            ConvBNReLU(widths[i + 1], widths[i], 1)
            for i in range(3, -1, -1)
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(DoubleConv(2 * widths[i], widths[i]),
                          nn.Dropout2d(dropout) if dropout else nn.Identity())
            for i in range(3, -1, -1)
        ])
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.base_channels = base_channels

    def forward(self, x: Tensor) -> Tensor:
        skip = []
        for i, encoder in enumerate(self.encoders):
            if i:
                x = F.max_pool2d(x, 2, 2, ceil_mode=True)
            x = encoder(x)
            skip.append(x)
        for index, (project, decode) in enumerate(zip(self.up_projections, self.decoders)):
            skip_feature = skip[3 - index]
            x = F.interpolate(x, size=skip_feature.shape[-2:], mode="bilinear", align_corners=False)
            x = decode(torch.cat((project(x), skip_feature), dim=1))
        return self.head(x)


def count_trainable(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def closest_unet_width(target_params: int, in_channels=1, dropout=0.1) -> int:
    """Find nearest parameter count, never imply exact equality."""
    if target_params < 1:
        raise ValueError("Expected positive target parameter count")
    best = None
    for width in range(4, 49):
        model = SameProtocolUNet(width, in_channels, dropout)
        score = abs(count_trainable(model) - target_params)
        if best is None or score < best[0]:
            best = (score, width)
        if count_trainable(model) >= target_params and width > best[1]:
            break
    return best[1]


def build_model(variant: str, **kwargs) -> nn.Module:
    """Create a variant. The public CGDD-Net defaults remain untouched."""
    if variant not in VARIANTS:
        raise ValueError(f"Choose a variant from {VARIANTS}, got {variant!r}")
    params = dict(kwargs)
    params.pop("ablation", None)
    if variant in {"unet_same_width", "unet_budget_matched"}:
        base = params.get("base_channels", 8)
        inch = params.get("in_channels", 1)
        dropout = params.get("dropout", 0.1)
        if variant == "unet_budget_matched":
            ref = CGDDNet(**params, ablation="full")
            base = closest_unet_width(count_trainable(ref), inch, dropout)
        return SameProtocolUNet(base, inch, dropout)
    if variant == "independent_detail":
        return IndependentDetailNet(**params)

    model = CGDDNet(**params, ablation="full")
    if variant == "fixed_offsets":
        for module in model.modules():
            if isinstance(module, DeformableAttentionHead):
                module.offset = ZeroOffsets()
    if variant in {"uniform_samg", "global_only_samg", "local_only_samg"}:
        mode = {"uniform_samg": "uniform", "global_only_samg": "global_only",
                "local_only_samg": "local_only"}[variant]
        for name, module in list(model.samg.items()):
            model.samg[name] = ControlledSAMG(module, mode)
    return model