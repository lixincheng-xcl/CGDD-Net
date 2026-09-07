from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    CBRBlock, CSDEBlock, CrissCrossAttention, DCDF, DecoderBlock,
    DownsamplingProjection, FusionUnit, SAMG, ConvBNReLU,
)

AblationName = Literal[
    "baseline", "csde", "samg", "dcdf", "detail_decoder", "selective_skip", "full"
]


@dataclass(frozen=True)
class AblationFlags:
    use_csde: bool
    use_samg: bool
    use_dcdf: bool
    detail_levels: int
    selective_skip: bool
    use_cca: bool


ABLATIONS: dict[str, AblationFlags] = {
    "baseline": AblationFlags(False, False, False, 0, False, False),
    "csde": AblationFlags(True, False, False, 0, False, False),
    "samg": AblationFlags(True, True, False, 1, False, False),
    "dcdf": AblationFlags(True, True, True, 1, False, False),
    "detail_decoder": AblationFlags(True, True, True, 3, False, False),
    "selective_skip": AblationFlags(True, True, True, 3, True, False),
    "full": AblationFlags(True, True, True, 3, True, True),
}


class CGDDNet(nn.Module):
    """CGDD-Net without MSPF, matching the final paper architecture.

    Args:
        in_channels: Model input channels. Paper preprocessing replicates grayscale to 3 channels.
        base_channels: C in the paper. Default 12.
        detail_channels: Canonical DCDF width. Default 12.
        ablation: One of the seven cumulative configurations in Table VIII.
        dropout: Spatial dropout applied after each decoder block.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 12,
                 detail_channels: int = 12, ablation: AblationName = "full",
                 dropout: float = 0.1):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation={ablation}. Choices: {list(ABLATIONS)}")
        self.ablation = ablation
        self.flags = ABLATIONS[ablation]
        c = base_channels
        self.channels = [c, 2*c, 4*c, 8*c, 16*c]
        c1, c2, c3, c4, c5 = self.channels

        self.stem = ConvBNReLU(in_channels, c1, 3)
        enc = CSDEBlock if self.flags.use_csde else CBRBlock
        self.enc1 = enc(c1)
        self.down1 = DownsamplingProjection(c1, c2)
        self.enc2 = enc(c2)
        self.down2 = DownsamplingProjection(c2, c3)
        self.enc3 = enc(c3)
        self.down3 = DownsamplingProjection(c3, c4)
        self.enc4 = enc(c4)
        self.down4 = DownsamplingProjection(c4, c5)
        self.enc5 = enc(c5)

        if self.flags.use_samg:
            self.samg2 = SAMG(c2)
            self.samg3 = SAMG(c3)
            self.samg4 = SAMG(c4)
        else:
            self.samg2 = self.samg3 = self.samg4 = None

        self.dcdf = DCDF(c2, c3, c4, detail_channels) if self.flags.use_dcdf else None
        self.samg3_to_detail = ConvBNReLU(c3, detail_channels, 1, padding=0)
        self.detail_to_d3 = ConvBNReLU(detail_channels, c3, 1, padding=0)
        self.detail_to_d2 = ConvBNReLU(detail_channels, c2, 1, padding=0)
        self.detail_to_d1 = ConvBNReLU(detail_channels, c1, 1, padding=0)

        self.d5 = DecoderBlock(c5, dropout)
        self.up54 = ConvBNReLU(c5, c4, 1, padding=0)
        self.up43 = ConvBNReLU(c4, c3, 1, padding=0)
        self.up32 = ConvBNReLU(c3, c2, 1, padding=0)
        self.up21 = ConvBNReLU(c2, c1, 1, padding=0)

        in4 = c4 + c4
        in3 = c3 + (0 if self.flags.selective_skip else c3) + (c3 if self.flags.detail_levels >= 1 else 0)
        in2 = c2 + (0 if self.flags.selective_skip else c2) + (c2 if self.flags.detail_levels >= 3 else 0)
        in1 = c1 + c1 + (c1 if self.flags.detail_levels >= 3 else 0)
        self.fuse4 = FusionUnit(in4, c4)
        self.fuse3 = FusionUnit(in3, c3)
        self.fuse2 = FusionUnit(in2, c2)
        self.fuse1 = FusionUnit(in1, c1)
        self.d4 = DecoderBlock(c4, dropout)
        self.d3 = DecoderBlock(c3, dropout)
        self.d2 = DecoderBlock(c2, dropout)
        self.d1 = DecoderBlock(c1, dropout)
        self.cca = CrissCrossAttention(c3) if self.flags.use_cca else nn.Identity()
        self.head = nn.Conv2d(c1, 1, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            if getattr(m, "_is_zero_init", False):
                return
            if not torch.all(m.weight == 0):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None and not torch.all(m.bias == 0):
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _up(x: torch.Tensor, proj: nn.Module, size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return proj(x)

    def encode(self, x: torch.Tensor):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        e5 = self.enc5(self.down4(e4))
        return e1, e2, e3, e4, e5

    def _detail(self, e2, e3, e4):
        if not self.flags.use_samg:
            return None
        f2 = self.samg2(e2)
        f3 = self.samg3(e3)
        f4 = self.samg4(e4)
        if self.flags.use_dcdf:
            return self.dcdf(f2, f3, f4)
        return self.samg3_to_detail(f3)

    def forward(self, x: torch.Tensor, return_logits: bool = True):
        e1, e2, e3, e4, e5 = self.encode(x)
        detail = self._detail(e2, e3, e4)

        d5 = self.d5(e5)
        u4 = self._up(d5, self.up54, e4.shape[-2:])
        d4 = self.d4(self.fuse4([u4, e4]))

        u3 = self._up(d4, self.up43, e3.shape[-2:])
        x3 = [u3]
        if not self.flags.selective_skip:
            x3.append(e3)
        if self.flags.detail_levels >= 1:
            x3.append(self.detail_to_d3(detail))
        d3 = self.d3(self.cca(self.fuse3(x3)))

        u2 = self._up(d3, self.up32, e2.shape[-2:])
        x2 = [u2]
        if not self.flags.selective_skip:
            x2.append(e2)
        if self.flags.detail_levels >= 3:
            det2 = F.interpolate(detail, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            x2.append(self.detail_to_d2(det2))
        d2 = self.d2(self.fuse2(x2))

        u1 = self._up(d2, self.up21, e1.shape[-2:])
        x1 = [u1, e1]
        if self.flags.detail_levels >= 3:
            det1 = F.interpolate(detail, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            x1.append(self.detail_to_d1(det1))
        d1 = self.d1(self.fuse1(x1))

        logits = self.head(d1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


def build_model(ablation: AblationName = "full", **kwargs) -> CGDDNet:
    return CGDDNet(ablation=ablation, **kwargs)
