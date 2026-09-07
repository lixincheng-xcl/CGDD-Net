import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cgddnet import CGDDNet


def main():
    model = CGDDNet().eval()
    x = torch.randn(1, 3, 48, 48)
    with torch.no_grad():
        y = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("params:", f"{params:,}")


if __name__ == "__main__":
    main()
