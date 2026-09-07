import torch
from cgddnet.models import ABLATIONS, CGDDNet


def test_all_ablation_shapes():
    x = torch.randn(1, 3, 48, 48)
    for name in ABLATIONS:
        m = CGDDNet(ablation=name).eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 1, 48, 48)
        assert torch.isfinite(y).all()


def test_full_parameter_count_is_paper_scale():
    m = CGDDNet(ablation="full")
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert 2_950_000 <= n <= 3_000_000
