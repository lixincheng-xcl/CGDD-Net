"""CPU smoke and gradient checks; these do not establish dataset performance."""

import pytest
import torch
from torch.nn import functional as F

from cgddnet.model import (
    ABLATIONS, CGDDNet, CrissCrossAttention, DeformableAttentionHead, SAMG, count_parameters,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (17, 19), (32, 32), (47, 51)])
def test_inference_preserves_small_and_odd_sizes(shape):
    model = CGDDNet(base_channels=4, dropout=0).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, *shape))
    assert output.shape == (1, 1, *shape)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_all_cumulative_variants_have_finite_forward_and_backward(ablation):
    torch.manual_seed(27)
    model = CGDDNet(base_channels=4, ablation=ablation, dropout=0)
    inputs = torch.randn(2, 1, 17, 19)
    target = torch.randint(0, 2, (2, 1, 17, 19)).float()
    output = model(inputs)
    F.binary_cross_entropy_with_logits(output, target).backward()
    assert output.shape == target.shape
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Inactive registered parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"


def test_offsets_and_both_samg_gates_receive_gradients():
    torch.manual_seed(2026)
    model = CGDDNet(base_channels=4, dropout=0)
    output = model(torch.randn(2, 1, 31, 35))
    target = torch.randint(0, 2, output.shape).float()
    F.binary_cross_entropy_with_logits(output, target).backward()
    # At deep 1x1 maps border sampling is constant, so examine the E1 geometry.
    for head in model.encoders[0].attention_heads:
        grad = head.offset.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
    for module in model.samg.values():
        for gate in (module.global_gate, module.local_gate):
            grads = [parameter.grad for parameter in gate.parameters()]
            assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
            assert sum(grad.abs().sum() for grad in grads) > 0
    for index in (0, 3):
        grad = model.encoders[0].pointwise[index].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_samg_weights_are_pixelwise_probabilities():
    module = SAMG(8).eval()
    weights = module.scale_weights(torch.randn(2, 8, 7, 9))
    assert weights.shape == (2, 4, 7, 9)
    assert (weights >= 0).all() and (weights <= 1).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, 7, 9))


def test_query_is_sampled_with_the_center_offset():
    # The center query starts at 10, strongly selecting the rightmost value.
    # Moving its center sampling point left makes the query zero. Attention is
    # then uniform and exactly three of the nine sampled values are one.
    head = DeformableAttentionHead(channels=1, base_scale=1)
    query = torch.tensor([[[[0.0, 10.0, 0.0]]]])
    key = torch.tensor([[[[0.0, 1.0, 2.0]]]])
    value = torch.tensor([[[[0.0, 0.0, 1.0]]]])
    with torch.no_grad():
        original = head(query, key, value)
        head.offset.bias[8] = -1.0  # x offset of the center point j=4
        shifted = head(query, key, value)
    assert original[0, 0, 0, 1] > 0.99
    torch.testing.assert_close(shifted[0, 0, 0, 1], torch.tensor(1.0 / 3.0))


def test_default_width_forward_and_backward():
    model = CGDDNet(dropout=0)
    assert model.channels == (8, 16, 32, 64, 128)
    output = model(torch.randn(2, 1, 17, 19))
    F.binary_cross_entropy_with_logits(output, torch.ones_like(output)).backward()
    assert output.shape == (2, 1, 17, 19)
    assert torch.isfinite(output).all()
    assert torch.isfinite(model.head.weight.grad).all()


def test_one_cca_occurs_after_d3_fusion_and_before_decoder():
    model = CGDDNet(base_channels=4).eval()
    events = []
    handles = [module.register_forward_hook(
        lambda module, args, output, name=name: events.append(name)
    ) for name, module in (
        ("fusion3", model.fusions["3"]), ("cca", model.cca),
        ("decoder3", model.decoders["3"])
    )]
    with torch.no_grad():
        model(torch.randn(1, 1, 17, 19))
    for handle in handles:
        handle.remove()
    assert events == ["fusion3", "cca", "decoder3"]
    assert sum(isinstance(module, CrissCrossAttention) for module in model.modules()) == 1
    assert model.skip_levels == (4, 1)
    assert model.detail_levels == (3, 2, 1)


def test_singleton_training_and_rgb_input():
    model = CGDDNet(in_channels=3, base_channels=4, dropout=0)
    output = model(torch.randn(1, 3, 1, 1))
    F.binary_cross_entropy_with_logits(output, torch.ones_like(output)).backward()
    assert output.shape == (1, 1, 1, 1)
    assert torch.isfinite(output).all()


def test_parameter_count_is_computed_from_configuration():
    narrow = CGDDNet(base_channels=4)
    wider = CGDDNet(base_channels=8, detail_channels=12)
    assert count_parameters(narrow) == sum(parameter.numel() for parameter in narrow.parameters())
    assert count_parameters(wider) > count_parameters(narrow)


@pytest.mark.parametrize("kwargs", [
    {"base_channels": 5}, {"dropout": 1.0}, {"large_kernel": 4},
    {"ffn_expansion": 0}, {"attention_scales": ()}, {"ablation": "unknown"}
])
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CGDDNet(**kwargs)
