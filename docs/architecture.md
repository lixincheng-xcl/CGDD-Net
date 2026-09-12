# CGDD-Net architecture and implementation choices

This code implements the current manuscript Figures 2--5 and the equations in
`Sections/methods.tex`. It was written as a new package inside the existing
VesselSeg-Pytorch checkout. It does not load, modify, or copy the baseline's model
implementations. Existing baseline files remain separate.

The implementation has not been trained to reproduce the manuscript result
tables. Parameter counts and operation counts must be measured from this exact
configuration; historical values such as 1.82 M parameters or 20.05 G operations
are not targets built into the code.

## Public API and chosen defaults

```python
from cgddnet.model import CGDDNet, count_parameters

model = CGDDNet(
    in_channels=1,
    base_channels=8,
    detail_channels=8,  # None follows base_channels
    dropout=0.1,
    attention_scales=(1, 3, 5, 7),
    samg_hidden=8,
    ffn_expansion=2.0,
    large_kernel=7,
    ablation="full",
)
logits = model(images)  # B x 1 x H x W, without sigmoid
parameters = count_parameters(model)
```

The five encoder widths are 8, 16, 32, 64, and 128. The common detail width is
8. This keeps all four attention heads active with integer widths and provides
a compact implementation of the complete diagram, including dense SAMG filters.
The count is consequently different from the earlier reference model.
Alternative widths are configurable; each CSDE width must be divisible by the
number of attention heads. No width was optimized against an expected parameter
or FLOP total.

## Mapping to the figures

| Figure/component | Computation |
| --- | --- |
| Fig. 2 stem | 3x3 Conv, BatchNorm, ReLU; input defaults to one grayscale channel |
| Five encoder stages | CSDE at widths C, 2C, 4C, 8C, 16C |
| Downsampling projection | 2x2 max-pool, stride 2, then 1x1 Conv-BN-ReLU |
| Fig. 3 convolution branch | Two 3x3 Conv-BN-ReLU blocks |
| Fig. 3 Q/K/V | Channel LayerNorm, dense 1x1 projection to 3C, depthwise 3x3 refinement, channel split into Q/K/V and heads |
| Deformable local attention | Four heads; nine-point reference lattice per head; spacings 1, 3, 5, 7; predicted 2D offsets; local softmax attention |
| First attention residual | Concatenate head outputs, dense 1x1 projection, add input |
| Large-kernel residual | 7x7 depthwise Conv, GELU, BatchNorm, then residual addition |
| Two-PWConv residual | C to 2C pointwise Conv, GELU, BN; 2C to C pointwise Conv, GELU, BN; residual addition |
| CSDE branch fusion | Concatenate convolution and attention branches, 3x3 CBR, add input |
| Fig. 4 SAMG | Four **dense** 1x1/3x3/5x5/7x7 CBR branches; no hidden depthwise replacement |
| SAMG global gate | GAP, C_s to 8 pointwise Conv, ReLU, 8 to 4 pointwise Conv |
| SAMG local gate | C_s to C_s 3x3 CBR, C_s to 4 pointwise Conv |
| SAMG aggregation | Add global/local logits, softmax over four kernels, channel-shared pixelwise weighted sum |
| Fig. 5 DCDF | Pool E2 detail, keep E3, resize E4; separately project to C_d, concatenate, 1x1 CBR fusion |
| Decoder | Bilinear resize plus channel projection, fusion 1x1 CBR, two 3x3 CBR blocks, spatial dropout |
| Detail guidance | Reuse DCDF output at D3/D2/D1 after resize and separate projections |
| Selective skips | Fixed direct E4 and E1 connections; E2/E3 reach the decoder through the detail branch |
| CCA | Exactly one row/column attention module after D3 fusion and before Decoder 3 |
| Prediction | Direct 1x1 Conv from D1 to a single logit channel |

`GELU + BN` in Figure 3 is interpreted in the displayed order: convolution,
GELU, then batch normalization. Both pointwise convolutions are included. The
CSDE output follows the manuscript residual equation without adding another
activation after the final addition.

## Numerical assumptions where the diagram is unspecified

- The nine reference offsets are the 3x3 lattice `{-1, 0, 1}^2`. Four base
  spacings `(1, 3, 5, 7)` were retained as a documented choice. Each head learns
  18 offset channels through a 3x3 convolution, initially zero.
- The query is bilinearly sampled with the offset of the center lattice point;
  keys and values are sampled with all nine offsets. Thus Q, K, and V all pass
  through deformable sampling as drawn in Figure 3, while each output location
  still has one query. The center point is index 4 in row-major lattice order,
  so its unshifted displacement is zero. The sharing of this center offset with
  the corresponding key/value sample is a documented choice where the figure
  does not specify a separate query offset predictor.
- Attention uses `softmax(q^T k / sqrt(head_width))` over the nine points. The
  old code's extra sigmoid sampling-modulation gate is not included because it
  is absent from the current Methods equation. SAMG contains the depicted
  global/local gating mechanism.
- Sampling uses pixel centers, `align_corners=False`, and border padding.
  Decoder/detail interpolation also uses `align_corners=False`.
- The QKV and attention output pointwise projections are dense. The context
  depthwise kernel is 7; the two-PWConv expansion ratio is 2; the global SAMG
  gate width is 8. All of these are explicit constructor settings. The SAMG
  local hidden width remains C_s, matching Figure 4.
- CCA uses query/key width `max(C_D3 // 8, 1)`, separately normalized row and
  column attention, and a scalar residual weight initialized to zero. The
  query pixel belongs to each direction. There is no recurrence or extra CCA.
- Decoder dropout is a configurable regularization setting retained from the
  reference configuration; the diagram omits this operator.

## Odd and very small image sizes

The network returns the original input height and width. Max-pooling uses
`ceil_mode=True`; for odd dimensions each stage has `ceil(previous_size / 2)`.
Upsampling targets the actual saved encoder size. DCDF applies the same ceil
pooling before channel projection, so its three inputs still align. For input
dimensions divisible by 16 the stage sizes exactly match Figure 2.

Batch normalization normally computes batch statistics during training. When
only one value per channel is available (`B=H=W=1`), it uses existing running
statistics instead. This makes tiny-image tests possible; it does not alter
the normalization of ordinary training batches. Zero-sized inputs are rejected.

## Cumulative ablations

| Name | Active components | Registered / backward-connected parameters |
| --- | --- | ---: |
| `baseline` | Two-CBR encoder blocks, all direct skips, regular decoder | 821,345 / 821,345 |
| `csde` | Replace all encoder blocks with CSDE | 1,451,129 / 1,451,129 |
| `samg` | Gate E3 and provide its projected response to D3 | 1,548,729 / 1,548,729 |
| `dcdf` | Gate E2/E3/E4 and replace the single-level D3 detail with DCDF | 1,956,201 / 1,956,201 |
| `detail_decoder` | Reuse shared detail at D3, D2, D1 | 1,956,761 / 1,956,761 |
| `selective_skip` | Keep only direct E4/E1 skips | 1,955,481 / 1,955,481 |
| `full` | Add one CCA before Decoder 3 | 1,956,802 / 1,956,802 |

Only active components are registered, so parameter counts do not include
unused detail modules from other ablations. The cumulative ordering follows the
reference experiment sequence; it does not assert reproduction of its scores.

The table uses the current default C=8, C_d=8, and one input channel. Each
variant was separately checked on a real `2 x 1 x 64 x 64` training-mode forward
and BCE backward pass with seed 2026. Every registered parameter was connected
to the backward graph and all computed gradients were finite; no unused module
was found. "Connected" means `.grad is not None`, not that every gradient is
nonzero. The full model's six CCA query/key/value parameter tensors initially
have zero gradients because the learnable residual gamma starts at zero.

## Validation and measured complexity

Run `python -m pytest tests/test_model.py -q` from the package root. Tests cover
small and odd output sizes, all cumulative variants, finite BCE backward passes,
offset gradients, global/local SAMG gate gradients, both new pointwise layers,
the single D3 CCA location, and invalid configurations.

On 2026-09-10, these 25 tests passed on CPU using
`/opt/anaconda3/bin/python` and PyTorch 2.3.1. The query-sampling test verifies an
analytic case: shifting the sampled query to a zero feature changes a peaked
attention response into the expected uniform average. The default C=8 model is
also exercised in a BCE forward/backward check. These are local smoke results;
they do not validate H200 execution or a trained segmentation score.

`scripts/profile_model.py --fvcore` was also run at 64x64 and 96x96 on CPU.
The machine-readable observations, full constructor settings, source hashes,
per-operator counts, and ablation participation checks are archived in
[`profile_measured.json`](profile_measured.json).

| Input shape | Registered parameters | fvcore supported-operation estimate |
| --- | ---: | ---: |
| 1 x 1 x 64 x 64 | 1,956,802 | 183,813,088 (0.183813 G) |
| 1 x 1 x 96 x 96 | 1,956,802 | 413,909,984 (0.413910 G) |

These are **partial supported-operation estimates, not complete FLOP totals**.
fvcore 0.1.5.post20221221 counts one fused multiply-add as one operation and
includes convolution, batch normalization, grid sampling, adaptive average
pooling, bilinear upsampling, and batched matrix multiplication here. The
unsupported operators (identical occurrence counts at both shapes) are:

| Unsupported operator | Occurrences |
| --- | ---: |
| `aten::mean` | 10 |
| `aten::sub` | 30 |
| `aten::square` | 5 |
| `aten::add` | 95 |
| `aten::rsqrt` | 5 |
| `aten::mul` | 140 |
| `aten::meshgrid` | 20 |
| `aten::reciprocal` | 20 |
| `aten::sum` | 43 |
| `aten::div` | 22 |
| `aten::softmax` | 25 |
| `aten::gelu` | 15 |
| `aten::max_pool2d` | 5 |
| `aten::feature_dropout` | 5 |

An unsupported occurrence does not itself establish a nonzero runtime cost;
for example, spatial dropout is inactive in evaluation mode. The lists are
retained as returned by the profiler, without silently claiming full coverage.

The following parameter totals were obtained directly from instantiated models
on 2026-09-10 with PyTorch 2.3.1, one input channel, detail width equal to base
width, four attention heads, and pointwise expansion 2. These are counts of this
new implementation, not verification of the manuscript's historical table.

| Base width C | Encoder widths | Actual parameters |
| --- | --- | ---: |
| 4 | 4, 8, 16, 32, 64 | 506,510 |
| **8 (default)** | **8, 16, 32, 64, 128** | **1,956,802** |
| 12 | 12, 24, 48, 96, 192 | 4,351,382 |
| 16 | 16, 32, 64, 128, 256 | 7,690,250 |

C=10 is incompatible with four equal-width heads. Changing to two or five heads
would alter the attention setting and was not used for selecting the default.
The default is approximately 1.96 M, not 1.82 M. There are no unused parameter
arrays or special adjustments to force an advertised total.

Use `count_parameters` on the exact instantiated configuration. FLOP/MAC reports
must state batch size, channels, spatial dimensions, counting library, and
unsupported operations, especially bilinear sampling. An architecture smoke
test or a complexity report is not a completed segmentation experiment.
