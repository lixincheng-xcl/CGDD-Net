# Architecture Notes

CGDD-Net follows a three-stage design:

1. **Geometry-adaptive encoding**
   - The encoder uses `CSDEBlock`.
   - A fixed-grid convolution branch preserves stable local evidence.
   - A scale-adaptive deformable-attention branch samples a 3×3 lattice with learned offsets and modulation.
   - Four heads use base sampling scales `(1, 3, 5, 7)`.

2. **Dynamic detail modeling**
   - `SAMG` forms 1/3/5/7 receptive-field responses and predicts location-dependent softmax weights.
   - `DCDF` aligns the selected responses from encoder levels E2, E3, and E4 to the E3 resolution and compresses them into one canonical detail representation.

3. **Selective reconstruction**
   - The detail representation can guide D3, D2, and D1.
   - The full model retains direct E4 and E1 skips but suppresses direct E3/E2 bypasses.
   - One `CrissCrossAttention` block refines the D3-stage fused feature.

The default channel schedule for `base_channels=12` is:

```text
E1: 12
E2: 24
E3: 48
E4: 96
E5: 192
detail: 12
```

The final prediction head is a 1×1 convolution that produces one logit channel.

See `cgddnet/models/cgddnet.py` and `cgddnet/models/blocks.py` for the executable definition.
