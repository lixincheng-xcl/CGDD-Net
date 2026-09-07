# Verification record

The packaged source was smoke-tested before release.

- All seven cumulative ablation variants produce a `B x 1 x H x W` logit map for a `48 x 48` input.
- A forward + BCE backward pass on the complete network produces finite gradients.
- The complete model contains **2,970,278 trainable parameters (2.97 M)**.
- `pytest -q` passes the included architecture tests.
- The final prediction head is a direct `1x1` convolution on the highest-resolution decoder output.
- The model source contains CSDE, SAMG, DCDF, detail-guided decoding, selective skips, and one CCA module.

Dataset files and trained checkpoints are intentionally not bundled. Populate the JSON manifests under `splits/` after placing the public datasets on the target server.
