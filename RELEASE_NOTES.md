# CGDD-Net source release — 13 September 2026

This release synchronizes the public implementation and documentation with the final manuscript configuration.

## Manuscript-aligned defaults

- Grayscale input with one channel.
- Encoder widths: `8 / 16 / 32 / 64 / 128`.
- Shared detail width: `8`.
- Full model: **1,956,802 trainable parameters (1.96 M)**.
- Adam base learning rate: **0.001**.
- FOV-masked BCEWithLogits objective.
- Linear warmup followed by cosine cycles with restarts.
- `64x64` training patches and overlapping `96x96` inference patches with stride `16`.
- Probability averaging in overlapping regions before thresholding at `0.5`.

## Architecture coverage

The release implements the manuscript's CSDE, SAMG, DCDF, shared detail guidance, selective E4/E1 skips, and a single CCA block after D3 fusion. Main-path channel projections use `1x1 Conv-BN-ReLU`; the prediction head is a bare `1x1` convolution.

Seven cumulative ablations are supported:

`baseline -> csde -> samg -> dcdf -> detail_decoder -> selective_skip -> full`

## Results and reproducibility

The `results/` directory contains machine-readable transcriptions of the manuscript's main, ablation, and cross-dataset tables. New training and evaluation runs write separate outputs and do not overwrite these manuscript values.

Training records the resolved configuration, environment, data manifests and hashes, training history, and checkpoints. Evaluation can export per-image metrics, probabilities, binary predictions, and provenance.

## Validation

The repository includes unit and synthetic end-to-end tests for model execution, gradient flow, patch inference, metrics, scheduler behavior, checkpoint resume, and output export. These checks verify software behavior; the retinal benchmark results are those reported in the manuscript.

## Attribution

Upstream workflow attribution and corresponding license material are retained in `NOTICE` and `LICENSES/`. Dataset images are not redistributed and remain subject to the original providers' terms.
