# Release status — 12 October 2026

## Current revision

The author requested a Neurocomputing revision reference document while leaving the supplied LaTeX archive unchanged. This repository update synchronizes reporting metadata and documentation, adds a mathematical p-value-bound utility, and removes redundant release documentation and an optional packaging wrapper. It does not retrain the model, recompute the manuscript scores, change the architecture, or assert journal acceptance.

The author order follows the supplied manuscript front matter: Xincheng Li, Xiaoqi Sheng, Xinyu Zhang. README headings, HTML layout, badges, command blocks, and result-table format are retained.

## Implementation and configuration

The reference implementation uses one grayscale input channel; encoder widths 8/16/32/64/128; shared detail width 8; four CSDE heads with spacings 1/3/5/7; dense SAMG branches; DCDF fusion of E2/E3/E4; shared guidance at D3/D2/D1; direct E4/E1 skips; and one CCA block after D3 fusion. DCDF fuses input-dependent SAMG responses using learned projections; it does not predict an additional set of dynamic fusion weights.

The full model contains 1,956,802 trainable parameters. Training uses FOV-masked BCEWithLogits, Adam at base learning rate 0.001, and a maximum of 100 epochs. The earlier public 50-epoch default was corrected following the author's 25 September 2026 confirmation. Overlapping-patch inference averages probabilities before thresholding at 0.5. Full operation details and dated diagnostic measurements remain in `architecture.md` and `profile_measured.json`.

Cumulative ablations trace one addition sequence and are not independent factorial tests. The SAMG-only variant routes gated E3 to D3; DCDF next replaces this with fused E2/E3/E4 guidance to D3; the detail-decoder variant then reuses it at D3/D2/D1. The final variant adds CCA. This describes the current implementation, not recovery of archived trained checkpoints.

## Data and results

The author-requested train/test counts are DRIVE 20/20, CHASE_DB1 20/8, STARE 10/10, and HRF 25/20. Acquisition FOVs are respectively 45°, 30°, 35°, and 60°. HRF 60° is verified from the original segmentation-database paper; it is distinct from the separate 45° image-quality dataset. See `data_protocol.md` for sources, mask semantics, and new-experiment examples.

Existing `results/paper_*.csv` files remain unchanged transcriptions of the supplied manuscript. The source manuscript described STARE as five-fold evaluation; changing the requested reporting count to 10/10 does not prove which protocol produced those numbers. Exact historical split IDs and the score-to-protocol association remain pending in `results/provenance.json`. The stale HRF 15/20 status entry is corrected to the supplied manuscript's 25/20 count without fabricating image assignments.

Sections 4.2 and 4.4 retain the author's stated rank-sum analysis. The new `scripts/exact_p_bounds.py` calculates only sample-size lower bounds under an untied, two-sided exact model. It does not generate AUC samples, empirical p-values, or significance claims. Original sample vectors are still needed for experimental recalculation.

## Validation and complexity

The existing CPU tests and synthetic end-to-end smoke test exercise software behavior, not retinal benchmark reproduction. A focused mathematical test module is added for the lower-bound utility. Check the workflow run attached to the exact commit for its CI outcome; no benchmark training is implied by CI success.

The author-reported benchmark environment is NVIDIA H200 141 GB, PyTorch 2.13.0, and CUDA 13.2; CI may use a different CPU/PyTorch environment. The manuscript reports 29.96 GFLOPs, whereas `profile_measured.json` contains partial supported-operation counts for specified input shapes. Those diagnostics are neither a replacement for the reported model-level total nor hardware latency measurements. A complete comparison requires the actual input shape, counting convention, tool version, and unsupported-operator policy. There is no MDF-Net complexity claim in this repository update.

## Retained provenance and cleanup

Required runtime code, configurations, data preparation, inference, profiling, statistical utilities, tests, CI, result provenance, measured diagnostics, `LICENSE`, `NOTICE`, and `LICENSES/` are retained. `RELEASE_NOTES.md` and `docs/reproducibility_zh.md` duplicate guidance consolidated here and in the README/data protocol; the unused optional `scripts/build_release.py` packaging wrapper is removed. Deletions remain recoverable in Git history.

For reproducible experiments, archive the exact source commit, configuration, image/subject split IDs, environment, logs, checkpoints, and original statistical observations with hashes. New runs already record configurations and manifests and can export per-image predictions. These facilities do not replace missing historical records. No permanent archive DOI or new trained weights are claimed. Dataset images are not redistributed.
