# Release status — 12 October 2026

## Current revision

This update accompanies the revised Neurocomputing LaTeX manuscript and English-Chinese Word version. It synchronizes the author-confirmed experimental protocol, 64x64 inference patches, fixed 100-epoch training budget, and Table 7 observation unit. No research figure, architecture, benchmark score, or original Table 7 p-value is regenerated. Vitae and photographs remain clearly identified author-completion placeholders in the submission materials, not fabricated personal records.

The author order remains Xincheng Li, Xiaoqi Sheng, Xinyu Zhang. README headings, HTML layout, badges, command-block structure and result-table format are preserved.

## Implementation and configuration

The implementation uses one grayscale input channel; encoder widths 8/16/32/64/128; detail width 8; four CSDE heads with spacings 1/3/5/7; dense SAMG branches; DCDF fusion of E2/E3/E4; shared guidance at D3/D2/D1; direct E4/E1 skips; and one CCA block after D3 fusion. DCDF combines input-dependent SAMG responses through learned projections, without an extra dynamically generated fusion-weight module.

The full model contains 1,956,802 trainable parameters. Training uses FOV-masked BCEWithLogits, Adam at learning rate 0.001, batch size 64, dropout 0.1, and 100 epochs. Early stopping is disabled in the default configuration. Training and inference both use 64x64 patches; the earlier 96x96 inference default is corrected. Stride 16, inference mini-batch 8 and threshold 0.5 remain unchanged. Reconstruction averages overlapping probabilities before full-image metrics are calculated.

Cumulative ablations trace one addition sequence, not independent factorial tests. The SAMG-only variant routes gated E3 to D3; DCDF combines E2/E3/E4 guidance at D3; the detail decoder reuses it at D3/D2/D1. The final variant adds CCA. Architecture and dated diagnostic records remain available in `architecture.md` and `profile_measured.json`.

## Data and results

Actual author-confirmed train/test counts are DRIVE 20/20, CHASE_DB1 20/8, STARE 10/10, and HRF 25/20. All STARE experiments used 10/10; five-fold was an erroneous manuscript description, not an experiment change. The earlier pending score-to-protocol association is resolved by this explicit author confirmation. Original split IDs are still separate, unprovided reproducibility records.

Acquisition FOVs are 45°, 30°, 35°, and 60°, respectively. HRF 60° refers to the segmentation database, not the separate 45° image-quality collection. Result CSVs are preserved, and new experiment outputs stay separate from their manuscript transcriptions.

Each Table 7 observation is one complete test image's AUC: vector lengths 20, 8, and 10. The original means, standard deviations, rank-sum p-values and stars remain author-retained values. This synchronization is not an independent statistical recalculation. `exact_p_bounds.py` provides mathematical diagnostics only; `summarize_runs.py` analyzes new repeated training runs.

## Validation and complexity

The test suite includes regression checks for the confirmed defaults, 64x64 full-image reconstruction, explicit profiling batch shape, and invalid profiler arguments. CI and synthetic smoke tests check software behavior; their success does not imply a retinal benchmark rerun. Consult the workflow attached to the exact commit for its status.

The author confirms H200 141 GB with Python/PyTorch/CUDA, and a Mac as the local workstation. The GPU execution-host OS and exact historical software versions are not inferred from the Mac. Earlier unverified version strings are omitted from current documentation.

Table 10 retains the reported 1.96 M parameters and 29.96 GFLOPs under the common comparison setup. The original profiling shape, tool version and counting convention have not been recovered. The revised profiler accepts explicit batch size, height, width and device, records the runtime environment, and optionally reports fvcore supported-operation counts plus unsupported operators. Its new 64x1x64x64 example is not asserted to reproduce the historical table. Equal pixel area between a patch batch and a 512x512 image does not establish equal network cost. No hardware-latency or MDF-Net complexity claim is introduced.

## Retained provenance and cleanup

Core model/training/evaluation code, data preparation, test and CI infrastructure, result CSVs, diagnostic records and licenses are retained. Previously removed duplicate release documents and the optional packaging wrapper are not reintroduced. Original benchmark outputs are not overwritten with theoretical bounds or new profiler estimates.

For reproducibility, retain the source commit, original image/subject split IDs, resolved configuration, execution environment, training logs, checkpoints and per-image statistics. New runs already export these records where available. No new trained weights, archive DOI or journal acceptance is asserted. Dataset images are not redistributed.
