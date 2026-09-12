# Release status — 13 September 2026

## Implemented and checked locally

The archived 10 September local test suite passed **43 tests** with zero failures. The detailed scope is recorded in `validation.json`.

- A complete figure-aligned CGDD-Net, seven cumulative ablations and explicit architecture settings.
- Data discovery and pair loading for all 133 original images: DRIVE 40, STARE 20, CHASE_DB1 28 and HRF 45. Data was read in place and is not distributed here.
- Image/subject/original-identity checks before patch sampling, plus byte-level duplicate detection.
- Adam, linear warmup with cosine restarts, masked binary cross entropy, validation-AUC model selection, checkpoint/resume support and bounded sliding-window inference.
- Synthetic end-to-end checks with the actual full model: two training epochs, checkpoint loading/resume, evaluation and probability/metric export.
- Unit/behavior checks for forward/backward, sampled-query semantics, offset and gate gradients, all ablations, data isolation, undefined metrics, overlap averaging and scheduler state recovery.

The local validation environment is Python 3.12.2, PyTorch 2.3.1, CPU. The separately reported server environment is NVIDIA H200 141 GB, PyTorch 2.13.0 and CUDA 13.2. Local functional checks do not establish results on the server or on retinal benchmarks. The GitHub Actions workflow runs CPU unit and synthetic checks on pushes and pull requests. Its actual status is available in the repository Actions tab.

## Measured implementation versus reported experiment

The default new implementation has **1,956,802 trainable parameters**. All registered parameters participate in the forward/backward graph; initially zero gradients in CCA's Q/K/V projections follow from its zero-initialized residual scale and are not unused parameters.

The manuscript uses the measured **1.96 M** parameter count, as confirmed by the author. **20.05 G FLOPs** remains an earlier server profile; its input and counting convention need archived evidence before comparison to this implementation.

The supplied `docs/profile_measured.json` includes 64×64 and 96×96 input profiles, source/config hashes, supported operator counts and unsupported operators. Partial fvcore counts are not full FLOP totals.

## Historical information not supplied in the attachment

The original server configuration and train/validation/test image lists are absent. The inherited toolkit's examples do not establish the manuscript's STARE folds, CHASE 20/8 partition or HRF partition. Newly generated protocols explicitly identify themselves as new.

The paper's statistical means and standard deviations do not provide the missing sample unit, number of seeds, pairing or p-value protocol. `scripts/summarize_runs.py` is a prospective tool for actual AUC run records; its available Wilcoxon/Holm method is not retroactively attributed to the paper.

The author confirmed the actual base learning rate as **0.001** on 10 September; the manuscript and default configuration now agree. Other release defaults remain distinct from recovered server settings. Hyperparameter Figure 8 remains under revision. Author names and CRediT have been restored in the manuscript; the AI declaration and experiment archive statement remain author-managed submission items. This code release includes neither fabricated checkpoints nor synthetic values presented as paper results.

## Source of the workflow

The outer `VesselSeg-Pytorch-master` is the supplied baseline. Its original files remain unchanged. `NOTICE` identifies the upstream project and the major changes. The executable pipeline here is self-contained: it does not import modules from the outer directory or the older CGDD-Net reference implementation.
