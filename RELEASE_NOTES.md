# Source update — 13 September 2026

The repository now uses the figure-aligned implementation prepared in the author-specified VesselSeg-Pytorch checkout. The measured full model has 1,956,802 trainable parameters (1.96 M); Adam uses the author-confirmed base learning rate 0.001. Training uses FOV-masked BCE from logits.

## Migration from the earlier implementation

- Import `CGDDNet` from `cgddnet.model`; the default input is grayscale with one channel.
- Use `configs/default.json` instead of the earlier YAML configuration. The previous 2.97 M architecture and its configuration are not interchangeable with this release.
- Use `train.py` and `evaluate.py` with explicit train/validation/test manifests. The README documents the current CLI; the old `test.py`, `cross_dataset.py` and shell wrappers belong to the earlier interface.
- Empty split placeholders have been removed. Supply actual manifests or explicitly generate a new experiment protocol with `scripts/prepare_data.py`.
- The earlier repository remains accessible in Git history, including commit `8d6203424534563f984efb149de9afb212b430e5`.
- The current source retains the supplied VesselSeg workflow's Apache-2.0 attribution. The previous repository's MIT notice is retained under `LICENSES/legacy-MIT.txt`.

## Evidence scope

The manuscript uses 1.96 M parameters and learning rate 0.001. Original server checkpoints, exact split manifests and seed-level outputs are still not included. `results/` contains author-supplied table transcriptions; local synthetic tests are not retinal benchmark reproduction. Historical profiling JSON and provenance hashes retain their measurement-time values. The earlier 20.05 G FLOPs figure is not a measured complete FLOP count for this release.

No manuscript draft, private author response form, original dataset images, checkpoints or local caches are included in this source update.
