# Kaggle-controlled CGDD-Net experiments: first batch (20 October 2026)

**Status: runnable preparation, not a completed retinal benchmark.** Source code is developed on branch `experiments/pr-controlled-ablations-20261020`. Experiments require the researcher's Kaggle account, accepted provider usage terms, an enabled Kaggle GPU, and original dataset archives. No historical run, split membership, or missing checkpoint has been recovered. This experiment is **not** a reproduction of the existing paper results.

## 1. Data rights and official sources

| Dataset | Official source | Legally relevant fact | Plan |
|---|---|---|---|
| CHASE_DB1 | [Kingston University](https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/) | The dataset page explicitly identifies the CHASEDB1 archive as **CC BY 4.0**, 28 retinal images from **14 children**, with first/second observer labels. Attribute dataset authors when publishing. | **First** experiment. Download `CHASEDB1.zip` directly or manually attach it as private Kaggle Input. |
| STARE | [Clemson STARE](https://cecas.clemson.edu/~ahoover/stare/probing/) | Official 20 experimental images and **Adam Hoover** manual labels are distributed there as archives. Permission to use/archive does not imply unrestricted third-party redistribution. Cite Hoover et al. (2000), check site terms. | Second experiment. Download `stare-images.tar`, `labels-ah.tar` from provider or attach original archives privately. |
| DRIVE | [Grand Challenge DRIVE](https://drive.grand-challenge.org/) | The *current* official site describes 20 training images with labels and 20 test images whose labels are **not directly exposed**; test predictions go to the hosted evaluation. Account/terms may apply. | Do not fabricate DRIVE test labels or AUC. Only add after acquiring authorized data and a compatible evaluation protocol. |
| HRF | [FAU fundus dataset](https://www5.cs.fau.de/research/data/fundus-images/) | The 45-image vessel-segmentation dataset (15 healthy, 15 DR, 15 glaucoma) is published under **CC BY 4.0** for research. Do not confuse with the separate image-quality dataset. | Future extension; manual review of category- and subject-aware 25/20 split before training. |

Kaggle's Dataset uploader can **automatically unpack ZIP/TAR archives**. The preparation script also accepts `--source-folder /kaggle/input/<your-private-dataset>` and hashes the expected extracted original provider files. The Notebook detects a unique suitable extracted directory; if several are attached, set `ATTACHED_SOURCE_DIR` explicitly. This still requires sourcing from the original university archive, not an unverified repackaged mirror.

When a university endpoint returns HTTP 403 or redirects to a login/HTML page, **do not bypass it** or pull unknown mirrors. Obtain the original archive through the official browser access process, and add it as a **private Kaggle Dataset** (Dataset → New Dataset → Private), then attach the input to the Notebook. Only use data within its stated terms. The preparation script checks exact expected image IDs and matching first-observer vessel annotations; it records SHA-256 checksums of archives.

The first-batch university image/label archives are **not assumed** to provide an authoritative per-pixel FOV mask. The script deliberately stores `fov: null`; the existing evaluator interprets this as **full rectangular-image evaluation**. This convention must be declared whenever reporting scores and **must not** be mixed with the original FOV-masked manuscript scores. An estimated FOV should not be silently labeled official; any new FOV policy requires a reproducible algorithm and visual inspection of masks before the experiment.

**Data isolation:** raw/extracted images are stored under `/kaggle/temp/cgdd_data` (ephemeral), **not** under Kaggle's automatically versioned `/kaggle/working` output. Only JSON manifests/hashes and experiment outputs go to `/kaggle/working`. Do not accidentally publish raw images or patient-derived prediction arrays under licensing constraints.

## 2. How to start the first experiment

1. On [Kaggle Notebooks](https://www.kaggle.com/code), create a notebook and use **File → Import Notebook** to select `notebooks/CGDDNet_PR_Kaggle_First_Batch.ipynb`. Kaggle UI labels may vary. Select **Settings → Accelerator → GPU** and ensure **Internet On** for GitHub cloning/provider downloads.
2. Keep `DATASET = "CHASEDB1"`, `RUN_MODE = "first_run"`, `SPLIT_SEED = 20261020`, `SEED = 42` initially. Once the experiment series starts, do not change the split or budget in order to improve a result.
3. Run cells in order. The data acquisition cell tries the official Kingston archive; if denied, download the official file in a browser, create a **private Kaggle Input**, attach it via **Add Input**, and rerun the data cell.
4. Read the printed subject- and image-ID lists and review the image/label alignment plots before training. CHASE's 20/8 split is **10 train subjects / 4 test subjects**, with both eyes grouped together.
5. The Notebook first executes a **one-epoch tiny smoke budget** on real images, marked `smoke_test_only` and excluding it from manuscript tables. The pilot assesses whether the code and GPU can run; it is not a biological benchmark.
6. `RUN_MODE="first_run"` starts **CGDD-Net full** for a target **100 fixed epochs, 1024 sampled 64×64 patches per epoch, batch 8**. This equal-budget protocol is **NEW**, far smaller than the manuscript's 150,000 patches/epoch. Default `EPOCHS_PER_SESSION=25` trains 25 epochs per successful Kaggle version and exits with `latest.pt`, `progress.json`, **no test evaluation**. You must resume enough segments to finish all 100 epochs; only the final checkpoint is tested.
7. Use **Save Version → Save & Run All** as the Kaggle run launcher: it runs cells top-to-bottom in a **separate clean session** and persists `/kaggle/working` outputs if the run completes successfully. **Do not** run the complete training interactively and then choose Save & Run All to snapshot it; this would launch a second run. `Quick Save` skips running, but preserving interactive output may depend on its Advanced Output options.
8. In the next session, **Add Input → Notebook Output** of the previous successful version; set `RESTORE_FROM` to its attached `cgdd_pr_runs` directory **in the FIRST configuration cell**. Keep `RUN_MODE="first_run"` until the checkpoint reaches epoch100, then `RUN_MODE="first_batch"` for the remaining four variants. The training code validates identical configuration and manifest hashes before resuming.
9. After completing all five models for seed 42, repeat with **predeclared** seeds 2026 and 3407, ensuring budget, archive versions, FOV protocol, and IDs are unchanged. Then repeat on STARE using a **new, fixed 10/10 split** (`SPLIT_SEED` unchanged).

Only one active Kaggle GPU Notebook can normally be planned per account/session; the quota may be limited and sessions can be interrupted. Kaggle [official Notebook docs](https://www.kaggle.com/docs/notebooks) explain GPU, Internet, Add Input, and Output. Kaggle [GPU efficiency guide](https://www.kaggle.com/docs/efficient-gpu-usage) describes quota usage, typically 30 hours/week or dependent on account and availability. There is no guarantee of a specific accelerator.

## 3. Training budget and interpretation

Original manuscript default: **64×64 patches, training batch 64, 100 epochs, 150,000 patches/epoch**, overlap-averaged inference **64×64 stride 16**. The reference default is exceptionally expensive for free Kaggle; a new comparison budget of **100 epochs × 1024 patches/epoch × batch 8** is introduced to make initial controlled research feasible. The 1024 value is not claimed to reproduce original Table 3–10. If you raise/lower it after seeing a result, you must restart the entire set of comparisons using the newly declared budget, with a separate protocol name and archive.

Test/validation protocol: the current experimental training tool explicitly tests the **last checkpoint after fixed training**, without a separate validation set. This removes test-driven model selection but is a different protocol from historical experiment checkpointing. All five models use the same train/test sets; no test pixels enter training.

The initial matrix is five architectures per dataset: `full`, `independent_detail`, `fixed_offsets`, `uniform_samg`, `unet_same_width` (additional `unet_budget_matched` is available as an optional sixth comparison). The current independent-detail model carries independently parameterized fusion at D3, D2, D1; its exact parameters and FLOPs are reported rather than hidden. A slight parameter mismatch is disclosed.

### Data manifest and subject policy

- **CHASE_DB1**: random seed 20261020 shuffles 14 subjects, chooses four whole subjects (8 eyes) as **held-out test**, and puts the remaining ten subjects (20 eyes) in training; no eye of one child enters both partitions.
- **STARE**: seed 20261020 chooses 10 original image IDs as held-out test and 10 for train. This **new** split does *not* recover the author's historical 10/10 image assignment.
- Both datasets use the first named observer's segmentation mask and the same preprocessing settings across variants. Keep data masks as labeled; no extra morphological “repair” of ground truth is performed.
- `provenance.json`, `train.json` and `test.json` include image identity and archive hashes; store this metadata, not original images, in the paper's archival record.

### Checkpoint/output layout

```
/kaggle/working/
  pr_configs/<dataset>_kaggle_budget.json
  pr_split_records/<DATASET>/{train.json,test.json,provenance.json}
  cgdd_pr_runs/<DATASET>/<variant>/seed_<SEED>/
    config.json, environment.json, history.csv, manifests.json
    latest.pt  # saved each completed epoch, resumable
    progress.json  # only while checkpointed/incomplete; NO test evaluated
    final.pt   # created only after completed 100 epochs
    experiment_summary.json  # only after final epoch
    test/per_image.csv, test/summary.json, test/predictions/
```

Resume requires restoring the same saved run directory, a matching resolved configuration and unchanged manifest hashes. `Save & Run All` starts a **new** execution and is not a live-session snapshot. A new session may not retain prior `/kaggle/working` output unless the output of a successfully saved Version is attached as Input. Use `EPOCHS_PER_SESSION=25` to ensure each segment completes before the default Kaggle Save & Run All runtime limit (official guidance says 12 hours), and check your run time. An unsuccessful session may lose files that were not versioned; reducing segment length does not eliminate platform failures.

## 4. What constitutes completed evidence

- `python -m pytest -q` passing is a **software result only**.
- One-epoch pilot performance and standalone toy predictions cannot be reported as retinal segmentation performance.
- A 100-epoch run with a preserved checkpoint, exact input provenance, per-image test metrics and a logged environment is one **new experimental observation**.
- Only after the full predeclared matrix has run can relative performance, between-seed variation, paired analysis and meaningful comparison claims be written.
- Table 7's historical p-values **cannot** be inferred or replaced by these new data; Table 10's historical GFLOPs **cannot** be “verified” by dividing the previously reported values by the batch size.
- All complex operation profiling must record unsupported operators and its counting convention; real latency measurement needs synchronized GPU benchmarking.

## 5. Command-line preparation (outside Notebook)

```bash
# Must first obtain official provider archives with permission, then:
python -m experiments.kaggle_data --dataset CHASEDB1 \
  --chase-zip /path/to/CHASEDB1.zip \
  --output /path/to/ephemeral_data --split-seed 20261020
python -m experiments.kaggle_data --dataset STARE \
  --stare-images /path/to/stare-images.tar \
  --stare-labels /path/to/labels-ah.tar \
  --output /path/to/ephemeral_data --split-seed 20261020
# Or if Kaggle has automatically expanded the original university archive:
python -m experiments.kaggle_data --dataset CHASEDB1 \
  --source-folder /kaggle/input/my-private-chase \
  --output /kaggle/temp/cgdd_data --split-seed 20261020
```

The manifest filenames are `${output}/manifests/${dataset}/{train,test}.json`; `--data-root` in training points to `${output}`. Official data may not be redistributed in a public GitHub commit or Kaggle output without checking applicable rights.

### References and official links

- Kingston University, *CHASE_DB1 retinal vessel reference dataset* (2012), CC BY 4.0, https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/
- Hoover et al., *Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response*, IEEE TMI 19(3), 2000; Clemson University project, https://cecas.clemson.edu/~ahoover/stare/probing/
- Grand Challenge, DRIVE official source https://drive.grand-challenge.org/
- FAU, HRF public research database https://www5.cs.fau.de/research/data/fundus-images/
- Kaggle official Notebook documentation https://www.kaggle.com/docs/notebooks; GPU efficiency https://www.kaggle.com/docs/efficient-gpu-usage