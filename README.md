# TADA

Implementation of **Multi-Scale Temporal Fusion Network with Encoding-Guided
Reciprocal Re-ranking for ECoG-to-Text Semantic Decoding**.

This repository provides data preparation, lag-resolved GPT-2 encoding,
SRM alignment, model training, and source/held-out participant evaluation.

## Install

Use Linux/WSL with Python 3.10-3.12. Install a PyTorch build matching your GPU,
then install the package and SRM dependencies:

```bash
python -m pip install -e '.[srm]'
python -c "from mpi4py import MPI; from brainiak.funcalign.srm import SRM; print(MPI.Get_library_version())"
```

## Prepare inputs

Run from the repository root. Download the official preprocessed high-gamma
and stimuli from [Podcast ECoG / OpenNeuro ds005574](https://openneuro.org/datasets/ds005574)
(Bash, Python and `wget` required), then obtain the significant-electrode mask:

```bash
bash scripts/download_official_preprocessed.sh data/source/ds005574
python scripts/fetch_electrode_metadata.py
python scripts/run_experiments.py --stage prepare
```

Expected source files:

```text
data/source/ds005574/
  derivatives/ecogprep/sub-XX/ieeg/sub-XX_task-podcast_desc-highgamma_ieeg.fif
  stimuli/podcast_transcript.csv
  stimuli/gpt2-xl/transcript.tsv
  stimuli/gpt2-xl/features.hdf5    # layer-24
data/metadata/all_subject_sig.csv
```

The electrode mask is pinned to Podcast Benchmark revision
`beac7e932d77f08a934520a7e84f369a7e6d03c8` and checked by SHA-256.
The dataset downloader resolves upstream `main`; strict replication requires
the original validated source snapshot. Dataset and external inputs retain
their upstream license and attribution requirements.

## Run the paper experiments

```bash
# Figure 1: lag-resolved encoding and its plot
python scripts/run_experiments.py --stage figure

# Tables I/II: train and evaluate every decoding method
python scripts/run_experiments.py --stage decoding --device cuda
```

Select an individual method with `--stage pca`, `srm-cnn`, `medium`, `long`,
`multi`, or `egr`. `--stage all` performs preparation, the figure analysis, and
all decoding experiments sequentially. `--dry-run` prints commands without
executing them or writing files.

| Paper column | Stage | Variant | Generated output |
|---|---|---|---|
| PCA / PCA+HA | `pca` | `pca` / `pca_ha` | `results/pca/` |
| SRM-CNN | `srm-cnn` | `direct` | `results/srm_cnn/` |
| Medium | `medium` | `matched_direct_combined` | `results/medium/` |
| Long-only | `long` | `matched_long_combined` | `results/long/` |
| Multi-scale | `multi` | `multiwindow_combined` | `results/multi/` |
| Multi+EGR | `egr` | base decoder + ridge re-ranking | `results/egr/` |

Every method evaluates eight held-out configurations: 01, 03, 04, 05, 06, 07,
08, 09. Each configuration trains the decoder on the other seven participants.
`sub-XX/summary.json` records per-fold source and held-out AUC-ROC, pairwise
accuracy, Top-10, Top-50 and MRR, together with the run configuration.
For EGR these are `source_base_mean`, `source_egr_mean`, `held_out_base_mean`
and `held_out_egr_mean`. Other methods use variant-specific `source_mean`
and `held_out_mean`.

Defaults: five chronological folds, target PCA50, SRM k=5/100 iterations,
seed 0, maximum 100 epochs, patience 10, batch size 32, source validation 0.1.
Medium/long/multi use the combined point-wise and hard-negative ranking loss.
PCA and PCA+HA use the matched medium-window decoder. EGR's entry point trains
and evaluates its base decoder in the same run; ridge alpha and score weight
are selected on source validation data. Re-ranking does not update the decoder.

To recover table means, average source participants and folds within each
held-out configuration, then calculate mean and sample SD across the eight
configurations. Held-out statistics use the eight participant-level means.
Optional reproduction of the paper's AUC significance tests:

```bash
python scripts/analyze_statistics.py
```

This optional step requires completed five-fold medium, long, multi and EGR runs.
It is not needed to train or evaluate models.

## Reference results and tables

Selected reference configurations and fold metrics are included under
`results/`, with generated paper tables in `reports/`. Check all 140 printed
means and sample SDs without downloading data or installing scientific packages:

```bash
python scripts/summarize_paper.py --check-manuscript
```

For tables from a complete new run, use:

```bash
python scripts/summarize_paper.py --layout run --results-root results --output-dir results/tables
```

See [the paper-to-code mapping](docs/PAPER_MAP.md),
[reproduction instructions](docs/REPRODUCE.md), and
[data layout](data/README.md) for details.

## Paths and test runs

The runner accepts `--source-root`, `--electrode-file`, `--cache-root` (parent
of the three cache folders), `--output-root`, and `--held-out-subjects`.
Use these options to reuse previously prepared inputs. For a short test run:

```bash
python scripts/run_experiments.py --stage decoding --held-out-subjects 03 --max-words 300 --epochs 2 --patience 2 --outer-folds 2 --output-root results/smoke
```

This limited run checks training/evaluation wiring, not scientific performance.
It uses all eight participants' existing caches but evaluates one held-out
participant. Full paper runs omit the debugging limits.

## Code

```text
src/tada/models.py              residual temporal encoder and adaptive fusion
src/tada/losses.py              semantic hard-negative mining and ranking loss
src/tada/srm_cnn_baseline.py    SRM, preprocessing, baseline and retrieval metrics
src/tada/multiwindow_srm.py      multi-window projections and decoder training
scripts/                       data preparation, training/evaluation, figure analysis
```

The figure analysis uses GPT-2 semantic encoding with 200-ms windows, a 50-ms
step, and ridge alpha 100.

References: [Podcast Benchmark](https://github.com/hassonlab/podcast-benchmark),
[SRM-Encoding](https://github.com/pritamarnab/SRM-Encoding).
