# Paper-to-code mapping

**Multi-Scale Temporal Fusion Network with Encoding-Guided Reciprocal
Re-ranking for ECoG-to-Text Semantic Decoding**.

| Paper item | Entry point | Included reference results |
|---|---|---|
| Figure 1: GPT-2 lag encoding | `run_lag_resolved_encoding.py`, `plot_lag_resolved_encoding.py` | `lag_resolved_semantic_encoding/` |
| PCA / PCA+HA | `run_pca_alignment_baselines.py` | `pca_alignment_baselines/`, variants `pca`, `pca_ha` |
| SRM-CNN | `run_srm_cnn_baseline.py` | `heo2026_srm_cnn_baseline_all/`, variant `direct` |
| Medium | `run_multiwindow_srm.py` | `multiwindow_srm_all/`, variant `matched_direct_combined` |
| Long-only | `run_multiwindow_srm.py` | `matched_long_control/`, variant `matched_long_combined` |
| Multi-scale | `run_multiwindow_srm.py` | `multiwindow_srm_all/`, variant `multiwindow_combined` |
| Multi+EGR | `run_encoding_guided_rerank.py` | `encoding_guided_rerank/` |
| Table aggregation and paired tests | `summarize_paper.py` | `reports/paper_tables.md`, `reports/paired_tests.json` |
| Paired tests on new runs | `analyze_statistics.py` | generated `results/statistics.json` |

Entry points are in `scripts/`; reference result paths are relative to
`results/`. Original result directory names are retained. New experiments use
the output directories listed in the README.

## Implementation

* `src/tada/models.py`: residual temporal encoder and adaptive window fusion.
* `src/tada/losses.py`: semantic hard-negative mining and ranking loss.
* `src/tada/srm_cnn_baseline.py`: SRM projection, preprocessing, retrieval
  metrics and SRM-CNN baseline.
* `src/tada/multiwindow_srm.py`: window-specific projection and decoder training.

## Reference result conventions

* Medium and Multi-scale use `multiwindow_srm_sub03/sub-03/summary.json` for
  participant 03 and `multiwindow_srm_all/` for the other seven configurations.
* Some reference summaries omit top-level source means. `summarize_paper.py`
  obtains these from the five per-fold `source_mean` dictionaries.
* Long-only corresponds to `matched_long_combined`; SRM-CNN corresponds to
  `direct`.
* EGR significance pairs `held_out_base_mean` and `held_out_egr_mean` from the
  same run.
* Reference JSON files contain configurations and fold metrics. Validation
  histories and machine-specific absolute paths are omitted.
* `configs/manuscript_tables.json` contains the 140 printed means and sample
  SDs used to check Tables I and II.
