#!/usr/bin/env python3
"""Fit word-locked semantic ECoG encoding models at successive neural lags.

For every participant and 200-ms high-gamma window, this script fits a
train-only PCA + ridge model from GPT-2 XL word representations to the
participant's significant ECoG electrodes.  The held-out chronological-fold
Pearson r is the encoding score.  It is deliberately an *encoding* analysis
(language -> ECoG), rather than the lag-resolved retrieval analysis
(ECoG -> language).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SUBJECTS = ("01", "03", "04", "05", "06", "07", "08", "09")
PUBLIC_TO_ORIGINAL = {
    "sub-01": 661, "sub-03": 717, "sub-04": 723, "sub-05": 741,
    "sub-06": 742, "sub-07": 743, "sub-08": 763, "sub-09": 798,
}


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--cache-root", type=Path,
                        default=project / "data/ecog_caches/wide_m1000_p1000")
    parser.add_argument("--source-root", type=Path,
                        default=project / "data/source/ds005574")
    parser.add_argument("--electrode-file", type=Path,
                        default=project / "data/metadata/all_subject_sig.csv")
    parser.add_argument("--output-root", type=Path,
                        default=project / "results/lag_resolved_semantic_encoding")
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--step-ms", type=float, default=50.0)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--target-components", type=int, default=50)
    parser.add_argument("--target-source", choices=("gpt2", "whisper"), default="gpt2")
    parser.add_argument(
        "--whisper-features-dir", type=Path,
        default=project / "data/whisper_layer_features/whisper-small",
    )
    parser.add_argument(
        "--whisper-layer", default="encoder/layer_00",
        help="HDF5 feature key used when --target-source whisper.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=100.0,
                        help="Fixed ridge strength after train-fold standardization.")
    parser.add_argument("--max-words", type=int, help="Smoke test only")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_targets(source_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    root = source_root / "stimuli/gpt2-xl"
    transcript = pd.read_csv(root / "transcript.tsv", sep="\t", index_col=0)
    with h5py.File(root / "features.hdf5", "r") as handle:
        tokens = np.asarray(handle["layer-24"], dtype=np.float32)
    transcript = transcript.copy()
    transcript["feature_row"] = np.arange(len(transcript))
    groups = transcript.groupby("word_idx", sort=True)
    target = np.stack([
        tokens[group.feature_row.to_numpy()].mean(axis=0)
        for _, group in groups
    ]).astype(np.float32)
    words = groups.agg(word=("word", "first"), start=("start", "first"),
                       end=("end", "last")).reset_index()
    valid = words.start.notna().to_numpy() & np.isfinite(target).all(axis=1)
    words, target = words.loc[valid].reset_index(drop=True), target[valid]
    order = np.argsort(words.start.to_numpy(float), kind="stable")
    return words.iloc[order].reset_index(drop=True), target[order]


def load_whisper_targets(directory: Path, layer: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load frozen word-level Whisper features and align them by word index."""
    transcript = pd.read_csv(directory / "transcript.tsv", sep="\t")
    with h5py.File(directory / "features.hdf5", "r") as handle:
        if layer not in handle:
            raise KeyError(f"{layer!r} is unavailable in {directory / 'features.hdf5'}")
        values = np.asarray(handle[layer], dtype=np.float32)
    if len(transcript) != len(values):
        raise ValueError("Whisper transcript and feature rows differ")
    transcript = transcript.copy()
    transcript["feature_row"] = np.arange(len(transcript))
    groups = transcript.groupby("word_idx", sort=True)
    target = np.stack([
        values[group.feature_row.to_numpy()].mean(axis=0)
        for _, group in groups
    ]).astype(np.float32)
    words = groups.agg(word=("word", "first"), start=("start", "first"),
                       end=("end", "last")).reset_index()
    valid = words.start.notna().to_numpy() & np.isfinite(target).all(axis=1)
    words, target = words.loc[valid].reset_index(drop=True), target[valid]
    order = np.argsort(words.start.to_numpy(float), kind="stable")
    return words.iloc[order].reset_index(drop=True), target[order]


def aligned_subject(subject: str, cache_root: Path, target_words: pd.DataFrame,
                    target: np.ndarray, selected_channels: dict[str, set[str]]) -> tuple[np.ndarray, np.ndarray]:
    name = f"sub-{int(subject):02d}"
    root = cache_root / name
    words = pd.read_csv(root / "words.tsv", sep="\t")
    ecog = np.asarray(np.load(root / "ecog.npy", mmap_mode="r"), dtype=np.float32)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    channels = np.asarray(metadata["channel_names"])
    keep_channels = np.isin(channels, list(selected_channels[name]))
    if not keep_channels.any():
        raise ValueError(f"{name}: no released significant electrodes found in cache")
    lookup = {int(value): row for row, value in enumerate(target_words.word_idx)}
    keep_words = words.word_idx.astype(int).isin(lookup).to_numpy()
    rows = np.flatnonzero(keep_words)
    target_rows = np.asarray([lookup[int(value)] for value in words.loc[keep_words, "word_idx"]])
    order = np.argsort(words.loc[keep_words, "start"].to_numpy(float), kind="stable")
    return ecog[rows][:, keep_channels][order], target[target_rows][order]


def mean_electrode_correlation(prediction: np.ndarray, observation: np.ndarray) -> float:
    pred = prediction - prediction.mean(axis=0, keepdims=True)
    obs = observation - observation.mean(axis=0, keepdims=True)
    denominator = np.sqrt((pred * pred).sum(axis=0) * (obs * obs).sum(axis=0))
    correlation = (pred * obs).sum(axis=0) / np.maximum(denominator, 1e-12)
    return float(np.nanmean(correlation))


def run_subject(subject: str, target_words: pd.DataFrame, target: np.ndarray,
                selected_channels: dict[str, set[str]], args: argparse.Namespace) -> dict:
    name = f"sub-{int(subject):02d}"
    output = args.output_root / name / "summary.json"
    if output.exists() and not args.overwrite:
        return json.loads(output.read_text(encoding="utf-8"))
    ecog, target = aligned_subject(subject, args.cache_root, target_words, target, selected_channels)
    if args.max_words:
        ecog, target = ecog[:args.max_words], target[:args.max_words]
    metadata = json.loads((args.cache_root / name / "metadata.json").read_text(encoding="utf-8"))
    start_ms = 1000.0 * float(metadata["response_start_s"])
    end_ms = 1000.0 * float(metadata["response_end_s"])
    bin_width_ms = (end_ms - start_ms) / ecog.shape[2]
    bins_per_window = int(round(args.window_ms / bin_width_ms))
    step_bins = int(round(args.step_ms / bin_width_ms))
    if bins_per_window < 1 or step_bins < 1 or bins_per_window > ecog.shape[2]:
        raise ValueError("window/step must be compatible with cache temporal bins")
    starts = list(range(0, ecog.shape[2] - bins_per_window + 1, step_bins))
    centers = np.asarray([
        start_ms + (first + bins_per_window / 2.0) * bin_width_ms for first in starts
    ])
    # The representation PCA is independent of neural lag.  Fit it once per
    # outer fold, rather than repeating the same decomposition at every lag.
    fold_designs = []
    for train, test in KFold(args.outer_folds, shuffle=False).split(target):
        n_components = min(args.target_components, len(train) - 1, target.shape[1])
        pca = PCA(n_components=n_components, random_state=0)
        fold_designs.append((train, test, pca.fit_transform(target[train]), pca.transform(target[test])))
    rows = []
    for index, (first, center) in enumerate(zip(starts, centers)):
        response = ecog[:, :, first:first + bins_per_window].mean(axis=2)
        scores, alphas = [], []
        for train, test, x_train, x_test in fold_designs:
            model = make_pipeline(
                StandardScaler(), Ridge(alpha=args.ridge_alpha)
            )
            model.fit(x_train, response[train])
            scores.append(mean_electrode_correlation(model.predict(x_test), response[test]))
            alphas.append(float(args.ridge_alpha))
        rows.append({
            "lag_center_ms": float(center),
            "mean_encoding_r": float(np.mean(scores)),
            "fold_encoding_r": [float(score) for score in scores],
            "mean_ridge_alpha": float(np.mean(alphas)),
        })
        print(f"{name}: {index + 1:02d}/{len(starts)} lag={center:+.0f} ms r={rows[-1]['mean_encoding_r']:.4f}", flush=True)
    report = {
        "method": "lag-resolved source-subject GPT-2-to-ECoG semantic encoding",
        "interpretation": "Each point is a 200-ms word-locked high-gamma response window; score is held-out mean Pearson r across released significant electrodes.",
        "subject": name,
        "configuration": {"window_ms": args.window_ms, "step_ms": args.step_ms,
                          "outer_folds": args.outer_folds,
                          "target_components": args.target_components,
                          "n_words": int(len(target)), "n_electrodes": int(ecog.shape[1])},
        "lags": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    if args.target_source == "gpt2":
        target_words, target = load_targets(args.source_root)
    else:
        target_words, target = load_whisper_targets(args.whisper_features_dir, args.whisper_layer)
    table = pd.read_csv(args.electrode_file)
    selected_channels = {
        name: set(table.loc[table.subject.astype(int) == original, "elec"].astype(str))
        for name, original in PUBLIC_TO_ORIGINAL.items()
    }
    reports = [run_subject(subject, target_words, target, selected_channels, args)
               for subject in args.subjects]
    lags = np.asarray([row["lag_center_ms"] for row in reports[0]["lags"]])
    values = np.asarray([[row["mean_encoding_r"] for row in report["lags"]] for report in reports])
    aggregate = {
        "method": reports[0]["method"], "subjects": [report["subject"] for report in reports],
        "target_source": args.target_source,
        "target_layer": args.whisper_layer if args.target_source == "whisper" else "gpt2-xl/layer-24",
        "lag_center_ms": lags.tolist(), "subject_encoding_r": values.tolist(),
        "mean_encoding_r": values.mean(axis=0).tolist(),
        "sem_encoding_r": (values.std(axis=0, ddof=1) / np.sqrt(len(values))).tolist(),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "aggregate.json"
    output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
