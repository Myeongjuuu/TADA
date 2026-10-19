#!/usr/bin/env python3
"""Cross-subject retrieval baselines: subject-PCA and PCA + hyperalignment.

PCA follows the matched low-dimensional control in Bhattacharjee et al.
PCA+HA first derives train-only subject PCA scores and then uses generalized
orthogonal Procrustes alignment (hyperalignment) on matched source stimuli.
All downstream decoder, target, folds, and retrieval metrics match the
medium-window SRM baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tada.srm_cnn_baseline import SUBJECTS, device_for, metrics, mean_metrics, predict, target_pca
from tada.multiwindow_srm import WINDOWS, fit_decoder, load_multiwindow_data


def parse_args():
    project = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    p.add_argument("--held-out-subjects", nargs="+", default=list(SUBJECTS))
    p.add_argument("--variants", nargs="+", choices=("pca", "pca_ha"), default=("pca", "pca_ha"))
    p.add_argument("--cache-root", type=Path, default=project / "data/ecog_caches/heo2026_multiwindow_t10")
    p.add_argument("--output-root", type=Path, default=project / "results/pca_alignment_baselines")
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--neural-components", type=int, default=5)
    p.add_argument("--ha-iterations", type=int, default=20)
    p.add_argument("--target-components", type=int, default=50)
    p.add_argument("--validation-fraction", type=float, default=.1)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--matched-direct-hidden-dim", type=int, default=113)
    p.add_argument("--dropout", type=float, default=.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=.07)
    p.add_argument("--hard-negatives", type=int, default=32)
    p.add_argument("--ranking-weight", type=float, default=.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--max-words", type=int)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if not set(a.held_out_subjects).issubset(set(a.subjects)):
        p.error("held-out subjects must be included in subjects")
    return a


def zscore(values, train):
    mean = values[train].mean(0, dtype=np.float64).astype(np.float32)
    scale = values[train].std(0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.
    return ((values - mean) / scale).astype(np.float32)


def orthogonal_map(source, destination):
    """Return R minimizing ||source R - destination||_F."""
    u, _, vt = np.linalg.svd(source.T @ destination, full_matrices=False)
    return (u @ vt).astype(np.float32)


def generalized_procrustes(scores, subjects, train, iterations):
    """Fit a common template and subject rotations with matched training words."""
    template = scores[subjects[0]][train].copy()
    template /= np.maximum(np.linalg.norm(template, axis=1, keepdims=True), 1e-12)
    rotations = {}
    for _ in range(iterations):
        aligned = []
        for subject in subjects:
            rotations[subject] = orthogonal_map(scores[subject][train], template)
            aligned.append(scores[subject][train] @ rotations[subject])
        template = np.mean(aligned, axis=0)
        template /= np.maximum(np.linalg.norm(template, axis=1, keepdims=True), 1e-12)
    return {subject: (scores[subject] @ rotations[subject]).astype(np.float32) for subject in subjects}


def pca_project(raw, subjects, train, components, variant, ha_iterations, seed):
    """Create word x component x time-bin features from the medium response."""
    normalized = {s: zscore(raw[s], train) for s in subjects}
    n_words, _, n_bins = next(iter(normalized.values())).shape
    output = {s: np.empty((n_words, components, n_bins), np.float32) for s in subjects}
    for bin_index in range(n_bins):
        scores = {}
        for subject_index, subject in enumerate(subjects):
            values = normalized[subject][:, :, bin_index]
            pca = PCA(n_components=min(components, len(train) - 1, values.shape[1]),
                      svd_solver="randomized", random_state=seed + bin_index * 100 + subject_index)
            scores[subject] = pca.fit_transform(values[train])
            # Refit only on the outer-training words, then transform all words.
            scores[subject] = pca.transform(values).astype(np.float32)
        if variant == "pca_ha":
            scores = generalized_procrustes(scores, subjects, train, ha_iterations)
        for subject in subjects:
            output[subject][:, :, bin_index] = scores[subject]
    return output


def pack_medium(projected):
    packed = {}
    for subject, values in projected.items():
        result = np.zeros((len(values), len(WINDOWS), values.shape[1], values.shape[2]), np.float32)
        result[:, WINDOWS.index("medium")] = values
        packed[subject] = result
    return packed


def run_subject(held_value, words, raw_target, ecog, args, device):
    held = f"sub-{int(held_value):02d}"
    output = args.output_root / held / "summary.json"
    sources = [s for s in ecog["medium"] if s != held]
    all_subjects = sources + [held]
    labels = np.unique(words.word.astype(str), return_inverse=True)[1].astype(np.int64)
    if output.exists() and not args.overwrite:
        report = json.loads(output.read_text())
        # A later invocation may request an additional baseline (e.g., add
        # PCA+HA after PCA); retain completed folds and append only missing work.
        report.setdefault("variants", {})
        for variant in args.variants:
            report["variants"].setdefault(variant, {"folds": []})
    else:
        report = {"method": "matched PCA and PCA+hyperalignment retrieval baselines",
                  "protocol": {"PCA": "per-subject, per-bin PCA fitted on outer-train ECoG only",
                               "PCA+HA": "PCA followed by generalized orthogonal Procrustes hyperalignment using matched outer-train words only",
                               "decoder": "same capacity-matched medium-window TCN and combined retrieval loss as SRM baseline"},
                  "held_out_subject": held, "source_subjects": sources,
                  "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "variants": {variant: {"folds": []} for variant in args.variants}}
    for fold, (train, test) in enumerate(KFold(args.outer_folds, shuffle=False).split(words), 1):
        target = target_pca(raw_target, train, args.target_components, args.seed + fold)
        for variant in args.variants:
            completed_folds = {row["fold"] for row in report["variants"][variant]["folds"]}
            if fold in completed_folds:
                continue
            projected = pca_project(ecog["medium"], all_subjects, train, args.neural_components,
                                    variant, args.ha_iterations, args.seed + fold * 1000)
            projected = pack_medium(projected)
            model, epoch, _ = fit_decoder("matched_direct_combined", projected, target, labels, sources, train,
                                          args, device, args.seed + int(held_value) * 10000 + fold)
            held_metrics = metrics(predict(model, projected[held][test], 512, device), target[test],
                                   words.iloc[test].word.astype(str).to_numpy(), args.batch_size)
            source_rows = [metrics(predict(model, projected[s][test], 512, device), target[test],
                                   words.iloc[test].word.astype(str).to_numpy(), args.batch_size) for s in sources]
            report["variants"][variant]["folds"].append({"fold": fold, "best_epoch": int(epoch),
                "held_out": held_metrics, "source_mean": mean_metrics(source_rows)})
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"held={held} fold={fold}/{args.outer_folds} {variant} AUC={held_metrics['auc_roc']:.4f}", flush=True)
    for variant in args.variants:
        rows = report["variants"][variant]["folds"]
        if len(rows) != args.outer_folds:
            raise RuntimeError(f"{held} {variant}: expected {args.outer_folds} folds, found {len(rows)}")
        report["variants"][variant]["held_out_mean"] = mean_metrics([r["held_out"] for r in rows])
        report["variants"][variant]["source_mean"] = mean_metrics([r["source_mean"] for r in rows])
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    args = parse_args(); args.output_root.mkdir(parents=True, exist_ok=True); device = device_for(args.device)
    words, target, ecog = load_multiwindow_data(args.cache_root, args.subjects, args.max_words)
    reports = [run_subject(subject, words, target, ecog, args, device) for subject in args.held_out_subjects]
    aggregate = {"n_held_out_subjects": len(reports), "variants": {}}
    for variant in args.variants:
        aggregate["variants"][variant] = {section: mean_metrics([r["variants"][variant][section] for r in reports])
                                             for section in ("source_mean", "held_out_mean")}
    (args.output_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))

if __name__ == "__main__":
    main()
