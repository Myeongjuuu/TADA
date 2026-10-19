#!/usr/bin/env python3
"""Evaluate encoding-guided reciprocal re-ranking (EGR).

The base decoder predicts a contextual text embedding from ECoG.  EGR adds a
second, train-only check: a ridge encoder predicts the expected SRM trajectory
for every candidate text embedding.  Candidate rankings combine the normal
brain-to-text cosine score and agreement with that expected neural trajectory.
All ridge strength and fusion-weight selection uses source-subject validation
data only; the held-out participant is never used for model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tada.heo2026_benchmark import (  # noqa: E402
    SUBJECTS, device_for, mean_metrics, predict, target_pca, unit,
)
from tada.multiwindow_srm import (  # noqa: E402
    WINDOWS, fit_decoder, load_multiwindow_data, project_windows,
)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out-subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--cache-root", type=Path,
                        default=project / "data/ecog_caches/heo2026_multiwindow_t10")
    parser.add_argument("--output-root", type=Path,
                        default=project / "results/encoding_guided_rerank")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--srm-features", type=int, default=5)
    parser.add_argument("--srm-iterations", type=int, default=100)
    parser.add_argument("--target-components", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument("--ranking-weight", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--ridge-alphas", nargs="+", type=float,
                        default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--fusion-weights", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-words", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not set(args.held_out_subjects).issubset(set(args.subjects)):
        parser.error("every held-out subject must be in --subjects")
    if any(value < 0 for value in args.fusion_weights):
        parser.error("fusion weights must be non-negative")
    return args


def metrics_from_scores(scores: np.ndarray, labels: np.ndarray, batch_size: int) -> dict[str, float]:
    """Same token-class metrics as the main benchmark, from any score matrix."""
    labels = np.asarray(labels, dtype=str)
    classes, class_index = np.unique(labels, return_inverse=True)
    class_scores = np.column_stack([
        scores[:, class_index == index].mean(axis=1)
        for index in range(len(classes))
    ])
    correct = class_scores[np.arange(len(labels)), class_index]
    ranks = 1 + np.sum(class_scores > correct[:, None], axis=1)
    auc_rows = []
    for index in range(len(classes)):
        positives = class_index == index
        if positives.any() and (~positives).any():
            # Rank-based binary AUC without a second sklearn dependency here.
            pos, neg = class_scores[positives, index], class_scores[~positives, index]
            auc_rows.append(np.mean(pos[:, None] > neg[None, :]) +
                            0.5 * np.mean(pos[:, None] == neg[None, :]))
    wins = ties = comparisons = 0
    for first in range(0, len(scores), batch_size):
        block = scores[first:first + batch_size, first:first + batch_size]
        positive = np.diag(block)
        mask = ~np.eye(len(block), dtype=bool)
        negative = block[mask].reshape(len(block), len(block) - 1)
        wins += int(np.sum(positive[:, None] > negative))
        ties += int(np.sum(positive[:, None] == negative))
        comparisons += negative.size
    return {
        "auc_roc": float(np.mean(auc_rows)),
        "pairwise_accuracy": float((wins + 0.5 * ties) / comparisons),
        "top_10": float(np.mean(ranks <= 10)),
        "top_50": float(np.mean(ranks <= 50)),
        "mrr": float(np.mean(1.0 / ranks)),
        "n_queries": int(len(ranks)),
    }


def trajectory(values: np.ndarray) -> np.ndarray:
    """Flatten [word, window, bin, SRM-feature] while preserving temporal bins."""
    return values.reshape(len(values), -1).astype(np.float32, copy=False)


def score_matrix(predicted_text: np.ndarray, observed: np.ndarray,
                 candidate_text: np.ndarray, candidate_trajectory: np.ndarray,
                 weight: float) -> np.ndarray:
    semantic = unit(predicted_text) @ unit(candidate_text).T
    if weight == 0:
        return semantic
    neural = unit(observed) @ unit(candidate_trajectory).T
    return semantic + weight * neural


def fit_forward_encoder(target: np.ndarray, projected: dict[str, np.ndarray],
                        sources: list[str], train: np.ndarray, alpha: float) -> Ridge:
    x = np.concatenate([target[train] for _ in sources])
    y = np.concatenate([trajectory(projected[subject][train]) for subject in sources])
    return Ridge(alpha=alpha, fit_intercept=True).fit(x, y)


def choose_hyperparameters(model, projected, target, sources, core, validation,
                           labels, args, device):
    validation_prediction = {
        subject: predict(model, projected[subject][validation], 512, device)
        for subject in sources
    }
    best = (-np.inf, args.ridge_alphas[0], args.fusion_weights[0])
    for alpha in args.ridge_alphas:
        forward = fit_forward_encoder(target, projected, sources, core, alpha)
        expected = forward.predict(target[validation]).astype(np.float32)
        for weight in args.fusion_weights:
            rows = [metrics_from_scores(
                score_matrix(validation_prediction[subject], trajectory(projected[subject][validation]),
                             target[validation], expected, weight),
                labels[validation], args.batch_size,
            ) for subject in sources]
            score = float(np.mean([row["mrr"] for row in rows]))
            if score > best[0]:
                best = (score, alpha, weight)
    return best


def run_subject(held_value, words, raw_target, ecog, args, device) -> None:
    held = f"sub-{int(held_value):02d}"
    sources = [name for name in ecog[WINDOWS[0]] if name != held]
    output = args.output_root / held / "summary.json"
    report = {"method": "Encoding-guided reciprocal re-ranking (EGR)",
              "protocol": {"base_decoder": "multi-window combined retrieval loss",
                           "forward_encoder": "source-trained ridge: text embedding to flattened SRM trajectory",
                           "selection": "ridge alpha and fusion weight selected on source validation only",
                           "held_out_used_for_training_or_selection": False},
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()},
              "held_out_subject": held, "source_subjects": sources,
              "n_words": len(words), "folds": []}
    if output.exists() and not args.overwrite:
        report = json.loads(output.read_text(encoding="utf-8"))
    labels = np.asarray(words["word"].astype(str))
    splits = list(KFold(args.outer_folds, shuffle=False).split(words))
    for fold_index, (train, test) in enumerate(splits, 1):
        if len(report["folds"]) >= fold_index:
            continue
        seed = args.seed + int(held_value) * 1000 + fold_index
        print(f"===== held={held} fold={fold_index}/{args.outer_folds} =====", flush=True)
        projected = project_windows(ecog, sources, held, train, args.srm_features,
                                    args.srm_iterations, required_windows=WINDOWS)
        target = target_pca(raw_target, train, args.target_components, seed)
        model, best_epoch, history = fit_decoder("multiwindow_combined", projected, target,
                                                  np.unique(labels, return_inverse=True)[1].astype(np.int64),
                                                  sources, train, args, device, seed)
        validation_count = max(1, round(len(train) * args.validation_fraction))
        core, validation = train[:-validation_count], train[-validation_count:]
        validation_mrr, alpha, weight = choose_hyperparameters(
            model, projected, target, sources, core, validation, labels, args, device)
        forward = fit_forward_encoder(target, projected, sources, train, alpha)
        expected = forward.predict(target[test]).astype(np.float32)
        rows = {}
        for subject in [held] + sources:
            predicted = predict(model, projected[subject][test], 512, device)
            observed = trajectory(projected[subject][test])
            base_scores = score_matrix(predicted, observed, target[test], expected, 0.0)
            egr_scores = score_matrix(predicted, observed, target[test], expected, weight)
            rows[subject] = {"base": metrics_from_scores(base_scores, labels[test], args.batch_size),
                             "egr": metrics_from_scores(egr_scores, labels[test], args.batch_size)}
        row = {"fold": fold_index, "n_train": len(train), "n_test": len(test),
               "best_epoch": best_epoch, "validation_history": history,
               "source_validation_mrr": validation_mrr, "ridge_alpha": alpha,
               "fusion_weight": weight, "held_out_base": rows[held]["base"],
               "held_out_egr": rows[held]["egr"],
               "source_base": mean_metrics([rows[name]["base"] for name in sources]),
               "source_egr": mean_metrics([rows[name]["egr"] for name in sources])}
        report["folds"].append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"held_base": row["held_out_base"], "held_egr": row["held_out_egr"],
                          "alpha": alpha, "weight": weight}, indent=2), flush=True)
        del model
        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
    if len(report["folds"]) == len(splits):
        report["held_out_base_mean"] = mean_metrics([row["held_out_base"] for row in report["folds"]])
        report["held_out_egr_mean"] = mean_metrics([row["held_out_egr"] for row in report["folds"]])
        report["source_base_mean"] = mean_metrics([row["source_base"] for row in report["folds"]])
        report["source_egr_mean"] = mean_metrics([row["source_egr"] for row in report["folds"]])
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}", flush=True)


def main() -> None:
    args = parse_args()
    device = device_for(args.device)
    words, raw_target, ecog = load_multiwindow_data(args.cache_root, args.subjects, args.max_words)
    print(f"subjects={list(ecog[WINDOWS[0]])} words={len(words)} device={device}", flush=True)
    for subject in args.held_out_subjects:
        run_subject(subject, words, raw_target, ecog, args, device)


if __name__ == "__main__":
    main()
