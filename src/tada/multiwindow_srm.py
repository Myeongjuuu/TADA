"""Factorial test of real temporal windows and retrieval-aware SRM decoding."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .srm_cnn_baseline import (
    SUBJECTS,
    device_for,
    fit_srm,
    fit_temporally_consistent_srm,
    mean_metrics,
    metrics,
    objective,
    predict,
    set_seed,
    standardize_shared,
    target_pca,
    unit,
)
from .models import MultiWindowTemporalEncoder, TemporalConvEncoder
from .losses import hard_negative_loss, mine_hard_negatives


WINDOWS = ("short", "medium", "long")
VARIANTS = ("matched_direct_combined", "matched_long_combined", "multiwindow_combined")


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out-subjects", nargs="+", default=["03"])
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--cache-root", type=Path,
        default=project / "data/ecog_caches/heo2026_multiwindow_t10",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=project / "results/multiwindow_srm",
    )
    parser.add_argument(
        "--target-source", choices=("gpt2", "whisper-layer"), default="gpt2",
        help="Target family used for decoder training and retrieval evaluation.",
    )
    parser.add_argument(
        "--whisper-layer", default="decoder/layer_10",
        help="HDF5 key used when --target-source whisper-layer.",
    )
    parser.add_argument(
        "--whisper-features-dir", type=Path,
        default=project / "data/whisper_layer_features/whisper-small",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--srm-features", type=int, default=5)
    parser.add_argument("--srm-iterations", type=int, default=100)
    parser.add_argument(
        "--temporal-consistent-srm", action="store_true",
        help="Fit one SRM basis per window across all temporal bins, rather than one basis per bin.",
    )
    parser.add_argument("--target-components", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument(
        "--matched-direct-hidden-dim", type=int, default=113,
        help="Width matching the default multi-window model's parameter count",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument("--ranking-weight", type=float, default=0.1)
    parser.add_argument(
        "--neural-ridge-alpha", type=float, default=10.0,
        help="Ridge penalty for train-only text-to-SRM encoding used to mine neural hard negatives.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-words", type=int, help="Smoke test only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not set(args.held_out_subjects).issubset(set(args.subjects)):
        parser.error("every held-out subject must also appear in --subjects")
    if args.temperature <= 0 or args.hard_negatives < 1 or args.ranking_weight < 0:
        parser.error("retrieval-loss hyperparameters are invalid")
    return args


def load_multiwindow_data(
    cache_root: Path, subjects: list[str], max_words: int | None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    canonical = pd.read_csv(cache_root / "canonical_words.tsv", sep="\t")
    target = np.load(cache_root / "gpt2_xl_layer24.npy")
    raw: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray]]] = {}
    word_sets = []
    for value in subjects:
        subject = f"sub-{int(value):02d}"
        words = pd.read_csv(cache_root / subject / "words.tsv", sep="\t")
        arrays = {
            name: np.load(cache_root / subject / f"ecog_{name}.npy", mmap_mode="r")
            for name in WINDOWS
        }
        if any(len(array) != len(words) for array in arrays.values()):
            raise ValueError(f"{subject}: cache row mismatch")
        shapes = {array.shape[1:] for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(f"{subject}: all windows must share channel/bin dimensions")
        raw[subject] = (words, arrays)
        word_sets.append(set(words.word_idx.astype(int)))
    common = set(canonical.word_idx.astype(int)).intersection(*word_sets)
    words = canonical.loc[canonical.word_idx.astype(int).isin(common)].copy()
    words = words.sort_values("start", kind="stable").reset_index(drop=True)
    if max_words:
        words = words.iloc[:max_words].reset_index(drop=True)
    canonical_lookup = {int(value): index for index, value in enumerate(canonical.word_idx)}
    target = target[
        [canonical_lookup[int(value)] for value in words.word_idx]
    ].astype(np.float32)
    aligned = {name: {} for name in WINDOWS}
    for subject, (subject_words, arrays) in raw.items():
        lookup = {int(value): index for index, value in enumerate(subject_words.word_idx)}
        rows = [lookup[int(value)] for value in words.word_idx]
        for name in WINDOWS:
            aligned[name][subject] = np.asarray(arrays[name][rows], dtype=np.float32)
    return words, target, aligned


def load_whisper_layer_target(
    words: pd.DataFrame, directory: Path, layer: str,
) -> np.ndarray:
    """Align one frozen Whisper layer representation to canonical word rows."""
    layer_words = pd.read_csv(directory / "transcript.tsv", sep="\t")
    lookup = {int(value): row for row, value in enumerate(layer_words.word_idx)}
    missing = [int(value) for value in words.word_idx if int(value) not in lookup]
    if missing:
        raise ValueError(f"Whisper layer transcript is missing {len(missing)} canonical words")
    rows = np.asarray([lookup[int(value)] for value in words.word_idx], dtype=int)
    with h5py.File(directory / "features.hdf5", "r") as handle:
        if layer not in handle:
            raise KeyError(f"Whisper layer {layer!r} is unavailable")
        # Load before reordering because h5py fancy indices must be monotonic.
        values = handle[layer][...]
    return values[rows].astype(np.float32)


def project_windows(
    ecog: dict[str, dict[str, np.ndarray]], sources: list[str], held: str,
    train: np.ndarray, features: int, iterations: int,
    required_windows: tuple[str, ...] = WINDOWS,
    temporal_consistent: bool = False,
) -> dict[str, np.ndarray]:
    per_window = {}
    aligner = fit_temporally_consistent_srm if temporal_consistent else fit_srm
    for name in required_windows:
        print(f"  fitting SRM window={name}", flush=True)
        projected = aligner(ecog[name], sources, held, train, features, iterations)
        per_window[name] = standardize_shared(projected, sources, train)
    template_name = required_windows[0]
    return {
        subject: np.stack(
            [
                per_window[name][subject]
                if name in per_window
                else np.zeros_like(per_window[template_name][subject])
                for name in WINDOWS
            ],
            axis=1,
        ).astype(np.float32)
        for subject in ecog[WINDOWS[0]]
    }


def windows_needed(variants: list[str]) -> tuple[str, ...]:
    if any(variant.startswith("multiwindow_") for variant in variants):
        return WINDOWS
    needed = []
    if any("long" in variant for variant in variants):
        needed.append("long")
    if any("short" in variant for variant in variants):
        needed.append("short")
    if any("direct" in variant and "long" not in variant for variant in variants):
        needed.append("medium")
    if not needed:
        raise ValueError(f"cannot infer temporal windows for variants: {variants}")
    return tuple(name for name in WINDOWS if name in needed)


class SingleWindowDecoder(nn.Module):
    """Direct control that sees exactly one temporal window."""

    def __init__(
        self, channels: int, output_dim: int, hidden_dim: int,
        dropout: float, window_index: int,
    ):
        super().__init__()
        self.window_index = window_index
        self.decoder = TemporalConvEncoder(
            channels, output_dim, hidden_dim=hidden_dim, depth=3, dropout=dropout
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(values[:, self.window_index])


def build_decoder(
    variant: str, channels: int, output_dim: int,
    args: argparse.Namespace, device: torch.device,
) -> nn.Module:
    single_window = (
        variant.startswith("direct_")
        or variant.startswith("matched_direct_")
        or variant.startswith("short_")
        or variant.startswith("matched_short_")
        or variant.startswith("long_")
        or variant.startswith("matched_long_")
    )
    if single_window:
        hidden_dim = (
            args.matched_direct_hidden_dim
            if variant.startswith("matched_") else args.hidden_dim
        )
        window_name = (
            "short" if "short" in variant else
            "long" if "long" in variant else "medium"
        )
        model = SingleWindowDecoder(
            channels, output_dim, hidden_dim, args.dropout,
            window_index=WINDOWS.index(window_name),
        )
    else:
        model = MultiWindowTemporalEncoder(
            channels, output_dim, n_windows=len(WINDOWS),
            hidden_dim=args.hidden_dim, depth=3, dropout=args.dropout,
        )
    return model.to(device)


def mine_neural_hard_negatives(
    target: np.ndarray, projected: dict[str, np.ndarray], sources: list[str],
    indices: np.ndarray, labels: np.ndarray, count: int, ridge_alpha: float,
) -> np.ndarray:
    """Mine wrong transcript tokens with similar *encoded* SRM trajectories.

    The ridge map and all candidate similarities are fitted exclusively on the
    outer-training source-participant data.  Thus it supplies a neural notion
    of confusability without touching held-out data or labels.
    """
    source_mean = np.mean(
        [projected[subject][indices].reshape(len(indices), -1) for subject in sources],
        axis=0,
    ).astype(np.float32)
    encoder = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(target[indices], source_mean)
    expected = encoder.predict(target[indices]).astype(np.float32)
    similarity = unit(expected) @ unit(expected).T
    local_labels = labels[indices]
    similarity[local_labels[:, None] == local_labels[None, :]] = -np.inf
    count = min(count, max(1, len(indices) - 1))
    partition = np.argpartition(similarity, -count, axis=1)[:, -count:]
    values = np.take_along_axis(similarity, partition, axis=1)
    order = np.argsort(values, axis=1)[:, ::-1]
    return np.take_along_axis(partition, order, axis=1).astype(np.int64)


def validation_score(
    model: nn.Module, projected: dict[str, np.ndarray], target: np.ndarray,
    sources: list[str], validation: np.ndarray, labels: np.ndarray,
    variant: str, batch_size: int, device: torch.device,
) -> tuple[float, str]:
    if variant.endswith("_point"):
        x = np.concatenate([projected[subject][validation] for subject in sources])
        y = np.concatenate([target[validation] for _ in sources])
        prediction = predict(model, x, 512, device)
        return float(np.mean(np.sum(unit(prediction) * unit(y), axis=1))), "validation_cosine"
    scores = []
    for subject in sources:
        row = metrics(
            predict(model, projected[subject][validation], 512, device),
            target[validation], labels[validation], batch_size,
        )
        scores.append(row["mrr"])
    return float(np.mean(scores)), "validation_mrr"


def fit_decoder(
    variant: str, projected: dict[str, np.ndarray], target: np.ndarray,
    word_labels: np.ndarray, sources: list[str], train: np.ndarray,
    args: argparse.Namespace, device: torch.device, seed: int,
) -> tuple[nn.Module, int, list[dict[str, float]]]:
    validation_count = max(1, round(len(train) * args.validation_fraction))
    core, validation = train[:-validation_count], train[-validation_count:]
    train_x = np.concatenate([projected[subject][core] for subject in sources])
    train_y = np.concatenate([target[core] for _ in sources])
    local_index = np.tile(np.arange(len(core), dtype=np.int64), len(sources))
    hard_index = mine_hard_negatives(target, core, word_labels, args.hard_negatives)
    neural_hard_index = None
    if variant == "multiwindow_neural_hard":
        neural_hard_index = mine_neural_hard_negatives(
            target, projected, sources, core, word_labels, args.hard_negatives,
            args.neural_ridge_alpha,
        )
    target_bank = torch.from_numpy(target[core]).to(device)
    hard_index_tensor = torch.from_numpy(hard_index).to(device)
    neural_hard_index_tensor = (
        torch.from_numpy(neural_hard_index).to(device)
        if neural_hard_index is not None else None
    )

    set_seed(seed)
    model = build_decoder(
        variant, train_x.shape[2], target.shape[1], args, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x), torch.from_numpy(train_y),
            torch.from_numpy(local_index),
        ),
        batch_size=args.batch_size, shuffle=True, generator=generator,
    )
    best_state, best_epoch, best_score, stale = None, 1, -np.inf, 0
    history: list[dict[str, float]] = []
    combined = variant.endswith("_combined") or variant == "multiwindow_neural_hard"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, point_losses, ranking_losses = [], [], []
        for batch_x, batch_y, batch_local in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_local = batch_local.to(device)
            prediction = model(batch_x)
            point = objective(prediction, batch_y)
            if combined:
                negatives = target_bank[hard_index_tensor[batch_local]]
                if neural_hard_index_tensor is not None:
                    neural_negatives = target_bank[neural_hard_index_tensor[batch_local]]
                    negatives = torch.cat((negatives, neural_negatives), dim=1)
                ranking = hard_negative_loss(
                    prediction, batch_y, negatives, args.temperature
                )
                loss = point + args.ranking_weight * ranking
            else:
                ranking = prediction.new_zeros(())
                loss = point
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            point_losses.append(float(point.detach().cpu()))
            ranking_losses.append(float(ranking.detach().cpu()))
        score, selection_name = validation_score(
            model, projected, target, sources, validation, word_labels,
            variant, args.batch_size, device,
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "point_loss": float(np.mean(point_losses)),
            "ranking_loss": float(np.mean(ranking_losses)),
            selection_name: score,
        })
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch} loss={np.mean(losses):.5f} "
                f"point={np.mean(point_losses):.5f} rank={np.mean(ranking_losses):.5f} "
                f"{selection_name}={score:.5f} best={best_epoch}", flush=True,
            )
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("decoder training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, history


@torch.no_grad()
def mean_window_weights(
    model: nn.Module, values: np.ndarray, device: torch.device,
) -> dict[str, float] | None:
    if not isinstance(model, MultiWindowTemporalEncoder):
        return None
    model.eval()
    rows = []
    for first in range(0, len(values), 512):
        batch = torch.from_numpy(values[first : first + 512]).to(device)
        _, details = model(batch, return_details=True)
        rows.append(details["window_weights"].cpu().numpy())
    average = np.concatenate(rows).mean(axis=0)
    return {name: float(average[index]) for index, name in enumerate(WINDOWS)}


def run_held_out(
    held_value: str, words: pd.DataFrame, raw_target: np.ndarray,
    ecog: dict[str, dict[str, np.ndarray]], args: argparse.Namespace,
    device: torch.device,
) -> None:
    held = f"sub-{int(held_value):02d}"
    sources = [subject for subject in ecog[WINDOWS[0]] if subject != held]
    output = args.output_root / held / "summary.json"
    if output.exists() and not args.overwrite:
        report = json.loads(output.read_text(encoding="utf-8"))
        for variant in args.variants:
            report["variants"].setdefault(variant, {"folds": []})
    else:
        report = {
            "method": "Multi-window retrieval-aware SRM decoder",
            "protocol": {
                "alignment": (
                    "train-only BrainIAK SRM with one basis shared across temporal bins"
                    if args.temporal_consistent_srm else
                    "train-only BrainIAK SRM fitted independently for each temporal bin"
                ),
                "windows": list(WINDOWS),
                "factorial_design": "direct versus multi-window crossed with point versus combined retrieval loss",
                "hard_negative_mining": "outer-train semantic targets only",
                "test_used_for_training_or_selection": False,
            },
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "held_out_subject": held,
            "source_subjects": sources,
            "n_words": len(words),
            "variants": {variant: {"folds": []} for variant in args.variants},
        }
    labels = pd.factorize(words["word"].astype(str), sort=True)[0].astype(np.int64)
    splits = list(KFold(args.outer_folds, shuffle=False).split(words))
    for fold_index, (train, test) in enumerate(splits, 1):
        needed = [
            variant for variant in args.variants
            if len(report["variants"][variant]["folds"]) < fold_index
        ]
        if not needed:
            continue
        seed = args.seed + int(held_value) * 1000 + fold_index
        print(f"===== held={held} fold={fold_index}/{args.outer_folds} =====", flush=True)
        projected = project_windows(
            ecog, sources, held, train, args.srm_features, args.srm_iterations,
            required_windows=windows_needed(needed),
            temporal_consistent=args.temporal_consistent_srm,
        )
        target = target_pca(raw_target, train, args.target_components, seed)
        test_labels = words.iloc[test]["word"].astype(str).to_numpy()
        for variant in needed:
            print(f"----- decoder={variant} -----", flush=True)
            model, best_epoch, history = fit_decoder(
                variant, projected, target, labels, sources, train,
                args, device, seed,
            )
            held_metrics = metrics(
                predict(model, projected[held][test], 512, device),
                target[test], test_labels, args.batch_size,
            )
            source_rows = [
                metrics(
                    predict(model, projected[subject][test], 512, device),
                    target[test], test_labels, args.batch_size,
                )
                for subject in sources
            ]
            row = {
                "fold": fold_index,
                "n_train": len(train),
                "n_test": len(test),
                "parameter_count": int(sum(p.numel() for p in model.parameters())),
                "best_epoch": best_epoch,
                "validation_history": history,
                "held_out": held_metrics,
                "source_mean": mean_metrics(source_rows),
                "held_out_mean_window_weights": mean_window_weights(
                    model, projected[held][test], device
                ),
            }
            report["variants"][variant]["folds"].append(row)
            print(json.dumps({
                "variant": variant,
                "held_out": held_metrics,
                "window_weights": row["held_out_mean_window_weights"],
                "parameters": row["parameter_count"],
                "best_epoch": best_epoch,
            }), flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for variant in args.variants:
        folds = report["variants"][variant]["folds"]
        if len(folds) == len(splits):
            report["variants"][variant]["held_out_mean"] = mean_metrics(
                [row["held_out"] for row in folds]
            )
            report["variants"][variant]["source_mean"] = mean_metrics(
                [row["source_mean"] for row in folds]
            )
            weights = [
                row["held_out_mean_window_weights"] for row in folds
                if row["held_out_mean_window_weights"] is not None
            ]
            if weights:
                report["variants"][variant]["mean_window_weights"] = {
                    name: float(np.mean([row[name] for row in weights]))
                    for name in WINDOWS
                }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}", flush=True)


def main() -> None:
    args = parse_args()
    device = device_for(args.device)
    words, target, ecog = load_multiwindow_data(
        args.cache_root, args.subjects, args.max_words
    )
    if args.target_source == "whisper-layer":
        target = load_whisper_layer_target(
            words, args.whisper_features_dir, args.whisper_layer
        )
        print(f"target=Whisper {args.whisper_layer}", flush=True)
    else:
        print("target=GPT-2 XL layer-24", flush=True)
    print(
        f"subjects={list(ecog[WINDOWS[0]])} words={len(words)} "
        f"windows={list(WINDOWS)} target={target.shape[1]} device={device}",
        flush=True,
    )
    for held in args.held_out_subjects:
        run_held_out(held, words, target, ecog, args, device)


if __name__ == "__main__":
    main()
