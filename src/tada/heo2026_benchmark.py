"""SRM-CNN baseline and shared cross-subject evaluation utilities.

The SRM stage follows the public Bhattacharjee et al. implementation:
BrainIAK SRM, five shared features, and subject-specific projection of a
held-out participant. The direct temporal CNN is the paper's SRM-CNN baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import TemporalConvEncoder


SUBJECTS = ("01", "03", "04", "05", "06", "07", "08", "09")
VARIANTS = ("direct",)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out-subjects", nargs="+", default=["03"])
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--cache-root", type=Path, default=project / "data/ecog_caches/heo2026_625ms_sw200_t10")
    parser.add_argument("--output-root", type=Path, default=project / "results/heo2026_benchmark")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--srm-features", type=int, default=5)
    parser.add_argument("--srm-iterations", type=int, default=100)
    parser.add_argument("--target-components", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-words", type=int, help="Smoke test only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not set(args.held_out_subjects).issubset(set(args.subjects)):
        parser.error("every held-out subject must also appear in --subjects")
    if len(args.subjects) < 3:
        parser.error("cross-subject SRM requires at least three subjects")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def load_data(cache_root: Path, subjects: list[str], max_words: int | None) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    canonical = pd.read_csv(cache_root / "canonical_words.tsv", sep="\t")
    target = np.load(cache_root / "gpt2_xl_layer24.npy")
    if len(canonical) != len(target):
        raise ValueError("canonical words and GPT-2 targets differ in length")
    word_sets, raw = [], {}
    for value in subjects:
        subject = f"sub-{int(value):02d}"
        words = pd.read_csv(cache_root / subject / "words.tsv", sep="\t")
        values = np.load(cache_root / subject / "ecog.npy", mmap_mode="r")
        if len(words) != len(values):
            raise ValueError(f"{subject}: cache row mismatch")
        raw[subject] = (words, values)
        word_sets.append(set(words.word_idx.astype(int)))
    common = set(canonical.word_idx.astype(int)).intersection(*word_sets)
    words = canonical.loc[canonical.word_idx.astype(int).isin(common)].copy()
    words = words.sort_values("start", kind="stable").reset_index(drop=True)
    if max_words:
        words = words.iloc[:max_words].reset_index(drop=True)
    canonical_lookup = {int(v): i for i, v in enumerate(canonical.word_idx)}
    target = target[[canonical_lookup[int(v)] for v in words.word_idx]].astype(np.float32)
    aligned = {}
    for subject, (subject_words, values) in raw.items():
        lookup = {int(v): i for i, v in enumerate(subject_words.word_idx)}
        aligned[subject] = np.asarray(values[[lookup[int(v)] for v in words.word_idx]], dtype=np.float32)
    return words, target, aligned


def fit_srm(
    ecog: dict[str, np.ndarray], source_subjects: list[str], held_out: str,
    train: np.ndarray, features: int, iterations: int,
) -> dict[str, np.ndarray]:
    try:
        from brainiak.funcalign.srm import SRM
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "This benchmark requires BrainIAK and an MPI runtime. In the active "
            "WSL environment run: python -m pip install brainiak mpi4py mpich; "
            "then verify with: python -c \"from mpi4py import MPI; from "
            "brainiak.funcalign.srm import SRM; print(MPI.Get_library_version())\""
        ) from exc
    # The released high-gamma FIF files are not amplitude-compatible across
    # participants (sub-01 is stored on a scale roughly 10^6 larger than the
    # others).  The cited public SRM-Encoding implementation z-scores every
    # electrode at every temporal bin.  Fit those statistics on the outer
    # training words only so the test split cannot influence preprocessing.
    normalized = {}
    for subject, values in ecog.items():
        mean = values[train].mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values[train].std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-12] = 1.0
        normalized[subject] = ((values - mean[None]) / scale[None]).astype(np.float32)
    n_words = next(iter(normalized.values())).shape[0]
    n_bins = next(iter(normalized.values())).shape[2]
    projected = {subject: np.empty((n_words, features, n_bins), np.float32) for subject in ecog}
    for time_bin in range(n_bins):
        train_data = [normalized[s][train, :, time_bin].T for s in source_subjects]
        model = SRM(n_iter=iterations, features=features)
        model.fit(train_data)
        for index, subject in enumerate(source_subjects):
            projected[subject][:, :, time_bin] = (
                model.w_[index].T @ normalized[subject][:, :, time_bin].T
            ).T
        held_train = normalized[held_out][train, :, time_bin].T
        held_basis = model.transform_subject(held_train)
        projected[held_out][:, :, time_bin] = (
            held_basis.T @ normalized[held_out][:, :, time_bin].T
        ).T
    return projected


def fit_temporally_consistent_srm(
    ecog: dict[str, np.ndarray], source_subjects: list[str], held_out: str,
    train: np.ndarray, features: int, iterations: int,
) -> dict[str, np.ndarray]:
    """Fit one SRM projection per temporal window, shared by all its bins.

    The standard benchmark fits a distinct SRM for every temporal bin.  That
    makes component axes independently rotatable across bins, even though the
    downstream TCN treats a component index as a temporally continuous input
    channel.  Here all word--bin samples of a window are concatenated before
    SRM fitting, yielding one subject basis that is consistent over time.
    """
    try:
        from brainiak.funcalign.srm import SRM
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("This benchmark requires BrainIAK and an MPI runtime.") from exc
    normalized = {}
    for subject, values in ecog.items():
        mean = values[train].mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values[train].std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-12] = 1.0
        normalized[subject] = ((values - mean[None]) / scale[None]).astype(np.float32)
    n_words = next(iter(normalized.values())).shape[0]
    n_bins = next(iter(normalized.values())).shape[2]

    def flatten(values: np.ndarray) -> np.ndarray:
        # [word, electrode, bin] -> [electrode, word * bin]
        return values.transpose(1, 0, 2).reshape(values.shape[1], -1)

    model = SRM(n_iter=iterations, features=features)
    model.fit([flatten(normalized[s][train]) for s in source_subjects])
    projected = {subject: np.empty((n_words, features, n_bins), np.float32) for subject in ecog}
    for index, subject in enumerate(source_subjects):
        scores = model.w_[index].T @ flatten(normalized[subject])
        projected[subject] = scores.reshape(features, n_words, n_bins).transpose(1, 0, 2).astype(np.float32)
    held_basis = model.transform_subject(flatten(normalized[held_out][train]))
    held_scores = held_basis.T @ flatten(normalized[held_out])
    projected[held_out] = held_scores.reshape(features, n_words, n_bins).transpose(1, 0, 2).astype(np.float32)
    return projected


def target_pca(target: np.ndarray, train: np.ndarray, components: int, seed: int) -> np.ndarray:
    count = min(components, len(train) - 1, target.shape[1])
    model = PCA(n_components=count, svd_solver="randomized", random_state=seed)
    model.fit(target[train])
    transformed = model.transform(target).astype(np.float32)
    # Match the cited public implementation: PCA scores are L2-normalized per
    # word before optimizing the MSE + cosine objective.
    return transformed / np.maximum(
        np.linalg.norm(transformed, axis=1, keepdims=True), 1e-12
    )


def standardize_shared(projected: dict[str, np.ndarray], sources: list[str], train: np.ndarray) -> dict[str, np.ndarray]:
    pooled = np.concatenate([projected[s][train] for s in sources], axis=0)
    mean = pooled.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = pooled.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return {s: ((x - mean[None]) / scale[None]).astype(np.float32) for s, x in projected.items()}


def build_model(variant: str, channels: int, bins: int, output_dim: int, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if variant != "direct":
        raise ValueError(f"Unknown SRM-CNN baseline: {variant}")
    return TemporalConvEncoder(channels, output_dim, hidden_dim=args.hidden_dim, depth=3, dropout=args.dropout).to(device)


def objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
    return 0.7 * mse + 0.3 * cosine


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict(model: nn.Module, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    result = []
    for first in range(0, len(values), batch_size):
        batch = torch.from_numpy(values[first:first + batch_size]).to(device)
        result.append(model(batch).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def fit_decoder(
    variant: str, projected: dict[str, np.ndarray], target: np.ndarray,
    sources: list[str], train: np.ndarray, args: argparse.Namespace,
    device: torch.device, seed: int,
) -> tuple[nn.Module, int, list[dict[str, float]]]:
    val_count = max(1, round(len(train) * args.validation_fraction))
    core, validation = train[:-val_count], train[-val_count:]
    train_x = np.concatenate([projected[s][core] for s in sources])
    train_y = np.concatenate([target[core] for _ in sources])
    val_x = np.concatenate([projected[s][validation] for s in sources])
    val_y = np.concatenate([target[validation] for _ in sources])
    set_seed(seed)
    model = build_model(variant, train_x.shape[1], train_x.shape[2], target.shape[1], args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = make_loader(train_x, train_y, args.batch_size, True, seed)
    best_state, best_epoch, best_cosine, stale, history = None, 1, -np.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = objective(model(batch_x), batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_prediction = predict(model, val_x, 512, device)
        val_cosine = float(np.mean(np.sum(unit(val_prediction) * unit(val_y), axis=1)))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_cosine": val_cosine})
        if val_cosine > best_cosine:
            best_cosine, best_epoch, stale = val_cosine, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch={epoch} loss={np.mean(losses):.5f} val_cos={val_cosine:.5f} best={best_epoch}", flush=True)
        if stale >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_epoch, history


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    word_labels: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    """Implement the paper's class-based AUC/Top-k and batched pairwise score.

    A class is an exact transcript token.  Its score is the mean cosine
    similarity to every contextual target occurrence of that token in the test
    split.  Pairwise accuracy instead follows the paper's mini-batch protocol
    and compares each prediction with its matched versus all unmatched target
    occurrences in that evaluation mini-batch.
    """
    scores = unit(prediction) @ unit(target).T
    labels = np.asarray(word_labels, dtype=str)
    classes, class_index = np.unique(labels, return_inverse=True)
    class_scores = np.column_stack([
        scores[:, class_index == index].mean(axis=1)
        for index in range(len(classes))
    ])
    auc_rows = [
        roc_auc_score((class_index == index).astype(np.int8), class_scores[:, index])
        for index in range(len(classes))
        if np.any(class_index == index) and np.any(class_index != index)
    ]
    correct_scores = class_scores[np.arange(len(class_scores)), class_index]
    ranks = 1 + np.sum(class_scores > correct_scores[:, None], axis=1)
    wins = ties = comparisons = 0
    for first in range(0, len(scores), batch_size):
        block = scores[first:first + batch_size, first:first + batch_size]
        positive = np.diag(block)
        block_mask = ~np.eye(len(block), dtype=bool)
        negative = block[block_mask].reshape(len(block), len(block) - 1)
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


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ("auc_roc", "pairwise_accuracy", "top_10", "top_50", "mrr")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def run_held_out(held_value: str, words: pd.DataFrame, raw_target: np.ndarray, ecog: dict[str, np.ndarray], args: argparse.Namespace, device: torch.device) -> None:
    held = f"sub-{int(held_value):02d}"
    sources = [s for s in ecog if s != held]
    output = args.output_root / held / "summary.json"
    if output.exists() and not args.overwrite:
        report = json.loads(output.read_text(encoding="utf-8"))
    else:
        report = {
            "protocol": {
                "reference": "Heo et al. 2026 cross-subject semantic decoding",
                "srm_reference_implementation": "Bhattacharjee et al. SRM-Encoding; BrainIAK SRM",
                "published_reference": {
                    "source_pairwise_accuracy": 0.667,
                    "held_out_pairwise_accuracy": 0.656,
                    "source_top_10": 0.055, "held_out_top_10": 0.051,
                    "source_top_50": 0.222, "held_out_top_50": 0.209,
                },
                "decoder": "direct temporal CNN baseline on SRM-aligned medium-window responses",
                "known_ambiguity": "Heo et al. do not report SRM feature count or complete CNN layer specification; k=5 follows the cited public SRM implementation",
                "participant_note": "the cited released 183-electrode list maps eight participants and excludes public sub-02, despite the decoding paper stating nine participants",
                "metric_definition": "AUC/Top-k use exact transcript-token word classes; pairwise accuracy uses all non-matches within sequential evaluation mini-batches",
                "normalization": "outer-train-only z-score per subject/electrode/temporal-bin before SRM; train-fitted PCA followed by per-word L2 normalization for GPT-2 targets",
            },
            "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "held_out_subject": held,
            "source_subjects": sources,
            "n_words": len(words),
            "variants": {},
        }
    splits = list(KFold(args.outer_folds, shuffle=False).split(words))
    for variant in args.variants:
        report["variants"].setdefault(variant, {"folds": []})
    complete = {
        variant for variant in args.variants
        if len(report["variants"][variant].get("folds", [])) >= len(splits)
    }
    if complete == set(args.variants):
        print(f"{held}: requested variants are complete; skipping", flush=True)
        return
    for fold_index, (train, test) in enumerate(splits, 1):
        needed = [
            variant for variant in args.variants
            if len(report["variants"][variant]["folds"]) < fold_index
        ]
        if not needed:
            continue
        seed = args.seed + int(held_value) * 1000 + fold_index
        print(f"===== held={held} SRM fold={fold_index}/{args.outer_folds} =====", flush=True)
        projected = fit_srm(ecog, sources, held, train, args.srm_features, args.srm_iterations)
        projected = standardize_shared(projected, sources, train)
        target = target_pca(raw_target, train, args.target_components, seed)
        for variant in needed:
            print(f"----- decoder={variant} -----", flush=True)
            model, best_epoch, history = fit_decoder(variant, projected, target, sources, train, args, device, seed)
            test_labels = words.iloc[test]["word"].astype(str).to_numpy()
            held_metrics = metrics(
                predict(model, projected[held][test], 512, device),
                target[test], test_labels, args.batch_size,
            )
            source_rows = [
                metrics(
                    predict(model, projected[s][test], 512, device),
                    target[test], test_labels, args.batch_size,
                )
                for s in sources
            ]
            source_metrics = mean_metrics(source_rows)
            print(json.dumps({"held_out": held_metrics, "source_mean": source_metrics, "best_epoch": best_epoch}), flush=True)
            report["variants"][variant]["folds"].append({"fold": fold_index, "n_train": len(train), "n_test": len(test), "best_epoch": best_epoch, "validation_history": history, "held_out": held_metrics, "source_subjects": dict(zip(sources, source_rows)), "source_mean": source_metrics})
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for variant in args.variants:
        folds = report["variants"][variant]["folds"]
        report["variants"][variant]["held_out_mean"] = mean_metrics([row["held_out"] for row in folds])
        report["variants"][variant]["source_mean"] = mean_metrics([row["source_mean"] for row in folds])
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}", flush=True)


def main() -> None:
    args = parse_args()
    device = device_for(args.device)
    words, target, ecog = load_data(args.cache_root, args.subjects, args.max_words)
    print(f"subjects={list(ecog)} words={len(words)} target={target.shape[1]} device={device}", flush=True)
    for held in args.held_out_subjects:
        run_held_out(held, words, target, ecog, args, device)


if __name__ == "__main__":
    main()
