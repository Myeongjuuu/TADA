#!/usr/bin/env python3
"""Prepare -1..+1 s high-gamma caches for the manuscript lag encoding figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["01", "03", "04", "05", "06", "07", "08", "09"],
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project / "data/source/ds005574",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=project / "data/source/ds005574/stimuli/podcast_transcript.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=project / "data/ecog_caches/wide_m1000_p1000",
    )
    parser.add_argument("--response-start-s", type=float, default=-1.0)
    parser.add_argument("--response-end-s", type=float, default=1.0)
    parser.add_argument("--temporal-bins", type=int, default=40)
    parser.add_argument("--epoch-batch-size", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_subject(value: str) -> str:
    return f"sub-{int(value):02d}"


def extract_word_responses(
    raw: mne.io.BaseRaw,
    onsets_s: np.ndarray,
    start_s: float,
    end_s: float,
    temporal_bins: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if end_s <= start_s:
        raise ValueError("response-end-s must be greater than response-start-s")
    if temporal_bins < 1:
        raise ValueError("temporal-bins must be at least one")

    events = np.zeros((len(onsets_s), 3), dtype=int)
    events[:, 0] = (onsets_s * float(raw.info["sfreq"])).astype(int)
    events[:, 2] = 1
    responses: list[np.ndarray] = []
    selections: list[np.ndarray] = []

    for first in range(0, len(events), batch_size):
        last = min(first + batch_size, len(events))
        print(f"  epoch batch {first}:{last}/{len(events)}", flush=True)
        epochs = mne.Epochs(
            raw,
            events[first:last],
            event_id=None,
            tmin=start_s,
            tmax=end_s,
            baseline=None,
            proj=None,
            preload=True,
            event_repeated="merge",
            reject_by_annotation=True,
            verbose=False,
        )
        if len(epochs) == 0:
            continue
        batch = epochs.get_data(copy=True)
        if temporal_bins > batch.shape[2]:
            raise ValueError("temporal-bins exceeds the number of samples in a window")
        edges = np.linspace(0, batch.shape[2], temporal_bins + 1, dtype=int)
        binned = np.stack(
            [
                batch[:, :, edges[index] : edges[index + 1]].mean(axis=2)
                for index in range(temporal_bins)
            ],
            axis=2,
        )
        responses.append(binned.astype(np.float32, copy=False))
        selections.append(epochs.selection + first)

    if not responses:
        raise RuntimeError("all word epochs were dropped")
    return np.concatenate(responses), np.concatenate(selections)


def prepare_subject(subject_arg: str, words: pd.DataFrame, args: argparse.Namespace) -> None:
    subject = normalize_subject(subject_arg)
    destination = args.cache_root / subject
    ecog_path = destination / "ecog.npy"
    words_path = destination / "words.tsv"
    metadata_path = destination / "metadata.json"
    if (
        not args.overwrite
        and ecog_path.exists()
        and words_path.exists()
        and metadata_path.exists()
    ):
        print(f"{subject}: cache already exists; skipping", flush=True)
        return

    fif = (
        args.source_root
        / "derivatives/ecogprep"
        / subject
        / "ieeg"
        / f"{subject}_task-podcast_desc-highgamma_ieeg.fif"
    )
    if not fif.exists():
        raise FileNotFoundError(f"official high-gamma file not found: {fif}")

    print(f"{subject}: reading {fif}", flush=True)
    raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
    raw.pick("ecog", exclude="bads")
    channel_names = list(raw.ch_names)
    channel_positions_m = [
        [float(value) for value in channel["loc"][:3]]
        for channel in raw.info["chs"]
    ]
    positions = np.asarray(channel_positions_m, dtype=float)
    if not np.isfinite(positions).all() or np.any(np.linalg.norm(positions, axis=1) == 0):
        raise ValueError(f"{subject}: missing or invalid electrode coordinates")
    ecog, selection = extract_word_responses(
        raw,
        words.start.to_numpy(dtype=float),
        args.response_start_s,
        args.response_end_s,
        args.temporal_bins,
        args.epoch_batch_size,
    )
    raw.close()
    selected_words = words.iloc[selection].reset_index(drop=True)
    finite = np.isfinite(ecog).all(axis=(1, 2))
    if not finite.all():
        selected_words = selected_words.loc[finite].reset_index(drop=True)
        ecog = ecog[finite]

    destination.mkdir(parents=True, exist_ok=True)
    np.save(ecog_path, ecog.astype(np.float32, copy=False))
    selected_words[["word_idx", "word", "start", "end"]].to_csv(
        words_path, sep="\t", index=False
    )
    metadata_path.write_text(
        json.dumps(
            {
                "subject": subject,
                "source": str(fif),
                "response_start_s": args.response_start_s,
                "response_end_s": args.response_end_s,
                "temporal_bins": args.temporal_bins,
                "channel_names": channel_names,
                "channel_positions_m": channel_positions_m,
                "shape": list(ecog.shape),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{subject}: cached ECoG={ecog.shape}", flush=True)


def main() -> None:
    args = parse_args()
    words = pd.read_csv(args.transcript)
    words.insert(0, "word_idx", np.arange(len(words)))
    required = {"word_idx", "word", "start", "end"}
    if not required.issubset(words.columns):
        raise ValueError(f"transcript must contain {sorted(required)}")
    words = words.sort_values("start", kind="stable").reset_index(drop=True)
    if np.any(np.diff(words.start.to_numpy(dtype=float)) < 0):
        raise RuntimeError("word onset sorting failed")
    for subject in args.subjects:
        prepare_subject(subject, words, args)


if __name__ == "__main__":
    main()
