#!/usr/bin/env python3
"""Prepare the word-locked ECoG representation described by Heo et al. (2026)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    source = project / "data/source/ds005574"
    parser = argparse.ArgumentParser(description=__doc__)
    # The released significant-electrode list contains the eight participants
    # used by Bhattacharjee et al.; public sub-02 was added later and is absent.
    parser.add_argument("--subjects", nargs="+", default=["01", "03", "04", "05", "06", "07", "08", "09"])
    parser.add_argument("--source-root", type=Path, default=source)
    parser.add_argument("--output-root", type=Path, default=project / "data/ecog_caches/heo2026_625ms_sw200_t10")
    parser.add_argument(
        "--electrode-file",
        type=Path,
        default=project / "data/metadata/all_subject_sig.csv",
    )
    parser.add_argument("--center-lag-ms", type=float, default=150.0)
    parser.add_argument("--segment-ms", type=float, default=625.0)
    parser.add_argument("--sliding-window-ms", type=float, default=200.0)
    parser.add_argument("--temporal-bins", type=int, default=10)
    parser.add_argument("--epoch-batch-size", type=int, default=100)
    parser.add_argument("--max-words", type=int, help="Smoke test only")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


PUBLIC_TO_ORIGINAL = {
    "sub-01": 661,
    "sub-03": 717,
    "sub-04": 723,
    "sub-05": 741,
    "sub-06": 742,
    "sub-07": 743,
    "sub-08": 763,
    "sub-09": 798,
}


def load_words_and_targets(source_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    root = source_root / "stimuli/gpt2-xl"
    transcript = pd.read_csv(root / "transcript.tsv", sep="\t", index_col=0)
    with h5py.File(root / "features.hdf5", "r") as handle:
        tokens = np.asarray(handle["layer-24"], dtype=np.float32)
    if len(transcript) != len(tokens):
        raise ValueError("GPT-2 transcript and layer-24 rows differ")
    transcript = transcript.copy()
    transcript["feature_row"] = np.arange(len(transcript))
    groups = transcript.groupby("word_idx", sort=True)
    targets = np.stack([
        tokens[group.feature_row.to_numpy()].mean(axis=0) for _, group in groups
    ]).astype(np.float32)
    words = groups.agg(word=("word", "first"), start=("start", "first"), end=("end", "last")).reset_index()
    valid = words.start.notna().to_numpy() & np.isfinite(targets).all(axis=1)
    words, targets = words.loc[valid].reset_index(drop=True), targets[valid]
    order = np.argsort(words.start.to_numpy(float), kind="stable")
    return words.iloc[order].reset_index(drop=True), targets[order]


def extract_subject(raw: mne.io.BaseRaw, onsets: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[float]]:
    half = args.segment_ms / 2000.0
    center = args.center_lag_ms / 1000.0
    start, end = center - half, center + half
    width = args.sliding_window_ms / 1000.0
    if width > end - start:
        raise ValueError("sliding window cannot exceed the segment")
    window_starts = np.linspace(start, end - width, args.temporal_bins)
    centers = (window_starts + width / 2.0).tolist()
    events = np.zeros((len(onsets), 3), dtype=int)
    events[:, 0] = np.rint(onsets * float(raw.info["sfreq"])).astype(int)
    events[:, 2] = 1
    outputs, selections = [], []
    for first in range(0, len(events), args.epoch_batch_size):
        last = min(first + args.epoch_batch_size, len(events))
        print(f"  epoch {first}:{last}/{len(events)}", flush=True)
        epochs = mne.Epochs(
            raw, events[first:last], tmin=start, tmax=end, baseline=None,
            preload=True, proj=None, event_repeated="merge",
            reject_by_annotation=True, verbose=False,
        )
        if not len(epochs):
            continue
        values = epochs.get_data(copy=True)
        binned = []
        for window_start in window_starts:
            mask = (epochs.times >= window_start) & (epochs.times < window_start + width)
            if not mask.any():
                raise RuntimeError("a sliding window contains no ECoG samples")
            binned.append(values[:, :, mask].mean(axis=2))
        outputs.append(np.stack(binned, axis=2).astype(np.float32))
        selections.append(epochs.selection + first)
    if not outputs:
        raise RuntimeError("all epochs were dropped")
    return np.concatenate(outputs), np.concatenate(selections), centers


def main() -> None:
    args = parse_args()
    words, targets = load_words_and_targets(args.source_root)
    electrode_table = pd.read_csv(args.electrode_file)
    if args.max_words:
        words, targets = words.iloc[:args.max_words].reset_index(drop=True), targets[:args.max_words]
    args.output_root.mkdir(parents=True, exist_ok=True)
    words.to_csv(args.output_root / "canonical_words.tsv", sep="\t", index=False)
    np.save(args.output_root / "gpt2_xl_layer24.npy", targets)
    for value in args.subjects:
        subject = f"sub-{int(value):02d}"
        destination = args.output_root / subject
        if (destination / "ecog.npy").exists() and not args.overwrite:
            print(f"{subject}: cache exists; skipping", flush=True)
            continue
        fif = args.source_root / "derivatives/ecogprep" / subject / "ieeg" / f"{subject}_task-podcast_desc-highgamma_ieeg.fif"
        print(f"{subject}: reading {fif}", flush=True)
        raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
        raw.pick("ecog", exclude="bads")
        if subject not in PUBLIC_TO_ORIGINAL:
            raise ValueError(f"{subject} is absent from the released 183-electrode SRM list")
        original_id = PUBLIC_TO_ORIGINAL[subject]
        selected = electrode_table.loc[
            electrode_table.subject.astype(int) == original_id, "elec"
        ].astype(str).tolist()
        missing = sorted(set(selected) - set(raw.ch_names))
        if missing:
            raise ValueError(f"{subject}: selected electrodes absent from FIF: {missing}")
        raw.pick(selected)
        ecog, selection, centers = extract_subject(raw, words.start.to_numpy(float), args)
        channel_names = list(raw.ch_names)
        raw.close()
        selected_words = words.iloc[selection].reset_index(drop=True)
        finite = np.isfinite(ecog).all(axis=(1, 2))
        ecog, selected_words = ecog[finite], selected_words.loc[finite].reset_index(drop=True)
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "ecog.npy", ecog)
        selected_words.to_csv(destination / "words.tsv", sep="\t", index=False)
        metadata = {
            "subject": subject,
            "source": str(fif),
            "band": "released high-gamma (70-200 Hz)",
            "center_lag_ms": args.center_lag_ms,
            "segment_ms": args.segment_ms,
            "sliding_window_ms": args.sliding_window_ms,
            "temporal_bins": args.temporal_bins,
            "window_centers_s_relative_to_onset": centers,
            "channel_names": channel_names,
            "electrode_selection": str(args.electrode_file),
            "original_subject_id": original_id,
            "shape": list(ecog.shape),
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"{subject}: saved {ecog.shape}", flush=True)


if __name__ == "__main__":
    main()
