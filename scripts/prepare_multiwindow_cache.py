#!/usr/bin/env python3
"""Prepare 250/625/1500-ms word-locked ECoG windows for SRM decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd


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


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    source = project / "data/source/ds005574"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects", nargs="+",
        default=["01", "03", "04", "05", "06", "07", "08", "09"],
    )
    parser.add_argument("--source-root", type=Path, default=source)
    parser.add_argument(
        "--output-root", type=Path,
        default=project / "data/ecog_caches/heo2026_multiwindow_t10",
    )
    parser.add_argument(
        "--electrode-file", type=Path,
        default=project / "data/metadata/all_subject_sig.csv",
    )
    parser.add_argument("--center-lag-ms", type=float, default=150.0)
    parser.add_argument(
        "--window-center-ms", nargs=3, type=float,
        help="Optional independent centers for the three windows (ms). Overrides --center-lag-ms.",
    )
    parser.add_argument("--window-ms", nargs=3, type=float, default=[250.0, 625.0, 1500.0])
    parser.add_argument("--smoothing-ms", nargs=3, type=float, default=[80.0, 200.0, 400.0])
    parser.add_argument("--window-names", nargs=3, default=["short", "medium", "long"])
    parser.add_argument("--temporal-bins", type=int, default=10)
    parser.add_argument("--epoch-batch-size", type=int, default=100)
    parser.add_argument("--max-words", type=int, help="Smoke test only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(set(args.window_names)) != 3:
        parser.error("--window-names must contain three unique names")
    for extent, smoothing in zip(args.window_ms, args.smoothing_ms):
        if extent <= 0 or smoothing <= 0 or smoothing > extent:
            parser.error("each smoothing width must be positive and no larger than its window")
    return args


def load_words_and_targets(source_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    root = source_root / "stimuli/gpt2-xl"
    transcript = pd.read_csv(root / "transcript.tsv", sep="\t", index_col=0)
    with h5py.File(root / "features.hdf5", "r") as handle:
        tokens = np.asarray(handle["layer-24"], dtype=np.float32)
    transcript = transcript.copy()
    transcript["feature_row"] = np.arange(len(transcript))
    groups = transcript.groupby("word_idx", sort=True)
    targets = np.stack([
        tokens[group.feature_row.to_numpy()].mean(axis=0) for _, group in groups
    ]).astype(np.float32)
    words = groups.agg(
        word=("word", "first"), start=("start", "first"), end=("end", "last")
    ).reset_index()
    valid = words.start.notna().to_numpy() & np.isfinite(targets).all(axis=1)
    words, targets = words.loc[valid].reset_index(drop=True), targets[valid]
    order = np.argsort(words.start.to_numpy(float), kind="stable")
    return words.iloc[order].reset_index(drop=True), targets[order]


def window_definitions(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    centers_ms = (
        args.window_center_ms if args.window_center_ms is not None
        else [args.center_lag_ms] * 3
    )
    definitions = {}
    for name, extent_ms, smoothing_ms, center_ms in zip(
        args.window_names, args.window_ms, args.smoothing_ms, centers_ms
    ):
        center = center_ms / 1000.0
        extent, smoothing = extent_ms / 1000.0, smoothing_ms / 1000.0
        start, end = center - extent / 2.0, center + extent / 2.0
        starts = np.linspace(start, end - smoothing, args.temporal_bins)
        definitions[name] = {
            "extent_ms": extent_ms,
            "smoothing_ms": smoothing_ms,
            "center_ms": center_ms,
            "start_s": start,
            "end_s": end,
            "bin_starts_s": starts,
            "bin_centers_s": starts + smoothing / 2.0,
        }
    return definitions


def extract_subject(
    raw: mne.io.BaseRaw, onsets: np.ndarray, args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, dict[str, object]]]:
    definitions = window_definitions(args)
    epoch_start = min(float(value["start_s"]) for value in definitions.values())
    epoch_end = max(float(value["end_s"]) for value in definitions.values())
    events = np.zeros((len(onsets), 3), dtype=int)
    events[:, 0] = np.rint(onsets * float(raw.info["sfreq"])).astype(int)
    events[:, 2] = 1
    outputs = {name: [] for name in definitions}
    selections = []
    for first in range(0, len(events), args.epoch_batch_size):
        last = min(first + args.epoch_batch_size, len(events))
        print(f"  epoch {first}:{last}/{len(events)}", flush=True)
        epochs = mne.Epochs(
            raw, events[first:last], tmin=epoch_start, tmax=epoch_end,
            baseline=None, preload=True, proj=None, event_repeated="merge",
            reject_by_annotation=True, verbose=False,
        )
        if not len(epochs):
            continue
        values = epochs.get_data(copy=True)
        for name, definition in definitions.items():
            smoothing = float(definition["smoothing_ms"]) / 1000.0
            binned = []
            for window_start in definition["bin_starts_s"]:
                mask = (
                    (epochs.times >= float(window_start))
                    & (epochs.times < float(window_start) + smoothing)
                )
                if not mask.any():
                    raise RuntimeError(f"{name} contains an empty temporal bin")
                binned.append(values[:, :, mask].mean(axis=2))
            outputs[name].append(np.stack(binned, axis=2).astype(np.float32))
        selections.append(epochs.selection + first)
    if not selections:
        raise RuntimeError("all epochs were dropped")
    return (
        {name: np.concatenate(rows) for name, rows in outputs.items()},
        np.concatenate(selections),
        definitions,
    )


def main() -> None:
    args = parse_args()
    words, targets = load_words_and_targets(args.source_root)
    if args.max_words:
        words, targets = words.iloc[: args.max_words].reset_index(drop=True), targets[: args.max_words]
    electrode_table = pd.read_csv(args.electrode_file)
    args.output_root.mkdir(parents=True, exist_ok=True)
    words.to_csv(args.output_root / "canonical_words.tsv", sep="\t", index=False)
    np.save(args.output_root / "gpt2_xl_layer24.npy", targets)
    for value in args.subjects:
        subject = f"sub-{int(value):02d}"
        destination = args.output_root / subject
        expected = [destination / f"ecog_{name}.npy" for name in args.window_names]
        if all(path.exists() for path in expected) and not args.overwrite:
            print(f"{subject}: cache exists; skipping", flush=True)
            continue
        fif = (
            args.source_root / "derivatives/ecogprep" / subject / "ieeg"
            / f"{subject}_task-podcast_desc-highgamma_ieeg.fif"
        )
        print(f"{subject}: reading {fif}", flush=True)
        raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
        raw.pick("ecog", exclude="bads")
        original_id = PUBLIC_TO_ORIGINAL[subject]
        selected = electrode_table.loc[
            electrode_table.subject.astype(int) == original_id, "elec"
        ].astype(str).tolist()
        missing = sorted(set(selected) - set(raw.ch_names))
        if missing:
            raise ValueError(f"{subject}: selected electrodes absent from FIF: {missing}")
        raw.pick(selected)
        arrays, selection, definitions = extract_subject(
            raw, words.start.to_numpy(float), args
        )
        channel_names = list(raw.ch_names)
        raw.close()
        selected_words = words.iloc[selection].reset_index(drop=True)
        finite = np.ones(len(selection), dtype=bool)
        for array in arrays.values():
            finite &= np.isfinite(array).all(axis=(1, 2))
        selected_words = selected_words.loc[finite].reset_index(drop=True)
        destination.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            np.save(destination / f"ecog_{name}.npy", array[finite])
        selected_words.to_csv(destination / "words.tsv", sep="\t", index=False)
        serializable_definitions = {
            name: {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in definition.items()
            }
            for name, definition in definitions.items()
        }
        metadata = {
            "subject": subject,
            "source": str(fif),
            "band": "released high-gamma (70-200 Hz)",
            "center_lag_ms": args.center_lag_ms,
            "window_centers_ms": args.window_center_ms,
            "temporal_bins": args.temporal_bins,
            "window_order": args.window_names,
            "windows": serializable_definitions,
            "channel_names": channel_names,
            "electrode_selection": str(args.electrode_file),
            "original_subject_id": original_id,
            "shape_per_window": {
                name: list(array[finite].shape) for name, array in arrays.items()
            },
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{subject}: saved {len(selected_words)} words; "
            + ", ".join(f"{name}={arrays[name][finite].shape}" for name in args.window_names),
            flush=True,
        )


if __name__ == "__main__":
    main()
