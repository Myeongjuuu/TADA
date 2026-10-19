#!/usr/bin/env python3
"""Plot group-level and participant-level semantic encoding dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=project / "results/lag_resolved_semantic_encoding")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path. Defaults depend on the selected layout.")
    parser.add_argument("--group-only", action="store_true",
                        help="Export only the group panel for a compact single-panel figure.")
    parser.add_argument("--horizontal", action="store_true",
                        help="Place group and participant panels side by side.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads((args.results_dir / "aggregate.json").read_text(encoding="utf-8"))
    lags = np.asarray(report["lag_center_ms"], dtype=float)
    curves = np.asarray(report["subject_encoding_r"], dtype=float)
    mean = np.asarray(report["mean_encoding_r"], dtype=float)
    sem = np.asarray(report["sem_encoding_r"], dtype=float)
    peak = int(np.argmax(mean))

    plt.rcParams.update({"font.size": 6, "font.family": "DejaVu Sans"})
    if args.group_only:
        fig, ax = plt.subplots(figsize=(3.55, 2.75))
        axes = [ax]
    elif args.horizontal:
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), sharey=True,
                                 gridspec_kw={"wspace": 0.23})
        axes = list(axes)
    else:
        # Readable one-column layout: group result above participant curves.
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.85), sharex=True,
                                 gridspec_kw={"hspace": 0.30})
        axes = list(axes)
    ax = axes[0]
    ax.plot(lags, mean, color="#e67e22", lw=2.5, label="GPT-2 semantic encoding")
    ax.fill_between(lags, mean - sem, mean + sem, color="#e67e22", alpha=0.18,
                    label="Across-subject SEM")
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.axvline(0, color="#bbbbbb", lw=0.8, ls=":")
    ax.scatter(lags[peak], mean[peak], color="#c0392b", s=28, zorder=3)
    ax.annotate(f"peak {lags[peak]:+.0f} ms", (lags[peak], mean[peak]),
                xytext=(4, 6), textcoords="offset points", fontsize=6.2, color="#a52a2a")
    ax.set_title("a  Group semantic encoding", loc="left", fontweight="bold", fontsize=7.2)
    if args.group_only or args.horizontal:
        ax.set_ylabel("Held-out encoding performance (Pearson r)", fontsize=6.6)
    ax.legend(frameon=True, fontsize=5.2, loc="lower right",
              framealpha=0.94, facecolor="white", edgecolor="none")

    if not args.group_only:
        ax = axes[1]
        colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
        for color, subject, curve in zip(colors, report["subjects"], curves):
            ax.plot(lags, curve, lw=1.35, color=color, label=subject.replace("sub-", "S"))
        ax.axhline(0, color="#777777", lw=0.8, ls="--")
        ax.axvline(0, color="#bbbbbb", lw=0.8, ls=":")
        ax.set_title("b  Participant-specific encoding", loc="left", fontweight="bold", fontsize=7.2)
        ax.legend(frameon=False, fontsize=5.0, ncol=2, loc="upper left", handlelength=1.2)

    for ax in axes:
        ax.set_xlim(lags.min() - 35, lags.max() + 35)
        ax.set_xticks(np.arange(-1000, 1001, 500))
        ax.set_yticks([0.0, 0.1, 0.2])
        ax.grid(axis="y", color="#e5e5e5", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    if args.horizontal:
        for ax in axes:
            ax.set_xlabel("Lag from spoken-word onset (ms)", fontsize=6.6)
    else:
        axes[-1].set_xlabel("Lag from spoken-word onset (ms)", fontsize=6.6)
    if args.group_only:
        fig.subplots_adjust(left=0.20, right=0.985, bottom=0.21, top=0.88)
    elif args.horizontal:
        fig.subplots_adjust(left=0.09, right=0.995, bottom=0.23, top=0.88)
    else:
        fig.subplots_adjust(left=0.22, right=0.985, bottom=0.15, top=0.93)
        fig.text(0.095, 0.52, "Held-out encoding performance (Pearson r)",
                 rotation=90, va="center", ha="center", fontsize=6.6)
    if args.output is None:
        suffix = "lag_resolved_semantic_encoding_horizontal" if args.horizontal else "lag_resolved_semantic_encoding"
        output = args.results_dir.parent / "figures" / f"{suffix}.png"
    else:
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
