#!/usr/bin/env python3
"""Render saved lag-resolved encoding curves as SVG."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lag_resolved_semantic_encoding" / "aggregate.json"
OUTPUT = ROOT / "results" / "figures" / "lag_resolved_semantic_encoding.svg"


def points(xs, ys, xmap, ymap):
    return " ".join(f"{xmap(x):.2f},{ymap(y):.2f}" for x, y in zip(xs, ys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=RESULTS.parent)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = json.loads((args.results_dir / 'aggregate.json').read_text(encoding="utf-8"))
    lags = report["lag_center_ms"]
    curves = report["subject_encoding_r"]
    mean = report["mean_encoding_r"]
    sem = report["sem_encoding_r"]

    width, height = 1040, 490
    left, right, top, bottom = 92, 38, 74, 82
    pw, ph = width - left - right, height - top - bottom
    xmin, xmax = -950, 950
    ymin, ymax = 0.0, 0.28
    xmap = lambda x: left + (x - xmin) / (xmax - xmin) * pw
    ymap = lambda y: top + (ymax - y) / (ymax - ymin) * ph

    peak_idx = max(range(len(mean)), key=mean.__getitem__)
    peak_lag, peak_r = lags[peak_idx], mean[peak_idx]

    # Reverse the lower bound to close the SEM confidence polygon.
    upper = points(lags, [m + s for m, s in zip(mean, sem)], xmap, ymap)
    lower = points(list(reversed(lags)), [m - s for m, s in zip(reversed(mean), reversed(sem))], xmap, ymap)

    out = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, Helvetica, sans-serif; fill: #202020; }}
  .axis {{ stroke: #222; stroke-width: 1.4; }}
  .grid {{ stroke: #dce1e7; stroke-width: 1; }}
  .tick {{ font-size: 15px; fill: #333; }}
  .label {{ font-size: 17px; font-weight: 600; }}
  .small {{ font-size: 14px; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="31" font-size="22" font-weight="700">Lag-resolved contextual semantic encoding</text>
<text x="{left}" y="53" class="small" fill="#555">Source-participant evaluation; mean across 8 participants (shading: SEM)</text>
''']

    # Faint spans show the response windows used by the temporal ablation.
    windows = [
        ("Long: -600 to +900 ms", -600, 900, "#8d6bb8"),
        ("Medium: -162.5 to +462.5 ms", -162.5, 462.5, "#e69a55"),
        ("Short: +25 to +275 ms", 25, 275, "#3a9fd8"),
    ]
    for label, lo, hi, color in windows:
        x1, x2 = xmap(lo), xmap(hi)
        out.append(f'<rect x="{x1:.2f}" y="{top}" width="{x2-x1:.2f}" height="{ph}" fill="{color}" opacity="0.075"/>')

    # Horizontal grid and y labels.
    for v in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]:
        yy = ymap(v)
        out.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" class="grid"/>')
        out.append(f'<text x="{left-13}" y="{yy+5:.2f}" text-anchor="end" class="tick">{v:.2f}</text>')

    # Word-onset reference.
    zero = xmap(0)
    out.append(f'<line x1="{zero:.2f}" y1="{top}" x2="{zero:.2f}" y2="{height-bottom}" stroke="#333" stroke-width="1.4" stroke-dasharray="5 4"/>')
    out.append(f'<text x="{zero+7:.2f}" y="{top+20}" class="small" fill="#333">word onset</text>')

    # Participant trajectories use the actual saved curves.
    for curve in curves:
        out.append(f'<polyline points="{points(lags, curve, xmap, ymap)}" fill="none" stroke="#aeb6c2" stroke-width="1.45" stroke-linejoin="round" opacity="0.72"/>')

    out.append(f'<polygon points="{upper} {lower}" fill="#7f5aa2" opacity="0.19"/>')
    out.append(f'<polyline points="{points(lags, mean, xmap, ymap)}" fill="none" stroke="#68458b" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')

    px, py = xmap(peak_lag), ymap(peak_r)
    out.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.5" fill="#68458b" stroke="white" stroke-width="2"/>')
    out.append(f'<text x="{px+12:.2f}" y="{py-12:.2f}" class="small" font-weight="700" fill="#68458b">Peak: r = {peak_r:.3f} at +{int(peak_lag)} ms</text>')

    # Axes and x labels.
    out.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    for v in [-900, -600, -300, 0, 300, 600, 900]:
        xx = xmap(v)
        out.append(f'<line x1="{xx:.2f}" y1="{height-bottom}" x2="{xx:.2f}" y2="{height-bottom+6}" class="axis"/>')
        sign = "" if v >= 0 else "−"
        out.append(f'<text x="{xx:.2f}" y="{height-bottom+28}" text-anchor="middle" class="tick">{sign}{abs(v)}</text>')
    out.append(f'<text x="{left+pw/2:.2f}" y="{height-22}" text-anchor="middle" class="label">Lag from spoken-word onset (ms)</text>')
    out.append(f'<text x="24" y="{top+ph/2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 24 {top+ph/2:.2f})">Held-out encoding performance (Pearson r)</text>')

    # Legend.
    lx, ly = width - right - 230, height - bottom - 132
    out.append(f'<rect x="{lx}" y="{ly}" width="223" height="122" rx="5" fill="white" opacity="0.94" stroke="#d4d8dd"/>')
    out.append(f'<line x1="{lx+13}" y1="{ly+18}" x2="{lx+44}" y2="{ly+18}" stroke="#aeb6c2" stroke-width="2"/>')
    out.append(f'<text x="{lx+53}" y="{ly+23}" class="small">individual participant</text>')
    out.append(f'<line x1="{lx+13}" y1="{ly+39}" x2="{lx+44}" y2="{ly+39}" stroke="#68458b" stroke-width="4"/>')
    out.append(f'<text x="{lx+53}" y="{ly+44}" class="small">group mean ± SEM</text>')
    for idx, (label, _lo, _hi, color) in enumerate(windows):
        yy = ly + 64 + idx * 18
        out.append(f'<rect x="{lx+13}" y="{yy-11}" width="29" height="12" fill="{color}" opacity="0.35"/>')
        out.append(f'<text x="{lx+53}" y="{yy}" class="small">{label}</text>')
    out.append('</svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out), encoding="utf-8")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
