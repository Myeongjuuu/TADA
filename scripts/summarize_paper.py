#!/usr/bin/env python3
"""Rebuild Tables I/II and paired tests from saved metrics (standard library only)."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from statistics import mean, stdev

SUBJECTS = ('01', '03', '04', '05', '06', '07', '08', '09')
METRICS = ('auc_roc', 'pairwise_accuracy', 'top_10', 'top_50', 'mrr')
METHODS = {
    'PCA': ('pca_alignment_baselines', 'pca'),
    'PCA+HA': ('pca_alignment_baselines', 'pca_ha'),
    'SRM-CNN': ('heo2026_srm_cnn_baseline_all', 'direct'),
    'Medium': ('multiwindow_srm_all', 'matched_direct_combined'),
    'Long-only': ('matched_long_control', 'matched_long_combined'),
    'Multi-scale': ('multiwindow_srm_all', 'multiwindow_combined'),
    'Multi+EGR': ('encoding_guided_rerank', None),
}
RUN_DIRECTORIES = {
    'PCA': 'pca', 'PCA+HA': 'pca', 'SRM-CNN': 'srm_cnn',
    'Medium': 'medium', 'Long-only': 'long', 'Multi-scale': 'multi', 'Multi+EGR': 'egr',
}


def read(root, directory, subject):
    path = root / directory / f'sub-{subject}' / 'summary.json'
    return json.loads(path.read_text(encoding='utf-8'))


def load(root, layout='reference'):
    values = {'source': {}, 'held_out': {}}
    for method, (directory, variant) in METHODS.items():
        if layout == 'run':
            directory = RUN_DIRECTORIES[method]
        for section in values:
            values[section][method] = []
        for subject in SUBJECTS:
            location = directory
            if layout == 'reference' and directory == 'multiwindow_srm_all' and subject == '03' and not (
                root / directory / 'sub-03/summary.json'
            ).exists():
                location = 'multiwindow_srm_sub03'
            report = read(root, location, subject)
            if report.get('held_out_subject') != f'sub-{subject}':
                raise ValueError(f'Participant mismatch: {location}/sub-{subject}')
            if variant is None:
                held, source = report['held_out_egr_mean'], report['source_egr_mean']
            else:
                row = report['variants'][variant]
                held = row['held_out_mean']
                # Early sub-03 summaries omit source_mean at the top level.
                source = row.get('source_mean') or {
                    metric: mean(fold['source_mean'][metric] for fold in row['folds'])
                    for metric in METRICS
                }
            values['source'][method].append(source)
            values['held_out'][method].append(held)
    return values


def sign_flip(first, second):
    differences = [b-a for a,b in zip(first, second)]
    observed = abs(mean(differences))
    extreme = sum(abs(mean(s*d for s,d in zip(signs, differences))) >= observed-1e-15
                  for signs in itertools.product((-1,1), repeat=len(differences)))
    return {'n': len(differences), 'mean_difference': mean(differences),
            'exact_two_sided_p': extreme / 2**len(differences)}


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=project/'results')
    parser.add_argument('--output-dir', type=Path, default=project/'reports')
    parser.add_argument('--layout', choices=('reference','run'), default='reference',
                        help='Included paper results or new run_experiments.py outputs')
    parser.add_argument('--check-manuscript', action='store_true')
    args = parser.parse_args()
    values = load(args.results_root, args.layout)
    expected = (json.loads((project/'configs/manuscript_tables.json').read_text(encoding='utf-8'))
                if args.check_manuscript else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown = []
    checked = 0
    for section in values:
        table = [['Metric', *METHODS]]
        numeric = []
        for metric in METRICS:
            cells = [metric]
            for method in METHODS:
                data = [row[metric] for row in values[section][method]]
                scale = 100 if metric in ('pairwise_accuracy','top_10','top_50') else 1
                digits = 2 if scale == 100 else 4
                avg, sd = mean(data)*scale, stdev(data)*scale
                cells.append(f'{avg:.{digits}f} +/- {sd:.{digits}f}')
                numeric.append([method, metric, avg, sd, len(data)])
                if args.check_manuscript:
                    for actual, saved in zip((avg,sd), expected[section][method][metric]):
                        if abs(actual-saved) > 0.5*10**(-digits)+1e-10:
                            raise AssertionError(f'Manuscript mismatch: {section}/{method}/{metric}: {actual} vs {saved}')
                        checked += 1
            table.append(cells)
        markdown += [f'## {section}', '', '| '+' | '.join(table[0])+' |',
                     '| '+' | '.join(['---']*len(table[0]))+' |']
        markdown += ['| '+' | '.join(row)+' |' for row in table[1:]]
        markdown += ['']
        with (args.output_dir/f'{section}_table.tsv').open('w',encoding='utf-8',newline='') as handle:
            writer = csv.writer(handle,delimiter='\t')
            writer.writerow(['method','metric','mean','sample_sd','n_configurations'])
            writer.writerows(numeric)
    held = {method:[row['auc_roc'] for row in rows] for method,rows in values['held_out'].items()}
    comparisons = [('Medium','Long-only'),('Long-only','Multi-scale'),('Medium','Multi-scale')]
    tests = {f'{b} minus {a}':sign_flip(held[a],held[b]) for a,b in comparisons}
    ordered = sorted(tests, key=lambda k:tests[k]['exact_two_sided_p'])
    running = 0
    for rank,name in enumerate(ordered):
        running = max(running, (len(ordered)-rank)*tests[name]['exact_two_sided_p'])
        tests[name]['holm_p_across_3_auc_tests'] = min(1,running)
    # Pair EGR against its own base decoder evaluation, as in the manuscript.
    egr_directory = 'egr' if args.layout == 'run' else 'encoding_guided_rerank'
    base = [read(args.results_root,egr_directory,s)['held_out_base_mean']['auc_roc'] for s in SUBJECTS]
    tests['EGR minus same-run base'] = sign_flip(base,held['Multi+EGR'])
    if args.check_manuscript:
        for name in ('Long-only minus Medium','Multi-scale minus Medium'):
            assert round(tests[name]['holm_p_across_3_auc_tests'],4) == .0234
        assert round(tests['EGR minus same-run base']['exact_two_sided_p'],4) == .0078
    (args.output_dir/'paper_tables.md').write_text('\n'.join(markdown),encoding='utf-8')
    (args.output_dir/'paired_tests.json').write_text(json.dumps(tests,indent=2)+'\n',encoding='utf-8')
    print('\n'.join(markdown))
    print(json.dumps(tests,indent=2))
    if args.check_manuscript:
        print(f'PASS: {checked} manuscript mean/SD cells and the reported AUC p-values.')


if __name__ == '__main__':
    main()
