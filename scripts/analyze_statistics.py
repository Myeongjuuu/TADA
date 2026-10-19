#!/usr/bin/env python3
"""Optional participant-level AUC tests on newly generated decoding outputs."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean

SUBJECTS = ('01','03','04','05','06','07','08','09')


def read(root,method,subject):
    path = root/method/f'sub-{subject}/summary.json'
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise SystemExit(f'Missing experiment output: {path}. Run the corresponding method first.')
    if report.get('held_out_subject') != f'sub-{subject}':
        raise ValueError(f'Participant mismatch: {path}')
    if len(report.get('folds',next(iter(report.get('variants',{}).values()),{}).get('folds',[]))) != 5:
        raise ValueError(f'Expected five completed folds: {path}')
    return report


def sign_flip(first,second):
    difference = [b-a for a,b in zip(first,second)]
    observed = abs(mean(difference))
    extreme = sum(abs(mean(s*d for s,d in zip(signs,difference))) >= observed-1e-15
                  for signs in itertools.product((-1,1),repeat=len(difference)))
    return {'n':len(difference),'mean_auc_difference':mean(difference),
            'exact_two_sided_p':extreme/2**len(difference)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    args = parser.parse_args()
    variants = {'medium':'matched_direct_combined','long':'matched_long_combined','multi':'multiwindow_combined'}
    auc = {method:[read(args.results_root,method,s)['variants'][variant]['held_out_mean']['auc_roc']
                   for s in SUBJECTS] for method,variant in variants.items()}
    comparisons = [('medium','long'),('long','multi'),('medium','multi')]
    tests = {f'{b} minus {a}':sign_flip(auc[a],auc[b]) for a,b in comparisons}
    ordered = sorted(tests,key=lambda k:tests[k]['exact_two_sided_p'])
    running = 0
    for index,name in enumerate(ordered):
        running = max(running,(len(ordered)-index)*tests[name]['exact_two_sided_p'])
        tests[name]['holm_p_across_3_temporal_auc_tests'] = min(1,running)
    egr = [read(args.results_root,'egr',s) for s in SUBJECTS]
    tests['EGR minus same-run base'] = sign_flip(
        [r['held_out_base_mean']['auc_roc'] for r in egr],
        [r['held_out_egr_mean']['auc_roc'] for r in egr])
    output = args.results_root/'statistics.json'
    output.write_text(json.dumps(tests,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(tests,indent=2))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
