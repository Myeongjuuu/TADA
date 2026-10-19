#!/usr/bin/env python3
"""Run the manuscript's preparation, figure analysis, and decoding methods."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

SUBJECTS = ('01', '03', '04', '05', '06', '07', '08', '09')
METHODS = ('pca', 'srm-cnn', 'medium', 'long', 'multi', 'egr')


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'figure', 'decoding', 'all', *METHODS), required=True)
    parser.add_argument('--source-root', type=Path, default=root/'data/source/ds005574')
    parser.add_argument('--electrode-file', type=Path, default=root/'data/metadata/all_subject_sig.csv')
    parser.add_argument('--cache-root', type=Path, default=root/'data/ecog_caches')
    parser.add_argument('--output-root', type=Path, default=root/'results')
    parser.add_argument('--held-out-subjects', nargs='+', choices=SUBJECTS, default=list(SUBJECTS))
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--outer-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-words', type=int, help='Debugging only; omit for paper experiments')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running or writing files')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.outer_folds < 2:
        parser.error('epochs/patience must be positive and outer-folds must be at least 2')
    if args.max_words is not None and args.max_words < 2*args.outer_folds:
        parser.error('max-words is too small for the requested folds')
    for name in ('source_root','electrode_file','cache_root','output_root'):
        setattr(args,name,getattr(args,name).resolve())
    return root,args


def commands(root, args):
    def command(script, *options):
        return [sys.executable, str(root/'scripts'/script), *map(str, options)]

    medium = args.cache_root/'heo2026_625ms_sw200_t10'
    multi = args.cache_root/'heo2026_multiwindow_t10'
    wide = args.cache_root/'wide_m1000_p1000'
    preparation = [
        command('prepare_heo2026_cache.py', '--subjects', *SUBJECTS,
                '--source-root', args.source_root, '--electrode-file', args.electrode_file,
                '--output-root', medium),
        command('prepare_multiwindow_cache.py', '--subjects', *SUBJECTS,
                '--source-root', args.source_root, '--electrode-file', args.electrode_file,
                '--output-root', multi),
        command('prepare_lag_cache.py', '--subjects', *SUBJECTS,
                '--source-root', args.source_root,
                '--transcript', args.source_root/'stimuli/podcast_transcript.csv', '--cache-root', wide),
    ]
    figure_options = ['--max-words', str(args.max_words)] if args.max_words else []
    figure = [
        command('run_lag_resolved_encoding.py', '--subjects', *SUBJECTS,
                '--cache-root', wide, '--source-root', args.source_root,
                '--electrode-file', args.electrode_file, '--outer-folds', args.outer_folds,
                '--output-root', args.output_root/'lag_encoding', *figure_options),
        command('plot_lag_resolved_encoding.py', '--results-dir', args.output_root/'lag_encoding',
                '--horizontal', '--output', args.output_root/'figures/lag_encoding.png'),
    ]
    common = ['--subjects', *SUBJECTS, '--held-out-subjects', *args.held_out_subjects,
              '--device', args.device, '--epochs', str(args.epochs), '--patience', str(args.patience),
              '--outer-folds', str(args.outer_folds), '--seed', str(args.seed)]
    if args.max_words:
        common += ['--max-words', str(args.max_words)]
    decoding = {
        'pca': command('run_pca_alignment_baselines.py', *common, '--cache-root', multi,
                       '--variants', 'pca', 'pca_ha', '--output-root', args.output_root/'pca'),
        'srm-cnn': command('run_heo2026_benchmark.py', *common, '--cache-root', medium,
                           '--variants', 'direct', '--output-root', args.output_root/'srm_cnn'),
        'medium': command('run_multiwindow_srm.py', *common, '--cache-root', multi,
                          '--variants', 'matched_direct_combined', '--output-root', args.output_root/'medium'),
        'long': command('run_multiwindow_srm.py', *common, '--cache-root', multi,
                        '--variants', 'matched_long_combined', '--output-root', args.output_root/'long'),
        'multi': command('run_multiwindow_srm.py', *common, '--cache-root', multi,
                         '--variants', 'multiwindow_combined', '--output-root', args.output_root/'multi'),
        'egr': command('run_encoding_guided_rerank.py', *common, '--cache-root', multi,
                       '--output-root', args.output_root/'egr'),
    }
    if args.stage == 'prepare':
        return preparation
    if args.stage == 'figure':
        return figure
    if args.stage == 'decoding':
        return list(decoding.values())
    if args.stage == 'all':
        return preparation + figure + list(decoding.values())
    return [decoding[args.stage]]


def main():
    root,args = parse_args()
    planned = commands(root,args)
    for index,cmd in enumerate(planned,1):
        print(f'[{index}/{len(planned)}] {shlex.join(cmd)}',flush=True)
        if not args.dry_run:
            subprocess.run(cmd,cwd=root,check=True)


if __name__ == '__main__':
    main()
