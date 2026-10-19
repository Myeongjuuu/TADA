#!/usr/bin/env python3
"""Fetch the exact public significant-electrode mask used in the saved runs."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

REVISION = 'beac7e932d77f08a934520a7e84f369a7e6d03c8'
URL = f'https://raw.githubusercontent.com/hassonlab/podcast-benchmark/{REVISION}/processed_data/all_subject_sig.csv'
SHA256 = '565d6f6f7fb8b069da6aceea8241f33fe0873dbca28da52f647bcdd3f0b6a099'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'data/metadata/all_subject_sig.csv')
    args = parser.parse_args()
    if args.output.exists():
        if hashlib.sha256(args.output.read_bytes()).hexdigest() != SHA256:
            raise SystemExit(f'Existing file differs; refusing to overwrite: {args.output}')
        print(f'Already verified: {args.output}')
        return
    with urlopen(URL, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise SystemExit('Upstream CSV checksum mismatch; no file was written')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f'Saved verified electrode mask: {args.output}')


if __name__ == '__main__':
    main()
