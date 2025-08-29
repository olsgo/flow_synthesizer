#!/usr/bin/env python3
"""
Prepare a consistent PolyMAX v2 dataset by cloning the existing dataset and
augmenting it with additional samples (single + phrase), all matching
polymax_dataset feature shapes (mel 128x173, mfcc 13x173, mono@22050).

This script:
1) Copies datasets/polymax_dataset -> datasets/polymax_dataset_v2
2) Generates additional samples into the v2 directory via isolated workers

Usage:
  eval "$(conda shell.zsh hook)" && conda activate flow-synth
  python tools/prepare_polymax_v2_dataset.py \
    --src datasets/polymax_dataset \
    --dst datasets/polymax_dataset_v2 \
    --add-single 6000 \
    --add-phrase 2500 \
    --bpm 115 \
    --pattern random
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def copy_dataset(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ('raw', 'mel', 'mfcc', 'wav'):
        (dst / sub).mkdir(exist_ok=True)
        if (src / sub).exists():
            for p in (src / sub).iterdir():
                if p.is_file():
                    target = dst / sub / p.name
                    if not target.exists():
                        shutil.copy2(str(p), str(target))


def run_isolated(dst: Path, count: int, mode: str, extra_args=None, seed=1234):
    if count <= 0:
        return
    cmd = [
        sys.executable, str(Path(__file__).parent / 'polymax_augment_isolated.py'),
        '--outdir', str(dst),
        '--count', str(count),
        '--mode', mode,
        '--retries', '1',
        '--seed', str(seed),
    ]
    if extra_args:
        cmd += extra_args
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description='Prepare PolyMAX v2 dataset (copy + augment)')
    ap.add_argument('--src', default='datasets/polymax_dataset')
    ap.add_argument('--dst', default='datasets/polymax_dataset_v2')
    ap.add_argument('--add-single', type=int, default=6000)
    ap.add_argument('--add-phrase', type=int, default=2500)
    ap.add_argument('--bpm', type=float, default=115.0)
    ap.add_argument('--pattern', choices=['up','down','random'], default='random')
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print(f'Cloning {src} -> {dst}')
    copy_dataset(src, dst)

    print('Augmenting: single notes')
    run_isolated(dst, args.add_single, 'single', seed=1001)

    print('Augmenting: phrases')
    run_isolated(dst, args.add_phrase, 'phrase', extra_args=['--bpm', str(args.bpm), '--pattern', args.pattern], seed=2001)

    print(f'Dataset ready at {dst}')


if __name__ == '__main__':
    main()
