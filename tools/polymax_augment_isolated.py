#!/usr/bin/env python3
"""
Isolated augmentation driver: spawns a fresh process per sample as per
DATASET_GENERATION_STANDARD.md. Includes basic quality validation.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def validate_wav(path: Path, min_rms: float = 0.005, max_peak: float = 0.99, tail_secs: float = 0.1) -> bool:
    try:
        y, sr = sf.read(str(path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if len(y) < 10:
            return False
        rms = float(np.sqrt(np.mean(y**2)))
        peak = float(np.max(np.abs(y)))
        tail_n = max(1, int(sr * tail_secs))
        tail_rms = float(np.sqrt(np.mean(y[-tail_n:]**2)))
        # Require adequate RMS, not clipping, and tail RMS not absurdly larger than overall
        if rms < min_rms:
            return False
        if peak > max_peak:
            return False
        if tail_rms > 3.0 * rms:
            return False
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description='Isolated PolyMAX augmentation driver')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--duration', type=float, default=4.0)
    ap.add_argument('--note', type=int, default=-1)
    ap.add_argument('--velocity', type=int, default=100)
    ap.add_argument('--plugin', default='/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3')
    ap.add_argument('--strategy', choices=['uniform', 'jitter'], default='jitter')
    ap.add_argument('--mode', choices=['single', 'phrase', 'sweep', 'preset_jitter'], default='single')
    ap.add_argument('--bpm', type=float, default=120.0)
    ap.add_argument('--pattern', choices=['up', 'down', 'random'], default='random')
    ap.add_argument('--sweep-params', default='filter_cutoff_freq')
    ap.add_argument('--sweep-segments', type=int, default=4)
    ap.add_argument('--preset-dir', type=str, default='')
    ap.add_argument('--jitter-std', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--retries', type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        base_seed = int(args.seed) + i
        cmd = [
            sys.executable, str(Path(__file__).parent / 'polymax_augment_worker.py'),
            '--outdir', str(args.outdir),
            '--index', str(i),
            '--duration', str(args.duration),
            '--note', str(args.note),
            '--velocity', str(args.velocity),
            '--plugin', args.plugin,
            '--strategy', args.strategy,
            '--mode', args.mode,
            '--bpm', str(args.bpm),
            '--pattern', args.pattern,
            '--sweep-params', args.sweep_params,
            '--sweep-segments', str(args.sweep_segments),
            '--preset-dir', args.preset_dir,
            '--jitter-std', str(args.jitter_std),
            '--seed', str(base_seed)
        ]

        ok = False
        attempts = 0
        while attempts <= max(0, args.retries) and not ok:
            attempts += 1
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                pass
            # Validate
            wav = outdir / 'wav' / f'aug_{i:06d}.wav'
            ok = validate_wav(wav)
            if not ok:
                # tweak seed
                cmd[-1] = str(base_seed + attempts)
                time.sleep(0.2)
        # Inter-sample delay to let plugin clean up per standard
        time.sleep(0.2)

    print(f'Completed isolated augmentation: {args.count} samples -> {args.outdir}')


if __name__ == '__main__':
    main()

