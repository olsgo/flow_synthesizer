#!/usr/bin/env python3
"""
Compute perceptual sensitivity weights for each of the 66 PolyMAX parameters.

Approach:
- Sample K random baseline patches (normalized params in [0,1])
- For each parameter i, perturb by +/- delta in normalized domain while clamping [0,1]
- Render audio and compute log-mel distance vs baseline
- Aggregate across K and normalize weights to median=1.0

Outputs: sensitivity_weights.json mapping {name: weight}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import librosa

import sys
sys.path.append(str(Path(__file__).parent.parent))
from code.synth.synthesize import create_synth, midiname2num


def load_schema(schema_path: Path) -> List[str]:
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    names = list(schema.get('parameter_order', []))
    if len(names) != 66:
        raise ValueError('params_schema.json must list exactly 66 parameters')
    return names


def sample_params(names: List[str]) -> Dict[str, float]:
    # Mid-biased sampling for stability
    v = {}
    for n in names:
        u = np.random.rand()
        v[n] = float(0.3 * u + 0.35)
    return v


def render_note(engine, note: int, velocity: int, duration: float) -> np.ndarray:
    engine.render_patch(int(note), int(velocity), max(0.1, duration - 0.05), float(duration), warm_up=False)
    y = engine.get_audio_frames()
    y = np.asarray(y)
    if y.ndim == 2:
        y = y.mean(axis=0)
    return y.astype(np.float32)


def logmel(y: np.ndarray, sr: int = 44100) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64, fmin=30, fmax=sr//2)
    S_db = librosa.power_to_db(S + 1e-10)
    return S_db


def main():
    ap = argparse.ArgumentParser(description='Compute parameter sensitivity weights')
    ap.add_argument('--schema', type=str, default='params_schema.json')
    ap.add_argument('--plugin', type=str, default='/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3')
    ap.add_argument('--out', type=str, default='sensitivity_weights.json')
    ap.add_argument('--note', type=int, default=60)
    ap.add_argument('--velocity', type=int, default=100)
    ap.add_argument('--duration', type=float, default=4.0)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--trials', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    names = load_schema(Path(args.schema))
    engine, generator, defaults, rev_idx = create_synth('polymax_dataset', 'polymax', args.plugin)

    sr = 44100
    T = int(args.duration * sr)

    # Accumulate distances per parameter
    dists = {n: [] for n in names}

    for t in range(args.trials):
        base = sample_params(names)
        patch = midiname2num(base, rev_idx)
        engine.set_patch(patch)
        y0 = render_note(engine, args.note, args.velocity, args.duration)
        if y0.size < T:
            y0 = np.pad(y0, (0, T - y0.size))
        elif y0.size > T:
            y0 = y0[:T]
        F0 = logmel(y0, sr)

        for n in names:
            v = float(base[n])
            # +delta
            vp = min(1.0, v + args.delta)
            base_p = dict(base)
            base_p[n] = vp
            engine.set_patch(midiname2num(base_p, rev_idx))
            yp = render_note(engine, args.note, args.velocity, args.duration)
            if yp.size < T:
                yp = np.pad(yp, (0, T - yp.size))
            elif yp.size > T:
                yp = yp[:T]
            Fp = logmel(yp, sr)

            # -delta
            vm = max(0.0, v - args.delta)
            base_m = dict(base)
            base_m[n] = vm
            engine.set_patch(midiname2num(base_m, rev_idx))
            ym = render_note(engine, args.note, args.velocity, args.duration)
            if ym.size < T:
                ym = np.pad(ym, (0, T - ym.size))
            elif ym.size > T:
                ym = ym[:T]
            Fm = logmel(ym, sr)

            # Distance as average of +/- deltas
            dp = float(np.mean((Fp - F0) ** 2) ** 0.5)
            dm = float(np.mean((Fm - F0) ** 2) ** 0.5)
            d = 0.5 * (dp + dm)
            dists[n].append(d)

    # Aggregate and normalize
    weights = {}
    vals = []
    for n in names:
        w = float(np.median(dists[n])) if dists[n] else 0.0
        weights[n] = w
        vals.append(w)
    med = float(np.median(vals)) if vals else 1.0
    if med > 0:
        for n in names:
            weights[n] = float(weights[n] / med)

    with open(args.out, 'w') as f:
        json.dump({'weights': weights, 'median_normalized': True, 'delta': args.delta, 'trials': args.trials}, f, indent=2)

    print(f"Saved sensitivity weights to {args.out}")


if __name__ == '__main__':
    main()

