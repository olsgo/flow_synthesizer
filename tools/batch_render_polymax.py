#!/usr/bin/env python3
"""
Batch render PolyMAX dataset (audio + 66-dim parameter vectors) and manifest.

- Uses pedalboard backend via code.synth.synthesize.create_synth
- Enforces exactly 66 parameters in the order from params_schema.json
- Saves WAVs and JSON arrays under polymax_dataset/{audio,params}/{train,val,test}
- Writes manifest.csv with: split,audio_path,params_path,preset_name

Example:
  python tools/batch_render_polymax.py \
    --outdir polymax_dataset \
    --count 1000 \
    --duration 4.0 \
    --sr 44100 \
    --plugin "/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3"
"""

import argparse
import csv
import json
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))  # repo root
from code.synth.synthesize import create_synth, midiname2num


def ensure_dataset_dirs(root: Path) -> Dict[str, Dict[str, Path]]:
    audio = {
        'train': root / 'audio' / 'train',
        'val': root / 'audio' / 'val',
        'test': root / 'audio' / 'test',
    }
    params = {
        'train': root / 'params' / 'train',
        'val': root / 'params' / 'val',
        'test': root / 'params' / 'test',
    }
    for d in list(audio.values()) + list(params.values()):
        d.mkdir(parents=True, exist_ok=True)
    return {'audio': audio, 'params': params}


def load_schema(schema_path: Path) -> List[str]:
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    names = list(schema.get('parameter_order', []))
    if len(names) != 66:
        raise ValueError(f"Schema must define exactly 66 parameters, got {len(names)}")
    return names


def sample_params(names: List[str], strategy: str = 'jitter') -> Dict[str, float]:
    # Lightweight sampler, biased toward mid-range for stability
    vec: Dict[str, float] = {}
    for n in names:
        if strategy == 'jitter':
            base = 0.5
            val = base + np.random.normal(0.0, 0.2)
        else:
            u = random.random()
            val = 0.25 * u + 0.375
        vec[n] = float(max(0.0, min(1.0, val)))
    return vec


def to_vector(param_map: Dict[str, float], names: List[str]) -> List[float]:
    v = [float(param_map.get(n, 0.5)) for n in names]
    if len(v) != 66:
        raise RuntimeError("Parameter vector must be length 66")
    # Clip strictly to [0,1]
    return [max(0.0, min(1.0, float(x))) for x in v]


def render_note(engine, note: int, velocity: int, duration: float, note_len: float) -> np.ndarray:
    engine.render_patch(int(note), int(velocity), float(note_len), float(duration), warm_up=False)
    y = engine.get_audio_frames()  # shape (2, N) or (N,)
    y = np.asarray(y)
    if y.ndim == 2:
        # average to mono
        y = y.mean(axis=0)
    y = y.astype(np.float32)
    # Optional sanity trim/pad to expected len
    return y


def validate_audio(y: np.ndarray, sr: int, min_rms: float = 0.003) -> bool:
    if y.size < max(16, sr // 100):
        return False
    rms = float(np.sqrt(np.mean((y.astype(np.float32) ** 2))))
    return rms >= min_rms


def main():
    ap = argparse.ArgumentParser(description='Batch render PolyMAX dataset')
    ap.add_argument('--outdir', type=str, default='polymax_dataset')
    ap.add_argument('--count', type=int, default=1000)
    ap.add_argument('--train-ratio', type=float, default=0.9)
    ap.add_argument('--val-ratio', type=float, default=0.05)
    ap.add_argument('--test-ratio', type=float, default=0.05)
    ap.add_argument('--duration', type=float, default=4.0)
    ap.add_argument('--sr', type=int, default=44100)
    ap.add_argument('--note', type=int, default=60)
    ap.add_argument('--velocity', type=int, default=100)
    ap.add_argument('--plugin', type=str, default='/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3')
    ap.add_argument('--strategy', choices=['uniform','jitter'], default='jitter')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.outdir)
    paths = ensure_dataset_dirs(root)

    # Copy schema into dataset root for portability
    repo_schema = Path('params_schema.json')
    if not repo_schema.exists():
        raise SystemExit('params_schema.json not found in repo root')
    dataset_schema_path = root / 'params_schema.json'
    root.mkdir(parents=True, exist_ok=True)
    if not dataset_schema_path.exists():
        dataset_schema_path.write_text(repo_schema.read_text())

    names = load_schema(dataset_schema_path)

    # Init engine (pedalboard)
    engine, generator, defaults, rev_idx = create_synth('polymax_dataset', 'polymax', args.plugin)

    # Prepare manifest
    manifest_path = root / 'manifest.csv'
    mf = open(manifest_path, 'w', newline='')
    writer = csv.writer(mf)
    writer.writerow(['split', 'audio_path', 'params_path', 'preset_name'])

    # Compute splits
    n_train = int(round(args.count * args.train_ratio))
    n_val = int(round(args.count * args.val_ratio))
    n_test = args.count - n_train - n_val
    split_bounds = (
        ('train', 0, n_train),
        ('val', n_train, n_train + n_val),
        ('test', n_train + n_val, args.count),
    )
    index_to_split = {}
    for split, start, end in split_bounds:
        for i in range(start, end):
            index_to_split[i] = split

    expected_len = int(args.duration * args.sr)

    for i in range(args.count):
        split = index_to_split[i]
        base = f"aug_{i:06d}"
        # Sample parameters and convert to vector
        pmap = sample_params(names, args.strategy)
        pvec = to_vector(pmap, names)

        # Apply patch
        patch = midiname2num(pmap, rev_idx)
        engine.set_patch(patch)

        # Render
        y = render_note(engine, args.note, args.velocity, args.duration, max(0.2, args.duration - 0.05))

        # Enforce sample rate/length: engine runs at 44100; enforce length anyway
        if y.size == 0:
            # Try render one more time
            y = render_note(engine, args.note, args.velocity, args.duration, max(0.2, args.duration - 0.05))
        if y.size < expected_len:
            y = np.pad(y, (0, expected_len - y.size))
        elif y.size > expected_len:
            y = y[:expected_len]

        # Validate; if too quiet, resample params once
        if not validate_audio(y, args.sr):
            pmap = sample_params(names, args.strategy)
            pvec = to_vector(pmap, names)
            patch = midiname2num(pmap, rev_idx)
            engine.set_patch(patch)
            y = render_note(engine, args.note, args.velocity, args.duration, max(0.2, args.duration - 0.05))
            if y.size < expected_len:
                y = np.pad(y, (0, expected_len - y.size))
            elif y.size > expected_len:
                y = y[:expected_len]

        # Save
        wav_path = paths['audio'][split] / f"{base}.wav"
        prm_path = paths['params'][split] / f"{base}.json"
        sf.write(str(wav_path), y, args.sr)
        with open(prm_path, 'w') as f:
            json.dump({
                'parameter_vector': pvec,
                'parameter_count': 66,
                'schema': 'params_schema.json'
            }, f, indent=2)

        writer.writerow([
            split,
            str(wav_path.relative_to(root)),
            str(prm_path.relative_to(root)),
            base
        ])

    mf.close()
    print(f"Wrote manifest: {manifest_path}")
    print(f"Audio dir: {root / 'audio'} | Params dir: {root / 'params'}")


if __name__ == '__main__':
    main()
