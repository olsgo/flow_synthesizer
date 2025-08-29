#!/usr/bin/env python3
"""
PolyMAX augmentation worker: generates exactly one sample in an isolated process.
Conforms to DATASET_GENERATION_STANDARD.md by reinitializing plugin per sample.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa
import soundfile as sf

# Project imports
import sys
sys.path.append(str(Path(__file__).parent.parent))  # repo root
sys.path.append(str(Path(__file__).parent.parent / 'code'))
from synth.synthesize import create_synth, midiname2num


def ensure_dirs(base: Path) -> Dict[str, Path]:
    raw = base / 'raw'
    mel = base / 'mel'
    mfcc = base / 'mfcc'
    wav = base / 'wav'
    for d in (raw, mel, mfcc, wav):
        d.mkdir(parents=True, exist_ok=True)
    return {'raw': raw, 'mel': mel, 'mfcc': mfcc, 'wav': wav}


def load_param_schema() -> Tuple[List[str], Dict[str, dict]]:
    schema_path = Path('params_schema.json')
    details_path = Path('polymax_param_details.json')
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    names: List[str] = list(schema.get('parameter_order', []))
    details: Dict[str, dict] = {}
    if details_path.exists():
        try:
            with open(details_path, 'r') as f:
                det = json.load(f)
            details = det.get('parameter_details', {})
        except Exception:
            details = {}
    return names, details


ENUM_CHOICES = {
    'polyphony': ['MONO', 'POLY'],
    'note_trigger_mode': ['LEGATO', 'RETRIG'],
    'lfo_shape': ['TRIANGLE', 'RAMP_UP', 'RAMP_DOWN', 'SQUARE', 'RANDOM'],
    'noise_color': ['PINK', 'WHITE'],
    'filter_mode': ['LP', 'HP', 'BP', 'NOTCH'],
    'filter_slope': ['2P', '4P'],
    'unison': ['OFF', 'ON'],
    'mod_fx_type': ['PHASER', 'FLANGER', 'CHORUS'],
    'space_fx_type': ['HALL', 'SPRING'],
    'arp_enable': ['OUT', 'IN'],
}

SAFE_DEFAULTS = {
    'master_bypass': 0.0,
    'polyphony': 1.0,
    'osc_1_volume': 0.8,
    'osc_2_volume': 0.1,
    'noise_volume': 0.0,
    'filter_cutoff_freq': 0.85,
    'filter_resonance': 0.15,
    'filter_env_amt': 0.0,
    'amp_env_attack': 0.02,
    'amp_env_decay': 0.25,
    'amp_env_sustain': 0.75,
    'amp_env_release': 0.25,
}


def sample_param_vector(names: List[str], details: Dict[str, dict], strategy: str = 'jitter', jitter_std: float = 0.1) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for n in names:
        d = details.get(n, {})
        min_v = d.get('min_value', None)
        max_v = d.get('max_value', None)
        if n in ENUM_CHOICES:
            choices = ENUM_CHOICES[n]
            idx = random.randrange(len(choices))
            vec[n] = 0.0 if len(choices) == 1 else idx / float(len(choices) - 1)
            continue
        if isinstance(min_v, bool) and isinstance(max_v, bool):
            vec[n] = float(random.random() > 0.5)
            continue
        if isinstance(min_v, (int, float)) and isinstance(max_v, (int, float)) and max_v > min_v:
            if strategy == 'jitter' and n in SAFE_DEFAULTS:
                base = SAFE_DEFAULTS[n]
                val = base + np.random.normal(0.0, jitter_std)
            else:
                u = random.random()
                val = 0.25 * u + 0.375
            vec[n] = float(max(0.0, min(1.0, val)))
            continue
        vec[n] = float(random.random())
    return vec


def render_note(engine, note: int, velocity: int, duration: float, note_length: float) -> np.ndarray:
    engine.render_patch(note, velocity, note_length, duration, warm_up=False)
    audio = engine.get_audio_frames()
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    return audio.astype(np.float32)


def build_phrase_messages(duration: float, bpm: float, base_note: int, pattern: str, velocity: int) -> List[object]:
    beat = 60.0 / max(1e-3, bpm)
    step = max(0.125 * beat, 0.05)
    n_steps = max(1, int(duration / step))
    offsets = [-12, -7, -5, 0, 2, 4, 7, 12]
    if pattern == 'up':
        seq = [(base_note + offsets[i % len(offsets)]) for i in range(n_steps)]
    elif pattern == 'down':
        seq = [(base_note + offsets[::-1][i % len(offsets)]) for i in range(n_steps)]
    else:
        seq = [base_note + random.choice(offsets) for _ in range(n_steps)]
    try:
        from pedalboard import MIDIMessage as _MM
        HAVE_MM = True
    except Exception:
        HAVE_MM = False
        _MM = None
    msgs = []
    cur = 0.0
    for i in range(n_steps):
        t0 = cur
        t1 = min(duration, t0 + (step * 0.9))
        n = int(np.clip(seq[i], 24, 108))
        if HAVE_MM:
            msgs.append(_MM.note_on(note=n, velocity=int(velocity), time=float(t0)))
            msgs.append(_MM.note_off(note=n, velocity=int(velocity), time=float(max(t0, t1 - 0.02))))
        else:
            class _CompatMidi:
                def __init__(self, status, note, velocity, time):
                    self._data = bytes([status, note, velocity])
                    self.time = time
                def bytes(self):
                    return self._data
            msgs.append(_CompatMidi(0x90, n, int(velocity), float(t0)))
            msgs.append(_CompatMidi(0x80, n, int(velocity), float(max(t0, t1 - 0.02))))
        cur += step
    return msgs


def render_phrase(engine, messages: List[object], duration: float) -> np.ndarray:
    if hasattr(engine, 'render_midi'):
        engine.render_midi(messages, duration, warm_up=False)
        audio = engine.get_audio_frames()
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        return audio.astype(np.float32)
    return np.zeros(int(duration * 44100), dtype=np.float32)


def sweep_values(n_segments: int, start: float, end: float) -> List[float]:
    return [float(start + (end - start) * (i / max(1, n_segments - 1))) for i in range(n_segments)]


def render_sweep(engine, rev_idx, base_params: Dict[str, float], sweep_params: List[str],
                 duration: float, velocity: int, note: int, n_segments: int = 4) -> Tuple[np.ndarray, List[dict]]:
    seg_len = duration / max(1, n_segments)
    audio_pieces = []
    automation = []
    sweep_points = {p: (random.random(), random.random()) for p in sweep_params}
    for i in range(max(1, n_segments)):
        t0 = i * seg_len
        cur_params = base_params.copy()
        for p, (s, e) in sweep_points.items():
            vals = sweep_values(max(1, n_segments), s, e)
            cur_params[p] = vals[i]
            automation.append({'time': float(t0), 'param': p, 'value': float(vals[i])})
        engine.set_patch(midiname2num(cur_params, rev_idx))
        y = render_note(engine, note, velocity, seg_len, max(0.1, seg_len - 0.02))
        audio_pieces.append(y)
    audio = np.concatenate(audio_pieces) if audio_pieces else np.zeros(int(duration * 44100), dtype=np.float32)
    return audio, automation


def audio_to_features(y: np.ndarray, sr: int = 22050) -> Tuple[np.ndarray, np.ndarray]:
    """Compute features consistent with existing polymax_dataset:
    - mel: 128 x 173 (n_fft=2048, hop=512, center=True)
    - mfcc: 13 x 173 (n_mfcc=13, hop=512)
    """
    target_frames = 173
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, n_mels=128, hop_length=512, fmin=30, fmax=11000, center=True
    )
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])), mode='constant')
    elif mel.shape[1] > target_frames:
        mel = mel[:, :target_frames]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512, center=True)
    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])), mode='edge')
    elif mfcc.shape[1] > target_frames:
        mfcc = mfcc[:, :target_frames]
    return mel.astype(np.float32), mfcc.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description='Isolated PolyMAX augmentation worker (one sample)')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--index', type=int, required=True)
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
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    out = ensure_dirs(Path(args.outdir))
    names, details = load_param_schema()

    # Create fresh engine (isolated process)
    engine, generator, param_defaults, rev_idx = create_synth('polymax_dataset', 'polymax', args.plugin)

    # Base params
    params = sample_param_vector(names, details, strategy=args.strategy, jitter_std=args.jitter_std)
    for k, v in SAFE_DEFAULTS.items():
        if k in params:
            if args.strategy == 'jitter':
                if k in ('osc_1_volume', 'amp_env_sustain'):
                    params[k] = float(max(v, params[k]))
            else:
                params[k] = float(v)
    if 'master_bypass' in params:
        params['master_bypass'] = 0.0
    if 'arp_enable' in params:
        params['arp_enable'] = 0.0

    # Optional preset jitter
    if args.mode == 'preset_jitter' and args.preset_dir and os.path.isdir(args.preset_dir):
        preset_files = [str(p) for p in Path(args.preset_dir).glob('**/*.vstpreset')]
        if preset_files:
            preset_path = random.choice(preset_files)
            try:
                engine.load_preset(preset_path)
            except Exception:
                pass
            plugin_param_names = list(getattr(engine.plugin, 'parameters').keys())
            for n, idx in rev_idx.items():
                if idx < len(plugin_param_names):
                    try:
                        base = float(engine.plugin.parameters[plugin_param_names[idx]].raw_value)
                    except Exception:
                        base = params.get(n, 0.5)
                    val = base + np.random.normal(0.0, args.jitter_std)
                    params[n] = float(np.clip(val, 0.0, 1.0))

    # Apply patch
    engine.set_patch(midiname2num(params, rev_idx))

    # Render according to mode
    note = args.note if args.note >= 0 else random.randint(48, 72)
    if args.mode == 'phrase':
        msgs = build_phrase_messages(args.duration, args.bpm, note, args.pattern, args.velocity)
        # minimal events info
        note_events = [{'t0': float(getattr(msgs[k], 'time', 0.0)), 't1': float(getattr(msgs[k+1], 'time', 0.0))}
                       for k in range(0, len(msgs), 2)]
        y44 = render_phrase(engine, msgs, args.duration)
        automation = []
    elif args.mode == 'sweep':
        sweep_list = [s.strip() for s in args.sweep_params.split(',') if s.strip()]
        y44, automation = render_sweep(engine, rev_idx, params.copy(), sweep_list, args.duration, args.velocity, note, args.sweep_segments)
        seg_len = args.duration / max(1, args.sweep_segments)
        note_events = [{'t0': float(s * seg_len), 't1': float(min(args.duration, (s + 1) * seg_len))}
                       for s in range(max(1, args.sweep_segments))]
    else:
        y44 = render_note(engine, note, args.velocity, args.duration, max(0.25, args.duration - 0.05))
        note_events, automation = [], []

    # Resample to dataset SR
    y = librosa.resample(y44, orig_sr=44100, target_sr=22050)
    # Normalize
    if np.max(np.abs(y)) > 0:
        y = 0.8 * y / np.max(np.abs(y))

    # Features
    mel, mfcc = audio_to_features(y, sr=22050)

    # Save
    base = f"aug_{args.index:06d}"
    raw_path = out['raw'] / f"{base}.npz"
    mel_path = out['mel'] / f"{base}.npy"
    mfcc_path = out['mfcc'] / f"{base}.npy"
    wav_path = out['wav'] / f"{base}.wav"

    sf.write(str(wav_path), y, 22050)
    np.save(str(mel_path), mel)
    np.save(str(mfcc_path), mfcc)

    chars = np.zeros((10, 3), dtype=np.float32)
    np.savez_compressed(
        str(raw_path),
        param={k: float(params.get(k, 0.5)) for k in names},
        chars=chars,
        audio=y.astype(np.float32)
    )

    sidecar = {
        'mode': args.mode,
        'note_events': note_events,
        'automation': automation,
    }
    with open(str(raw_path).replace('.npz', '.json'), 'w') as f:
        json.dump(sidecar, f, indent=2)


if __name__ == '__main__':
    main()
