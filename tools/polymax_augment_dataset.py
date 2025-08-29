#!/usr/bin/env python3
"""
PolyMAX Dataset Augmentation

Generates synthetic training samples by randomizing PolyMAX parameters/state,
rendering audio via pedalboard, and saving flow-synth compatible files:

- raw/*.npz with {'param': dict[name->value in 0..1], 'chars': (10,3) zeros, 'audio': mono @ 22050}
- mel/*.npy with 64x80 mel spectrogram (linear power)
- mfcc/*.npy with 64x80 MFCC (16 coefficients tiled to 64 rows)

Usage:
  eval "$(conda shell.zsh hook)" && conda activate flow-synth
  python tools/polymax_augment_dataset.py \
    --outdir datasets/polymax_aug \
    --count 1000 \
    --duration 4.0 \
    --seed 42

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


def sample_param_vector(names: List[str], details: Dict[str, dict], strategy: str = 'jitter') -> Dict[str, float]:
    """Return dict of name->normalized_value in [0,1]."""
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
        # Booleans
        if isinstance(min_v, bool) and isinstance(max_v, bool):
            vec[n] = float(random.random() > 0.5)
            continue
        # Numeric with known range
        if isinstance(min_v, (int, float)) and isinstance(max_v, (int, float)) and max_v > min_v:
            if strategy == 'jitter' and n in SAFE_DEFAULTS:
                base = SAFE_DEFAULTS[n]
                val = base + np.random.normal(0.0, 0.1)
            else:
                # Bias toward mid-range slightly for stability
                u = random.random()
                val = 0.25 * u + 0.375  # [~0.375, ~0.625]
            vec[n] = float(max(0.0, min(1.0, val)))
            continue
        # Fallback: uniform 0..1
        vec[n] = float(random.random())
    return vec


def render_note(engine, note: int, velocity: int, duration: float, note_length: float) -> np.ndarray:
    engine.render_patch(note, velocity, note_length, duration, warm_up=False)
    audio = engine.get_audio_frames()  # (2, N) or (N,)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    return audio.astype(np.float32)

def build_phrase_messages(duration: float, bpm: float, base_note: int, pattern: str, velocity: int) -> List[object]:
    """Create MIDI messages for a simple phrase over the given duration."""
    beat = 60.0 / max(1e-3, bpm)
    step = max(0.125 * beat, 0.05)  # 1/8 note or min 50ms
    n_steps = max(1, int(duration / step))
    offsets = [-12, -7, -5, 0, 2, 4, 7, 12]
    if pattern == 'up':
        seq = [(base_note + offsets[i % len(offsets)]) for i in range(n_steps)]
    elif pattern == 'down':
        seq = [(base_note + offsets[::-1][i % len(offsets)]) for i in range(n_steps)]
    else:
        seq = [base_note + random.choice(offsets) for _ in range(n_steps)]
    # Try to use pedalboard.MIDIMessage, fallback to compat wrapper
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
    # Use engine.render_midi if available
    if hasattr(engine, 'render_midi'):
        engine.render_midi(messages, duration, warm_up=False)
        audio = engine.get_audio_frames()
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        return audio.astype(np.float32)
    # Fallback to silence if not available
    return np.zeros(int(duration * 44100), dtype=np.float32)

def sweep_values(n_segments: int, start: float, end: float) -> List[float]:
    return [float(start + (end - start) * (i / max(1, n_segments - 1))) for i in range(n_segments)]

def render_sweep(engine, rev_idx, base_params: Dict[str, float], sweep_params: List[str],
                 duration: float, velocity: int, note: int, n_segments: int = 4) -> Tuple[np.ndarray, List[dict]]:
    seg_len = duration / max(1, n_segments)
    audio_pieces = []
    automation = []
    # Random start/end for each swept param
    sweep_points = {p: (random.random(), random.random()) for p in sweep_params}
    for i in range(max(1, n_segments)):
        t0 = i * seg_len
        cur_params = base_params.copy()
        for p, (s, e) in sweep_points.items():
            vals = sweep_values(max(1, n_segments), s, e)
            cur_params[p] = vals[i]
            automation.append({'time': float(t0), 'param': p, 'value': float(vals[i])})
        # Apply patch and render note segment
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
    # Mel 128x173
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, n_mels=128, hop_length=512, fmin=30, fmax=11000, center=True
    )
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])), mode='constant')
    elif mel.shape[1] > target_frames:
        mel = mel[:, :target_frames]
    # MFCC 13x173
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512, center=True)
    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])), mode='edge')
    elif mfcc.shape[1] > target_frames:
        mfcc = mfcc[:, :target_frames]
    return mel.astype(np.float32), mfcc.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description='Augment PolyMAX dataset by randomizing parameters and rendering audio')
    ap.add_argument('--outdir', required=True, help='Output dataset directory (will create raw/mel/mfcc/wav)')
    ap.add_argument('--count', type=int, default=100, help='Number of samples to generate')
    ap.add_argument('--duration', type=float, default=4.0, help='Render duration (seconds)')
    ap.add_argument('--note', type=int, default=-1, help='Fixed MIDI note (if <0, random in [48,72])')
    ap.add_argument('--velocity', type=int, default=100, help='MIDI velocity')
    ap.add_argument('--seed', type=int, default=None, help='Random seed')
    ap.add_argument('--plugin', default='/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3', help='PolyMAX VST3 path')
    ap.add_argument('--strategy', choices=['uniform', 'jitter'], default='jitter', help='Parameter sampling strategy')
    # Modes and options
    ap.add_argument('--mode', choices=['single', 'phrase', 'sweep', 'preset_jitter'], default='single', help='Augmentation mode')
    ap.add_argument('--bpm', type=float, default=120.0, help='Phrase tempo')
    ap.add_argument('--pattern', choices=['up', 'down', 'random'], default='random', help='Phrase pattern')
    ap.add_argument('--sweep-params', default='filter_cutoff_freq', help='Comma-separated list of params to sweep')
    ap.add_argument('--sweep-segments', type=int, default=4, help='Number of segments for sweep mode')
    ap.add_argument('--preset-dir', type=str, default='', help='Directory with .vstpreset files for preset_jitter mode')
    ap.add_argument('--jitter-std', type=float, default=0.1, help='Stddev for jitter around preset values (normalized)')
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    out = ensure_dirs(Path(args.outdir))
    names, details = load_param_schema()

    # Create synth engine
    engine, generator, param_defaults, rev_idx = create_synth('polymax_dataset', 'polymax', args.plugin)
    
    # Preset list if requested
    preset_files = []
    if args.preset_dir and os.path.isdir(args.preset_dir):
        preset_files = [str(p) for p in Path(args.preset_dir).glob('**/*.vstpreset')]

    for i in range(args.count):
        note_events: List[dict] = []
        automation: List[dict] = []
        # Sample parameter vector (normalized 0..1 per name)
        params = sample_param_vector(names, details, strategy=args.strategy)
        # Enforce safe defaults (hard constraints)
        for k, v in SAFE_DEFAULTS.items():
            if k in params:
                if args.strategy == 'jitter':
                    # Keep not less than default for critical loudness params
                    if k in ('osc_1_volume', 'amp_env_sustain'):
                        params[k] = float(max(v, params[k]))
                else:
                    params[k] = float(v)
        # Ensure master bypass off and arp out
        if 'master_bypass' in params:
            params['master_bypass'] = 0.0
        if 'arp_enable' in params:
            params['arp_enable'] = 0.0  # OUT

        # Optional preset-seeded jitter
        if args.mode == 'preset_jitter' and preset_files:
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

        # MIDI settings
        note = args.note if args.note >= 0 else random.randint(48, 72)
        # Render per mode
        if args.mode == 'phrase':
            msgs = build_phrase_messages(args.duration, args.bpm, note, args.pattern, args.velocity)
            for k in range(0, len(msgs), 2):
                try:
                    t0 = float(getattr(msgs[k], 'time', 0.0))
                    t1 = float(getattr(msgs[k+1], 'time', 0.0))
                    note_events.append({'t0': t0, 't1': t1})
                except Exception:
                    pass
            y44 = render_phrase(engine, msgs, args.duration)
        elif args.mode == 'sweep':
            sweep_list = [s.strip() for s in args.sweep_params.split(',') if s.strip()]
            y44, automation = render_sweep(engine, rev_idx, params.copy(), sweep_list, args.duration, args.velocity, note, args.sweep_segments)
            seg_len = args.duration / max(1, args.sweep_segments)
            for s in range(max(1, args.sweep_segments)):
                t0 = s * seg_len
                t1 = min(args.duration, (s + 1) * seg_len)
                note_events.append({'t0': float(t0), 't1': float(t1)})
        else:
            # single or preset_jitter
            y44 = render_note(engine, note, args.velocity, args.duration, max(0.25, args.duration - 0.05))

        # Resample to 22.05k for dataset
        y = librosa.resample(y44, orig_sr=44100, target_sr=22050)

        # Normalize to prevent clipping and keep consistent RMS
        if np.max(np.abs(y)) > 0:
            y = 0.8 * y / np.max(np.abs(y))

        # Features
        mel, mfcc = audio_to_features(y, sr=22050)

        # Filenames
        base = f"aug_{i:06d}"
        raw_path = out['raw'] / f"{base}.npz"
        mel_path = out['mel'] / f"{base}.npy"
        mfcc_path = out['mfcc'] / f"{base}.npy"
        wav_path = out['wav'] / f"{base}.wav"

        # Save audio (22050 mono)
        sf.write(str(wav_path), y, 22050)

        # Save features
        np.save(str(mel_path), mel)
        np.save(str(mfcc_path), mfcc)

        # Save raw npz (param dict and placeholder chars)
        chars = np.zeros((10, 3), dtype=np.float32)
        np.savez_compressed(
            str(raw_path),
            param={k: float(params.get(k, 0.5)) for k in names},
            chars=chars,
            audio=y.astype(np.float32)
        )

        # Write sidecar JSON with events/automation for richer training if needed
        meta_json = {
            'mode': args.mode,
            'note_events': note_events,
            'automation': automation,
        }
        with open(str(raw_path).replace('.npz', '.json'), 'w') as f:
            json.dump(meta_json, f, indent=2)

        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{args.count}")

    print(f"Done. Augmented dataset written to {args.outdir}")


if __name__ == '__main__':
    main()
