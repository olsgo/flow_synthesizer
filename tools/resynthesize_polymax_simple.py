#!/usr/bin/env python3
"""
Lightweight PolyMAX resynthesis (no Torch)

Given an input audio file, this script:
- Loads the UAD PolyMAX VST3 via pedalboard
- Estimates a simple amplitude envelope from the audio
- Applies a reasonable PolyMAX parameter setup
- Triggers a MIDI note to render comparable audio

This is a best-effort resynthesis when the ML model isn't available.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
import librosa
from pedalboard import load_plugin


VST3_PATH = '/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3'


def analyze_envelope(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Estimate simple amp envelope characteristics from audio RMS.

    Returns seconds for attack/decay/release and sustain level (0..100 for PolyMAX).
    """
    # Frame-based RMS
    hop = 512
    win = 1024
    rms = librosa.feature.rms(y=audio, frame_length=win, hop_length=hop, center=True)[0]
    if len(rms) == 0:
        return dict(attack=0.01, decay=0.2, sustain=70.0, release=0.3)

    t = np.arange(len(rms)) * (hop / sr)
    rms_norm = rms / (np.max(rms) + 1e-9)

    # Attack: time to reach 90% of peak from start
    try:
        peak_idx = int(np.argmax(rms_norm))
        first_idx = int(np.argmax(rms_norm > 0.05))
        ninety_idx = first_idx + int(np.argmax(rms_norm[first_idx:peak_idx+1] >= 0.9))
        attack = max(0.001, t[ninety_idx] - t[first_idx])
    except Exception:
        attack = 0.01

    # Sustain: median level over middle third of non-silent region
    nz = np.where(rms_norm > 0.05)[0]
    if len(nz) > 10:
        lo = nz[len(nz)//3]
        hi = nz[2*len(nz)//3]
        sustain_level = float(np.median(rms_norm[lo:hi]))
    else:
        sustain_level = float(np.median(rms_norm))
    sustain_percent = float(np.clip(sustain_level * 100.0, 0.0, 100.0))

    # Decay: time from peak to sustain level
    try:
        peak_idx = int(np.argmax(rms_norm))
        # first index after peak within 10% of sustain level
        target = max(0.1, sustain_level)
        decay_offset = np.argmax(rms_norm[peak_idx:] <= target)
        decay = float(np.clip(t[peak_idx + decay_offset] - t[peak_idx], 0.002, 8.0))
    except Exception:
        decay = 0.2

    # Release: tail from last 50% level to end
    try:
        half_idxs = np.where(rms_norm <= 0.5)[0]
        last_half = int(half_idxs[-1]) if len(half_idxs) else len(rms_norm) - 1
        release = float(np.clip(t[-1] - t[last_half], 0.01, 6.0))
    except Exception:
        release = 0.3

    return dict(attack=float(attack), decay=float(decay), sustain=sustain_percent, release=float(release))


def apply_polymax_params(polymax, params: Dict[str, float]) -> None:
    """Set a safe subset of PolyMAX parameters with appropriate units."""
    # Safe defaults
    try:
        if hasattr(polymax, 'master_volume'):
            polymax.master_volume = float(params.get('master_volume', 60.0))  # 0..100
    except Exception:
        pass

    # Envelope (units are seconds for A/D/R, 0..100 for sustain)
    try:
        if hasattr(polymax, 'amp_env_attack'):
            polymax.amp_env_attack = float(params.get('amp_env_attack', 0.01))
        if hasattr(polymax, 'amp_env_decay'):
            polymax.amp_env_decay = float(params.get('amp_env_decay', 0.2))
        if hasattr(polymax, 'amp_env_sustain'):
            polymax.amp_env_sustain = float(params.get('amp_env_sustain', 70.0))
        if hasattr(polymax, 'amp_env_release'):
            polymax.amp_env_release = float(params.get('amp_env_release', 0.3))
    except Exception:
        pass

    # Basic oscillator setup for a solid tone
    try:
        if hasattr(polymax, 'osc_1_volume'):
            polymax.osc_1_volume = float(params.get('osc_1_volume', 80.0))
        if hasattr(polymax, 'osc_2_volume'):
            polymax.osc_2_volume = float(params.get('osc_2_volume', 0.0))
        if hasattr(polymax, 'noise_volume'):
            polymax.noise_volume = float(params.get('noise_volume', 0.0))
        if hasattr(polymax, 'osc_1_shape') and isinstance(params.get('osc_1_shape', 'SAWTOOTH'), str):
            polymax.osc_1_shape = params.get('osc_1_shape', 'SAWTOOTH')
    except Exception:
        pass

    # Filter cutoff heuristic from spectral centroid (in Hz). PolyMAX expects Hz-like values.
    if 'filter_cutoff_freq' in params and hasattr(polymax, 'filter_cutoff_freq'):
        try:
            polymax.filter_cutoff_freq = float(params['filter_cutoff_freq'])
        except Exception:
            pass


def render_polymax(audio_seconds: float,
                   note_seconds: float,
                   sample_rate: int = 44100,
                   midi_note: int = 60,
                   midi_velocity: int = 110) -> np.ndarray:
    """Render audio_seconds of audio from PolyMAX given a note lasting note_seconds."""
    polymax = load_plugin(VST3_PATH)

    # Ensure instrument mode behavior and sane defaults
    try:
        polymax.reset()
    except Exception:
        pass

    # MIDI creation uses pedalboard.MIDIMessage if available
    try:
        from pedalboard import MIDIMessage
        midi_messages = [
            MIDIMessage.note_on(note=int(midi_note), velocity=int(midi_velocity), time=0.0),
            MIDIMessage.note_off(note=int(midi_note), velocity=int(midi_velocity), time=float(note_seconds)),
        ]
    except Exception:
        class _CompatMidi:
            def __init__(self, status, note, velocity, time):
                self._data = bytes([status, note, velocity])
                self.time = time
            def bytes(self):
                return self._data
        midi_messages = [
            _CompatMidi(0x90, int(midi_note), int(midi_velocity), 0.0),
            _CompatMidi(0x80, int(midi_note), int(midi_velocity), float(note_seconds)),
        ]

    audio = polymax(
        midi_messages,
        duration=float(audio_seconds),
        sample_rate=int(sample_rate),
        num_channels=2,
        reset=False,
    )

    if audio.ndim == 1:
        return audio
    # Return mono mixdown for comparison simplicity
    return np.mean(audio, axis=0)


def main():
    p = argparse.ArgumentParser(description='Resynthesize audio with PolyMAX without ML model')
    p.add_argument('--audio', required=True, help='Path to target audio (wav)')
    p.add_argument('--outdir', help='Output directory (default: auto timestamp)')
    p.add_argument('--seconds', type=float, default=None, help='Render length; defaults to input length (clipped to 6s)')
    args = p.parse_args()

    if not os.path.exists(VST3_PATH):
        raise FileNotFoundError(f"PolyMAX VST not found at {VST3_PATH}")

    # Load and analyze target
    y, sr = librosa.load(args.audio, sr=44100)
    target_len = len(y) / sr
    render_len = float(np.clip(args.seconds or target_len, 1.0, 6.0))
    note_len = max(0.2, render_len - 0.25)

    env = analyze_envelope(y, sr)

    # Spectral centroid heuristic for filter cutoff
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    filter_cut = float(np.clip(centroid * 1.5, 200.0, 18000.0))

    # Estimate fundamental and choose closest MIDI note
    try:
        f0 = librosa.yin(y, fmin=50, fmax=2000, sr=sr)
        f0 = float(np.median(f0[np.isfinite(f0)]))
        midi_est = 69 + 12 * np.log2(max(1e-6, f0) / 440.0)
        midi_note = int(np.clip(np.round(midi_est), 21, 108))
        # Fine detune in semitone units (-1..1)
        cents = float((midi_est - midi_note) * 100.0)
        fine_tune = float(np.clip(cents / 100.0, -1.0, 1.0))
    except Exception:
        midi_note = 60
        fine_tune = 0.0

    # Prepare output dir
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = args.outdir or f'resynth_results_{ts}'
    os.makedirs(outdir, exist_ok=True)

    # Configure plugin and render
    polymax = load_plugin(VST3_PATH)
    try:
        polymax.reset()
    except Exception:
        pass

    params = {
        'master_volume': 60.0,
        'amp_env_attack': env['attack'],
        'amp_env_decay': env['decay'],
        'amp_env_sustain': env['sustain'],
        'amp_env_release': env['release'],
        'osc_1_volume': 80.0,
        'osc_2_volume': 0.0,
        'noise_volume': 0.0,
        'osc_1_shape': 'SAWTOOTH',
        'filter_cutoff_freq': filter_cut,
        'osc_1_fine_tune': fine_tune,
    }
    apply_polymax_params(polymax, params)

    # Render
    audio = render_polymax(audio_seconds=render_len, note_seconds=note_len, sample_rate=44100, midi_note=midi_note)

    # Save outputs and a quick comparison metric
    out_orig = os.path.join(outdir, 'original_audio.wav')
    out_syn = os.path.join(outdir, 'resynthesized_audio.wav')
    sf.write(out_orig, y, 44100)
    sf.write(out_syn, audio, 44100)

    # Simple similarity metrics
    min_len = min(len(y), len(audio))
    y_t = y[:min_len]
    a_t = audio[:min_len]
    mse = float(np.mean((y_t - a_t) ** 2))
    corr = float(np.corrcoef(y_t, a_t)[0, 1]) if min_len > 1 else 0.0
    report = {
        'input': os.path.abspath(args.audio),
        'output_dir': os.path.abspath(outdir),
        'env': env,
        'filter_cutoff_hz': filter_cut,
        'midi_note': int(midi_note),
        'osc_1_fine_tune': float(fine_tune),
        'render_seconds': render_len,
        'note_seconds': note_len,
        'mse': mse,
        'correlation': corr,
    }
    with open(os.path.join(outdir, 'comparison_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
