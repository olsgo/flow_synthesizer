#!/usr/bin/env python3
"""
Resynthesize an input WAV using UAD PolyMAX via pedalboard.

Pipeline:
1) Predict PolyMAX parameters from the input audio using a trained model.
2) Load the PolyMAX VST3, set parameters, and render a sustained note.
3) Match duration to input and evaluate spectrogram similarity.

Requires the PolyMAX VST3 at /Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3
and the project environment with pedalboard + torch.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# Project imports
sys.path.append(str(Path(__file__).parent.parent))  # add repo root
sys.path.append(str(Path(__file__).parent.parent / 'code'))
from predict_polymax_params import PolyMAXPredictor
from synth.synthesize import create_synth, midiname2num

# Reuse metrics from the GL tool
from tools.resynthesize_audio import compute_metrics, GLConfig
from typing import List


def estimate_midi_note(y: np.ndarray, sr: int) -> int:
    try:
        f0 = librosa.yin(y, fmin=40.0, fmax=2000.0, sr=sr)
        f0 = f0[np.isfinite(f0)]
        if f0.size == 0:
            return 60
        hz = float(np.median(f0))
        midi = int(np.clip(np.round(librosa.hz_to_midi(hz)), 36, 96))
        return midi
    except Exception:
        return 60


def detect_note_sequence(y: np.ndarray, sr: int) -> List[tuple]:
    """Detect a simple monophonic note sequence: list of (start_time, end_time, midi_note)."""
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, units='time')
        if len(onsets) == 0:
            return [(0.0, float(len(y)/sr), estimate_midi_note(y, sr))]
        # Append audio end as last boundary
        times = list(onsets) + [len(y) / float(sr)]
        # Use YIN to estimate pitch per frame
        f0 = librosa.yin(y, fmin=40.0, fmax=2000.0, sr=sr)
        seq = []
        for i in range(len(times) - 1):
            t0, t1 = float(times[i]), float(times[i + 1])
            if t1 - t0 < 0.05:
                continue
            i0 = max(0, int(t0 * sr))
            i1 = min(len(y), int(t1 * sr))
            fseg = f0[i0:i1]
            fseg = fseg[np.isfinite(fseg)]
            if fseg.size == 0:
                midi = estimate_midi_note(y[i0:i1], sr)
            else:
                midi = int(np.clip(np.round(librosa.hz_to_midi(float(np.median(fseg)))), 36, 96))
            seq.append((t0, t1, midi))
        return seq
    except Exception:
        return [(0.0, float(len(y)/sr), estimate_midi_note(y, sr))]


def prepare_polymax_engine(params_map: dict, plugin_path: str) -> tuple:
    engine, generator, param_defaults, rev_idx = create_synth('polymax_dataset', 'polymax', plugin_path)
    patch = midiname2num(params_map, rev_idx)
    engine.set_patch(patch)
    return engine, generator, rev_idx


def render_with_engine(engine, duration_s: float, midi_note: int, velocity: int) -> np.ndarray:
    render_len = float(duration_s)
    note_len = max(0.25, render_len - 0.05)
    engine.render_patch(midi_note, velocity, note_len, render_len, warm_up=False)
    audio = engine.get_audio_frames()
    if audio.ndim == 2:
        audio_mono = audio.mean(axis=0)
    else:
        audio_mono = audio.astype(np.float32)
    return audio_mono.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description='Resynthesize a WAV using PolyMAX + pedalboard')
    ap.add_argument('--audio', '-a', required=True, help='Input audio file')
    ap.add_argument('--model', '-m', default='outputs_optimized/models/vae_mel_l1_0_optimized_gated_cnn_mlp.model',
                   help='Trained model path')
    ap.add_argument('--outdir', '-o', default='resynth_polymax', help='Output directory')
    ap.add_argument('--midi', type=int, help='Override MIDI note')
    ap.add_argument('--vel', type=int, default=100, help='MIDI velocity')
    ap.add_argument('--no-note-search', action='store_true', help='Disable MIDI note search')
    ap.add_argument('--optimize', action='store_true', help='Run parameter optimization on a short segment')
    ap.add_argument('--target-acc', type=float, default=0.8, help='Target spectrogram-cosine accuracy')
    ap.add_argument('--max-opt-secs', type=float, default=2.5, help='Max audio seconds for optimization segment')
    ap.add_argument('--max-iters', type=int, default=25, help='Max optimization iterations (DE)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load input audio
    y, sr = sf.read(args.audio, always_2d=True)
    y_mono = y.mean(axis=1).astype(np.float32)
    duration = len(y_mono) / float(sr)

    # Predict parameters
    predictor = PolyMAXPredictor(args.model, device='auto')
    pred = predictor.predict(args.audio)
    params_map = pred['parameters']  # name->float in [0,1]

    # Determine MIDI note (initial guess)
    midi_note = args.midi if args.midi is not None else estimate_midi_note(y_mono, sr)

    # Prepare engine once
    engine, _, rev_idx = prepare_polymax_engine(params_map, '/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3')

    # Optional short search over MIDI notes to maximize spectrogram similarity
    if not args.no_note_search and args.midi is None:
        search_len = float(min(duration, 3.0))
        cfg = GLConfig()
        best_acc = -1.0
        best_note = midi_note
        # Candidates: around estimate ±12, and a coarse octave set
        candidates = sorted(set([midi_note + d for d in range(-12, 13, 2)] + [36, 48, 60, 72]))
        for n in candidates:
            if n < 24 or n > 96:
                continue
            y_pred = render_with_engine(engine, search_len, n, args.vel)
            # Match SR
            if 44100 != sr:
                y_pred = librosa.resample(y_pred, orig_sr=44100, target_sr=sr)
            # Compare on the same slice
            y_ref = y_mono[: len(y_pred)]
            m = compute_metrics(y_ref, y_pred, sr, cfg)
            acc = m.get('accuracy', 0.0)
            if acc > best_acc:
                best_acc, best_note = acc, n
        midi_note = best_note

    # Optional parameter optimization on a short segment
    cfg = GLConfig()
    if args.optimize:
        try:
            from scipy.optimize import differential_evolution
            # Choose a subset of parameters to optimize (timbrally impactful)
            schema = json.load(open(str(Path(__file__).parent.parent / 'params_schema.json'), 'r'))
            schema_names: List[str] = list(schema.get('parameter_order', []))
            candidates = [
                'osc_1_shape','osc_2_shape','osc_1_volume','osc_2_volume','noise_volume',
                'filter_cutoff_freq','filter_resonance','filter_env_amt',
                'amp_env_attack','amp_env_decay','amp_env_sustain','amp_env_release',
                'unison','mod_fx_depth','space_fx_amount'
            ]
            opt_names = [n for n in candidates if n in schema_names]
            # Vectorize initial values
            x0 = np.array([float(params_map.get(n, 0.5)) for n in opt_names], dtype=float)
            # Segment for evaluation
            seg_len = float(min(duration, max(1.5, args.max_opt_secs)))
            y_ref = y_mono[: int(sr * seg_len)]
            # Objective: minimize (1 - accuracy)
            def _eval(vec):
                vec = np.clip(np.asarray(vec, dtype=float), 0.0, 1.0)
                cur = params_map.copy()
                for i, n in enumerate(opt_names):
                    cur[n] = float(vec[i])
                # Apply and render
                # Re-set patch each evaluation (engine persists)
                engine.set_patch(midiname2num(cur, rev_idx))
                y_pred = render_with_engine(engine, seg_len, midi_note, args.vel)
                # Resample to input sr
                if 44100 != sr:
                    y_pred = librosa.resample(y_pred, orig_sr=44100, target_sr=sr)
                y_pred = y_pred[: len(y_ref)]
                m = compute_metrics(y_ref, y_pred, sr, cfg)
                return 1.0 - float(m.get('accuracy', 0.0))
            bounds = [(0.0, 1.0)] * len(x0)
            if len(bounds) > 0:
                res = differential_evolution(_eval, bounds, maxiter=args.max_iters, popsize=8, polish=True, tol=1e-3, seed=42)
                x_best = np.clip(res.x, 0.0, 1.0)
                for i, n in enumerate(opt_names):
                    params_map[n] = float(x_best[i])
        except Exception as e:
            print(f"[WARN] Optimization skipped: {e}")

    # Render full duration with chosen note and (optionally) optimized params
    # Re-apply final params
    engine.set_patch(midiname2num(params_map, rev_idx))
    # Try detected note sequence for more realistic rendering
    seq = detect_note_sequence(y_mono, sr)
    if len(seq) > 1:
        try:
            from pedalboard import MIDIMessage as _MM
            HAVE_MM = True
        except Exception:
            HAVE_MM = False
            _MM = None
        midi_messages = []
        for (t0, t1, n) in seq:
            if HAVE_MM:
                midi_messages.append(_MM.note_on(note=int(n), velocity=int(args.vel), time=float(t0)))
                midi_messages.append(_MM.note_off(note=int(n), velocity=int(args.vel), time=float(max(t0, t1 - 0.02))))
            else:
                class _CompatMidi:
                    def __init__(self, status, note, velocity, time):
                        self._data = bytes([status, note, velocity])
                        self.time = time
                    def bytes(self):
                        return self._data
                midi_messages.append(_CompatMidi(0x90, int(n), int(args.vel), float(t0)))
                midi_messages.append(_CompatMidi(0x80, int(n), int(args.vel), float(max(t0, t1 - 0.02))))
        # Use new MIDI rendering API on the engine
        engine.render_midi(midi_messages, duration, warm_up=False)
        audio = engine.get_audio_frames()
        audio_pm = audio.mean(axis=0).astype(np.float32)
    else:
        audio_pm = render_with_engine(engine, duration, midi_note, args.vel)
    sr_pm = 44100

    # Resample to match input SR for comparison
    if sr_pm != sr:
        audio_pm = librosa.resample(audio_pm, orig_sr=sr_pm, target_sr=sr)

    base = os.path.splitext(os.path.basename(args.audio))[0]
    out_wav = os.path.join(args.outdir, f"{base}_polymax.wav")
    sf.write(out_wav, audio_pm, sr)

    # Evaluate similarity using spectrogram cosine (same config as GL tool)
    cfg = GLConfig()
    metrics = compute_metrics(y_mono, audio_pm, sr, cfg)

    # Save report
    report = {
        'input': args.audio,
        'model': args.model,
        'midi_note': midi_note,
        'velocity': args.vel,
        'output_wav': out_wav,
        'metrics': metrics,
        'predict_confidence': pred.get('confidence', None),
    }
    out_json = os.path.join(args.outdir, f"{base}_polymax_metrics.json")
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps({
        'output': out_wav,
        'accuracy': metrics.get('accuracy', 0.0),
        'metrics_file': out_json,
        'midi_note': midi_note,
    }, indent=2))


if __name__ == '__main__':
    main()
