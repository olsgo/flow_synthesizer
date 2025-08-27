#!/usr/bin/env python3
"""
Build a dataset from saved plugin state files (.dwstate) produced by DawDreamer.

This is useful for enumerating factory presets (e.g., via bank/program MIDI),
saving each state, and then rendering audio/features following this repo's
dataset conventions.

Output structure:
  out_dir/
    raw/*.npz           # {'audio': float32[nsamples], 'param': dict{name->0..1}, 'chars': int32[K,3], 'state_path': str}
    mel/*.npy           # 64 x 80 mel spectrogram
    mfcc/*.npy          # 64 x 80 MFCC
    params.txt          # Parameter names used (full vector)
    parameter_index_map.json  # name -> plugin parameter index

Usage example:
  python code/build_dataset_from_states.py \
    --plugin "/Library/Audio/Plug-Ins/Components/ZENOLOGY.component" \
    --state_dir datasets/zenology_preset_states \
    --out datasets/zenology_from_presets \
    --note 60 --velocity 110 --note_len 2.0 --render_len 3.0
"""

import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

from dd_renderer import DDRenderer


def ensure_dirs(base: Path):
    (base / "raw").mkdir(parents=True, exist_ok=True)
    (base / "mel").mkdir(parents=True, exist_ok=True)
    (base / "mfcc").mkdir(parents=True, exist_ok=True)


def _pad_crop(mat: np.ndarray, target_cols: int) -> np.ndarray:
    C, T = mat.shape
    if T >= target_cols:
        return mat[:, :target_cols]
    out = np.zeros((C, target_cols), dtype=mat.dtype)
    out[:, :T] = mat
    return out


def compute_features(audio: np.ndarray, sr: int = 22050):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=2048, n_mels=64, hop_length=1024, fmin=30, fmax=11000
    )
    mel = _pad_crop(mel[:64, :], 80)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=16, hop_length=256)
    mfcc = _pad_crop(mfcc[:16, :], 320).reshape(64, 80)
    return mel.astype(np.float32), mfcc.astype(np.float32)


def list_state_files(state_dir: Path):
    return sorted([p for p in state_dir.glob("*.dwstate")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True, help="Path to AU/VST3 instrument")
    ap.add_argument("--state_dir", required=True, help="Directory containing .dwstate files")
    ap.add_argument("--out", required=True, help="Output dataset directory")
    ap.add_argument("--sample_rate", type=int, default=22050)
    ap.add_argument("--note", type=int, default=60)
    ap.add_argument("--velocity", type=int, default=110)
    ap.add_argument("--note_len", type=float, default=2.0)
    ap.add_argument("--render_len", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of states (0 = all)")
    ap.add_argument("--skip_silence", type=float, default=0.0, help="If peak < threshold, skip saving the sample")
    ap.add_argument("--open_editor", action="store_true", help="Open plugin editor before rendering (stability workaround)")
    args = ap.parse_args()

    out = Path(args.out)
    ensure_dirs(out)

    states = list_state_files(Path(args.state_dir))
    if args.limit and args.limit > 0:
        states = states[: args.limit]
    if not states:
        raise SystemExit(f"No .dwstate files found in {args.state_dir}")

    # Renderer
    R = DDRenderer(sample_rate=args.sample_rate, block_size=512)
    R.load_plugin(args.plugin)
    if args.open_editor and getattr(R, 'inst', None) is not None:
        try:
            R.inst.open_editor("PluginEditor")
        except Exception:
            pass
    pdesc = R.get_parameters_description()
    idx_to_name = [d.get("name", str(i)) for i, d in enumerate(pdesc)]
    name_to_index = {idx_to_name[i]: i for i in range(len(idx_to_name))}
    (out / "parameter_index_map.json").write_text(json.dumps(name_to_index, indent=2))
    (out / "params.txt").write_text("\n".join(idx_to_name) + "\n")

    for n, state_path in enumerate(states):
        stem = f"ex_{n:06d}"
        raw_p = out / "raw" / f"{stem}.npz"
        mel_p = out / "mel" / f"{stem}.npy"
        mfcc_p = out / "mfcc" / f"{stem}.npy"

        # Load plugin state and get full param vector after load
        R.load_state(str(state_path))
        full_patch = R.get_patch()  # list[(index, value)]
        param_dict = {idx_to_name[i]: float(v) for (i, v) in full_patch}

        # Render audio
        audio = R.render_patch(
            midi_note=args.note,
            velocity=args.velocity,
            note_len_sec=args.note_len,
            render_len_sec=args.render_len,
        )
        mono = np.mean(audio, axis=0).astype(np.float32)

        # Optional silence check
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if args.skip_silence and peak < args.skip_silence:
            if (n + 1) % 10 == 0:
                print(f"Skipped {n+1}/{len(states)} (silent: peak={peak:.5f})")
            continue

        # Save raw + metadata
        np.savez(
            raw_p,
            audio=mono,
            param=param_dict,
            chars=np.zeros((1, 3), dtype=np.int32),
            state_path=str(state_path),
        )

        # Features
        mel, mfcc = compute_features(mono, sr=args.sample_rate)
        np.save(mel_p, mel)
        np.save(mfcc_p, mfcc)

        if (n + 1) % 50 == 0 or n == len(states) - 1:
            print(f"Rendered {n+1}/{len(states)}  ({state_path.name})  peak={peak:.3f}")

    print("\nDone. Dataset folders created at:")
    print(f"  {out}")
    print("To train: set --path to parent dir and --dataset to this folder name.")


if __name__ == "__main__":
    main()
