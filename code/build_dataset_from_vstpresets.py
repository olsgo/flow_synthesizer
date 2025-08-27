#!/usr/bin/env python3
"""
Build a dataset by loading .vstpreset files (VST3) for an instrument like ZENOLOGY.

This approach uses DawDreamer's load_vst3_preset, which is the recommended
programmatic way to change presets for VST3 plugins.

Output structure aligns with the repo's training pipeline.

Usage example:
  python code/build_dataset_from_vstpresets.py \
    --plugin "/Library/Audio/Plug-Ins/VST3/ZENOLOGY.vst3" \
    --presets_dir "/Library/Audio/Presets/Roland/ZENOLOGY" \
    --out datasets/zenology_vstpresets \
    --sample_rate 22050 --note 60 --velocity 110 --note_len 2.0 --render_len 4.0 \
    --skip_silence 0.0005
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
    (base / "state").mkdir(parents=True, exist_ok=True)
    (base / "audio").mkdir(parents=True, exist_ok=True)


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


def find_vstpresets(root: Path) -> list[Path]:
    return sorted(root.rglob("*.vstpreset"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True, help="Path to VST3 instrument")
    ap.add_argument("--presets_dir", required=False, help="Directory containing .vstpreset files (search recursively)")
    ap.add_argument("--out", required=True, help="Output dataset directory")
    ap.add_argument("--sample_rate", type=int, default=22050)
    ap.add_argument("--note", type=int, default=60)
    ap.add_argument("--velocity", type=int, default=110)
    ap.add_argument("--note_len", type=float, default=2.0)
    ap.add_argument("--render_len", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_silence", type=float, default=0.0, help="If peak < threshold, skip saving the sample")
    ap.add_argument("--save_wav_first", type=int, default=8, help="Write WAV for first N presets for quick inspection")
    args = ap.parse_args()

    out = Path(args.out)
    ensure_dirs(out)

    # Renderer
    R = DDRenderer(sample_rate=args.sample_rate, block_size=512)
    R.load_plugin(args.plugin)
    pdesc = R.get_parameters_description()
    idx_to_name = [d.get("name", str(i)) for i, d in enumerate(pdesc)]
    name_to_index = {idx_to_name[i]: i for i in range(len(idx_to_name))}
    (out / "parameter_index_map.json").write_text(json.dumps(name_to_index, indent=2))
    (out / "params.txt").write_text("\n".join(idx_to_name) + "\n")

    # Presets
    candidates = []
    if args.presets_dir:
        candidates = find_vstpresets(Path(args.presets_dir))
    else:
        # Try common locations
        common = [
            Path.home() / "Library/Audio/Presets",
            Path("/Library/Audio/Presets"),
        ]
        for root in common:
            if root.exists():
                candidates.extend(find_vstpresets(root))
        # Filter for ZENOLOGY
        candidates = [p for p in candidates if "ZENOLOGY" in p.as_posix()]

    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("No .vstpreset files found. Provide --presets_dir or install factory/user presets.")

    for n, preset in enumerate(candidates):
        stem = f"ex_{n:06d}"
        raw_p = out / "raw" / f"{stem}.npz"
        mel_p = out / "mel" / f"{stem}.npy"
        mfcc_p = out / "mfcc" / f"{stem}.npy"
        state_p = out / "state" / f"{stem}.state"
        wav_p = out / "audio" / f"{stem}.wav"

        try:
            R.load_vst3_preset(str(preset))
        except Exception as e:
            print(f"Failed to load preset {preset}: {e}")
            continue

        # Render audio
        audio = R.render_patch(
            midi_note=args.note,
            velocity=args.velocity,
            note_len_sec=args.note_len,
            render_len_sec=args.render_len,
        )
        mono = np.mean(audio, axis=0).astype(np.float32)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if args.skip_silence and peak < args.skip_silence:
            if (n + 1) % 20 == 0:
                print(f"Skipped {n+1}/{len(candidates)} (silent) -> {preset.name}")
            continue

        # Persist state for exact recall and the full parameter vector
        try:
            R.save_state(str(state_p))
        except Exception:
            pass
        full_patch = R.get_patch()
        param_dict = {idx_to_name[i]: float(v) for (i, v) in full_patch}

        # Save raw + features
        np.savez(
            raw_p,
            audio=mono,
            param=param_dict,
            chars=np.zeros((1, 3), dtype=np.int32),
            preset_path=str(preset),
            state_path=str(state_p),
        )
        mel, mfcc = compute_features(mono, sr=args.sample_rate)
        np.save(mel_p, mel)
        np.save(mfcc_p, mfcc)
        if n < args.save_wav_first:
            try:
                sf.write(str(wav_p), mono, args.sample_rate)
            except Exception:
                pass

        if (n + 1) % 50 == 0 or n == len(candidates) - 1:
            print(f"Rendered {n+1}/{len(candidates)}  peak={peak:.3f}  src={preset.name}")

    print("\nDone. Dataset folders created at:")
    print(f"  {out}")
    print("To train: set --path to parent dir and --dataset to this folder name.")


if __name__ == "__main__":
    main()

