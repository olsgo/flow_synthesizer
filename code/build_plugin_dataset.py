#!/usr/bin/env python3
"""
Build a training dataset from any AU/VST3 instrument plugin using DawDreamer.

This script renders N randomized patches and writes a dataset compatible with
the existing training pipeline in this repo:

out_dir/
  raw/*.npz           # {'audio': float32[nsamples], 'param': dict{name->0..1}, 'chars': int32[K,3]}
  mel/*.npy           # 64 x 80 mel spectrograms
  mfcc/*.npy          # 64 x 80 MFCC matrices
  params.txt          # Ordered list of parameter names used by the dataset
  parameter_index_map.json  # name -> plugin parameter index

Usage:
  python code/build_plugin_dataset.py \
    --plugin "/Library/Audio/Plug-Ins/Components/ZENOLOGY.component" \
    --out datasets/zenology \
    --count 2000 \
    --keep 64

Notes:
  - Prefer VST3 if available (e.g., /Library/Audio/Plug-Ins/VST3/ZENOLOGY.vst3)
  - DawDreamer expects normalized parameters in [0, 1].
  - Some plugins expose non-automatable or UI-only params; we filter by name
    and keep the first K usable parameters by default.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from dd_renderer import DDRenderer


def ensure_dirs(base: Path, save_state: bool = False):
    (base / "raw").mkdir(parents=True, exist_ok=True)
    (base / "mel").mkdir(parents=True, exist_ok=True)
    (base / "mfcc").mkdir(parents=True, exist_ok=True)
    if save_state:
        (base / "state").mkdir(parents=True, exist_ok=True)


def compute_features(audio: np.ndarray, sr: int = 22050):
    # Mel: 64 x 80 (match repository conventions)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=2048, n_mels=64, hop_length=1024, fmin=30, fmax=11000
    )
    mel = mel[:64, :80]
    # MFCC: 64 x 80 (reshape from 16 x 320)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=16, hop_length=256)
    mfcc = mfcc[:16, :320].reshape(64, 80)
    return mel.astype(np.float32), mfcc.astype(np.float32)


def pick_parameters(param_desc, keep: int | None, use_filter: bool = True):
    names = []
    for i, d in enumerate(param_desc):
        name = d.get("name", f"Param {i}")
        # Heuristic filter: discard obvious by/bypass/display-only names
        if use_filter:
            lname = name.lower()
            if any(k in lname for k in ["bypass", "ui", "display", "about"]) :
                continue
        names.append(name)
    if not names:
        # Fallback: keep positional names
        names = [d.get("name", f"Param {i}") for i, d in enumerate(param_desc)]
    if keep is not None and keep > 0:
        names = names[:keep]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True, help="Path to AU/VST3 instrument")
    ap.add_argument("--out", required=True, help="Output dataset directory")
    ap.add_argument("--count", type=int, default=1000, help="Number of examples to render in this run")
    ap.add_argument("--start", type=int, default=0, help="Starting index for filenames (enables chunked/resumable runs)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip items whose raw/mel/mfcc already exist")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed for parameter sampling")
    ap.add_argument("--keep", type=int, default=64, help="Number of parameters to randomize (0 = all)")
    ap.add_argument("--full_params", action="store_true", help="Store the full parameter vector (all params) in each sample")
    ap.add_argument("--save_state", action="store_true", help="Save plugin state file for each rendered example")
    ap.add_argument("--no_filter", action="store_true", help="Do not filter parameter names (include all)")
    ap.add_argument("--sample_rate", type=int, default=22050)
    ap.add_argument("--note", type=int, default=60)
    ap.add_argument("--velocity", type=int, default=110)
    ap.add_argument("--note_len", type=float, default=3.0)
    ap.add_argument("--render_len", type=float, default=4.0)
    args = ap.parse_args()

    out = Path(args.out)
    ensure_dirs(out, save_state=args.save_state)

    # Renderer
    R = DDRenderer(sample_rate=args.sample_rate, block_size=512)
    R.load_plugin(args.plugin)
    pdesc = R.get_parameters_description()
    # Build mappings
    idx_to_name = [d.get("name", str(i)) for i, d in enumerate(pdesc)]
    name_to_index = {idx_to_name[i]: i for i in range(len(idx_to_name))}
    (out / "parameter_index_map.json").write_text(json.dumps(name_to_index, indent=2))

    use_param_names = pick_parameters(pdesc, keep=args.keep, use_filter=(not args.no_filter))
    (out / "params.txt").write_text("\n".join(use_param_names) + "\n")

    rng = np.random.default_rng(args.seed)

    for n in range(args.count):
        idx = args.start + n
        stem = f"ex_{idx:06d}"
        raw_p = out / "raw" / f"{stem}.npz"
        mel_p = out / "mel" / f"{stem}.npy"
        mfcc_p = out / "mfcc" / f"{stem}.npy"

        if args.skip_existing and raw_p.exists() and mel_p.exists() and mfcc_p.exists():
            if (n + 1) % 100 == 0 or n == args.count - 1:
                print(f"Skipped/rendered {n+1}/{args.count}")
            continue

        # Random patch over the selected parameters (uniform 0..1)
        patch = []
        pvals = {}
        for name in use_param_names:
            idxp = name_to_index[name]
            val = float(rng.random())
            pvals[name] = val
            patch.append((idxp, val))

        # Render audio
        R.set_patch(patch)
        audio = R.render_patch(
            midi_note=args.note,
            velocity=args.velocity,
            note_len_sec=args.note_len,
            render_len_sec=args.render_len,
        )
        # DawDreamer returns (channels, samples)
        audio_mono = np.mean(audio, axis=0).astype(np.float32)

        # Optionally capture full parameter vector after setting the patch
        full_param_dict = None
        if args.full_params:
            full_patch = R.get_patch()  # list of (index, value)
            full_param_dict = {idx_to_name[i]: float(v) for (i, v) in full_patch}
            # If we want the dataset to learn all parameters, override pvals
            pvals = full_param_dict

        # Persist raw npz with params
        np.savez(
            raw_p,
            audio=audio_mono,
            param=pvals,
            chars=np.zeros((1, 3), dtype=np.int32),
            param_full=full_param_dict if full_param_dict is not None else None,
            state_path=str((out / "state" / f"{stem}.state")) if args.save_state else "",
        )

        # Optionally save the plugin binary state for exact recall
        if args.save_state:
            try:
                R.save_state(str(out / "state" / f"{stem}.state"))
            except Exception:
                pass

        # Features to match training pipeline convenience
        mel, mfcc = compute_features(audio_mono, sr=args.sample_rate)
        np.save(mel_p, mel)
        np.save(mfcc_p, mfcc)

        # Optional WAV dump for quick inspection (first few only in this chunk)
        if n < 8 and args.start == 0:
            try:
                sf.write(str(out / f"{stem}.wav"), audio_mono, args.sample_rate)
            except Exception:
                pass

        if (n + 1) % 100 == 0 or n == args.count - 1:
            print(f"Rendered {n+1}/{args.count}")

    print("\nDone. Dataset folders created at:")
    print(f"  {out}")
    print("To train: set --path to parent dir and --dataset to this folder name.")


if __name__ == "__main__":
    main()
