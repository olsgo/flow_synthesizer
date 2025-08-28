#!/usr/bin/env python3
"""Compute log-mel feature statistics for each audio file in the manifest.

This script follows the "Option B" recipe from the PolyMAX Resynth plan: for
32 kHz audio, compute a log-mel spectrogram with 96 bins and derive global
statistics (mean, std, p10, p50, p90) for each mel band. Additionally, compute
spectral centroid, spectral rolloff, and RMS energy. The resulting feature
vector has 483 dimensions.

The script expects a manifest CSV with columns: split,audio_path,params_path,
preset_name. Features are written as .npy files to the specified directory using
the audio file stem as the filename.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import librosa

SR = 32_000
N_MELS = 96
N_FFT = int(0.025 * SR)  # 25 ms
HOP_LENGTH = int(0.010 * SR)  # 10 ms

def feature_from_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y, sr=SR, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel = librosa.power_to_db(mel + 1e-10, ref=1.0)
    stats = [
        mel.mean(axis=1),
        mel.std(axis=1),
        np.percentile(mel, 10, axis=1),
        np.percentile(mel, 50, axis=1),
        np.percentile(mel, 90, axis=1),
    ]
    centroid = librosa.feature.spectral_centroid(S=mel, sr=SR).mean(axis=1)
    rolloff = librosa.feature.spectral_rolloff(S=mel, sr=SR).mean(axis=1)
    rms = librosa.feature.rms(S=librosa.db_to_power(mel)).mean(axis=1)
    feat = np.concatenate(stats + [centroid, rolloff, rms])
    assert feat.shape[0] == 483, f"Unexpected feature dim: {feat.shape[0]}"
    return feat.astype(np.float32)

def main(manifest, features_dir):
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = Path(row["audio_path"])
            feat_path = features_dir / (audio_path.stem + ".npy")
            if feat_path.exists():
                continue
            feat = feature_from_audio(audio_path)
            np.save(feat_path, feat)
            print("wrote", feat_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features_dir", default="data/features")
    args = ap.parse_args()
    main(args.manifest, args.features_dir)
